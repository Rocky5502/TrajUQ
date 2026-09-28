from types import SimpleNamespace

import pytest

from scripts.resolve_experiment_revisions import (
    collect_dataset_specs,
    collect_model_specs,
    is_retryable_remote_error,
    resolve_remote_sha,
)


def test_collect_dataset_specs_follows_primary_order():
    primary = {"datasets": ["a", "b"]}
    manifest = {
        "datasets": {
            "a": {"hf_id": "org/a", "revision": None},
            "b": {"hf_id": "org/b", "revision": "tag"},
        }
    }
    specs = collect_dataset_specs(primary, manifest)
    assert [item["name"] for item in specs] == ["a", "b"]
    assert specs[1]["requested_revision"] == "tag"


def test_collect_model_specs_reads_primary_templates(tmp_path):
    cfg = tmp_path / "llada.yaml"
    cfg.write_text(
        "adapter: llada\nmodel_id: org/model\nrevision: null\n",
        encoding="utf-8",
    )
    primary = {"models": {"primary": [str(cfg)], "diagnostic": []}}
    specs = collect_model_specs(primary)
    assert specs[0]["adapter"] == "llada"
    assert specs[0]["model_id"] == "org/model"
    assert specs[0]["requested_revision"] is None


def test_resolve_remote_sha_retries_transient_connect_error():
    class ConnectError(Exception):
        pass

    attempts = 0
    sleeps: list[float] = []

    def fetch():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred"
            )
        return SimpleNamespace(sha="abc123")

    resolved = resolve_remote_sha(
        fetch,
        label="model org/model",
        max_attempts=5,
        base_delay_seconds=0.5,
        sleep_fn=sleeps.append,
    )

    assert resolved == "abc123"
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_resolve_remote_sha_fails_fast_for_forbidden_http_error():
    class HfHubHTTPError(Exception):
        def __init__(self):
            super().__init__("forbidden")
            self.response = SimpleNamespace(status_code=403)

    attempts = 0

    def fetch():
        nonlocal attempts
        attempts += 1
        raise HfHubHTTPError()

    with pytest.raises(HfHubHTTPError):
        resolve_remote_sha(
            fetch,
            label="dataset gated/example",
            max_attempts=5,
            base_delay_seconds=0.0,
            sleep_fn=lambda _: None,
        )

    assert attempts == 1


def test_retryable_remote_error_recognizes_server_status():
    class HfHubHTTPError(Exception):
        def __init__(self):
            super().__init__("server error")
            self.response = SimpleNamespace(status_code=503)

    assert is_retryable_remote_error(HfHubHTTPError()) is True
