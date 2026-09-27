import sys
from types import SimpleNamespace

from scripts.preflight import (
    MIN_TRANSFORMERS_VERSION,
    probe_remote_assets,
    validate_remote_asset_contract,
    version_triplet,
)


def test_version_triplet_parses_release_versions():
    assert version_triplet("4.52.4") == (4, 52, 4)
    assert version_triplet("4.57.1") == (4, 57, 1)


def test_version_triplet_accepts_suffix_after_patch():
    assert version_triplet("4.52.4.dev0") == (4, 52, 4)
    assert version_triplet("4.52.4rc1") == (4, 52, 4)


def test_transformers_floor_contract():
    assert version_triplet("4.52.3") < MIN_TRANSFORMERS_VERSION
    assert version_triplet("4.52.4") >= MIN_TRANSFORMERS_VERSION
    assert version_triplet("4.53.0") >= MIN_TRANSFORMERS_VERSION


def test_version_triplet_rejects_unparseable_value():
    assert version_triplet("unknown") is None


def test_remote_asset_probe_uses_exact_revision(monkeypatch):
    calls = []

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("config", model_id, kwargs))
            return object()

    class FakeTokenizerInstance:
        mask_token_id = 42

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizerInstance()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeConfig,
            AutoTokenizer=FakeTokenizer,
        ),
    )

    payload, failures = probe_remote_assets(
        {
            "model_id": "example/model",
            "revision": "resolved-sha",
            "trust_remote_code": True,
        }
    )

    assert failures == []
    assert payload["attempted"] is True
    assert payload["config_loaded"] is True
    assert payload["tokenizer_loaded"] is True
    assert payload["mask_token_id"] == 42
    assert calls == [
        (
            "config",
            "example/model",
            {
                "revision": "resolved-sha",
                "trust_remote_code": True,
            },
        ),
        (
            "tokenizer",
            "example/model",
            {
                "revision": "resolved-sha",
                "trust_remote_code": True,
            },
        ),
    ]


def test_remote_asset_probe_fails_closed(monkeypatch):
    class BrokenConfig:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise RuntimeError("remote config unavailable")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=BrokenConfig,
            AutoTokenizer=FakeTokenizer,
        ),
    )

    payload, failures = probe_remote_assets(
        {
            "model_id": "example/model",
            "revision": "resolved-sha",
        }
    )

    assert payload["attempted"] is True
    assert payload["config_loaded"] is False
    assert failures
    assert "remote config unavailable" in failures[0]


def test_remote_asset_probe_skips_unpinned_exploratory_config():
    payload, failures = probe_remote_assets(
        {
            "model_id": "example/model",
            "revision": None,
        }
    )

    assert failures == []
    assert payload["attempted"] is False


def test_remote_asset_contract_requires_mask_when_declared():
    failures = validate_remote_asset_contract(
        {
            "validation": {
                "require_mask_token_id": True,
            }
        },
        {
            "attempted": True,
            "tokenizer_loaded": True,
            "mask_token_id": None,
        },
    )

    assert failures == [
        "frozen model validation requires tokenizer.mask_token_id"
    ]


def test_remote_asset_contract_accepts_declared_mask():
    assert validate_remote_asset_contract(
        {
            "validation": {
                "require_mask_token_id": True,
            }
        },
        {
            "attempted": True,
            "tokenizer_loaded": True,
            "mask_token_id": 126336,
        },
    ) == []
