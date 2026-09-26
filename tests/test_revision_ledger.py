from scripts.resolve_experiment_revisions import (
    collect_dataset_specs,
    collect_model_specs,
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
