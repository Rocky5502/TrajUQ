from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rq4_registry_builder_preserves_mechanical_firewall():
    text = Path("scripts/build_rq4_scientific_claim_registry.py").read_text(encoding="utf-8")
    assert 'mechanical.get("claim_authorized") is not False' in text
    assert '"mechanical_manifest_claim_authorized": False' in text
    assert '"claim_authorized": True' in text
    assert "downstream-plan hash drift" in text


def test_rq5_census_detects_nul_and_has_deterministic_tree_binding(tmp_path):
    module = load("scripts/build_rq5_evidence_census.py", "rq5_census")
    nul = tmp_path / "manifest.json"
    nul.write_bytes(b"\x00" * 11)
    assert module.all_nul(nul)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b.json").write_text("b", encoding="utf-8")
    (tree / "a.json").write_text("a", encoding="utf-8")
    first = module.tree_binding(tree, ("*.json",))
    second = module.tree_binding(tree, ("*.json",))
    assert first == second
    expected = hashlib.sha256()
    for name, value in (("a.json", "a"), ("b.json", "b")):
        digest = hashlib.sha256(value.encode()).hexdigest()
        expected.update(f"{name}\0{digest}\0{1}\n".encode())
    assert first["tree_sha256"] == expected.hexdigest()


def test_closure_outputs_are_non_overwriting():
    for script in (
        "scripts/build_rq4_scientific_claim_registry.py",
        "scripts/build_rq5_evidence_census.py",
    ):
        text = Path(script).read_text(encoding="utf-8")
        assert 'with path.open("xb")' in text
