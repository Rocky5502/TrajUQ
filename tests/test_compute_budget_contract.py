import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_registry_is_exact_or_explicitly_undisclosed():
    data = json.loads((ROOT / "configs/operations/model_parameter_registry.json").read_text(encoding="utf-8"))
    models = {row["key"]: row for row in data["models"]}
    assert models["llada"]["total_parameters"] == 7_356_880_896
    assert models["llada"]["active_parameters"] == 1_400_000_000
    assert models["sdar"]["total_parameters"] == 2_031_739_904
    assert models["dream"]["total_parameters"] == 7_615_616_512
    assert models["llama31"]["total_parameters"] == 8_030_261_248
    for key in ("deepseek_v32", "glm51", "kimi_k26"):
        assert models[key]["total_parameters"] is None
        assert models[key]["active_parameters"] is None
        assert "not disclosed" in models[key]["notes"]
    assert all(row["parameter_evidence"] for row in data["models"])


def test_manuscript_uses_generated_compute_macros():
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    appendix = (ROOT / "paper/appendix.tex").read_text(encoding="utf-8")
    assert r"\input{generated_compute}" in main
    assert r"\TotalAuthoritativeGPUHours" in main
    assert r"\RQFiveCostRMB" in main
    assert r"\ComputeBudgetTable" in main
    assert r"\ComputeBudgetAppendixTable" in appendix
    assert "Provider-side hardware" in main


def test_builder_defines_required_macros_and_does_not_use_utilization_as_multiplier():
    source = (ROOT / "scripts/build_compute_budget.py").read_text(encoding="utf-8")
    for macro in (
        "TotalPrimaryGPUHours", "TotalRQTwoThreeGPUHours", "TotalRQFourGPUHours",
        "TotalAuthoritativeGPUHours", "TotalDiagnosticGPUHours", "TotalProjectGPUHours",
        "RQFiveRequests", "RQFiveInputTokens", "RQFiveOutputTokens", "RQFiveCostRMB",
    ):
        assert macro in source
    assert "utilization is not applied" in source


def test_secret_material_is_not_persisted_by_capture_script():
    source = (ROOT / "scripts/capture_compute_environment.py").read_text(encoding="utf-8")
    remote_payload = source.split("def parse_credentials", 1)[0]
    assert '"password"' not in remote_payload
    assert "host_id_sha256" in source
