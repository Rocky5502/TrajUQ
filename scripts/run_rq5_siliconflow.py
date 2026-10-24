#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import yaml


LOCK = threading.Lock()
NAMESPACE = uuid.UUID("6ddf4f6d-50dc-4a38-94d3-94ace947ff7c")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_secret(path: Path) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        key, value = raw.split("=", 1)
        if key.strip() == "SILICONFLOW_API_KEY" and value.strip():
            return value.strip().strip('"').strip("'")
    raise RuntimeError("SILICONFLOW_API_KEY is unavailable")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_content(content: str) -> tuple[str | None, int | None]:
    answer = re.search(r"(?im)^FINAL_ANSWER:\s*(.+?)\s*$", content)
    confidence = re.search(r"(?im)^CONFIDENCE:\s*(\d{1,3})\s*$", content)
    value = int(confidence.group(1)) if confidence else None
    if value is not None and not 0 <= value <= 100:
        value = None
    return (answer.group(1).strip() if answer else None, value)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        self.freeze_path = Path(args.freeze)
        self.freeze = json.loads(self.freeze_path.read_text(encoding="utf-8"))
        self.freeze_sha = sha256_bytes(self.freeze_path.read_bytes())
        expected_sha = self.freeze_path.with_suffix(self.freeze_path.suffix + ".sha256").read_text().strip()
        if self.freeze["status"] != "PASS" or self.freeze_sha != expected_sha:
            raise RuntimeError("RQ5 freeze evidence is not PASS or its hash drifted")
        authorization_path = Path("configs/operations/rq5_budget_authorization_af0347a.yaml")
        authorization = yaml.safe_load(authorization_path.read_text(encoding="utf-8"))
        if authorization["freeze_sha256"] != self.freeze_sha:
            raise RuntimeError("Operational budget authorization is not bound to this freeze")
        self.budget_authorization_sha256 = sha256_bytes(authorization_path.read_bytes())
        self.preferred_target = float(authorization["preferred_target_rmb"])
        self.warning_threshold = float(authorization["warning_threshold_rmb"])
        self.absolute_hard_cap = float(authorization["absolute_hard_stop_rmb"])
        self.token = read_secret(Path(args.secret))
        pricing = json.loads(Path(self.config["provider"]["pricing_contract"]).read_text(encoding="utf-8"))
        self.prices = {row["model_id"]: {item["specification"]: float(item["price"]) for item in row["pricing"]} for row in pricing["models"]}
        self.output = Path(args.output_dir)
        self.worker_id = str(args.worker_id)
        self.status_path = self.output / f"campaign_status.{self.worker_id}.json"
        self.cost = 0.0
        for path in self.output.glob("records/*/*/*/*.json"):
            try:
                self.cost += float(json.loads(path.read_text(encoding="utf-8")).get("estimated_cost_rmb", 0.0))
            except (OSError, ValueError):
                pass

    def request(self, model: str, dataset: str, split: str, example_id: str, repeat: int, messages: list[dict], smoke: bool) -> dict:
        request_uuid = str(uuid.uuid5(NAMESPACE, f"{self.freeze_sha}|{model}|{dataset}|{split}|{example_id}|{repeat}"))
        safe_model = model.replace("/", "__")
        root = self.output / ("smoke" if smoke else "records") / safe_model / dataset / split
        result_path = root / f"{request_uuid}.json"
        pending_path = root / f"{request_uuid}.pending"
        error_path = self.output / "errors" / safe_model / dataset / split / f"{request_uuid}.json"
        if result_path.exists():
            return {"state": "SKIP_SUCCESS", "path": str(result_path)}
        if error_path.exists():
            return {"state": "SKIP_TERMINAL_ERROR", "path": str(error_path)}
        if pending_path.exists():
            return {"state": "AMBIGUOUS_MISSING_NOT_RESUBMITTED", "path": str(pending_path)}
        with LOCK:
            if self.cost >= self.absolute_hard_cap:
                raise RuntimeError("RQ5 absolute hard cost stop reached")
        root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(pending_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, (json.dumps({"request_uuid": request_uuid, "created_utc": datetime.now(timezone.utc).isoformat()}) + "\n").encode())
            os.close(descriptor)
        except FileExistsError:
            return {"state": "AMBIGUOUS_MISSING_NOT_RESUBMITTED", "path": str(pending_path)}

        decoding = self.config["decoding"]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "max_tokens": decoding["max_tokens"],
            "enable_thinking": decoding["enable_thinking"],
            "thinking_budget": decoding["thinking_budget"],
            "stream": False,
        }
        body = json.dumps(payload).encode()
        response_json: dict = {}
        status = 0
        trace_id = None
        latency = 0.0
        retry_number = 0
        try:
            for retry_number in range(int(self.config["execution"]["max_retries"]) + 1):
                started = time.perf_counter()
                request = urllib.request.Request(
                    self.config["provider"]["endpoint"],
                    data=body,
                    method="POST",
                    headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "X-Trace-Id": request_uuid},
                )
                try:
                    with urllib.request.urlopen(request, timeout=int(self.config["execution"]["timeout_seconds"])) as response:
                        status = int(response.status)
                        trace_id = response.headers.get("x-request-id") or response.headers.get("x-trace-id")
                        response_json = json.loads(response.read())
                    latency = time.perf_counter() - started
                    break
                except urllib.error.HTTPError as exc:
                    status = int(exc.code)
                    latency = time.perf_counter() - started
                    if status not in self.config["execution"]["retry_http_statuses"] or retry_number >= int(self.config["execution"]["max_retries"]):
                        raise
                    time.sleep(float(self.config["execution"]["retry_backoff_seconds"][retry_number]))

            choice = response_json.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = str(message.get("content") or "")
            answer, confidence = parse_content(content)
            usage = response_json.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            price = self.prices[model]
            estimated_cost = input_tokens / 1_000_000 * price["prompt"] + output_tokens / 1_000_000 * price["completion"]
            with LOCK:
                projected = self.cost + estimated_cost
                self.cost = projected
            record = {
                "freeze_sha256": self.freeze_sha,
                "request_uuid": request_uuid,
                "requested_model": model,
                "returned_model": response_json.get("model"),
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": self.config["provider"]["endpoint"],
                "prompt_template_sha256": self.freeze["prompt_template_sha256"],
                "dataset": dataset,
                "split": split,
                "example_id": example_id,
                "repeat_index": repeat,
                "decoding": decoding,
                "raw_provider_response": response_json,
                "parsed_final_answer": answer,
                "verbalized_confidence": confidence,
                "reasoning_content": message.get("reasoning_content"),
                "usage": usage,
                "finish_reason": choice.get("finish_reason"),
                "http_status": status,
                "latency_seconds": latency,
                "retry_number": retry_number,
                "provider_trace_id": trace_id,
                "system_fingerprint": response_json.get("system_fingerprint"),
                "estimated_cost_rmb": estimated_cost,
                "budget_authorization_sha256": self.budget_authorization_sha256,
                "transport_worker_id": self.worker_id,
            }
            atomic_json(result_path, record)
            pending_path.unlink()
            return {"state": "PASS", "path": str(result_path), "cost": estimated_cost}
        except Exception as exc:
            confirmed_provider_error = isinstance(exc, urllib.error.HTTPError)
            error = {
                "schema": "trajuq_rq5_request_error_v1",
                "freeze_sha256": self.freeze_sha,
                "request_uuid": request_uuid,
                "requested_model": model,
                "dataset": dataset,
                "split": split,
                "example_id": example_id,
                "repeat_index": repeat,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "http_status": status,
                "retry_number": retry_number,
                "confirmed_provider_error": confirmed_provider_error,
                "will_resubmit": False,
            }
            atomic_json(error_path, error)
            if confirmed_provider_error and pending_path.exists():
                pending_path.unlink()
                return {"state": "TERMINAL_PROVIDER_ERROR", "path": str(error_path)}
            # Network/process uncertainty retains the marker and is an explicit missing request.
            return {"state": "AMBIGUOUS_MISSING_NOT_RESUBMITTED", "path": str(pending_path)}

    def smoke(self) -> None:
        prompt = "What is 2 + 2? End with exactly: FINAL_ANSWER: <answer> and CONFIDENCE: <integer from 0 to 100>."
        messages = [{"role": "system", "content": self.config["prompt"]["system"]}, {"role": "user", "content": prompt}]
        results = [self.request(model, "synthetic", "smoke", "synthetic:2plus2", 0, messages, True) for model in self.config["models"]]
        manifest = {"schema": "trajuq_rq5_siliconflow_smoke_v1", "status": "PASS", "freeze_sha256": self.freeze_sha, "models": self.config["models"], "results": results, "benchmark_outcomes_observed": False}
        atomic_json(self.output / "smoke_manifest.json", manifest)
        print("PASS: three-model synthetic SiliconFlow smoke")

    def execute_model(self, model: str) -> None:
        system = self.config["prompt"]["system"]
        suffix = self.config["prompt"]["suffix"]
        for dataset in self.config["datasets"]:
            for split in ("calibration", "test"):
                path = Path(self.args.data_root) / dataset / f"{split}.jsonl"
                with path.open("r", encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                for row in rows:
                    messages = [{"role": "system", "content": system}, {"role": "user", "content": str(row["prompt"]) + suffix}]
                    for repeat in range(int(self.config["decoding"]["repeated_samples"])):
                        self.request(model, dataset, split, str(row["id"]), repeat, messages, False)
                        with LOCK:
                            atomic_json(self.status_path, {"schema": "trajuq_rq5_campaign_status_v1", "state": "RUNNING", "updated_utc": datetime.now(timezone.utc).isoformat(), "estimated_cost_rmb": self.cost, "preferred_target_rmb": self.preferred_target, "warning_threshold_rmb": self.warning_threshold, "absolute_hard_stop_rmb": self.absolute_hard_cap})

    def execute(self) -> None:
        smoke = json.loads((self.output / "smoke_manifest.json").read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS" or smoke.get("freeze_sha256") != self.freeze_sha:
            raise RuntimeError("Matching PASS smoke evidence is required before benchmark execution")
        with ThreadPoolExecutor(max_workers=len(self.config["models"]), thread_name_prefix="rq5-model") as pool:
            futures = [pool.submit(self.execute_model, model) for model in self.config["models"]]
            for future in futures:
                future.result()
        atomic_json(self.status_path, {"schema": "trajuq_rq5_campaign_status_v1", "state": "COMPLETE", "updated_utc": datetime.now(timezone.utc).isoformat(), "estimated_cost_rmb": self.cost, "preferred_target_rmb": self.preferred_target, "warning_threshold_rmb": self.warning_threshold, "absolute_hard_stop_rmb": self.absolute_hard_cap})


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotent SiliconFlow RQ5 executor.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--secret")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq5_blackbox.yaml")
    parser.add_argument("--freeze", default="results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output-dir", default="results/rq5/api")
    parser.add_argument("--worker-id", default="worker0")
    args = parser.parse_args()
    if not args.smoke and not args.execute:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        print(json.dumps({"mode": "DRY_RUN", "models": config["models"], "benchmark_requests_sent": 0}, indent=2))
        return
    if not args.secret:
        raise SystemExit("--secret is required for --smoke or --execute")
    runner = Runner(args)
    runner.smoke() if args.smoke else runner.execute()


if __name__ == "__main__":
    main()
