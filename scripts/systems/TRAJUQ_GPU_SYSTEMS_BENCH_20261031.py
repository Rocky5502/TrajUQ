from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

OUT = Path("/data/trajuq_gpu_systems_bench_20261031")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 2, 4, 8, 16]

WARMUP_REPEATS = 1
MEASURED_REPEATS = 5
NEW_TOKENS = 128

torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)

torch.backends.cuda.matmul.allow_tf32 = True

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "left"

print("Loading exact frozen model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
).to("cuda")

model.eval()

print("MODEL LOADED")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0))


def gpu_sample():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True).strip()

        p, u, m, t = [float(x.strip()) for x in out.split(",")]
        return {
            "power_w": p,
            "util_pct": u,
            "memory_mib": m,
            "temp_c": t,
        }
    except Exception:
        return None


def make_prompt(target_tokens: int):
    base = (
        "This is a synthetic inference-efficiency diagnostic. "
        "No benchmark answer or scientific outcome is being evaluated. "
        "Return a short neutral continuation. "
    )

    text = base
    while len(tok(text, add_special_tokens=True)["input_ids"]) < target_tokens:
        text += base

    ids = tok(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens,
    )["input_ids"]

    return ids[0]


rows = []

for prompt_len in PROMPT_LENGTHS:
    base_ids = make_prompt(prompt_len)

    for batch_size in BATCH_SIZES:
        print("=" * 72)
        print("prompt_len", prompt_len, "batch", batch_size)

        input_ids = base_ids.unsqueeze(0).repeat(batch_size, 1).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        cell_rows = []

        try:
            for run in range(WARMUP_REPEATS + MEASURED_REPEATS):
                measured = run >= WARMUP_REPEATS

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                samples = []
                stop = threading.Event()

                def monitor():
                    while not stop.is_set():
                        s = gpu_sample()
                        if s is not None:
                            samples.append(s)
                        time.sleep(0.20)

                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()

                start = time.perf_counter()

                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=NEW_TOKENS,
                        min_new_tokens=NEW_TOKENS,
                        use_cache=True,
                        pad_token_id=tok.pad_token_id,
                    )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                stop.set()
                thread.join(timeout=2)

                generated_tokens = (
                    generated.shape[1] - input_ids.shape[1]
                ) * batch_size

                throughput = generated_tokens / elapsed

                peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

                row = {
                    "classification":
                        "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
                    "scientific_claim_authority": False,
                    "model_revision": MODEL_REVISION,
                    "prompt_tokens_per_sequence": int(input_ids.shape[1]),
                    "batch_size": batch_size,
                    "generated_tokens_total": int(generated_tokens),
                    "new_tokens_per_sequence": NEW_TOKENS,
                    "run_index": run,
                    "measured": measured,
                    "elapsed_seconds": elapsed,
                    "generated_tokens_per_second": throughput,
                    "peak_allocated_gib": peak_alloc,
                    "peak_reserved_gib": peak_reserved,
                    "mean_power_w": (
                        statistics.mean(s["power_w"] for s in samples)
                        if samples else None
                    ),
                    "max_power_w": (
                        max(s["power_w"] for s in samples)
                        if samples else None
                    ),
                    "mean_gpu_util_pct": (
                        statistics.mean(s["util_pct"] for s in samples)
                        if samples else None
                    ),
                    "max_memory_mib_nvsmi": (
                        max(s["memory_mib"] for s in samples)
                        if samples else None
                    ),
                    "max_temp_c": (
                        max(s["temp_c"] for s in samples)
                        if samples else None
                    ),
                }

                rows.append(row)

                if measured:
                    cell_rows.append(row)

                print(json.dumps(row, sort_keys=True))

            summary = {
                "prompt_len": prompt_len,
                "batch_size": batch_size,
                "median_tokens_per_sec": statistics.median(
                    r["generated_tokens_per_second"] for r in cell_rows
                ),
                "median_latency_sec": statistics.median(
                    r["elapsed_seconds"] for r in cell_rows
                ),
                "max_peak_allocated_gib": max(
                    r["peak_allocated_gib"] for r in cell_rows
                ),
            }

            print("CELL SUMMARY:", json.dumps(summary))

        except torch.cuda.OutOfMemoryError:
            print(
                "OOM:",
                "prompt_len=", prompt_len,
                "batch_size=", batch_size,
            )

            rows.append({
                "classification":
                    "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
                "scientific_claim_authority": False,
                "model_revision": MODEL_REVISION,
                "prompt_tokens_per_sequence": prompt_len,
                "batch_size": batch_size,
                "status": "OOM",
            })

            torch.cuda.empty_cache()

        except Exception as exc:
            rows.append({
                "classification":
                    "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
                "scientific_claim_authority": False,
                "model_revision": MODEL_REVISION,
                "prompt_tokens_per_sequence": prompt_len,
                "batch_size": batch_size,
                "status": "ERROR",
                "error": repr(exc),
            })

            print("ERROR:", repr(exc))


jsonl = OUT / "systems_benchmark.jsonl"

with jsonl.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")


measured = [
    r for r in rows
    if r.get("measured") is True
]


summary_rows = []

for prompt_len in PROMPT_LENGTHS:
    for batch_size in BATCH_SIZES:
        cell = [
            r for r in measured
            if r["prompt_tokens_per_sequence"] == prompt_len
            and r["batch_size"] == batch_size
        ]

        if not cell:
            continue

        summary_rows.append({
            "prompt_tokens": prompt_len,
            "batch_size": batch_size,
            "median_latency_sec":
                statistics.median(r["elapsed_seconds"] for r in cell),
            "median_tokens_per_sec":
                statistics.median(
                    r["generated_tokens_per_second"] for r in cell
                ),
            "peak_allocated_gib":
                max(r["peak_allocated_gib"] for r in cell),
            "mean_power_w":
                statistics.mean(
                    r["mean_power_w"]
                    for r in cell
                    if r["mean_power_w"] is not None
                )
                if any(r["mean_power_w"] is not None for r in cell)
                else None,
        })


csv_path = OUT / "systems_benchmark_summary.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "prompt_tokens",
            "batch_size",
            "median_latency_sec",
            "median_tokens_per_sec",
            "peak_allocated_gib",
            "mean_power_w",
        ],
    )

    writer.writeheader()
    writer.writerows(summary_rows)


manifest = {
    "schema": "trajuq_secondary_systems_diagnostic_v1",
    "classification":
        "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
    "scientific_claim_authority": False,
    "purpose":
        "Inference-cost and reproducibility characterization only.",
    "not_part_of_confirmatory_rq1_rq5": True,
    "uses_benchmark_outcomes": False,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "model_revision": MODEL_REVISION,
    "dtype": "bfloat16",
    "device": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "prompt_lengths": PROMPT_LENGTHS,
    "batch_sizes": BATCH_SIZES,
    "new_tokens": NEW_TOKENS,
    "warmup_repeats": WARMUP_REPEATS,
    "measured_repeats": MEASURED_REPEATS,
}

manifest_path = OUT / "manifest.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


hashes = {
    "systems_benchmark.jsonl": sha(jsonl),
    "systems_benchmark_summary.csv": sha(csv_path),
    "manifest.json": sha(manifest_path),
}

(OUT / "SHA256SUMS.json").write_text(
    json.dumps(hashes, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("=" * 72)
print("SYSTEMS BENCHMARK COMPLETE")
print("output:", OUT)
print(json.dumps(hashes, indent=2))
