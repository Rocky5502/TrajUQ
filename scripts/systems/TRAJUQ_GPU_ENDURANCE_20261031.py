from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = os.environ["MODEL_DIR"]

MODEL_REVISION = (
    "0e9e39f249a16976918f6564b8830bc894c89659"
)

OUT = Path("/data/trajuq_gpu_endurance_20261031")
OUT.mkdir(parents=True, exist_ok=True)

RAW = OUT / "endurance_runs.jsonl"
PROGRESS = OUT / "progress.json"
SUMMARY = OUT / "endurance_summary.json"
MANIFEST = OUT / "manifest.json"

CLASSIFICATION = (
    "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC"
)

# Large, round-robin systems stress matrix.
# No benchmark prompts, references, labels, or outcomes.
CONFIGS = [
    {"prompt_tokens": 512,  "batch_size": 16},
    {"prompt_tokens": 2048, "batch_size": 8},
    {"prompt_tokens": 4096, "batch_size": 4},
    {"prompt_tokens": 8192, "batch_size": 2},
    {"prompt_tokens": 16384, "batch_size": 1},
]

OUTPUT_TOKENS = 256

# 75 cycles x 5 configurations = 375 measured inference runs.
CYCLES = 75

MONITOR_INTERVAL = 0.25

torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)

torch.backends.cuda.matmul.allow_tf32 = True


def nvidia_sample():
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "power.draw,"
                "utilization.gpu,"
                "memory.used,"
                "temperature.gpu,"
                "clocks.sm,"
                "clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

        vals = [
            float(x.strip())
            for x in raw.split(",")
        ]

        return {
            "power_w": vals[0],
            "util_pct": vals[1],
            "memory_mib": vals[2],
            "temperature_c": vals[3],
            "sm_clock_mhz": vals[4],
            "memory_clock_mhz": vals[5],
        }

    except Exception:
        return None


print("Loading exact tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("Loading exact frozen model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
).to("cuda")

model.eval()

# Remove irrelevant sampling options from inherited generation config.
model.generation_config.temperature = None
model.generation_config.top_p = None


print(
    "MODEL_READY",
    torch.cuda.get_device_name(0),
    flush=True,
)


def make_input(target_tokens: int):
    base = (
        "Synthetic long-duration inference systems diagnostic. "
        "No benchmark question, correctness label, reference answer, "
        "or scientific outcome is contained in this prompt. "
        "Continue with neutral text. "
    )

    text = base

    while True:
        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) >= target_tokens:
            break

        text += base

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=target_tokens,
        return_tensors="pt",
    )["input_ids"][0]

    return ids


inputs = {}

for cfg in CONFIGS:
    n = cfg["prompt_tokens"]
    inputs[n] = make_input(n)


manifest = {
    "schema":
        "trajuq_gpu_endurance_v1",
    "classification":
        CLASSIFICATION,
    "scientific_claim_authority":
        False,
    "confirmatory_rq1_rq5":
        False,
    "uses_benchmark_prompts":
        False,
    "uses_benchmark_labels":
        False,
    "uses_benchmark_outcomes":
        False,
    "purpose":
        "Post-hoc long-duration systems stability, "
        "energy, throughput, memory, and deterministic-output "
        "characterization.",
    "model":
        "meta-llama/Llama-3.1-8B-Instruct",
    "model_revision":
        MODEL_REVISION,
    "dtype":
        "bfloat16",
    "device":
        torch.cuda.get_device_name(0),
    "python":
        platform.python_version(),
    "torch":
        torch.__version__,
    "transformers":
        transformers.__version__,
    "configs":
        CONFIGS,
    "output_tokens":
        OUTPUT_TOKENS,
    "cycles":
        CYCLES,
    "expected_measured_runs":
        CYCLES * len(CONFIGS),
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# One warmup per configuration.
print("BEGIN_WARMUPS", flush=True)

for cfg in CONFIGS:

    prompt_tokens = cfg["prompt_tokens"]
    batch_size = cfg["batch_size"]

    ids = (
        inputs[prompt_tokens]
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to("cuda")
    )

    mask = torch.ones_like(ids)

    try:
        with torch.inference_mode():
            model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=32,
                min_new_tokens=32,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()

        print(
            "WARMUP_PASS",
            prompt_tokens,
            batch_size,
            flush=True,
        )

    except torch.cuda.OutOfMemoryError:

        print(
            "WARMUP_OOM",
            prompt_tokens,
            batch_size,
            flush=True,
        )

        torch.cuda.empty_cache()


print("END_WARMUPS", flush=True)


rows = []
completed_runs = 0


for cycle in range(CYCLES):

    print(
        "=" * 78,
        flush=True,
    )

    print(
        f"CYCLE {cycle + 1}/{CYCLES}",
        flush=True,
    )

    for cfg_index, cfg in enumerate(CONFIGS):

        prompt_tokens = cfg["prompt_tokens"]
        batch_size = cfg["batch_size"]

        ids = (
            inputs[prompt_tokens]
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to("cuda")
        )

        mask = torch.ones_like(ids)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        samples = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                sample = nvidia_sample()

                if sample is not None:
                    samples.append(sample)

                time.sleep(MONITOR_INTERVAL)

        monitor_thread = threading.Thread(
            target=monitor,
            daemon=True,
        )

        monitor_thread.start()

        started = time.time()
        start_perf = time.perf_counter()

        try:

            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    do_sample=False,
                    min_new_tokens=OUTPUT_TOKENS,
                    max_new_tokens=OUTPUT_TOKENS,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_perf
            )

            stop.set()
            monitor_thread.join(timeout=2)

            new_tokens = (
                generated[:, ids.shape[1]:]
                .detach()
                .cpu()
            )

            output_hash = hashlib.sha256(
                new_tokens.numpy().tobytes()
            ).hexdigest()

            generated_total = (
                int(new_tokens.numel())
            )

            throughput = (
                generated_total / elapsed
            )

            peak_allocated = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            peak_reserved = (
                torch.cuda.max_memory_reserved()
                / 1024**3
            )

            record = {
                "classification":
                    CLASSIFICATION,
                "scientific_claim_authority":
                    False,
                "cycle":
                    cycle + 1,
                "config_index":
                    cfg_index,
                "prompt_tokens":
                    prompt_tokens,
                "batch_size":
                    batch_size,
                "output_tokens_per_sequence":
                    OUTPUT_TOKENS,
                "generated_tokens_total":
                    generated_total,
                "elapsed_seconds":
                    elapsed,
                "tokens_per_second":
                    throughput,
                "peak_allocated_gib":
                    peak_allocated,
                "peak_reserved_gib":
                    peak_reserved,
                "output_sha256":
                    output_hash,
                "started_unix":
                    started,
                "mean_power_w":
                    (
                        statistics.mean(
                            x["power_w"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "max_power_w":
                    (
                        max(
                            x["power_w"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "mean_gpu_util_pct":
                    (
                        statistics.mean(
                            x["util_pct"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "min_gpu_util_pct":
                    (
                        min(
                            x["util_pct"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "max_gpu_util_pct":
                    (
                        max(
                            x["util_pct"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "max_memory_mib":
                    (
                        max(
                            x["memory_mib"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "mean_temperature_c":
                    (
                        statistics.mean(
                            x["temperature_c"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "max_temperature_c":
                    (
                        max(
                            x["temperature_c"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "mean_sm_clock_mhz":
                    (
                        statistics.mean(
                            x["sm_clock_mhz"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "mean_memory_clock_mhz":
                    (
                        statistics.mean(
                            x["memory_clock_mhz"]
                            for x in samples
                        )
                        if samples else None
                    ),
                "status":
                    "PASS",
            }

        except torch.cuda.OutOfMemoryError:

            stop.set()
            monitor_thread.join(timeout=2)

            record = {
                "classification":
                    CLASSIFICATION,
                "scientific_claim_authority":
                    False,
                "cycle":
                    cycle + 1,
                "config_index":
                    cfg_index,
                "prompt_tokens":
                    prompt_tokens,
                "batch_size":
                    batch_size,
                "output_tokens_per_sequence":
                    OUTPUT_TOKENS,
                "status":
                    "OOM",
            }

            torch.cuda.empty_cache()

        except Exception as exc:

            stop.set()
            monitor_thread.join(timeout=2)

            record = {
                "classification":
                    CLASSIFICATION,
                "scientific_claim_authority":
                    False,
                "cycle":
                    cycle + 1,
                "config_index":
                    cfg_index,
                "prompt_tokens":
                    prompt_tokens,
                "batch_size":
                    batch_size,
                "output_tokens_per_sequence":
                    OUTPUT_TOKENS,
                "status":
                    "ERROR",
                "error":
                    repr(exc),
            }

        rows.append(record)
        completed_runs += 1

        # Append immediately so interruption does not lose completed runs.
        with RAW.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
                + "\n"
            )

        PROGRESS.write_text(
            json.dumps(
                {
                    "completed_runs":
                        completed_runs,
                    "expected_runs":
                        CYCLES
                        * len(CONFIGS),
                    "current_cycle":
                        cycle + 1,
                    "status":
                        "RUNNING",
                    "last_record":
                        record,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                record,
                sort_keys=True,
            ),
            flush=True,
        )


# Aggregate final results.
by_config = defaultdict(list)

for row in rows:

    if row.get("status") != "PASS":
        continue

    key = (
        row["prompt_tokens"],
        row["batch_size"],
    )

    by_config[key].append(row)


summary = {
    "classification":
        CLASSIFICATION,
    "scientific_claim_authority":
        False,
    "completed_runs":
        completed_runs,
    "expected_runs":
        CYCLES * len(CONFIGS),
    "cells":
        [],
}


for (prompt_tokens, batch_size), cell in sorted(
    by_config.items()
):

    throughput = [
        r["tokens_per_second"]
        for r in cell
    ]

    latency = [
        r["elapsed_seconds"]
        for r in cell
    ]

    temperatures = [
        r["max_temperature_c"]
        for r in cell
        if r["max_temperature_c"] is not None
    ]

    power = [
        r["mean_power_w"]
        for r in cell
        if r["mean_power_w"] is not None
    ]

    hashes = [
        r["output_sha256"]
        for r in cell
    ]

    n = len(cell)

    q = max(1, n // 4)

    first_q = throughput[:q]
    last_q = throughput[-q:]

    mean_tp = statistics.mean(
        throughput
    )

    std_tp = (
        statistics.stdev(throughput)
        if len(throughput) > 1
        else 0.0
    )

    throughput_cv = (
        std_tp / mean_tp
        if mean_tp
        else None
    )

    first_median = (
        statistics.median(first_q)
    )

    last_median = (
        statistics.median(last_q)
    )

    drift_pct = (
        (
            last_median
            - first_median
        )
        / first_median
        * 100.0
        if first_median
        else None
    )

    summary["cells"].append(
        {
            "prompt_tokens":
                prompt_tokens,
            "batch_size":
                batch_size,
            "successful_runs":
                n,
            "median_tokens_per_second":
                statistics.median(
                    throughput
                ),
            "mean_tokens_per_second":
                mean_tp,
            "throughput_cv":
                throughput_cv,
            "median_latency_seconds":
                statistics.median(
                    latency
                ),
            "p95_latency_seconds":
                sorted(latency)[
                    min(
                        len(latency) - 1,
                        int(
                            0.95
                            * len(latency)
                        ),
                    )
                ],
            "max_temperature_c":
                (
                    max(temperatures)
                    if temperatures
                    else None
                ),
            "mean_power_w":
                (
                    statistics.mean(power)
                    if power
                    else None
                ),
            "unique_output_hashes":
                len(set(hashes)),
            "deterministic_outputs":
                len(set(hashes)) == 1,
            "throughput_drift_first_to_last_quartile_pct":
                drift_pct,
        }
    )


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


hashes = {}

for path in [
    RAW,
    PROGRESS,
    SUMMARY,
    MANIFEST,
]:
    hashes[path.name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


(OUT / "SHA256SUMS.json").write_text(
    json.dumps(
        hashes,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


PROGRESS.write_text(
    json.dumps(
        {
            "completed_runs":
                completed_runs,
            "expected_runs":
                CYCLES * len(CONFIGS),
            "status":
                "COMPLETE",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "=" * 78,
    flush=True,
)

print(
    "TRAJUQ_GPU_ENDURANCE_COMPLETE",
    flush=True,
)

print(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
