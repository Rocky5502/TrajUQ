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

MODEL_REVISION = (
    "0e9e39f249a16976918f6564b8830bc894c89659"
)

OUT = Path(
    "/data/trajuq_gpu_context_decode_scaling_20261031"
)
OUT.mkdir(parents=True, exist_ok=True)

# Systems-only matrix.
# Does NOT use any benchmark prompt/reference/outcome.
PROMPT_LENGTHS = [128, 512, 2048, 4096, 8192]
OUTPUT_LENGTHS = [32, 128, 512]
BATCH_SIZES = [1, 4, 8]

WARMUP = 1
REPEATS = 3

torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)

torch.backends.cuda.matmul.allow_tf32 = True


def gpu_sample():
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "power.draw,"
                "utilization.gpu,"
                "memory.used,"
                "temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

        power, util, memory, temp = [
            float(x.strip()) for x in raw.split(",")
        ]

        return {
            "power_w": power,
            "util_pct": util,
            "memory_mib": memory,
            "temp_c": temp,
        }
    except Exception:
        return None


print("Loading tokenizer...", flush=True)

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

# Avoid irrelevant greedy-decoding warnings inherited from
# generation_config.
model.generation_config.temperature = None
model.generation_config.top_p = None

print(
    "MODEL_READY",
    torch.cuda.get_device_name(0),
    flush=True,
)


def synthetic_prompt(target_tokens: int):
    text = (
        "Synthetic systems-efficiency diagnostic. "
        "This text contains no benchmark question, "
        "reference answer, correctness label, or scientific "
        "outcome. Continue with neutral text. "
    )

    content = text

    while len(
        tokenizer(
            content,
            add_special_tokens=False,
        )["input_ids"]
    ) < target_tokens:
        content += text

    encoded = tokenizer(
        content,
        add_special_tokens=False,
        truncation=True,
        max_length=target_tokens,
        return_tensors="pt",
    )["input_ids"][0]

    return encoded


rows = []

for prompt_length in PROMPT_LENGTHS:

    base_ids = synthetic_prompt(prompt_length)

    for output_length in OUTPUT_LENGTHS:

        for batch_size in BATCH_SIZES:

            print(
                "=" * 72,
                flush=True,
            )

            print(
                "CELL",
                f"prompt={prompt_length}",
                f"output={output_length}",
                f"batch={batch_size}",
                flush=True,
            )

            input_ids = (
                base_ids
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to("cuda")
            )

            attention_mask = torch.ones_like(input_ids)

            cell = []

            try:
                for run_index in range(
                    WARMUP + REPEATS
                ):

                    measured = run_index >= WARMUP

                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                    samples = []
                    stop = threading.Event()

                    def monitor():
                        while not stop.is_set():
                            sample = gpu_sample()

                            if sample is not None:
                                samples.append(sample)

                            time.sleep(0.2)

                    thread = threading.Thread(
                        target=monitor,
                        daemon=True,
                    )
                    thread.start()

                    start = time.perf_counter()

                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=False,
                            min_new_tokens=output_length,
                            max_new_tokens=output_length,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    torch.cuda.synchronize()

                    elapsed = (
                        time.perf_counter() - start
                    )

                    stop.set()
                    thread.join(timeout=2)

                    generated_per_sequence = (
                        generated.shape[1]
                        - input_ids.shape[1]
                    )

                    generated_total = (
                        generated_per_sequence
                        * batch_size
                    )

                    tokens_per_second = (
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
                            "SECONDARY_EXPLORATORY_"
                            "SYSTEMS_DIAGNOSTIC",
                        "scientific_claim_authority":
                            False,
                        "uses_benchmark_outcomes":
                            False,
                        "model_revision":
                            MODEL_REVISION,
                        "prompt_tokens":
                            int(input_ids.shape[1]),
                        "output_tokens_requested":
                            output_length,
                        "batch_size":
                            batch_size,
                        "run_index":
                            run_index,
                        "measured":
                            measured,
                        "elapsed_seconds":
                            elapsed,
                        "generated_tokens_total":
                            int(generated_total),
                        "tokens_per_second":
                            tokens_per_second,
                        "peak_allocated_gib":
                            peak_allocated,
                        "peak_reserved_gib":
                            peak_reserved,
                        "mean_power_w":
                            (
                                statistics.mean(
                                    x["power_w"]
                                    for x in samples
                                )
                                if samples
                                else None
                            ),
                        "mean_gpu_util_pct":
                            (
                                statistics.mean(
                                    x["util_pct"]
                                    for x in samples
                                )
                                if samples
                                else None
                            ),
                        "max_gpu_memory_mib":
                            (
                                max(
                                    x["memory_mib"]
                                    for x in samples
                                )
                                if samples
                                else None
                            ),
                        "max_temp_c":
                            (
                                max(
                                    x["temp_c"]
                                    for x in samples
                                )
                                if samples
                                else None
                            ),
                    }

                    rows.append(record)

                    if measured:
                        cell.append(record)

                    print(
                        json.dumps(
                            record,
                            sort_keys=True,
                        ),
                        flush=True,
                    )

                print(
                    "CELL_COMPLETE",
                    json.dumps(
                        {
                            "prompt_tokens":
                                prompt_length,
                            "output_tokens":
                                output_length,
                            "batch_size":
                                batch_size,
                            "median_tokens_per_second":
                                statistics.median(
                                    r["tokens_per_second"]
                                    for r in cell
                                ),
                            "median_latency_seconds":
                                statistics.median(
                                    r["elapsed_seconds"]
                                    for r in cell
                                ),
                            "peak_allocated_gib":
                                max(
                                    r[
                                        "peak_allocated_gib"
                                    ]
                                    for r in cell
                                ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            except torch.cuda.OutOfMemoryError:

                rows.append(
                    {
                        "classification":
                            "SECONDARY_EXPLORATORY_"
                            "SYSTEMS_DIAGNOSTIC",
                        "scientific_claim_authority":
                            False,
                        "prompt_tokens":
                            prompt_length,
                        "output_tokens_requested":
                            output_length,
                        "batch_size":
                            batch_size,
                        "status":
                            "OOM",
                    }
                )

                print(
                    "OOM",
                    prompt_length,
                    output_length,
                    batch_size,
                    flush=True,
                )

                torch.cuda.empty_cache()

            except Exception as exc:

                rows.append(
                    {
                        "classification":
                            "SECONDARY_EXPLORATORY_"
                            "SYSTEMS_DIAGNOSTIC",
                        "scientific_claim_authority":
                            False,
                        "prompt_tokens":
                            prompt_length,
                        "output_tokens_requested":
                            output_length,
                        "batch_size":
                            batch_size,
                        "status":
                            "ERROR",
                        "error":
                            repr(exc),
                    }
                )

                print(
                    "ERROR",
                    repr(exc),
                    flush=True,
                )


raw_path = OUT / "context_decode_scaling.jsonl"

with raw_path.open(
    "w",
    encoding="utf-8",
) as handle:
    for row in rows:
        handle.write(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
        )


measured = [
    row
    for row in rows
    if row.get("measured") is True
]

summary = []

for prompt_length in PROMPT_LENGTHS:
    for output_length in OUTPUT_LENGTHS:
        for batch_size in BATCH_SIZES:

            cell = [
                r
                for r in measured
                if r["prompt_tokens"]
                    == prompt_length
                and
                r["output_tokens_requested"]
                    == output_length
                and
                r["batch_size"]
                    == batch_size
            ]

            if not cell:
                continue

            summary.append(
                {
                    "prompt_tokens":
                        prompt_length,
                    "output_tokens":
                        output_length,
                    "batch_size":
                        batch_size,
                    "median_latency_sec":
                        statistics.median(
                            r["elapsed_seconds"]
                            for r in cell
                        ),
                    "median_tokens_per_sec":
                        statistics.median(
                            r["tokens_per_second"]
                            for r in cell
                        ),
                    "peak_allocated_gib":
                        max(
                            r["peak_allocated_gib"]
                            for r in cell
                        ),
                    "mean_power_w":
                        statistics.mean(
                            r["mean_power_w"]
                            for r in cell
                            if r["mean_power_w"]
                            is not None
                        )
                        if any(
                            r["mean_power_w"]
                            is not None
                            for r in cell
                        )
                        else None,
                }
            )


summary_path = (
    OUT
    / "context_decode_scaling_summary.csv"
)

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "prompt_tokens",
            "output_tokens",
            "batch_size",
            "median_latency_sec",
            "median_tokens_per_sec",
            "peak_allocated_gib",
            "mean_power_w",
        ],
    )

    writer.writeheader()
    writer.writerows(summary)


manifest = {
    "schema":
        "trajuq_secondary_context_decode_scaling_v1",
    "classification":
        "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
    "scientific_claim_authority":
        False,
    "not_part_of_confirmatory_rq1_rq5":
        True,
    "uses_benchmark_outcomes":
        False,
    "purpose":
        "Post-hoc systems characterization of "
        "inference cost only.",
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
    "prompt_lengths":
        PROMPT_LENGTHS,
    "output_lengths":
        OUTPUT_LENGTHS,
    "batch_sizes":
        BATCH_SIZES,
    "warmup":
        WARMUP,
    "measured_repeats":
        REPEATS,
}

manifest_path = OUT / "manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


def sha256(path: Path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


hash_manifest = {
    raw_path.name:
        sha256(raw_path),
    summary_path.name:
        sha256(summary_path),
    manifest_path.name:
        sha256(manifest_path),
}

hash_path = OUT / "SHA256SUMS.json"

hash_path.write_text(
    json.dumps(
        hash_manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("=" * 72)
print(
    "CONTEXT_DECODE_SCALING_COMPLETE",
    flush=True,
)
print(
    json.dumps(
        hash_manifest,
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
