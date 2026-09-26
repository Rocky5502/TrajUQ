# HybridDiffusion / FLARE native-trace bridge

Target upstream:
- repository: `yuchen-zhu-zyc/HybridDiffusion`
- commit: `6ca547aebb72bfe897e80e0a683c776789e5f38c`
- selected checkpoint: `yuchen-zhu-zyc/HybridDiffusion-2B`
- algorithm file: `eval/sglang/srt/dllm/algorithm/hybrid_diffusion_self_spec.py`

The pinned upstream SGLang runtime already contains an opt-in JSONL tracing path. TrajUQ therefore **does not patch or vendor the server** for the first diagnostic integration.

## Native trace switches

Linux:

```bash
export SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH=/absolute/path/hybrid_trace.jsonl
export SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_TOPK=16
```

Windows PowerShell:

```powershell
$env:SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH = "G:\TrajUQ\runs\hybrid_trace.jsonl"
$env:SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_TOPK = "16"
```

Use the pinned upstream runtime and freeze the exact server/algorithm YAML. The inspected upstream self-spec example uses `bd_bidir_shift`, `gen_block_size=4`, `mask_id=248077`, speculative verification, and truncated sampling. These are reference values, not paper defaults until frozen.

Native trace rows can include request ID, forward ID, source (`accepted_spec`, `correction`, `clean_sample`, `t0`), emitted token, acceptance state, rejection location, top-k logits/tokens, and argmax margin.

## Convert to TrajUQ diagnostics

```bash
python scripts/convert_hybrid_trace.py \
  --trace runs/hybrid_trace.jsonl \
  --output runs/hybrid_trace.trajuq.jsonl \
  --model-revision <EXACT_MODEL_REVISION> \
  --require-topk 16
```

The bridge computes request-level acceptance, correction, rejection-forward, margin, late-forward, and top-k entropy-proxy diagnostics.

## Critical entropy limitation

The native trace stores only top-k logits. Therefore `topk_entropy_proxy` is normalized entropy **after renormalizing within the traced top-k support**. It is not full-vocabulary predictive entropy.

For RQ1/RQ2 cross-model entropy comparisons, use integrations that expose full logits. HybridDiffusion's native trace is currently an **RQ4 diagnostic** unless a parity-safe full-logit summary hook is added.

## Parity gate

Run the same frozen prompts twice:
1. tracing disabled;
2. tracing enabled with the chosen top-k.

Require identical generated token IDs/text. Trace-enabled latency is expected to include extra top-k extraction and JSONL I/O, so measure logging overhead separately and never substitute traced wall-clock time for native serving latency.

## Same-checkpoint AR/diffusion diagnostic

Store AR-path and diffusion-path answers separately with stable prompt IDs. Compute disagreement only after both are frozen.

- AR output is **not** ground truth.
- AR/diffusion disagreement is a diagnostic feature.
- correctness comes from the external benchmark scorer.
- disagreement thresholds, if any, are selected on calibration data only.

## Licensing

The inspected upstream release states HybridDiffusion-authored code/modifications/checkpoints use PolyForm Noncommercial 1.0.0, while third-party portions retain their own terms. Keep the runtime external and re-check exact terms before public artifact release.


## Join frozen AR and diffusion outputs

Once both paths have been exported with the same stable example IDs, create the pre-outcome disagreement artifact:

```bash
python scripts/join_hybrid_paths.py \
  --ar runs/hybrid_ar.raw.jsonl \
  --diffusion runs/hybrid_diffusion.raw.jsonl \
  --output runs/hybrid_path_disagreement.jsonl \
  --dataset gsm8k \
  --require-checkpoint-metadata
```

The joiner refuses ID mismatches and prompt-hash mismatches (when hashes are available). With `--require-checkpoint-metadata`, it also requires the same model ID and immutable revision on both paths. The joiner rejects scored AR/diffusion inputs or outcome-like metadata, computes exact-text disagreement, parsed-answer disagreement, normalized token edit distance, common-prefix fraction, and length difference, and writes **no correctness labels**. MATH-500 answer disagreement uses the pinned symbolic-equivalence scorer.


## Held-out RQ4 evaluation

After calibration/test AR and diffusion outputs are frozen and the diffusion
path has been scored externally, evaluate whether same-checkpoint path
disagreement is associated with diffusion-answer error:

```bash
python scripts/evaluate_hybrid_diagnostic.py \
  --cal-target runs/flare_cal.scored.jsonl \
  --cal-diagnostic runs/flare_cal.path_disagreement.jsonl \
  --test-target runs/flare_test.scored.jsonl \
  --test-diagnostic runs/flare_test.path_disagreement.jsonl \
  --require-same-checkpoint \
  --output results/flare_rq4_hybrid.json \
  --predictions-output results/flare_rq4_hybrid.predictions.jsonl
```

The logistic diagnostic is fit on calibration data only. The report includes
held-out AUROC/AUPRC/calibration/selective-risk metrics and a descriptive
agreement-versus-disagreement error-rate split. This analysis is explicitly
secondary/RQ4 diagnostic evidence and is never substituted for the primary
three-family RQ1 endpoint.
