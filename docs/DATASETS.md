# Datasets

TrajUQ records exact upstream identifiers and revisions but does not redistribute raw benchmark rows. Dataset licenses, gated-access rules, and upstream terms remain authoritative.

## Frozen dataset inventory

| Key | Upstream source | Config | Frozen revision | Calibration source | Evaluation source | Selection |
|---|---|---|---|---|---|---|
| gsm8k | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | main | 740312add88f781978c0658806c59bc2815b9866 | train | test | SHA-256 ID rank; max 500/500 |
| math500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | default | 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be | internal 30% holdout of source test | remaining internal holdout | full frozen holdout |
| gpqa_diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | gpqa_diamond | 83022cefff930aea54f654c0b282e74b9eeda5c6 | internal 30% holdout of train | remaining internal holdout | full frozen holdout; gated |
| arc_challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | ARC-Challenge | 210d026faf9955653af8916fad021475a3f00453 | validation | test | keep calibration; cap test at 500 |
| mmlu_pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | default | b189ec765aa7ed75c8acfea42df31fdae71f97be | validation | test | keep calibration; cap test at 500 |
| truthfulqa | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | multiple_choice | 741b8276f2d1982aa3d5b832d3ee81ed3b896490 | internal 30% holdout of validation | remaining internal holdout | full frozen holdout |

The machine-readable source is data/manifests/evaluation_splits.json.

## Deterministic selection

The pre-generation cap protocol is sha256_id_rank_v1 with seed 2027. Selection uses namespaced example identifiers only. It does not consult labels, reference answers, prompt text, correctness, or model outputs.

## Single-split datasets

MATH-500, GPQA-Diamond, and TruthfulQA use deterministic internal held-out partitions. These results must be described as internal held-out evaluation, not as full-benchmark leaderboard scores.

## MATH-500 scoring

MATH-500 correctness uses math-verify[antlr4_13_2]==0.9.0 with antlr4 4.13.2. String equality is not substituted for the registered symbolic scorer.

## GPQA access

GPQA is gated. Users must obtain access under the upstream dataset terms. This repository does not include or bypass gated benchmark content.

## Redistribution boundary

The repository includes source identifiers, exact revisions, split/cap policies, deterministic preparation code, and hashes. It excludes raw benchmark rows, gated dataset contents, and local dataset caches.
