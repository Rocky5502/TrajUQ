# Data

TrajUQ does not commit raw benchmark snapshots.

Reasons:
1. benchmark licenses and redistribution terms differ;
2. GPQA is gated and asks users not to expose examples publicly;
3. dataset revisions change;
4. raw/model-generated artifacts are better stored outside Git.

## Workflow

\`\`\`bash
pip install -e ".[models]"
python scripts/download_data.py --dataset gsm8k math500 arc_challenge mmlu_pro truthfulqa
python scripts/prepare_benchmarks.py --dataset gsm8k --limit 50
\`\`\`

Downloads go to \`data/cache/\`; normalized prompts go to \`data/processed/\`. Both are gitignored.

Normalized JSONL:

\`\`\`json
{
  "id": "dataset-specific-id",
  "dataset": "gsm8k",
  "prompt": "...",
  "reference": "...",
  "metadata": {}
}
\`\`\`

The reference is scorer-only and must never be passed to the generator or trajectory feature extractor.

## GPQA

Accept GPQA's terms on Hugging Face and authenticate locally before downloading. Do **not** commit GPQA examples, prompts, or screenshots.

\`data/samples/synthetic_smoke.jsonl\` is synthetic and only tests plumbing.
