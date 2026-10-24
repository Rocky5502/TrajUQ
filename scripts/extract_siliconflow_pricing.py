#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


TARGETS = (
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "Pro/zai-org/GLM-5.1",
    "Pro/moonshotai/Kimi-K2.6",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen target pricing from captured page.")
    parser.add_argument("--input", default="results/rq5/provider_snapshot/pricing_page.html")
    parser.add_argument("--output", default="results/rq5/provider_snapshot/pricing_contract.json")
    args = parser.parse_args()
    source = Path(args.input)
    text = source.read_text(encoding="utf-8")
    decoded = text.replace('\\"', '"')
    records = []
    for model_id in TARGETS:
        pattern = re.compile(
            r'\{"modelId":"[^"]+","modelName":"'
            + re.escape(model_id)
            + r'".*?"contextLen":(?P<context>\d+).*?"status":"(?P<status>[^"]+)"'
            + r'.*?"pricing":(?P<pricing>\[.*?\])',
            re.DOTALL,
        )
        matches = list(pattern.finditer(decoded))
        if not matches:
            raise SystemExit(f"Pricing record unavailable for {model_id}")
        match = matches[-1]
        pricing = json.loads(match.group("pricing"))
        records.append(
            {
                "model_id": model_id,
                "context_length": int(match.group("context")),
                "status": match.group("status"),
                "pricing": pricing,
            }
        )
    payload = {
        "schema": "trajuq_siliconflow_pricing_contract_v1",
        "captured_page": str(source).replace("\\", "/"),
        "captured_page_sha256": sha256(source),
        "extracted_utc": datetime.now(timezone.utc).isoformat(),
        "currency": "RMB",
        "models": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
