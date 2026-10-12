#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request


TARGET_MODELS = (
    "Pro/deepseek-ai/DeepSeek-V3.2",
    "Pro/zai-org/GLM-5.1",
    "Pro/moonshotai/Kimi-K2.6",
)


def read_secret(path: Path) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "SILICONFLOW_API_KEY":
            token = value.strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError("SILICONFLOW_API_KEY is unavailable")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture outcome-blind SiliconFlow inventory.")
    parser.add_argument("--secret", required=True)
    parser.add_argument("--output-dir", default="results/rq5/provider_snapshot")
    args = parser.parse_args()
    token = read_secret(Path(args.secret))
    output = Path(args.output_dir)

    inventory_url = "https://api.siliconflow.cn/v1/models?sub_type=chat"
    request = urllib.request.Request(
        inventory_url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        inventory_bytes = response.read()
        status = int(response.status)
        content_type = response.headers.get("content-type")
    inventory = json.loads(inventory_bytes)
    canonical_inventory = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(output / "authenticated_models_chat.json", canonical_inventory)

    pricing_url = "https://siliconflow.cn/pricing"
    pricing_request = urllib.request.Request(pricing_url, headers={"User-Agent": "TrajUQ-RQ5/1.0"})
    with urllib.request.urlopen(pricing_request, timeout=60) as response:
        pricing_bytes = response.read()
        pricing_status = int(response.status)
        pricing_content_type = response.headers.get("content-type")
    atomic_bytes(output / "pricing_page.html", pricing_bytes)

    records = inventory.get("data", [])
    available_ids = {
        str(item.get("id")) for item in records if isinstance(item, dict) and item.get("id")
    }
    snapshot = {
        "schema": "trajuq_siliconflow_provider_snapshot_v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_url": inventory_url,
        "inventory_http_status": status,
        "inventory_content_type": content_type,
        "inventory_sha256": sha256(canonical_inventory),
        "inventory_model_count": len(available_ids),
        "pricing_url": pricing_url,
        "pricing_http_status": pricing_status,
        "pricing_content_type": pricing_content_type,
        "pricing_page_sha256": sha256(pricing_bytes),
        "target_models": [
            {"model_id": model_id, "available": model_id in available_ids}
            for model_id in TARGET_MODELS
        ],
        "authorization_persisted": False,
    }
    snapshot_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(output / "provider_snapshot.json", snapshot_bytes)
    print(json.dumps({
        "status": "PASS" if all(item["available"] for item in snapshot["target_models"]) else "FAIL",
        "inventory_sha256": snapshot["inventory_sha256"],
        "pricing_page_sha256": snapshot["pricing_page_sha256"],
        "target_models": snapshot["target_models"],
    }, indent=2))


if __name__ == "__main__":
    main()
