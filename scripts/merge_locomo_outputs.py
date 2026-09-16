#!/usr/bin/env python3
"""Merge disjoint per-conversation LoCoMo prediction files.

Each runner shard writes the usual list of ``{sample_id, qa}`` objects.  This
utility rejects duplicate conversation ids so a parallel run cannot silently
mix or overwrite predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction JSON files from disjoint shards.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = []
    seen = set()
    for raw_path in args.inputs:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a LoCoMo prediction list")
        for sample in payload:
            sample_id = sample.get("sample_id")
            if not sample_id:
                raise ValueError(f"{path} contains an item without sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id {sample_id}; shards must be disjoint")
            seen.add(sample_id)
            merged.append(sample)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(merged)} conversations into {target}")


if __name__ == "__main__":
    main()
