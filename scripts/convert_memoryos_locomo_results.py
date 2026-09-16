#!/usr/bin/env python3
"""Put MemoryOS predictions into the canonical LoCoMo JSON layout.

The upstream MemoryOS script writes one flat record per question.  This adapter
matches those records to the QA-masked source data by ``sample_id`` and question
text, then leaves gold labels solely in the evaluator input.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--memoryos-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-key", default="memoryos_prediction")
    args = parser.parse_args()

    source = json.loads(Path(args.source_data).read_text(encoding="utf-8"))
    records = json.loads(Path(args.memoryos_results).read_text(encoding="utf-8"))
    records_by_key = defaultdict(deque)
    for record in records:
        records_by_key[(record.get("sample_id"), record.get("question"))].append(record)

    output, missing = [], []
    for sample in source:
        result_sample = {"sample_id": sample.get("sample_id"), "qa": []}
        for qa in sample.get("qa", []):
            key = (sample.get("sample_id"), qa.get("question"))
            if not records_by_key[key]:
                missing.append(key)
                continue
            record = records_by_key[key].popleft()
            result_qa = dict(qa)
            result_qa[args.prediction_key] = record.get("system_answer", "")
            result_sample["qa"].append(result_qa)
        output.append(result_sample)

    extras = sum(len(items) for items in records_by_key.values())
    if missing or extras:
        raise SystemExit("MemoryOS result alignment failed: %d missing, %d extra records." % (len(missing), extras))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Converted %d MemoryOS predictions to %s" % (len(records), target))


if __name__ == "__main__":
    main()
