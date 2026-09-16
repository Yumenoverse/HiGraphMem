#!/usr/bin/env python3
"""Aggregate provider usage stored by ``eval/run_locomo_v2.py``."""
import argparse
import json
from pathlib import Path


FIELDS = ("calls", "calls_with_usage", "prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-key", default="higraphmem_prediction")
    args = parser.parse_args()
    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    events, recorded_qas, total_qas = [], 0, 0
    for sample in samples:
        for qa in sample.get("qa", []):
            total_qas += 1
            usage = qa.get(args.prediction_key + "_token_usage")
            if usage:
                recorded_qas += 1
                events.extend(usage.get("events", []))
    phases = {}
    for event in events:
        bucket = phases.setdefault(event.get("phase", "answer"), {key: 0 for key in FIELDS})
        bucket["calls"] += 1
        bucket["latency_seconds"] += float(event.get("latency_seconds") or 0.0)
        if event.get("usage_available"):
            bucket["calls_with_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                bucket[key] += int(event.get(key) or 0)
    def combine(names):
        return {key: sum(phases.get(name, {}).get(key, 0) for name in names) for key in FIELDS}
    online, end_to_end = combine(["answer"]), combine(list(phases))
    result = {
        "input": args.input,
        "prediction_key": args.prediction_key,
        "total_qas": total_qas,
        "qas_with_recorded_usage": recorded_qas,
        "coverage": recorded_qas / total_qas if total_qas else 0.0,
        "phases": phases,
        "online": online,
        "end_to_end": end_to_end,
        "mean_online_total_tokens_per_qa": online["total_tokens"] / recorded_qas if recorded_qas else None,
        "mean_end_to_end_total_tokens_per_qa": end_to_end["total_tokens"] / recorded_qas if recorded_qas else None,
        "mean_online_api_latency_seconds_per_call": online["latency_seconds"] / online["calls"] if online["calls"] else None,
        "mean_online_api_latency_seconds_per_qa": online["latency_seconds"] / recorded_qas if recorded_qas else None,
        "online_completion_tokens_per_second": online["completion_tokens"] / online["latency_seconds"] if online["latency_seconds"] else None,
        "online_total_tokens_per_second": online["total_tokens"] / online["latency_seconds"] if online["latency_seconds"] else None,
        "usage_source": "provider_response_usage; latency is successful provider API wall time only; unavailable usage is not estimated",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
