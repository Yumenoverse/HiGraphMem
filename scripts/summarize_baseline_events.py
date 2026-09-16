#!/usr/bin/env python3
"""Summarize provider usage events emitted by external baseline adapters."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.load(open(args.input, encoding="utf-8"))
events = payload.get("events", [])
recorded = [event for event in events if event.get("total_tokens") is not None]
totals = {key: sum(float(event.get(key) or 0) for event in recorded) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")}
sources = sorted({event.get("usage_source", "provider_response_usage") for event in recorded})
output = {"method": payload.get("method"), "calls": len(events), "calls_with_usage": len(recorded), **totals,
          "online_total_tokens_per_second": totals["total_tokens"] / totals["latency_seconds"] if totals["latency_seconds"] else None,
          "usage_source": "; ".join(sources)}
json.dump(output, open(args.output, "w", encoding="utf-8"), indent=2)
