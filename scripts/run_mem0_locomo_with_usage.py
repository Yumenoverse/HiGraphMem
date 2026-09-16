#!/usr/bin/env python3
"""Run Mem0's official LoCoMo benchmark and save OpenAI-compatible usage.

The Mem0 Cloud API does not expose the tokens used by its managed memory
pipeline.  This wrapper records every answerer/judge completion made by the
official benchmark, which is the externally observable model cost.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import time
from pathlib import Path


class UsageTracker:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict] = []

    def add(self, response, latency: float) -> None:
        usage = getattr(response, "usage", None)
        get = lambda key: getattr(usage, key, None) if usage is not None else None
        prompt, completion, total = get("prompt_tokens"), get("completion_tokens"), get("total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        self.events.append({
            "phase": "answer_or_judge",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "latency_seconds": round(latency, 6),
            "usage_source": "provider_response_usage" if total is not None else "provider_usage_unavailable",
        })

    def export(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        recorded = [e for e in self.events if e["total_tokens"] is not None]
        totals = {key: sum(float(e.get(key) or 0) for e in recorded)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")}
        payload = {
            "method": "Mem0",
            "scope": "Mem0 Cloud retrieval plus answerer/judge completions; Mem0-managed ingestion tokens are not exposed by its API and are excluded.",
            "events": self.events,
            "summary": {
                "calls": len(self.events),
                "calls_with_usage": len(recorded),
                **totals,
                "online_total_tokens_per_second": totals["total_tokens"] / totals["latency_seconds"] if totals["latency_seconds"] else None,
                "usage_source": "provider_response_usage (answerer/judge only)",
            },
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    if not (benchmark_dir / "benchmarks" / "locomo" / "run.py").is_file():
        raise SystemExit(f"Not a memory-benchmarks checkout: {benchmark_dir}")

    sys.path.insert(0, str(benchmark_dir))
    from openai.resources.chat.completions import AsyncCompletions

    tracker = UsageTracker(Path(args.events).resolve())
    original_create = AsyncCompletions.create

    async def tracked_create(self, *call_args, **call_kwargs):
        started = time.perf_counter()
        response = await original_create(self, *call_args, **call_kwargs)
        tracker.add(response, time.perf_counter() - started)
        return response

    AsyncCompletions.create = tracked_create
    atexit.register(tracker.export)
    sys.argv = ["benchmarks.locomo.run"] + [item for item in args.benchmark_args if item != "--"]
    runpy.run_module("benchmarks.locomo.run", run_name="__main__")


if __name__ == "__main__":
    main()
