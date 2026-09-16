#!/usr/bin/env python3
"""Create a compact efficiency table from a completed ablation batch."""
import argparse
import json
from pathlib import Path


VARIANTS = (
    ("Plain Text-RAG", "plain_text_rag"),
    ("w/o Update", "wo_update"),
    ("w/o Conflict Control", "wo_conflict_control"),
    ("w/o TWMem", "wo_twmem"),
    ("w/o Gated Activation", "wo_gated_activation"),
    ("w/o Router", "wo_router"),
)


def number(value, digits=2):
    return round(float(value), digits) if value is not None else None


def display(value, digits=2):
    return f"{value:.{digits}f}" if value is not None else "n/a"


def read_f1(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("Avg", "avg", "average"):
        row = data.get(key)
        if isinstance(row, dict):
            return row.get("f1", row.get("F1"))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Shared result prefix used by run_all_ablations_locomo.sh")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output", required=True, help="Output JSON path; a Markdown table is written beside it")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for label, suffix in VARIANTS:
        stem = f"{args.prefix}_{suffix}"
        usage_path = results_dir / f"{stem}_token_usage.json"
        if not usage_path.exists():
            rows.append({"method": label, "result_name": stem, "status": "missing"})
            continue
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
        online = usage.get("online", {})
        rows.append({
            "method": label,
            "result_name": stem,
            "status": "complete",
            "coverage": number(usage.get("coverage"), 4),
            "avg_f1": number(read_f1(results_dir / f"{stem}_official_eval.json")),
            "online_tokens_per_qa": number(usage.get("mean_online_total_tokens_per_qa")),
            "api_latency_seconds_per_call": number(usage.get("mean_online_api_latency_seconds_per_call")),
            "completion_tokens_per_second": number(usage.get("online_completion_tokens_per_second")),
            "api_calls": online.get("calls"),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"prefix": args.prefix, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "| Variant | Coverage | Avg F1 | Online tokens / QA | API s / call | Completion tokens / s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "complete":
            markdown.append(f"| {row['method']} | missing | missing | missing | missing | missing |")
            continue
        markdown.append(
            f"| {row['method']} | {display(row['coverage'] * 100)}% | {display(row['avg_f1'])} | "
            f"{display(row['online_tokens_per_qa'])} | {display(row['api_latency_seconds_per_call'])} | "
            f"{display(row['completion_tokens_per_second'])} |"
        )
    markdown.extend([
        "",
        "`Completion tokens / s` is completion tokens divided by successful provider API wall time. "
        "It excludes graph construction, local retrieval, failed retries, and retry backoff.",
    ])
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown))
    print(f"\nEfficiency JSON: {output}")
    print(f"Efficiency table: {markdown_path}")


if __name__ == "__main__":
    main()
