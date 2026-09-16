#!/usr/bin/env python3
"""Render a small, evidence-centered view of a HiGraphMem memory graph."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


TYPE_COLORS = {
    "Person": "#4C78A8",
    "Event": "#F58518",
    "State": "#54A24B",
    "Time": "#B279A2",
    "Preference": "#E45756",
    "Goal": "#72B7B2",
    "Object": "#9D755D",
    "OpenDomainClue": "#FF9DA6",
    "ObservationFact": "#79706E",
    "EvidenceSpan": "#BAB0AC",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def shorten(value: str, limit: int = 76) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-neighbors", type=int, default=14)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    nodes = {item["id"]: item for item in read_jsonl(graph_dir / "nodes.jsonl")}
    edges = read_jsonl(graph_dir / "edges.jsonl")

    active_ids = {item["node_id"] for item in bundle.get("evidence_nodes", []) if item.get("node_id") in nodes}
    if not active_ids:
        raise SystemExit("The evidence bundle has no graph-backed evidence nodes.")

    incident = [edge for edge in edges if edge["source"] in active_ids or edge["target"] in active_ids]
    incident.sort(key=lambda edge: (-float(edge.get("confidence", 0.0)), edge["type"]))
    selected_edges = incident[: args.max_neighbors]
    selected_ids = set(active_ids)
    for edge in selected_edges:
        selected_ids.add(edge["source"])
        selected_ids.add(edge["target"])

    dot_lines = [
        "digraph memory {",
        'graph [rankdir=LR, bgcolor="white", pad="0.25", nodesep="0.45", ranksep="0.7", fontname="Helvetica"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.10,0.07"];',
        'edge [fontname="Helvetica", fontsize=8, color="#68737D", arrowsize=0.65];',
        'label=' + quote("Long-term graph neighborhood with the QA-triggered short-term selection") + ";",
        'labelloc="t"; fontsize=16; fontname="Helvetica";',
    ]

    for node_id in sorted(selected_ids):
        node = nodes[node_id]
        node_type = node["type"]
        label = f"{node_type}\\n{shorten(str(node.get('label', '')))}"
        fill = TYPE_COLORS.get(node_type, "#D3D3D3")
        attrs = [f"fillcolor={quote(fill)}"]
        if node_id in active_ids:
            attrs.extend(['color="#1F4E79"', "penwidth=3"])
        else:
            attrs.extend(['color="#4B5563"', "penwidth=1"])
        dot_lines.append(f"{quote(node_id)} [label={quote(label)}, {', '.join(attrs)}];")

    for edge in selected_edges:
        if edge["source"] in selected_ids and edge["target"] in selected_ids:
            label = shorten(str(edge["type"]), 28)
            dot_lines.append(
                f"{quote(edge['source'])} -> {quote(edge['target'])} "
                f"[label={quote(label)}, penwidth={1 + float(edge.get('confidence', 0.0)):.2f}];"
            )

    question = shorten(str(bundle.get("question", "")), 120)
    question_label = "Question\\n" + question
    dot_lines.append(
        f'question [shape=note, fillcolor="#FFF2CC", color="#B8860B", label={quote(question_label)}];'
    )
    for node_id in sorted(active_ids):
        dot_lines.append(f"question -> {quote(node_id)} [style=dashed, color=\"#1F4E79\", label=\"triggers\"];")
    dot_lines.append("}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dot_path = output.with_suffix(".dot")
    dot_path.write_text("\n".join(dot_lines) + "\n", encoding="utf-8")
    subprocess.run(["dot", f"-T{output.suffix.lstrip('.')}", str(dot_path), "-o", str(output)], check=True)
    print(output)


if __name__ == "__main__":
    main()
