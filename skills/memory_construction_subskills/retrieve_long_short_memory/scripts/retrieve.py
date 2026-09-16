#!/usr/bin/env python3
"""Compose an evidence bundle from graph and Triggered Working Memory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--working-memory", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    nodes = _read_jsonl(Path(args.graph_dir) / "nodes.jsonl")
    edges = _read_jsonl(Path(args.graph_dir) / "edges.jsonl")
    working_memory = json.loads(Path(args.working_memory).read_text(encoding="utf-8"))
    bundle = retrieve(args.question, nodes, edges, working_memory, args.top_k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieve(question: str, nodes: list[dict], edges: list[dict], working_memory: dict, top_k: int) -> dict:
    tokens = set(_tokens(question))
    activated = {item["node_id"]: item.get("activation", 0.0) for item in working_memory.get("activated_nodes", [])}
    activation_enabled = working_memory.get("trigger_model", {}).get("gated_activation_enabled", True)
    static_buffer_ids = {item["node_id"] for item in working_memory.get("static_buffer_nodes", [])}
    node_by_id = {node["id"]: node for node in nodes}
    candidates = []
    for node in nodes:
        if not activation_enabled and node["id"] not in static_buffer_ids:
            continue
        lexical = len(tokens & set(_tokens(node.get("label", "") + " " + json.dumps(node.get("attrs", {}), ensure_ascii=False))))
        score = lexical * 0.35 + activated.get(node["id"], 0.0)
        if node["type"] in {"EvidenceSpan", "ObservationFact"}:
            score += 0.15
        if score > 0:
            candidates.append({
                "node_id": node["id"],
                "node_type": node["type"],
                "label": node["label"],
                "source_ids": node.get("source_ids", []),
                "attrs": node.get("attrs", {}),
                "score": round(score, 3),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected_ids = {item["node_id"] for item in candidates[:top_k]}
    paths = []
    for edge in edges:
        if edge["source"] in selected_ids or edge["target"] in selected_ids:
            src = node_by_id.get(edge["source"])
            dst = node_by_id.get(edge["target"])
            if src and dst:
                paths.append({
                    "path": [edge["source"], edge["type"], edge["target"]],
                    "labels": [src["label"], edge["type"], dst["label"]],
                    "source_ids": edge.get("source_ids", []),
                    "confidence": edge.get("confidence", 0.0),
                })
    paths.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "question": question,
        "source_scope": "question_plus_memory_only_no_gold_labels",
        "intent": working_memory.get("question_focus", {}).get("intent", "unknown"),
        "evidence_nodes": candidates[:top_k],
        "evidence_paths": paths[:top_k],
        "open_domain_clues": working_memory.get("open_domain_clues", [])[:top_k],
        "active_operations": working_memory.get("active_operations", []),
        "gated_activation_enabled": activation_enabled,
        "selection_mode": working_memory.get("trigger_model", {}).get("selection_mode", "twmem_gated" if activation_enabled else "static_buffer"),
        "audit": {
            "question_visible": True,
            "gold_answer_visible": False,
            "gold_evidence_visible": False,
            "test_error_visible": False,
        },
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
