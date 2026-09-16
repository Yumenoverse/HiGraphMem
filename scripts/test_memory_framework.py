#!/usr/bin/env python3
"""Run framework-level tests for HiGraphMem memory skills."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCRIPT = ROOT / "skills/memory_construction_subskills/build_typed_temporal_graph/scripts/build_graph.py"
STM_SCRIPT = ROOT / "skills/memory_construction_subskills/induce_short_term_working_memory/scripts/induce_memory.py"
RETRIEVE_SCRIPT = ROOT / "skills/memory_construction_subskills/retrieve_long_short_memory/scripts/retrieve.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--data-file", default="data/locomo/data/locomo10.json")
    parser.add_argument("--output-dir", default="experiments/framework_tests")
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--questions-per-category", type=int, default=10)
    args = parser.parse_args()

    graph_mod = _load_module(GRAPH_SCRIPT, "higraphmem2_skill_graph")
    stm_mod = _load_module(STM_SCRIPT, "higraphmem2_skill_stm")
    retrieve_mod = _load_module(RETRIEVE_SCRIPT, "higraphmem2_skill_retrieve")

    source_root = Path(args.source_root)
    samples = json.loads((source_root / args.data_file).read_text(encoding="utf-8"))
    if args.max_conversations > 0:
        samples = samples[: args.max_conversations]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_rows = []
    weight_rows = []
    trigger_rows = []
    clue_rows = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", "unknown"))
        graph = graph_mod.build_graph(sample)
        sample_dir = output_dir / sample_id
        graph_dir = sample_dir / "long_term_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(graph_dir / "nodes.jsonl", graph["nodes"])
        _write_jsonl(graph_dir / "edges.jsonl", graph["edges"])
        (graph_dir / "graph_metadata.json").write_text(json.dumps(graph["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")

        extraction_rows.append(_extraction_summary(sample_id, graph["nodes"], graph["edges"]))
        weight_rows.extend(_weight_summary(sample_id, graph["nodes"], graph["edges"]))
        clue_rows.extend(_clue_audit_rows(sample_id, graph["nodes"]))

        for qa in _select_questions(sample, args.questions_per_category):
            question = str(qa.get("question", ""))
            stm = stm_mod.induce_working_memory(graph["metadata"], graph["nodes"], graph["edges"], question)
            bundle = retrieve_mod.retrieve(question, graph["nodes"], graph["edges"], stm, top_k=10)
            trigger_rows.append(_trigger_summary(sample_id, qa, stm, bundle))

    extraction_report = {
        "audit": _audit(args),
        "rows": extraction_rows,
        "totals": _extraction_totals(extraction_rows),
    }
    weight_report = {
        "audit": _audit(args),
        "rows": weight_rows,
        "checks": _weight_checks(weight_rows),
    }
    trigger_report = {
        "audit": _audit(args, questions_visible=True),
        "rows": trigger_rows,
        "summary": _trigger_totals(trigger_rows),
    }
    clue_report = {
        "audit": _audit(args),
        "rows": clue_rows,
        "summary": _clue_totals(clue_rows),
    }
    (output_dir / "extraction_quality.json").write_text(json.dumps(extraction_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "weight_mechanism.json").write_text(json.dumps(weight_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "short_term_triggering.json").write_text(json.dumps(trigger_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "clue_audit.json").write_text(json.dumps(clue_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir)
    print("extraction:", extraction_report["totals"])
    print("weights:", weight_report["checks"])
    print("trigger:", trigger_report["summary"])
    print("clues:", clue_report["summary"])


def _extraction_summary(sample_id: str, nodes: list[dict], edges: list[dict]) -> dict:
    node_counts = Counter(node["type"] for node in nodes)
    edge_counts = Counter(edge["type"] for edge in edges)
    return {
        "sample_id": sample_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": dict(sorted(node_counts.items())),
        "edge_types": dict(sorted(edge_counts.items())),
        "top_nodes_by_weight": _top_items(nodes, "weight", fields=("id", "type", "label", "weight")),
        "top_edges_by_weight": _top_items(edges, "weight", fields=("source", "type", "target", "weight")),
    }


def _weight_summary(sample_id: str, nodes: list[dict], edges: list[dict]) -> list[dict]:
    rows = []
    for group_name, items in (("node", nodes), ("edge", edges)):
        by_type = defaultdict(list)
        for item in items:
            by_type[item["type"]].append(float(item.get("weight", 0.0)))
        for item_type, weights in sorted(by_type.items()):
            rows.append(
                {
                    "sample_id": sample_id,
                    "group": group_name,
                    "type": item_type,
                    "count": len(weights),
                    "avg_weight": round(mean(weights), 4),
                    "max_weight": round(max(weights), 4),
                    "min_weight": round(min(weights), 4),
                }
            )
    return rows


def _trigger_summary(sample_id: str, qa: dict, stm: dict, bundle: dict) -> dict:
    activated_types = Counter(item.get("node_type", "") for item in stm.get("activated_nodes", []))
    dominant_type, dominant_count = activated_types.most_common(1)[0] if activated_types else ("", 0)
    return {
        "sample_id": sample_id,
        "category": str(qa.get("category", "unknown")),
        "question": qa.get("question", ""),
        "intent": stm.get("question_focus", {}).get("intent", "unknown"),
        "answer_type": stm.get("question_focus", {}).get("answer_type", "unknown"),
        "activated_count": len(stm.get("activated_nodes", [])),
        "activated_types": dict(sorted(activated_types.items())),
        "dominant_activated_type": dominant_type,
        "dominant_activated_ratio": round(dominant_count / max(len(stm.get("activated_nodes", [])), 1), 4),
        "type_saturation_flag": len(stm.get("activated_nodes", [])) >= 4 and dominant_count / max(len(stm.get("activated_nodes", [])), 1) > 0.5,
        "working_evidence_count": len(stm.get("working_evidence", [])),
        "open_domain_clue_count": len(stm.get("open_domain_clues", [])),
        "active_operation_count": len(stm.get("active_operations", [])),
        "retrieved_nodes": len(bundle.get("evidence_nodes", [])),
        "retrieved_paths": len(bundle.get("evidence_paths", [])),
        "gold_answer_visible": False,
    }


def _clue_audit_rows(sample_id: str, nodes: list[dict]) -> list[dict]:
    rows = []
    for node in nodes:
        if node.get("type") != "OpenDomainClue":
            continue
        attrs = dict(node.get("attrs", {}))
        text = str(attrs.get("text", node.get("label", "")))
        rows.append(
            {
                "sample_id": sample_id,
                "node_id": node.get("id", ""),
                "bridge_type": attrs.get("bridge_type", ""),
                "value": attrs.get("value", ""),
                "label": node.get("label", ""),
                "weight": node.get("weight", 0.0),
                "source_ids": node.get("source_ids", []),
                "low_quality_flag": _low_quality_clue(text, attrs),
            }
        )
    return rows


def _select_questions(sample: dict, per_category: int) -> list[dict]:
    grouped = defaultdict(list)
    for qa in sample.get("qa", []):
        grouped[str(qa.get("category", "unknown"))].append(qa)
    selected = []
    for category in sorted(grouped):
        selected.extend(grouped[category][:per_category])
    return selected


def _extraction_totals(rows: list[dict]) -> dict:
    nodes = Counter()
    edges = Counter()
    for row in rows:
        nodes.update(row["node_types"])
        edges.update(row["edge_types"])
    return {
        "conversation_count": len(rows),
        "node_types": dict(sorted(nodes.items())),
        "edge_types": dict(sorted(edges.items())),
    }


def _weight_checks(rows: list[dict]) -> dict:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row["avg_weight"])
    return {
        group: {
            "type_count": len(values),
            "avg_of_avg_weights": round(mean(values), 4) if values else 0.0,
            "has_nonzero_weights": any(value > 0 for value in values),
        }
        for group, values in sorted(by_group.items())
    }


def _trigger_totals(rows: list[dict]) -> dict:
    by_intent = Counter(row["intent"] for row in rows)
    by_category = Counter(row["category"] for row in rows)
    open_rows = [row for row in rows if row["intent"] == "open_domain"]
    return {
        "question_count": len(rows),
        "by_intent": dict(sorted(by_intent.items())),
        "by_category": dict(sorted(by_category.items())),
        "open_domain_questions": len(open_rows),
        "open_domain_with_clues": sum(1 for row in open_rows if row["open_domain_clue_count"] > 0),
        "type_saturation_flags": sum(1 for row in rows if row.get("type_saturation_flag")),
        "avg_activated_count": round(mean([row["activated_count"] for row in rows]), 4) if rows else 0.0,
    }


def _clue_totals(rows: list[dict]) -> dict:
    by_type = Counter(row.get("bridge_type", "") for row in rows)
    low_quality = sum(1 for row in rows if row.get("low_quality_flag"))
    return {
        "clue_count": len(rows),
        "by_bridge_type": dict(sorted(by_type.items())),
        "low_quality_flag_count": low_quality,
        "low_quality_flag_ratio": round(low_quality / max(len(rows), 1), 4),
        "top_examples": sorted(rows, key=lambda row: float(row.get("weight", 0.0)), reverse=True)[:12],
    }


def _low_quality_clue(text: str, attrs: dict) -> bool:
    lowered = text.lower()
    if len(text.split()) > 35:
        return True
    if attrs.get("bridge_type") == "location_clue_to_place" and not any(token in lowered for token in ["visited", "trip", "moved", "country", "park", "camped"]):
        return True
    if attrs.get("bridge_type") == "entertainment_clue_to_named_item" and "game for" in lowered:
        return True
    return False


def _top_items(items: list[dict], key: str, fields: tuple[str, ...]) -> list[dict]:
    output = []
    for item in sorted(items, key=lambda value: float(value.get(key, 0.0)), reverse=True)[:8]:
        output.append({field: item.get(field, "") for field in fields})
    return output


def _audit(args: argparse.Namespace, questions_visible: bool = False) -> dict:
    return {
        "source_root": args.source_root,
        "data_file": args.data_file,
        "questions_visible": questions_visible,
        "gold_answer_visible": False,
        "gold_evidence_visible": False,
        "test_error_visible": False,
        "global_skill_update": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
