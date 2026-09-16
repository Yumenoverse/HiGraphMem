#!/usr/bin/env python3
"""Induce Triggered Working Memory from graph artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_CAPACITY = 6


def induce_global_graph_memory(metadata: dict, question: str = "") -> dict:
    """Describe a full-graph lexical retriever without constructing TWMem.

    This is the controlled replacement for TWMem ablation: GMGraph and the
    graph-evidence budget remain available, but there is no bounded,
    question-triggered buffer or graph propagation.
    """
    return {
        "conversation_id": metadata.get("conversation_id", ""),
        "memory_type": "global_graph_retrieval_baseline",
        "source_scope": "graph_and_dialogue_only_qa_masked",
        "trigger_model": {
            # retrieve.py uses this flag only to decide whether to restrict
            # candidates to a working buffer. True leaves all graph nodes in
            # the lexical candidate set; it does not denote gated activation.
            "gated_activation_enabled": True,
            "selection_mode": "full_graph_lexical",
            "capacity_limit": None,
            "graph_propagation": "disabled",
        },
        "activation_model": {"enabled": False, "propagation_steps": 0, "damping": 0.0},
        "question_focus": {"question": question, "intent": _intent(question)},
        "activated_nodes": [],
        "static_buffer_nodes": [],
        "working_evidence": [],
        "open_domain_clues": [],
        "active_operations": [],
        "audit": {"gold_answer_visible": False, "gold_evidence_visible": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question", default="")
    parser.add_argument("--disable-gated-activation", action="store_true", help="Ablation: disable query/intent activation and graph propagation; fill the bounded buffer using static graph weights.")
    parser.add_argument("--disable-relation-gate", action="store_true", help="Ablation: all graph relations propagate with equal coefficient.")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    nodes = _read_jsonl(graph_dir / "nodes.jsonl")
    edges = _read_jsonl(graph_dir / "edges.jsonl")
    metadata = json.loads((graph_dir / "graph_metadata.json").read_text(encoding="utf-8"))
    memory = induce_working_memory(
        metadata,
        nodes,
        edges,
        args.question,
        enable_gated_activation=not args.disable_gated_activation,
        enable_relation_gate=not args.disable_relation_gate,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def induce_working_memory(
    metadata: dict,
    nodes: list[dict],
    edges: list[dict],
    question: str = "",
    enable_gated_activation: bool = True,
    enable_relation_gate: bool = True,
) -> dict:
    question_tokens = set(_tokens(question))
    node_by_id = {node["id"]: node for node in nodes}
    intent = _intent(question)
    capacity = _capacity_for_intent(intent)
    balanced_activations = []
    static_buffer_nodes = []
    if enable_gated_activation:
        initial = {}
        activation_reasons = {}
        for node in nodes:
            score, reasons = _activation_score(node, question_tokens, intent)
            if score > 0:
                initial[node["id"]] = score
                activation_reasons[node["id"]] = reasons
        propagated, propagation_trace = _propagate_activation(
            initial,
            node_by_id,
            edges,
            steps=2,
            damping=0.58,
            enable_relation_gate=enable_relation_gate,
        )
        activations = []
        for node_id, score in propagated.items():
            node = node_by_id.get(node_id)
            if not node:
                continue
            activations.append(
                {
                    "node_id": node["id"],
                    "node_type": node["type"],
                    "label": node["label"],
                    "activation": round(score, 3),
                    "base_weight": node.get("weight", 0.0),
                    "reasons": activation_reasons.get(node_id, []) + propagation_trace.get(node_id, [])[:3],
                }
            )
        activations.sort(key=lambda item: item["activation"], reverse=True)
        balanced_activations = _select_balanced_activations(activations, intent, capacity)
        buffer_ids = {item["node_id"] for item in balanced_activations}
    else:
        # Preserve a bounded working buffer without performing any query- or
        # intent-conditioned activation. This keeps the ablation distinct from
        # w/o TWMem, which bypasses this module and injects no graph evidence.
        ranked_nodes = sorted(
            nodes,
            key=lambda node: (
                float(node.get("weight", node.get("confidence", 0.0))),
                node.get("id", ""),
            ),
            reverse=True,
        )
        static_buffer_nodes = [
            {
                "node_id": node["id"],
                "node_type": node["type"],
                "label": node["label"],
                "base_weight": node.get("weight", node.get("confidence", 0.0)),
                "selection_reason": "static_long_term_weight",
            }
            for node in ranked_nodes[:capacity]
        ]
        buffer_ids = {item["node_id"] for item in static_buffer_nodes}

    aliases = _alias_memory(nodes, edges, buffer_ids)
    open_domain_clues = _open_domain_clues(
        nodes,
        edges,
        buffer_ids,
        enable_gated_activation and intent == "open_domain",
        question_tokens if enable_gated_activation else set(),
    )
    working_evidence = _evidence_paths(edges, node_by_id, buffer_ids)
    operations = _active_operations(intent, open_domain_clues) if enable_gated_activation else []

    return {
        "conversation_id": metadata.get("conversation_id", ""),
        "memory_type": "short_term_triggered_working_memory",
        "source_scope": "graph_and_dialogue_only_qa_masked",
        "trigger_model": {
            "gated_activation_enabled": enable_gated_activation,
            "bottom_up_activation": "query match plus consolidated long-term node weight" if enable_gated_activation else "disabled",
            "top_down_control": "intent-specific operation templates bias graph node types and bridge clues" if enable_gated_activation else "disabled",
            "graph_propagation": "two-step weighted activation propagation over consolidated graph edges" if enable_gated_activation else "disabled",
            "relation_gate_enabled": enable_gated_activation and enable_relation_gate,
            "capacity_limit": capacity,
            "capacity_policy": "multi_hop=6, temporal=3, open_domain=3, single_hop=1",
            "type_balancing": "per-type caps prevent high-frequency person nodes from saturating working memory",
            "decay": "inactive nodes are excluded from the current working buffer",
        },
        "activation_model": {
            "enabled": enable_gated_activation,
            "formula": "a0(v|q)=query_match(q,v)+intent_bias(type(v))+prior_weight(v); a_{k+1}(v)=clip(a_k(v)+damping*sum_u a_k(u)*w(u,v))" if enable_gated_activation else "disabled",
            "propagation_steps": 2 if enable_gated_activation else 0,
            "damping": 0.58 if enable_gated_activation else 0.0,
        },
        "question_focus": {
            "question": question,
            "entities": _question_entities(question),
            "intent": intent,
            "answer_type": _answer_type(intent, question),
        },
        "aliases": aliases,
        "activated_nodes": balanced_activations,
        "static_buffer_nodes": static_buffer_nodes,
        "working_evidence": working_evidence[:capacity],
        "open_domain_clues": open_domain_clues[:capacity],
        "active_operations": operations,
        "constraints": [
            "Use only activated graph nodes, evidence spans, and explicit open-domain clues.",
            "Allow at most one public-knowledge bridge for open-domain questions.",
            "Return No information available when no evidence path supports the answer.",
        ],
        "audit": {
            "gold_answer_visible": False,
            "gold_evidence_visible": False,
            "test_error_visible": False,
            "global_skill_update": False,
        },
    }


def _activation_score(node: dict, question_tokens: set[str], intent: str) -> tuple[float, list[str]]:
    label_tokens = set(_tokens(node.get("label", "")))
    attrs_tokens = set(_tokens(json.dumps(node.get("attrs", {}), ensure_ascii=False)))
    reasons = []
    prior = 0.35 * float(node.get("weight", node.get("confidence", 0.0)))
    score = prior
    if prior:
        reasons.append("consolidated_prior:%.3f" % prior)
    overlap = question_tokens & (label_tokens | attrs_tokens)
    if overlap:
        score += 1.0 + 0.12 * len(overlap)
        reasons.append("question_overlap:" + ",".join(sorted(overlap)[:6]))
    intent_bias = _intent_node_bias(intent, node.get("type", ""))
    if intent_bias:
        score += intent_bias
        reasons.append("intent_bias:%s:%.2f" % (intent, intent_bias))
    if node.get("type") in {"OpenDomainClue", "Person", "Event", "State"}:
        score += 0.25
        reasons.append("high_value_memory_type")
    if node.get("type") == "EvidenceSpan":
        score += 0.05
        reasons.append("raw_text_fallback")
    return score, reasons


def _capacity_for_intent(intent: str) -> int:
    return {
        "multi_hop": 6,
        "temporal": 3,
        "open_domain": 3,
        "single_hop": 1,
    }.get(intent, DEFAULT_CAPACITY)


def _select_balanced_activations(activations: list[dict], intent: str, capacity: int) -> list[dict]:
    caps = {
        "Person": 2,
        "EvidenceSpan": 4,
        "ObservationFact": 3,
        "OpenDomainClue": 4,
        "Object": 4,
        "Event": 4,
        "State": 3,
        "Time": 3,
        "Preference": 3,
        "Goal": 3,
    }
    required_by_intent = {
        "open_domain": ["Person", "OpenDomainClue", "Object", "EvidenceSpan"],
        "temporal": ["Person", "Time", "Event", "State", "EvidenceSpan"],
        "multi_hop": ["Person", "Event", "State", "Preference", "EvidenceSpan"],
        "single_hop": ["Person", "EvidenceSpan", "ObservationFact"],
    }
    selected = []
    counts = {}
    used = set()
    for node_type in required_by_intent.get(intent, []):
        for item in activations:
            if item["node_id"] in used or item["node_type"] != node_type:
                continue
            selected.append(item)
            used.add(item["node_id"])
            counts[node_type] = counts.get(node_type, 0) + 1
            break
    for item in activations:
        if len(selected) >= capacity:
            break
        if item["node_id"] in used:
            continue
        node_type = item["node_type"]
        if counts.get(node_type, 0) >= caps.get(node_type, 3):
            continue
        selected.append(item)
        used.add(item["node_id"])
        counts[node_type] = counts.get(node_type, 0) + 1
    return selected[:capacity]


def _propagate_activation(initial: dict[str, float], nodes: dict[str, dict], edges: list[dict], steps: int, damping: float, enable_relation_gate: bool = True) -> tuple[dict[str, float], dict[str, list[str]]]:
    activation = {node_id: min(1.0, score) for node_id, score in initial.items()}
    trace = {node_id: [] for node_id in activation}
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], []).append((edge["target"], edge))
        adjacency.setdefault(edge["target"], []).append((edge["source"], edge))
    for step in range(steps):
        increments = {}
        for src, score in activation.items():
            src_type = nodes.get(src, {}).get("type", "")
            for dst, edge in adjacency.get(src, []):
                dst_type = nodes.get(dst, {}).get("type", "")
                if src_type == "EvidenceSpan" and dst_type not in {"Person", "Event", "State", "Time", "OpenDomainClue"}:
                    continue
                if dst_type == "EvidenceSpan" and edge.get("type") not in {"SUPPORTS", "HAPPENED_AT", "VALID_DURING"}:
                    continue
                edge_weight = float(edge.get("weight", edge.get("confidence", 0.0)))
                relation_gate = _propagation_gate(edge.get("type", "")) if enable_relation_gate else 1.0
                delta = damping * score * edge_weight * relation_gate
                if delta <= 0.015:
                    continue
                increments[dst] = increments.get(dst, 0.0) + delta
                trace.setdefault(dst, []).append("propagated_step_%s:%s:%.3f" % (step + 1, edge.get("type", ""), delta))
        for node_id, delta in increments.items():
            activation[node_id] = min(1.0, activation.get(node_id, 0.0) + delta)
    return activation, trace


def _intent_node_bias(intent: str, node_type: str) -> float:
    table = {
        "open_domain": {"OpenDomainClue": 0.45, "Object": 0.20, "Person": 0.15, "EvidenceSpan": 0.08},
        "temporal": {"Time": 0.45, "Event": 0.25, "State": 0.20, "EvidenceSpan": 0.08},
        "multi_hop": {"Person": 0.25, "Event": 0.25, "State": 0.18, "Preference": 0.18},
        "single_hop": {"EvidenceSpan": 0.20, "ObservationFact": 0.18, "Person": 0.12},
    }
    return table.get(intent, {}).get(node_type, 0.0)


def _propagation_gate(edge_type: str) -> float:
    return {
        "SUPPORTS": 0.75,
        "SUMMARIZES": 0.72,
        "HAS_STATE": 0.82,
        "HAS_PREFERENCE": 0.78,
        "HAS_GOAL": 0.74,
        "PARTICIPATES_IN": 0.75,
        "BRIDGES_TO": 0.68,
        "VALID_DURING": 0.62,
        "HAPPENED_AT": 0.55,
        "TEMPORAL_NEXT": 0.35,
        "MENTIONS_OBJECT": 0.48,
        "ALIAS_HINT": 0.42,
    }.get(edge_type, 0.35)


def _alias_memory(nodes: list[dict], edges: list[dict], activated_ids: set[str]) -> list[dict]:
    aliases = []
    person_ids = {node["id"]: node for node in nodes if node["type"] == "Person"}
    for edge in edges:
        if edge["type"] == "ALIAS_HINT" and edge["source"] in person_ids and (not activated_ids or edge["source"] in activated_ids):
            aliases.append({
                "canonical": person_ids[edge["source"]]["label"],
                "aliases": [edge.get("attrs", {}).get("alias", "")],
                "evidence_ids": edge.get("source_ids", []),
            })
    return aliases


def _open_domain_clues(nodes: list[dict], edges: list[dict], activated_ids: set[str], allow_top_clues: bool = False, question_tokens: set[str] | None = None) -> list[dict]:
    clues = []
    node_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        if node["type"] != "OpenDomainClue":
            continue
        label_tokens = set(_tokens(node.get("label", "")))
        has_question_overlap = bool((question_tokens or set()) & label_tokens)
        if activated_ids and node["id"] not in activated_ids:
            incoming = [edge for edge in edges if edge["target"] == node["id"] and edge["source"] in activated_ids]
            if not incoming and not (allow_top_clues and (has_question_overlap or float(node.get("weight", 0.0)) >= 0.52)):
                continue
        bridge_type = node.get("attrs", {}).get("bridge_type", "open_domain_clue")
        supporters = [edge for edge in edges if edge["target"] == node["id"] and edge["type"] == "SUPPORTS"]
        clues.append({
            "subject": _nearest_person_label(node["id"], edges, node_by_id),
            "clue": node["label"],
            "bridge_type": bridge_type,
            "allowed_external_steps": 1,
            "evidence_ids": sorted({sid for edge in supporters for sid in edge.get("source_ids", [])}),
            "weight": node.get("weight", 0.0),
        })
    return sorted(clues, key=lambda item: float(item.get("weight", 0.0)), reverse=True)


def _evidence_paths(edges: list[dict], node_by_id: dict, activated_ids: set[str]) -> list[dict]:
    paths = []
    for edge in edges:
        if edge["source"] in activated_ids or edge["target"] in activated_ids:
            src = node_by_id.get(edge["source"])
            dst = node_by_id.get(edge["target"])
            if not src or not dst:
                continue
            paths.append({
                "path": [src["id"], edge["type"], dst["id"]],
                "labels": [src["label"], edge["type"], dst["label"]],
                "raw_span_ids": edge.get("source_ids", []),
                "confidence": edge.get("confidence", 0.0),
            })
    return sorted(paths, key=lambda item: item["confidence"], reverse=True)


def _active_operations(intent: str, clues: list[dict]) -> list[dict]:
    operations = []
    if intent == "open_domain":
        for clue in clues:
            operations.append({
                "skill": "single_step_open_domain_bridge",
                "bridge_type": clue["bridge_type"],
                "input_clue": clue["clue"],
                "allowed_external_steps": 1,
            })
    if intent == "temporal":
        operations.append({"skill": "temporal_state_resolution", "input": "activated Time and VALID_DURING edges"})
    if intent == "multi_hop":
        operations.append({"skill": "multi_hop_graph_path_composition", "input": "activated graph paths"})
    return operations


def _nearest_person_label(node_id: str, edges: list[dict], node_by_id: dict) -> str:
    for edge in edges:
        if edge["target"] == node_id:
            src = node_by_id.get(edge["source"], {})
            if src.get("type") == "Person":
                return src.get("label", "")
    return ""


def _intent(question: str) -> str:
    lowered = question.lower()
    if lowered.startswith(("when", "how long", "what year", "what month")):
        return "temporal"
    if any(token in lowered for token in ["would", "could", "likely", "what console", "what game", "what country", "what state", "condition", "technique"]):
        return "open_domain"
    if any(token in lowered for token in ["both", "in common", "which events", "what activities", "where has", "how many"]):
        return "multi_hop"
    return "single_hop"


def _answer_type(intent: str, question: str) -> str:
    if intent == "temporal":
        return "date_or_duration"
    if intent == "open_domain":
        return "short_entity_or_concept"
    if "how many" in question.lower():
        return "number_or_list"
    return "short_span"


def _question_entities(question: str) -> list[str]:
    return re.findall(r"\b[A-Z][a-zA-Z]+\b", question)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
