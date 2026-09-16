#!/usr/bin/env python3
"""Build a conversation-local Grounded MemoryGraph from QA-masked LoCoMo data."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NODE_TYPES = {
    "Person",
    "Event",
    "State",
    "Time",
    "Place",
    "Object",
    "Topic",
    "Preference",
    "Goal",
    "EvidenceSpan",
    "ObservationFact",
    "OpenDomainClue",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="A single LoCoMo sample JSON file, or a LoCoMo list JSON.")
    parser.add_argument("--sample-id", default="", help="Required when --input contains multiple samples.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--disable-update", action="store_true", help="Ablation: initialize GMGraph from the first session and freeze it; do not ingest later sessions.")
    parser.add_argument("--disable-conflict-control", action="store_true", help="Ablation: retain Update but remove stale/conflicting-state inhibition.")
    args = parser.parse_args()

    sample = _load_sample(Path(args.input), args.sample_id)
    graph = build_graph(sample, enable_update=not args.disable_update, enable_conflict_control=not args.disable_conflict_control)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "nodes.jsonl", graph["nodes"])
    _write_jsonl(output_dir / "edges.jsonl", graph["edges"])
    (output_dir / "graph_metadata.json").write_text(
        json.dumps(graph["metadata"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_graph(sample: dict, enable_update: bool = True, enable_conflict_control: bool = True) -> dict:
    """Build GMGraph session by session without reading QA labels."""
    conversation = sample.get("conversation", {})
    sample_id = str(sample.get("sample_id") or sample.get("conversation_id") or "conversation")
    session_indices = sorted(
        int(match.group(1))
        for key, turns in conversation.items()
        if (match := re.fullmatch(r"session_([1-9]\d*)", key)) and turns
    )
    ingested_sessions = session_indices if enable_update else session_indices[:1]
    nodes = {}
    edges = {}
    entity_mentions = defaultdict(list)
    previous_session_node = ""
    previous_event_node = ""
    touched_nodes = set()
    touched_edges = set()
    consolidation_trace = {
        "algorithm": "incremental_session_update_v0.5" if enable_update else "disabled_initial_graph_frozen",
        "update_enabled": enable_update,
        "conflict_control_enabled": enable_update and enable_conflict_control,
        "steps": [],
        "node_reinforcements": 0,
        "edge_reinforcements": 0,
        "node_inhibitions": 0,
        "edge_inhibitions": 0,
        "num_steps": 0,
    }

    def add_node(node_type: str, label: str, source_ids: list[str], **attrs: object) -> str:
        assert node_type in NODE_TYPES
        node_id = attrs.pop("id", "") or _node_id(node_type, label)
        current = nodes.get(node_id)
        if current:
            current["source_ids"] = sorted(set(current["source_ids"]) | set(source_ids))
            current["confidence"] = max(current["confidence"], float(attrs.get("confidence", 0.7)))
            touched_nodes.add(node_id)
            return node_id
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "source_ids": sorted(set(source_ids)),
            "confidence": float(attrs.pop("confidence", 0.7)),
            "attrs": attrs,
        }
        touched_nodes.add(node_id)
        return node_id

    def add_edge(src: str, dst: str, edge_type: str, source_ids: list[str], confidence: float = 0.7, **attrs: object) -> None:
        if not src or not dst or src == dst:
            return
        key = (src, dst, edge_type)
        item = edges.get(key)
        if item:
            item["source_ids"] = sorted(set(item["source_ids"]) | set(source_ids))
            item["confidence"] = max(item["confidence"], confidence)
            touched_edges.add(key)
            return
        edges[key] = {
            "source": src,
            "target": dst,
            "type": edge_type,
            "source_ids": sorted(set(source_ids)),
            "confidence": confidence,
            "attrs": attrs,
        }
        touched_edges.add(key)

    for session_position, session_idx in enumerate(ingested_sessions):
        touched_nodes.clear()
        touched_edges.clear()
        session_key = f"session_{session_idx}"
        turns = conversation.get(session_key)
        date_time = str(conversation.get(f"{session_key}_date_time", ""))
        time_node = add_node("Time", date_time or f"session {session_idx}", [session_key], id=f"time:S{session_idx}", session=session_key)
        if previous_session_node:
            add_edge(previous_session_node, time_node, "TEMPORAL_NEXT", [session_key], 0.95)
        previous_session_node = time_node

        for turn in turns:
            source_id = str(turn.get("dia_id") or f"S{session_idx}:unknown")
            speaker = str(turn.get("speaker", "")).strip() or "Unknown"
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            evidence_node = add_node(
                "EvidenceSpan",
                f"{speaker}: {text}",
                [source_id],
                id=f"evidence:{source_id}",
                speaker=speaker,
                session=session_key,
                date_time=date_time,
                text=text,
            )
            person_node = add_node("Person", speaker, [source_id])
            entity_mentions[speaker.lower()].append(source_id)
            add_edge(evidence_node, person_node, "SUPPORTS", [source_id], 0.9, role="speaker")
            add_edge(evidence_node, time_node, "HAPPENED_AT", [source_id], 0.85)

            for extraction in _extract_memory_units(text, speaker=speaker, source_type="dialogue_turn"):
                subject = str(extraction.get("subject") or extraction.get("attrs", {}).get("subject") or speaker)
                subject_node = add_node("Person", subject, [source_id])
                unit_node = add_node(extraction["type"], extraction["label"], [source_id], confidence=extraction["confidence"], **extraction.get("attrs", {}))
                add_edge(subject_node, unit_node, extraction["person_edge"], [source_id], extraction["confidence"])
                add_edge(evidence_node, unit_node, "SUPPORTS", [source_id], 0.9)
                add_edge(unit_node, time_node, "HAPPENED_AT" if extraction["type"] == "Event" else "VALID_DURING", [source_id], 0.75)
                for temporal_ref in extraction.get("temporal_refs", []):
                    temporal_node = add_node(
                        "Time",
                        temporal_ref["label"],
                        [source_id],
                        id=f"timeexpr:{_safe_id(source_id)}:{_safe_id(temporal_ref['label'])[:48]}",
                        granularity=temporal_ref.get("granularity", "relative"),
                        normalized=temporal_ref.get("normalized", ""),
                        source_time=date_time,
                    )
                    add_edge(unit_node, temporal_node, temporal_ref.get("edge_type", "HAPPENED_AT"), [source_id], 0.88)
                    add_edge(temporal_node, time_node, "VALID_DURING", [source_id], 0.7, role="session_anchor")
                if extraction["type"] == "Event":
                    if previous_event_node:
                        add_edge(previous_event_node, unit_node, "TEMPORAL_NEXT", [source_id], 0.55)
                    previous_event_node = unit_node
                if extraction.get("open_domain_clue"):
                    clue_node = add_node(
                        "OpenDomainClue",
                        extraction["label"],
                        [source_id],
                        bridge_type=extraction["open_domain_clue"],
                    )
                    add_edge(unit_node, clue_node, "BRIDGES_TO", [source_id], 0.8)
                    add_edge(evidence_node, clue_node, "SUPPORTS", [source_id], 0.9)

        for observation_key, observations in sample.get("observation", {}).items():
            if not isinstance(observations, dict) or _max_session([observation_key]) != session_idx:
                continue
            time_node = f"time:S{session_idx}"
            for entity, facts in observations.items():
                person_node = add_node("Person", str(entity), [observation_key])
                for idx, fact_item in enumerate(facts):
                    fact_text, source_id = _observation_fact(fact_item, f"{observation_key}:{idx}")
                    fact_node = add_node(
                        "ObservationFact",
                        fact_text,
                        [source_id],
                        id=f"observation:{_safe_id(source_id)}",
                        entity=str(entity),
                        session=observation_key,
                    )
                    add_edge(fact_node, person_node, "SUMMARIZES", [source_id], 0.85)
                    if time_node in nodes:
                        add_edge(fact_node, time_node, "VALID_DURING", [source_id], 0.7)
                    for extraction in _extract_memory_units(fact_text, speaker=str(entity), source_type="observation_fact"):
                        subject = str(extraction.get("subject") or extraction.get("attrs", {}).get("subject") or entity)
                        subject_node = add_node("Person", subject, [source_id])
                        unit_node = add_node(extraction["type"], extraction["label"], [source_id], confidence=extraction["confidence"], **extraction.get("attrs", {}))
                        add_edge(subject_node, unit_node, extraction["person_edge"], [source_id], extraction["confidence"])
                        add_edge(fact_node, unit_node, "SUPPORTS", [source_id], 0.9)
                        for temporal_ref in extraction.get("temporal_refs", []):
                            temporal_node = add_node(
                                "Time",
                                temporal_ref["label"],
                                [source_id],
                                id=f"timeexpr:{_safe_id(source_id)}:{_safe_id(temporal_ref['label'])[:48]}",
                                granularity=temporal_ref.get("granularity", "relative"),
                                normalized=temporal_ref.get("normalized", ""),
                            )
                            add_edge(unit_node, temporal_node, temporal_ref.get("edge_type", "VALID_DURING"), [source_id], 0.88)

        alias_edges_before = set(edges)
        _add_alias_edges(nodes, edges, entity_mentions)
        touched_edges.update(set(edges) - alias_edges_before)
        # G1 is identical in full and frozen runs. Later sessions mutate G1
        # in place; strength must never depend on unseen sessions.
        if enable_update or session_position == 0:
            step_trace = _apply_streaming_update_step(
                nodes,
                edges,
                touched_nodes,
                touched_edges,
                step=session_idx,
                max_step=session_idx,
                enable_conflict_control=enable_conflict_control and session_position > 0,
            )
            step_trace["phase"] = "initialize" if session_position == 0 else "update"
            step_trace["node_count"] = len(nodes)
            step_trace["edge_count"] = len(edges)
            consolidation_trace["steps"].append(step_trace)
            for key in ("node_reinforcements", "edge_reinforcements", "node_inhibitions", "edge_inhibitions"):
                consolidation_trace[key] += step_trace[key]
            consolidation_trace["num_steps"] += 1

    return {
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(edges.values(), key=lambda item: (item["source"], item["target"], item["type"])),
        "metadata": {
            "schema": "higraphmem2-typed-temporal-graph-v0.3",
            "conversation_id": sample_id,
            "source_scope": "conversation_observation_only_qa_masked",
            "qa_answer_visible": False,
            "qa_evidence_visible": False,
            "qa_category_visible": False,
            "graph_style": "conversation_local_incremental_memory_graph",
            "graph_state_scope": "all_sessions_incrementally_updated" if enable_update else "initial_session_only_frozen",
            "available_sessions": session_indices,
            "ingested_sessions": ingested_sessions,
            "initialized_session": ingested_sessions[0] if ingested_sessions else None,
            "num_updates": max(0, len(ingested_sessions) - 1),
            "frozen_after_session": None if enable_update else (ingested_sessions[0] if ingested_sessions else None),
            "consolidation": {
                "algorithm": "incremental_session_update_v0.5" if enable_update else "disabled_initial_graph_frozen",
                "update_enabled": enable_update,
                "conflict_control_enabled": enable_update and enable_conflict_control,
                "decay_factor": 0.92 if enable_update else None,
                "reinforcement": "frequency+source_reliability+salience+extractor_confidence+temporal_relevance" if enable_update else "initialization_only",
                "inhibition": "same_entity_slot_conflict+stale_state_penalty" if enable_update and enable_conflict_control else "disabled",
                "activation": "query_prior_plus_weighted_graph_propagation",
                "trace": consolidation_trace,
            },
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def _extract_memory_units(text: str, speaker: str = "", source_type: str = "dialogue_turn") -> list[dict]:
    lowered = text.lower()
    units = []
    temporal_refs = _temporal_refs(text)
    subject = _subject_for_text(text, speaker)
    base_attrs = {
        "text": text,
        "speaker": speaker,
        "subject": subject,
        "source_type": source_type,
        "extraction_method": "schema_grounded_extractor_v0.3",
    }
    event = _event_frame(text, subject)
    if event:
        units.append({
            "type": "Event",
            "label": event["label"],
            "subject": subject,
            "person_edge": "PARTICIPATES_IN",
            "confidence": (0.80 if source_type == "observation_fact" else 0.70) + event.get("confidence_bonus", 0.0),
            "attrs": {
                **base_attrs,
                "slot": "event",
                "predicate": event["predicate"],
                "value": event["value"],
                "object": event.get("object", ""),
                "temporal_signal": event.get("temporal_signal", ""),
            },
            "temporal_refs": temporal_refs,
        })
    preference_match = re.search(r"\b(likes|loves|enjoys|favorite|interested in|prefers)\b", lowered)
    if preference_match:
        units.append({
            "type": "Preference",
            "label": _preference_label(text, subject, preference_match.group(1)),
            "subject": subject,
            "person_edge": "HAS_PREFERENCE",
            "confidence": 0.80 if source_type == "observation_fact" else 0.72,
            "attrs": {**base_attrs, "slot": "preference", "predicate": preference_match.group(1), "value": _preference_value(text)},
            "temporal_refs": temporal_refs,
        })
    goal_match = re.search(r"\b(plans to|planning|hopes to|wants to|aims to|goal)\b", lowered)
    if goal_match:
        units.append({
            "type": "Goal",
            "label": _goal_label(text, subject, goal_match.group(1)),
            "subject": subject,
            "person_edge": "HAS_GOAL",
            "confidence": 0.76 if source_type == "observation_fact" else 0.68,
            "attrs": {**base_attrs, "slot": "goal", "predicate": goal_match.group(1), "value": _goal_value(text)},
            "temporal_refs": temporal_refs,
        })
    state_slot = _state_slot(lowered)
    if state_slot:
        state_value = _state_value(text, state_slot)
        units.append({
            "type": "State",
            "label": _state_label(subject, state_slot, state_value, lowered),
            "subject": subject,
            "person_edge": "HAS_STATE",
            "confidence": _state_confidence(source_type, lowered),
            "attrs": {
                **base_attrs,
                "slot": state_slot,
                "value": state_value,
                "polarity": _state_polarity(lowered),
                "change_type": _state_change_type(lowered),
                "temporal_signal": ", ".join(item["label"] for item in temporal_refs),
            },
            "temporal_refs": temporal_refs,
        })
    bridge_type = _bridge_type(lowered)
    if bridge_type:
        units.append({
            "type": "Object",
            "label": _clue_label(text, bridge_type),
            "subject": subject,
            "person_edge": "MENTIONS_OBJECT",
            "confidence": 0.76 if source_type == "observation_fact" else 0.64,
            "attrs": {**base_attrs, "slot": "open_domain_clue", "bridge_type": bridge_type, "value": _clue_value(text, bridge_type)},
            "temporal_refs": temporal_refs,
            "open_domain_clue": bridge_type,
        })
    return units


def _event_frame(text: str, speaker: str = "") -> dict:
    lowered = text.lower()
    if re.search(r"\b(went wrong|get(?:ting)? lost in|lost in (?:a|the)?\s*(?:book|story|fantasy|world)|opened doors|opened me up|why you started this|started this)\b", lowered):
        return {}
    patterns = [
        ("lost_job", r"\b(lost)\s+(?:his|her|my|their)?\s*(?:job|position|work)\b", "lost job", 0.08),
        ("moved", r"\b(moved)\s+(?:to|from|into|back to)\s+([^.!?,;]+)", "moved", 0.06),
        ("visited", r"\b(visited|went to|traveled to|travelled to)\s+([^.!?,;]+)", "visited", 0.05),
        ("joined", r"\b(joined)\s+([^.!?,;]+)", "joined", 0.04),
        ("started", r"\b(started|launched)\s+([^.!?,;]+)", "started", 0.05),
        ("opened_business", r"\b(opened)\s+((?:an?|my|her|his|their|our|own)\s+[^.!?,;]*(?:store|studio|business|shop|company)[^.!?,;]*)", "opened", 0.06),
        ("attended", r"\b(attended|hosted|played|contacted|applied for)\s+([^.!?,;]+)", "participated", 0.03),
        ("read", r"\b(read|finished reading|picked up)\s+([^.!?,;]+)", "read", 0.03),
    ]
    for predicate, pattern, action, bonus in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        obj = _clean_phrase(match.group(2) if len(match.groups()) >= 2 else action)
        temporal_signal = ", ".join(item["label"] for item in _temporal_refs(text))
        subject = speaker or "entity"
        label = f"{subject} {action}: {obj}" if obj and obj != action else f"{subject} {action}"
        if temporal_signal:
            label = f"{label} @ {temporal_signal}"
        return {
            "predicate": predicate,
            "value": label,
            "object": obj,
            "label": _short_label(label),
            "temporal_signal": temporal_signal,
            "confidence_bonus": bonus,
        }
    return {}


def _subject_for_text(text: str, speaker: str = "") -> str:
    non_person_subjects = {
        "i", "it", "this", "that", "there", "where", "what", "when", "why", "how",
        "dance", "networking", "thanks", "sorry", "hey", "hi", "wow", "yes", "no",
    }
    explicit = re.search(
        r"\b([A-Z][a-zA-Z]+)\s+(?:lost|left|quit|started|opened|launched|joined|visited|moved|got accepted|works|worked|is|was)\b",
        text,
    )
    if explicit and explicit.group(1).lower() not in non_person_subjects:
        return explicit.group(1)
    possessive = re.search(r"\b([A-Z][a-zA-Z]+)'s\s+(?:job|store|studio|business|career|position|family|health)\b", text)
    if possessive and possessive.group(1).lower() not in non_person_subjects:
        return possessive.group(1)
    return speaker or "Unknown"


def _temporal_refs(text: str) -> list[dict]:
    refs = []
    patterns = [
        (r"\b(?:last|next|this)\s+(?:week|month|year|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b", "relative"),
        (r"\b(?:yesterday|today|tomorrow|tonight|recently|now|currently|finally)\b", "relative"),
        (r"\b(?:a few days|several days|two weeks|three weeks|a month|few months|several months|a year)\s+(?:ago|later|after|before)?\b", "duration"),
        (r"\b(?:after|before|since|during)\s+[^.!?,;]{1,50}", "relative_clause"),
        (r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b", "calendar_date"),
        (r"\b\d{4}\b", "year"),
    ]
    seen = set()
    for pattern, granularity in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _clean_phrase(match.group(0))
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "label": value,
                "normalized": key,
                "granularity": granularity,
                "edge_type": "HAPPENED_AT" if granularity != "duration" else "VALID_DURING",
            })
    return refs[:3]


def _preference_label(text: str, speaker: str, predicate: str) -> str:
    value = _preference_value(text)
    return _short_label(f"{speaker} {predicate}: {value}" if speaker else f"{predicate}: {value}")


def _preference_value(text: str) -> str:
    match = re.search(r"\b(?:likes|loves|enjoys|interested in|prefers)\s+([^.!?;]+)", text, flags=re.IGNORECASE)
    if match:
        return _clean_phrase(match.group(1))
    favorite = re.search(r"\bfavorite\s+([^.!?;]+)", text, flags=re.IGNORECASE)
    return _clean_phrase(favorite.group(0) if favorite else _short_label(text))


def _goal_label(text: str, speaker: str, predicate: str) -> str:
    value = _goal_value(text)
    return _short_label(f"{speaker} {predicate}: {value}" if speaker else f"{predicate}: {value}")


def _goal_value(text: str) -> str:
    match = re.search(r"\b(?:plans to|planning to|hopes to|wants to|aims to)\s+([^.!?;]+)", text, flags=re.IGNORECASE)
    return _clean_phrase(match.group(1) if match else _short_label(text))


def _state_value(text: str, state_slot: str) -> str:
    patterns = {
        "career_status": [
            r"\b(?:works as|working as|job as|position as)\s+(?:a|an)?\s*([^.!?,;]+)",
            r"\b(?:lost|left|quit)\s+(?:his|her|my|their)?\s*(?:job|position|work)(?:\s+as\s+([^.!?,;]+))?",
            r"\b(unemployed|between jobs|looking for work)\b",
        ],
        "education_status": [
            r"\b(?:studying|studies|study|degree in|majoring in)\s+([^.!?,;]+)",
            r"\b(?:student|college|education|degree)\b[^.!?;]*",
        ],
        "relationship_status": [
            r"\b(single|married|divorced|partnered|in a relationship)\b[^.!?;]*",
        ],
        "family_status": [
            r"\b(?:single parent|parent|children|kids|son|daughter|adoption|adopt)\b[^.!?;]*",
        ],
        "health_status": [
            r"\b(?:condition|diagnosed|symptom|health|injury|pain)\b[^.!?;]*",
        ],
    }
    for pattern in patterns.get(state_slot, []):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = next((group for group in match.groups() if group), match.group(0))
            return _clean_phrase(value)
    return _short_label(text)


def _state_label(speaker: str, state_slot: str, state_value: str, lowered: str) -> str:
    polarity = _state_polarity(lowered)
    change = _state_change_type(lowered)
    prefix = speaker or "entity"
    return _short_label(f"{prefix} {state_slot} {change}/{polarity}: {state_value}")


def _state_confidence(source_type: str, lowered: str) -> float:
    base = 0.82 if source_type == "observation_fact" else 0.70
    if any(token in lowered for token in ["lost his job", "lost her job", "lost my job", "unemployed", "no longer", "currently", "now"]):
        base += 0.08
    return min(0.92, base)


def _state_change_type(lowered: str) -> str:
    if any(token in lowered for token in ["lost", "no longer", "stopped", "quit", "left", "unemployed"]):
        return "ended"
    if any(token in lowered for token in ["started", "became", "now", "currently", "finally"]):
        return "started_or_current"
    return "stable"


def _clue_label(text: str, bridge_type: str) -> str:
    return _short_label(f"{bridge_type}: {_clue_value(text, bridge_type)}")


def _clean_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -,:;.!?\"'")
    return _short_label(value)


def _bridge_type(lowered: str) -> str:
    if any(token in lowered for token in ["zelda", "console", "video game", "card game", "board game"]):
        return "entertainment_clue_to_named_item"
    if any(token in lowered for token in ["national park", "visited", "trip", "moved from", "home country", "travel", "camped"]):
        return "location_clue_to_place"
    if any(token in lowered for token in ["education", "college", "career", "counseling", "counselling", "degree", "field"]):
        return "profile_clue_to_career_or_field"
    if any(token in lowered for token in ["book", "bookshelf", "novel", "read "]) or lowered.startswith("read "):
        return "reading_clue_to_likely_book_or_author"
    if any(token in lowered for token in ["fashion", "department", "company", "brand", "store", "shop"]):
        return "work_clue_to_industry_or_organization"
    if any(token in lowered for token in ["condition", "symptom", "technique", "method"]):
        return "descriptive_clue_to_concept"
    return ""


def _state_slot(lowered: str) -> str:
    if any(token in lowered for token in ["single", "married", "partner", "divorced"]):
        return "relationship_status"
    if any(token in lowered for token in ["single parent", "becoming a parent", "adopt", "adoption", "my children", "my kids", "my son", "my daughter"]):
        return "family_status"
    if re.search(r"\b(works as|working as|job as|position as|position in|unemployed|between jobs|lost (?:his|her|my|their)?\s*job|accepted for .*internship|part-time position|full-time position)\b", lowered):
        return "career_status"
    if re.search(r"\b(i am|i'm|he is|she is|they are|as a)\s+(?:a\s+)?(?:student|college student)\b|\b(studying|degree in|majoring in|my education|his education|her education)\b", lowered):
        return "education_status"
    if any(token in lowered for token in ["condition", "diagnosed", "symptom", "health"]):
        return "health_status"
    return ""


def _state_polarity(lowered: str) -> str:
    if any(token in lowered for token in ["lost", "left", "quit", "no longer", "not ", "unemployed", "stopped"]):
        return "negative_or_ended"
    return "positive_or_active"


def _clue_value(text: str, bridge_type: str) -> str:
    if bridge_type == "entertainment_clue_to_named_item":
        matches = re.findall(r"\b(?:Zelda|Nintendo|Switch|Xbox|PlayStation|Monopoly|Scrabble|Pokemon|Pokémon)\b", text)
        return ", ".join(matches) or _short_label(text)
    if bridge_type == "location_clue_to_place":
        matches = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", text)
        return ", ".join(matches[:4]) or _short_label(text)
    return _short_label(text)


def _load_sample(path: Path, sample_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    if not sample_id:
        if len(payload) == 1:
            return payload[0]
        raise SystemExit("--sample-id is required when --input contains multiple samples")
    for sample in payload:
        if str(sample.get("sample_id")) == sample_id:
            return sample
    raise SystemExit(f"sample_id not found: {sample_id}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _capitalized_entities(text: str) -> list[str]:
    entities = []
    for match in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", text):
        value = match.group(0)
        if value.lower() not in {"i", "the", "and", "but"}:
            entities.append(value)
    return entities[:8]


def _observation_fact(item: object, fallback_id: str) -> tuple[str, str]:
    if isinstance(item, list) and item:
        return str(item[0]), str(item[1] if len(item) > 1 else fallback_id)
    return str(item), fallback_id


def _short_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:120]


def _node_id(node_type: str, label: str) -> str:
    return f"{node_type.lower()}:{_safe_id(label)[:96]}"


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _add_alias_edges(nodes: dict, edges: dict, mentions: dict) -> None:
    people = [node for node in nodes.values() if node["type"] == "Person"]
    for person in people:
        label = person["label"].lower()
        for alias in ("he", "she", "they", "him", "her", "them"):
            if alias in mentions and label in mentions:
                key = (person["id"], person["id"], "ALIAS_HINT")
                edges[key] = {
                    "source": person["id"],
                    "target": person["id"],
                    "type": "ALIAS_HINT",
                    "source_ids": sorted(set(mentions[label] + mentions[alias])),
                    "confidence": 0.35,
                    "attrs": {"alias": alias},
                }


def _apply_streaming_update_step(
    nodes: dict,
    edges: dict,
    touched_nodes: set[str],
    touched_edges: set[tuple[str, str, str]],
    step: int,
    max_step: int,
    enable_conflict_control: bool = True,
) -> dict:
    """Update the persistent graph immediately after ingesting one session."""
    rho = 0.92
    conflict_groups = _conflict_groups(nodes, edges)
    trace = {
        "step": step,
        "touched_nodes": len(touched_nodes),
        "touched_edges": len(touched_edges),
        "node_reinforcements": 0,
        "edge_reinforcements": 0,
        "node_inhibitions": 0,
        "edge_inhibitions": 0,
    }

    for node_id, node in nodes.items():
        is_new = "weight" not in node
        old_weight = float(node.get("weight", 0.05))
        prior_weight = old_weight if is_new else old_weight * rho
        strength, components = (0.0, {})
        if node_id in touched_nodes:
            strength, components = _node_strength(node, step, max_step)
        inhibition = _node_inhibition(node, conflict_groups) if enable_conflict_control else 0.0
        node["memory_step"] = step
        node["prior_weight"] = round(prior_weight, 4)
        node["delta_strength"] = round(strength, 4)
        node["delta_inhibition"] = round(inhibition, 4)
        node["weight_components"] = components
        node["weight"] = round(_clip(prior_weight + strength - inhibition), 4)
        trace["node_reinforcements"] += int(strength > 0)
        trace["node_inhibitions"] += int(inhibition > 0)

    for edge_key, edge in edges.items():
        is_new = "weight" not in edge
        old_weight = float(edge.get("weight", 0.03))
        prior_weight = old_weight if is_new else old_weight * rho
        strength, components = (0.0, {})
        if edge_key in touched_edges:
            strength, components = _edge_strength(edge, step, max_step)
        inhibition = _edge_inhibition(edge, nodes, conflict_groups) if enable_conflict_control else 0.0
        edge["memory_step"] = step
        edge["prior_weight"] = round(prior_weight, 4)
        edge["delta_strength"] = round(strength, 4)
        edge["delta_inhibition"] = round(inhibition, 4)
        edge["weight_components"] = components
        edge["weight"] = round(_clip(prior_weight + strength - inhibition), 4)
        trace["edge_reinforcements"] += int(strength > 0)
        trace["edge_inhibitions"] += int(inhibition > 0)

    return trace


def _node_strength(node: dict, step: int, max_step: int) -> tuple[float, dict]:
    frequency = min(1.0, len(node.get("source_ids", [])) / 6.0)
    recency = step / max(max_step, 1)
    salience = _node_salience(node)
    reliability = _source_reliability(node)
    confidence = float(node.get("confidence", 0.7))
    temporal_relevance = 1.0 if node.get("type") in {"Time", "Event", "State"} else 0.45
    components = {
        "frequency": round(frequency, 4),
        "recency": round(recency, 4),
        "salience": round(salience, 4),
        "source_reliability": round(reliability, 4),
        "extractor_confidence": round(confidence, 4),
        "temporal_relevance": round(temporal_relevance, 4),
    }
    strength = (
        0.20 * frequency
        + 0.14 * recency
        + 0.20 * salience
        + 0.18 * reliability
        + 0.18 * confidence
        + 0.10 * temporal_relevance
    )
    return strength, components


def _edge_strength(edge: dict, step: int, max_step: int) -> tuple[float, dict]:
    evidence_count = min(1.0, len(edge.get("source_ids", [])) / 5.0)
    confidence = float(edge.get("confidence", 0.7))
    temporal = 1.0 if edge.get("type") in {"TEMPORAL_NEXT", "HAPPENED_AT", "VALID_DURING"} else 0.45
    relation_strength = _relation_strength(edge.get("type", ""))
    recency = step / max(max_step, 1)
    components = {
        "evidence_count": round(evidence_count, 4),
        "relation_confidence": round(confidence, 4),
        "temporal_consistency": round(temporal, 4),
        "relation_strength": round(relation_strength, 4),
        "recency": round(recency, 4),
    }
    strength = (
        0.24 * evidence_count
        + 0.22 * confidence
        + 0.16 * temporal
        + 0.24 * relation_strength
        + 0.14 * recency
    )
    return strength, components


def _node_inhibition(node: dict, conflict_groups: dict) -> float:
    if node.get("type") not in {"State", "Preference", "Goal", "OpenDomainClue"}:
        return 0.0
    group = _semantic_group(node)
    group_info = conflict_groups.get(group, {})
    group_size = int(group_info.get("count", 0))
    if group_size <= 1:
        return 0.0
    node_step = _max_session(node.get("source_ids", []))
    latest_step = int(group_info.get("latest_step", node_step))
    if node_step >= latest_step:
        return 0.0
    staleness = max(0.0, (latest_step - node_step) / max(latest_step, 1))
    polarity = str(node.get("attrs", {}).get("polarity", ""))
    polarities = set(group_info.get("polarities", []))
    polarity_conflict = bool(polarity and len(polarities) > 1)
    same_slot_pressure = min(0.16, 0.035 * (group_size - 1))
    stale_penalty = 0.22 * staleness
    conflict_penalty = 0.18 if polarity_conflict and node_step < latest_step else 0.0
    return min(0.42, same_slot_pressure + stale_penalty + conflict_penalty)


def _edge_inhibition(edge: dict, nodes: dict, conflict_groups: dict) -> float:
    if edge.get("type") not in {"HAS_STATE", "HAS_PREFERENCE", "HAS_GOAL", "BRIDGES_TO"}:
        return 0.0
    target = nodes.get(edge.get("target"), {})
    return 0.6 * _node_inhibition(target, conflict_groups)


def _conflict_groups(nodes: dict, edges: dict) -> dict:
    groups = {}
    for node in nodes.values():
        group = _semantic_group(node)
        if group:
            item = groups.setdefault(group, {"count": 0, "latest_step": 0, "polarities": set()})
            item["count"] += 1
            item["latest_step"] = max(item["latest_step"], _max_session(node.get("source_ids", [])))
            polarity = node.get("attrs", {}).get("polarity")
            if polarity:
                item["polarities"].add(str(polarity))
    for item in groups.values():
        item["polarities"] = sorted(item["polarities"])
    return groups


def _semantic_group(node: dict) -> str:
    node_type = node.get("type", "")
    if node_type not in {"State", "Preference", "Goal", "OpenDomainClue"}:
        return ""
    label = str(node.get("label", "")).lower()
    attrs = dict(node.get("attrs", {}))
    slot = str(attrs.get("slot") or node_type.lower())
    if any(token in label for token in ["job", "career", "work", "position", "department"]):
        slot = "career"
    elif any(token in label for token in ["single", "married", "parent", "family"]):
        slot = "family_state"
    elif any(token in label for token in ["book", "read", "author"]):
        slot = "reading"
    elif any(token in label for token in ["country", "state", "visited", "trip", "moved"]):
        slot = "location"
    speaker = str(attrs.get("speaker", "")).lower() or "unknown"
    return "%s:%s:%s" % (speaker, node_type, slot)


def _node_salience(node: dict) -> float:
    node_type = node.get("type", "")
    if node_type in {"Person", "OpenDomainClue", "Event", "State"}:
        return 0.9
    if node_type in {"Preference", "Goal", "ObservationFact"}:
        return 0.75
    if node_type in {"Object", "Place", "Time"}:
        return 0.55
    return 0.35


def _source_reliability(node: dict) -> float:
    source_ids = node.get("source_ids", [])
    attrs = node.get("attrs", {})
    if attrs.get("source_type") == "observation_fact":
        return 0.9
    if any(str(source).lower().startswith("obs") or "observation" in str(source).lower() for source in source_ids):
        return 0.9
    if node.get("type") == "OpenDomainClue":
        return 0.55
    if node.get("type") == "EvidenceSpan":
        return 0.75
    return 0.7


def _relation_strength(edge_type: str) -> float:
    return {
        "SUPPORTS": 0.95,
        "SUMMARIZES": 0.9,
        "HAS_STATE": 0.85,
        "HAS_PREFERENCE": 0.82,
        "HAS_GOAL": 0.78,
        "PARTICIPATES_IN": 0.78,
        "BRIDGES_TO": 0.72,
        "TEMPORAL_NEXT": 0.72,
        "VALID_DURING": 0.8,
        "HAPPENED_AT": 0.76,
        "MENTIONS_OBJECT": 0.58,
        "ALIAS_HINT": 0.45,
    }.get(edge_type, 0.5)


def _recency_score(source_ids: list[str]) -> float:
    sessions = []
    for source in source_ids:
        text = str(source)
        match = re.search(r"(?:D|S|session_)(\d+)", text)
        if match:
            sessions.append(int(match.group(1)))
    if not sessions:
        return 0.5
    return min(1.0, max(sessions) / 20.0)


def _max_session(source_ids: list[str]) -> int:
    sessions = []
    for source in source_ids:
        match = re.search(r"(?:D|S|session_)(\d+)", str(source))
        if match:
            sessions.append(int(match.group(1)))
    return max(sessions) if sessions else 1


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
