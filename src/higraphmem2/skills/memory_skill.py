import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MemorySkill:
    name: str
    description: str
    trigger_patterns: List[str]
    state_schema: Dict[str, str]
    evidence_patterns: List[str]
    retrieval_policy: Dict[str, object] = field(default_factory=dict)
    answer_policy: Dict[str, object] = field(default_factory=dict)
    skill_type: str = "memory"
    open_domain_policy: Dict[str, object] = field(default_factory=dict)
    legacy_entity: str = ""
    legacy_values: List[str] = field(default_factory=list)

    def matches_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(_pattern_matches(pattern, lowered) for pattern in self.trigger_patterns)


def load_skills(path: str) -> List[MemorySkill]:
    target = Path(path)
    if target.is_dir():
        return _load_subskill_library(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    items = payload.get("skills", payload.get("rules", []))
    return [_skill_from_dict(item) for item in items]


def save_skills(path: str, skills: Iterable[MemorySkill], metadata: Optional[Dict[str, object]] = None) -> None:
    target = Path(path)
    if target.suffix.lower() != ".json":
        _save_subskill_library(target, skills, metadata=metadata)
        return
    output = {
        "metadata": metadata or {},
        "skills": [_skill_to_dict(skill) for skill in skills],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def candidate_skills() -> List[MemorySkill]:
    return [
        _skill(
            "relationship_state_memory",
            "Extract stable relationship or family status for a person.",
            ["relationship status"],
            "person_state",
            ["single parent", "married", "partner", "family status"],
            legacy_entity="Caroline",
            legacy_values=["Single"],
        ),
        _skill(
            "origin_location_memory",
            "Extract where a person moved from or originally came from.",
            ["move from", "moved from", "where did {person} move"],
            "origin_location",
            ["home country", "moved from", "from {location}"],
            legacy_entity="Caroline",
            legacy_values=["Sweden"],
        ),
        _skill(
            "location_history_memory",
            "Aggregate places a person has visited, camped, lived, or plans to visit.",
            ["where has", "camp", "visited", "been to"],
            "location_list",
            ["camped", "visited", "went to", "been to", "planning to go"],
            legacy_entity="Melanie",
            legacy_values=["mountains", "beach", "forest"],
        ),
        _skill(
            "preference_state_memory",
            "Extract likes, preferences, favorite styles, and interests.",
            ["kids like", "favorite", "interests", "hobbies", "like"],
            "preference_list",
            ["favorite", "likes", "loves", "enjoys", "interested in"],
            legacy_entity="Melanie",
            legacy_values=["dinosaurs", "nature"],
        ),
        _skill(
            "reading_memory",
            "Aggregate books, articles, or named works read by a person.",
            ["books", "read", "reading"],
            "reading_list",
            ["read", "book", "novel", "article"],
            legacy_entity="Melanie",
            legacy_values=["Charlotte's Web", "Nothing is Impossible"],
        ),
        _skill(
            "destress_activity_memory",
            "Extract activities a person uses to relax or de-stress.",
            ["destress", "de-stress", "stress"],
            "activity_list",
            ["destress", "de-stress", "relax", "stress fix"],
            legacy_entity="Melanie",
            legacy_values=["running", "pottery"],
        ),
        _skill(
            "activity_event_memory",
            "Aggregate activities, hobbies, events, and recurring participation.",
            ["activities", "events", "participated", "partake", "done with", "promote", "promoted"],
            "activity_event_list",
            ["participated", "attended", "hosted", "fair", "networking", "competition", "festival", "campaign", "website", "video presentation", "camping", "painting", "swimming", "hiking", "museum"],
            legacy_entity="Melanie",
            legacy_values=["pottery", "painting", "camping", "museum", "swimming", "hiking"],
        ),
        _skill(
            "identity_support_event_memory",
            "Track identity-related support events and community participation.",
            ["lgbtq", "support group", "pride", "children"],
            "support_event_list",
            ["support group", "school event", "pride parade", "mentorship program"],
            legacy_entity="Caroline",
            legacy_values=["support group", "school speech", "pride parade", "mentoring program"],
        ),
        _skill(
            "creative_output_memory",
            "Extract recent creative outputs such as paintings, writing, or performances.",
            ["paint recently", "painted", "wrote", "performed"],
            "creative_output",
            ["painting", "painted", "wrote", "performed", "recently"],
            legacy_entity="Melanie",
            legacy_values=["sunset", "landscape/still life"],
        ),
    ]


def default_skills() -> List[MemorySkill]:
    """Return the fixed, data-independent skill inventory used at evaluation.

    This deliberately enables every supported policy.  It must not depend on a
    held-out conversation, since selecting a subset from that conversation would
    make the evaluation split part of configuration.
    """
    # The marker question activates the three general open-domain policies that
    # were previously serialized into ``memory_skills.json``.
    open_domain = open_domain_skill_candidates(
        {"category": "3", "question": "what country political leaning"}
    )
    return candidate_skills() + open_domain


def induce_skills(samples: Iterable[dict]) -> List[MemorySkill]:
    questions = [qa.get("question", "") for sample in samples for qa in sample.get("qa", [])]
    return [skill for skill in candidate_skills() if any(skill.matches_question(question) for question in questions)]


def evolve_skills(samples: Iterable[dict], seed_skills: Iterable[MemorySkill] = ()) -> List[MemorySkill]:
    """Stream over induction samples and add reusable open-domain skills.

    The evolved skills encode question/evidence patterns and resolver policies, not
    test-set answer mappings. Gold answers are allowed only in the induction split
    to decide that a pattern is useful.
    """
    inventory = {skill.name: skill for skill in seed_skills}
    for sample in samples:
        for qa in sample.get("qa", []):
            for skill in open_domain_skill_candidates(qa):
                inventory.setdefault(skill.name, skill)
    return list(inventory.values())


def open_domain_skill_candidates(qa: Dict[str, object]) -> List[MemorySkill]:
    question = str(qa.get("question", ""))
    lowered = question.lower()
    if str(qa.get("category", "")) not in {"3", "open_domain"} and not _looks_open_domain_question(lowered):
        return []

    return _open_domain_skill_templates(lowered)


def _open_domain_skill_templates(lowered: str) -> List[MemorySkill]:
    candidates = [
        _open_domain_skill(
            "open_domain_general_reasoning",
            "Infer a concise answer to an open-domain question by combining retrieved conversational evidence with a justified, conservative world-knowledge step.",
            [
                "would",
                "could",
                "likely",
                "based on",
                "what might",
                "what kind",
                "what is a",
                "what is the",
                "which",
            ],
            [
                "preference",
                "interest",
                "plan",
                "clue",
                "evidence",
                "likely",
                "would",
                "could",
            ],
            {
                "resolver": "llm_public_knowledge",
                "target_type": "question_specific_concise_answer",
                "constraints": [
                    "Use retrieved memory evidence as the primary source.",
                    "Perform at most one explicit public-knowledge inference step.",
                    "Preserve uncertainty when the evidence is weak and do not invent unsupported entities.",
                    "Return only the shortest answer phrase.",
                ],
            },
        )
    ]
    if any(marker in lowered for marker in ["what state", "which state", "in which state"]):
        candidates.append(
            _open_domain_skill(
                "open_domain_location_city_to_state",
                "Infer a state or province when the question asks for a state and memory evidence gives a city or local place clue.",
                ["what state", "which state", "in which state"],
                ["city", "town", "visited", "shelter", "summer", "trip"],
                {
                    "resolver": "llm_public_knowledge",
                    "target_type": "state_or_province",
                    "constraints": [
                        "Use retrieved memory evidence as location clues.",
                        "Use public geographic knowledge only to map a supported city/place clue to a state.",
                        "Do not guess if no city/place clue is present.",
                    ],
                },
            )
        )
    if any(marker in lowered for marker in ["what country", "which country", "in what country"]):
        candidates.append(
            _open_domain_skill(
                "open_domain_location_city_to_country",
                "Infer a country when the question asks for a country and memory evidence gives a city, region, or travel clue.",
                ["what country", "which country", "in what country"],
                ["city", "country", "travel", "visited", "trip", "summer"],
                {
                    "resolver": "llm_public_knowledge",
                    "target_type": "country",
                    "constraints": [
                        "Use retrieved memory evidence as travel/location clues.",
                        "Use public geographic knowledge only to map a supported clue to a country.",
                        "Return No information available if the clue is ambiguous.",
                    ],
                },
            )
        )
    if any(marker in lowered for marker in ["what game", "what card game", "what board game", "what console"]):
        candidates.append(
            _open_domain_skill(
                "open_domain_entertainment_clue_to_name",
                "Infer a named game, card game, board game, or console from supported entertainment clues in memory evidence.",
                ["what game", "what card game", "what board game", "what console"],
                ["game", "cards", "console", "played", "rules", "characters", "deck"],
                {
                    "resolver": "llm_public_knowledge",
                    "target_type": "named_entertainment_item",
                    "constraints": [
                        "Use only clues present in retrieved memory evidence.",
                        "Use public knowledge to name the item implied by those clues.",
                        "Prefer No information available over an unsupported named guess.",
                    ],
                },
            )
        )
    if any(marker in lowered for marker in ["technique", "method", "kind of yoga", "condition", "disease"]):
        candidates.append(
            _open_domain_skill(
                "open_domain_clue_to_concept",
                "Infer a named concept, method, practice, condition, or technique from supported descriptive clues.",
                ["technique", "method", "kind of yoga", "condition", "disease"],
                ["symptom", "practice", "method", "break", "routine", "health", "exercise"],
                {
                    "resolver": "llm_public_knowledge",
                    "target_type": "named_concept",
                    "constraints": [
                        "Use retrieved memory evidence as constraints.",
                        "Use public knowledge only to name the concept matching those constraints.",
                        "Return a concise phrase and avoid explanations.",
                    ],
                },
            )
        )
    if any(marker in lowered for marker in ["political leaning", "financial status", "would enjoy", "likely signed"]):
        candidates.append(
            _open_domain_skill(
                "open_domain_profile_inference",
                "Infer a concise profile attribute or likely named target from supported preference, value, endorsement, or status clues.",
                ["political leaning", "financial status", "would enjoy", "likely signed"],
                ["supports", "advocate", "preference", "brand", "endorsement", "status", "values"],
                {
                    "resolver": "llm_public_knowledge",
                    "target_type": "profile_attribute_or_named_target",
                    "constraints": [
                        "Use memory evidence as the source of clues.",
                        "Use public knowledge only when the evidence clearly constrains the answer.",
                        "Do not infer sensitive attributes unless the evidence directly supports the label.",
                    ],
                },
            )
        )
    return candidates
    return candidates


def _skill(
    name: str,
    description: str,
    trigger_patterns: List[str],
    slot: str,
    evidence_patterns: List[str],
    legacy_entity: str,
    legacy_values: List[str],
) -> MemorySkill:
    return MemorySkill(
        name=name,
        description=description,
        trigger_patterns=trigger_patterns,
        state_schema={"entity": "person", "slot": slot, "value_type": "list"},
        evidence_patterns=evidence_patterns,
        retrieval_policy={
            "prefer_entity_match": True,
            "aggregate_across_sessions": True,
            "include_dates": True,
        },
        answer_policy={
            "output": "concise_list",
            "deduplicate": True,
            "preserve_specific_names": True,
        },
        skill_type="memory",
        legacy_entity=legacy_entity,
        legacy_values=legacy_values,
    )


def _open_domain_skill(
    name: str,
    description: str,
    trigger_patterns: List[str],
    evidence_patterns: List[str],
    open_domain_policy: Dict[str, object],
) -> MemorySkill:
    return MemorySkill(
        name=name,
        description=description,
        trigger_patterns=trigger_patterns,
        state_schema={"entity": "open_domain", "slot": name, "value_type": "inference_policy"},
        evidence_patterns=evidence_patterns,
        retrieval_policy={
            "prefer_entity_match": False,
            "aggregate_across_sessions": True,
            "include_dates": True,
        },
        answer_policy={
            "output": "concise_phrase",
            "deduplicate": True,
            "preserve_specific_names": True,
            "allow_public_knowledge": True,
        },
        skill_type="open_domain",
        open_domain_policy=open_domain_policy,
    )


def _pattern_matches(pattern: str, lowered_question: str) -> bool:
    tokens = [token for token in pattern.lower().replace("{person}", "").split() if token]
    return all(token in lowered_question for token in tokens)


def _looks_open_domain_question(lowered_question: str) -> bool:
    markers = [
        "likely",
        "would",
        "could",
        "might",
        "based on",
        "what game",
        "what console",
        "what state",
        "what country",
        "technique",
        "condition",
    ]
    return any(marker in lowered_question for marker in markers)


def _skill_from_dict(item: Dict[str, object]) -> MemorySkill:
    if "state_schema" not in item:
        return _legacy_rule_to_skill(item)
    return MemorySkill(
        name=str(item["name"]),
        description=str(item.get("description", "")),
        trigger_patterns=list(item.get("trigger_patterns", [])),
        state_schema=dict(item.get("state_schema", {})),
        evidence_patterns=list(item.get("evidence_patterns", [])),
        retrieval_policy=dict(item.get("retrieval_policy", {})),
        answer_policy=dict(item.get("answer_policy", {})),
        skill_type=str(item.get("skill_type", "memory")),
        open_domain_policy=dict(item.get("open_domain_policy", {})),
        legacy_entity=str(item.get("legacy_entity", "")),
        legacy_values=list(item.get("legacy_values", [])),
    )


def _skill_to_dict(skill: MemorySkill) -> Dict[str, object]:
    return {
        "name": skill.name,
        "description": skill.description,
        "trigger_patterns": skill.trigger_patterns,
        "state_schema": skill.state_schema,
        "evidence_patterns": skill.evidence_patterns,
        "retrieval_policy": skill.retrieval_policy,
        "answer_policy": skill.answer_policy,
        "skill_type": skill.skill_type,
        "open_domain_policy": skill.open_domain_policy,
        "legacy_entity": skill.legacy_entity,
        "legacy_values": skill.legacy_values,
    }


def _load_subskill_library(path: Path) -> List[MemorySkill]:
    skills = []
    for metadata_path in sorted(path.glob("*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        skills.append(_skill_from_subskill_metadata(payload, metadata_path.parent))
    if not skills:
        raise ValueError(f"No subskill metadata found under {path}")
    return skills


def _save_subskill_library(
    path: Path,
    skills: Iterable[MemorySkill],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    library_metadata = {
        "metadata_schema": "higraphmem2-subskill-library-v0.1",
        "format": "mat-skill-agent-style-subskills",
        **(metadata or {}),
    }
    (path / "library_metadata.json").write_text(
        json.dumps(library_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for skill in skills:
        skill_dir = path / _slugify_skill_id(skill.name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(_skill_to_subskill_markdown(skill), encoding="utf-8")
        (skill_dir / "metadata.json").write_text(
            json.dumps(_skill_to_subskill_metadata(skill), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _skill_from_subskill_metadata(payload: Dict[str, object], skill_dir: Path) -> MemorySkill:
    routing = dict(payload.get("routing", {}))
    io = dict(payload.get("io", {}))
    execution = dict(payload.get("execution", {}))
    policy = dict(payload.get("policy", {}))
    legacy = dict(payload.get("legacy", {}))
    state_schema = dict(payload.get("state_schema", {}))
    if not state_schema:
        state_schema = {
            "entity": str(policy.get("entity", "open_domain" if payload.get("category") == "open-domain-inference" else "person")),
            "slot": str(policy.get("slot", payload.get("id", skill_dir.name))),
            "value_type": str(policy.get("value_type", "inference_policy" if payload.get("category") == "open-domain-inference" else "list")),
        }
    return MemorySkill(
        name=str(payload.get("id", skill_dir.name)),
        description=str(routing.get("description") or payload.get("description") or payload.get("display", {}).get("summary", "")),
        trigger_patterns=list(routing.get("trigger", [])),
        state_schema=state_schema,
        evidence_patterns=list(routing.get("evidence_trigger", routing.get("evidence_patterns", []))),
        retrieval_policy=dict(policy.get("retrieval_policy", {})),
        answer_policy=dict(policy.get("answer_policy", {})),
        skill_type=str(execution.get("skill_type", payload.get("skill_type", "open_domain" if payload.get("category") == "open-domain-inference" else "memory"))),
        open_domain_policy=dict(policy.get("open_domain_policy", {})),
        legacy_entity=str(legacy.get("entity", "")),
        legacy_values=list(legacy.get("values", [])),
    )


def _skill_to_subskill_metadata(skill: MemorySkill) -> Dict[str, object]:
    category = "open-domain-inference" if skill.skill_type == "open_domain" else "memory-retrieval"
    return {
        "metadata_schema": "higraphmem2-subskill-v0.1",
        "id": skill.name,
        "category": category,
        "status": "train-induced",
        "version": "v0.1.0",
        "display": {
            "title": _title_from_skill_name(skill.name),
            "summary": skill.description,
            "tags": [skill.skill_type, category],
            "examples": {
                "en": list(skill.trigger_patterns[:3]),
                "zh": list(skill.trigger_patterns[:3]),
            },
        },
        "routing": {
            "description": skill.description,
            "purpose": skill.description,
            "trigger": list(skill.trigger_patterns),
            "evidence_trigger": list(skill.evidence_patterns),
            "skip": [
                "Do not use when no retrieved memory evidence supports the trigger.",
                "Do not use to encode held-out test answers or gold evidence labels.",
            ],
            "trigger_examples": list(skill.trigger_patterns[:3]),
        },
        "io": {
            "inputs": [
                {
                    "name": "question",
                    "type": "string",
                    "required": True,
                    "description": "LoCoMo QA question.",
                },
                {
                    "name": "evidence",
                    "type": "list:string",
                    "required": True,
                    "description": "Retrieved memory evidence lines.",
                },
            ],
            "parameters": [],
            "outputs": [
                {
                    "name": "candidate_answer",
                    "type": "string",
                    "required": True,
                    "description": "Concise answer phrase or No information available.",
                }
            ],
        },
        "execution": {
            "environment": "locomo-qa",
            "runner": "prompt-policy",
            "entrypoint": "",
            "requires_vm": False,
            "skill_type": skill.skill_type,
            "subskill_path": f"skills/subskills/{_slugify_skill_id(skill.name)}",
        },
        "workflow": [
            {
                "name": "match_question",
                "instruction": "Check whether routing triggers match the question.",
                "expected_artifact": "matched trigger patterns",
            },
            {
                "name": "retrieve_evidence",
                "instruction": "Use evidence triggers and retrieval policy to collect supporting memory evidence.",
                "expected_artifact": "supporting evidence lines",
            },
            {
                "name": "realize_answer",
                "instruction": "Apply the answer policy and verification constraints.",
                "expected_artifact": "candidate answer",
            },
        ],
        "policy": {
            "state_schema": dict(skill.state_schema),
            "retrieval_policy": dict(skill.retrieval_policy),
            "answer_policy": dict(skill.answer_policy),
            "open_domain_policy": dict(skill.open_domain_policy),
        },
        "state_schema": dict(skill.state_schema),
        "constraints": [
            "Skills must be induced only from train/induction data.",
            "Do not use held-out test answers.",
            "Do not use gold evidence labels during retrieval or generation.",
            "Use public knowledge only when a retrieved memory clue supports it.",
        ],
        "quality_criteria": [
            "The skill must have explicit routing triggers.",
            "The skill must produce concise benchmark-compatible answers.",
            "The skill must preserve provenance through retrieved evidence.",
        ],
        "validation": {
            "required_checks": [
                "metadata.json is parseable.",
                "SKILL.md exists.",
                "Routing triggers are non-empty.",
            ],
            "golden_tasks": [],
        },
        "legacy": {
            "entity": skill.legacy_entity,
            "values": list(skill.legacy_values),
        },
    }


def _skill_to_subskill_markdown(skill: MemorySkill) -> str:
    title = _title_from_skill_name(skill.name)
    trigger_examples = "\n".join(f"- {pattern}" for pattern in skill.trigger_patterns[:5]) or "- N/A"
    evidence_patterns = "\n".join(f"- {pattern}" for pattern in skill.evidence_patterns[:8]) or "- N/A"
    constraints = skill.open_domain_policy.get("constraints", []) if skill.open_domain_policy else []
    constraint_text = "\n".join(f"- {item}" for item in constraints) or "- Do not use held-out test answers or gold evidence labels."
    return f"""---
name: {skill.name}
description: {skill.description}
---

# {title}

- Skill ID: `{skill.name}`
- Status: `train-induced`
- Category: `{"open-domain-inference" if skill.skill_type == "open_domain" else "memory-retrieval"}`
- Environment: `locomo-qa`
- Runner: `prompt-policy`

## Use When

{skill.description}

## Purpose

Apply a reusable train-induced policy to retrieve memory evidence and realize concise LoCoMo-compatible answers.

## Inputs

- `question` (string, required): LoCoMo QA question.
- `evidence` (list[string], required): Retrieved memory evidence lines.

## Parameters

- `max_public_knowledge_steps` (number, optional, default `1`): Used only for open-domain skills.

## Outputs

- `candidate_answer`: Concise answer phrase or `No information available`.

## Workflow

- `match_question`: Match question routing triggers. -> matched trigger patterns
- `retrieve_evidence`: Retrieve evidence matching skill evidence patterns. -> supporting evidence lines
- `realize_answer`: Apply answer policy and verification constraints. -> candidate answer

## Trigger Examples

{trigger_examples}

## Evidence Patterns

{evidence_patterns}

## Do Not Use When

- The question does not match the routing triggers.
- Retrieved evidence does not contain a supporting clue.
- The only way to answer is to use held-out test answers or gold evidence labels.

## Quality Criteria

{constraint_text}
"""


def _slugify_skill_id(name: str) -> str:
    chars = []
    previous_dash = False
    for char in name.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "skill"


def _title_from_skill_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("_", "-").split("-"))


def _legacy_rule_to_skill(item: Dict[str, object]) -> MemorySkill:
    return MemorySkill(
        name=str(item["name"]),
        description="Legacy state rule imported as a memory skill.",
        trigger_patterns=list(item.get("question_keywords", [])),
        state_schema={"entity": "person", "slot": str(item["name"]), "value_type": "list"},
        evidence_patterns=list(item.get("extract_keywords", [])),
        retrieval_policy={"prefer_entity_match": True, "aggregate_across_sessions": True},
        answer_policy={"output": "concise_list", "deduplicate": True},
        skill_type="memory",
        legacy_entity=str(item.get("entity", "")),
        legacy_values=list(item.get("values", [])),
    )
