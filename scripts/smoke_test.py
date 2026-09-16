import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_runner_helpers():
    spec = importlib.util.spec_from_file_location("higraphmem2_runner", ROOT / "eval" / "run_locomo_v2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

from higraphmem2.answer.multi_hop import aggregate_multi_hop_answer
from higraphmem2.answer.normalizer import normalize_answer_text
from higraphmem2.answer.evidence_verifier import verify_answer
from higraphmem2.answer.open_domain_cbi import (
    build_open_domain_skill_prompt,
    matching_open_domain_skills,
    normalize_open_domain_answer,
    select_cbi_answer,
    select_rescue_answer,
    should_refine_open_domain_answer,
    should_rescue_no_information,
)
from higraphmem2.answer.single_hop import extract_single_hop_answer
from higraphmem2.answer.temporal import canonicalize_temporal_answer, resolve_temporal_answer
from higraphmem2.memory.entity_state import build_entity_state_memory
from higraphmem2.memory.graph_memory import SkillGuidedMemoryGraph
from higraphmem2.memory.open_domain_state import OpenDomainStateAggregator
from higraphmem2.retrieval.evidence_chain import EvidenceChainComposer
from higraphmem2.retrieval.evidence_budget import budget_for_plan
from higraphmem2.retrieval.multihop_reranker import rerank_multihop_evidence
from higraphmem2.retrieval.open_domain_expansion import expand_open_domain_query
from higraphmem2.retrieval.query_decomposition import decompose_question
from higraphmem2.retrieval.skill_router import route_question
from higraphmem2.skills.memory_skill import candidate_skills, evolve_skills


def main() -> None:
    runner_helpers = _load_runner_helpers()
    anchor = datetime(2023, 5, 25)
    checks = [
        canonicalize_temporal_answer("last year", anchor) == "2022",
        canonicalize_temporal_answer("next month", anchor) == "June 2023",
        canonicalize_temporal_answer("tomorrow", anchor) == "26 May 2023",
        resolve_temporal_answer("When is Jon planning to open?", "Tomorrow", [], [anchor])[0] == "26 May 2023",
        normalize_answer_text("What is Caroline's identity?", "Caroline is a trans woman.") == "transgender woman",
        normalize_answer_text("What did Caroline research?", "Caroline researched adoption agencies.") == "adoption agencies",
        extract_single_hop_answer("What is Gina's favorite style of dance?", "Gina's favorite dance style is contemporary", [])[0] == "contemporary",
        extract_single_hop_answer("What does Jon tell Gina he won't do?", "Jon tells Gina he won't give up or quit", [])[0] == "quit",
        aggregate_multi_hop_answer("Which events has Jon participated in to promote his business venture?", "Jon participated in networking events and hosted a dance competition.", [])[0] == "networking events, dance competition",
        aggregate_multi_hop_answer("Where has Melanie camped?", "camping, painting, hiking", ["Melanie camped at the beach."]) is None,
        normalize_open_domain_answer("Did John study together with James?", "No information available") == "No information available",
    ]
    sample = {
        "conversation": {
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "It will be tough as a single parent."},
                {"speaker": "Caroline", "dia_id": "D1:2", "text": "My home country, Sweden, means a lot to me."},
                {"speaker": "Melanie", "dia_id": "D1:3", "text": "We went camping at the beach and in the forest."},
            ]
        },
        "observation": {},
    }
    memory = build_entity_state_memory(sample)
    graph = SkillGuidedMemoryGraph(sample, candidate_skills())
    graph_hits = graph.retrieve("Where has Melanie camped?", seed_memory_ids=["D1:3"], top_k=2)
    plan = route_question("Which events has Jon participated in?", candidate_skills())
    open_plan = route_question("What fields would Caroline be likely to pursue?", candidate_skills())
    routed_console = route_question("What Console does Nate own?", candidate_skills())
    routed_children = route_question("How many children does Melanie have?", candidate_skills())
    routed_symbols = route_question("What symbols are important to Caroline?", candidate_skills())
    budget = budget_for_plan(plan, default_bm25=10, default_graph=4)
    open_state = OpenDomainStateAggregator(sample)
    open_hits = open_state.retrieve("What fields would Caroline be likely to pursue?", top_k=2)
    open_expansions = expand_open_domain_query("What Console does Nate own?")
    chains = EvidenceChainComposer().compose("What activities has Melanie done?", [graph_hits], top_k=2)
    reranked = rerank_multihop_evidence("What do Caroline and Melanie both have in common?", [graph_hits], max_items=2)
    subqueries = decompose_question("What do Jon and Gina both have in common?", "multi_hop")
    verified = verify_answer(
        "What did Caroline research?",
        "Caroline is researching adoption agencies with the dream of having a family",
        ["Caroline: She researched adoption agencies."],
        "span",
    )
    cbi = select_cbi_answer(
        "Would Caroline pursue writing as a career option?",
        "No information available",
        "Likely no, she wants to be a counselor",
        ["Caroline wants to pursue counseling as a career."],
    )
    rescue = select_rescue_answer(
        "Would Caroline want to move back to her home country soon?",
        "No information available",
        "Likely no",
        ["Caroline is in the process of adopting children and building a family here."],
    )
    evolved = evolve_skills(
        [
            {
                "sample_id": "train-smoke",
                "qa": [
                    {
                        "category": 3,
                        "question": "What state did Alex visit last summer?",
                        "answer": "A state name",
                    }
                ],
            }
        ],
        candidate_skills(),
    )
    open_domain_skills = matching_open_domain_skills("What state did Alex visit last summer?", evolved)
    skill_prompt = build_open_domain_skill_prompt(
        "What state did Alex visit last summer?",
        ["Alex visited a city during summer."],
        open_domain_skills,
    )
    checks.extend(
        [
            memory.answer("What is Caroline's relationship status?")[0] == "Single",
            memory.answer("Where did Caroline move from 4 years ago?")[0] == "Sweden",
            memory.answer("Where has Melanie camped?")[0] == "beach, forest",
            isinstance(graph_hits, list),
            plan.query_type == "multi_hop",
            open_plan.query_type == "open_domain",
            routed_console.query_type == "open_domain",
            routed_children.query_type == "multi_hop",
            routed_symbols.query_type == "multi_hop",
            budget.graph_top_k >= 4,
            isinstance(open_hits, list),
            any("gaming" in item.query for item in open_expansions),
            isinstance(chains, list),
            isinstance(reranked.context_ids, list),
            len(subqueries) >= 2,
            verified.answer == "adoption agencies",
            cbi.accepted and cbi.answer == "Likely no",
            should_rescue_no_information(
                "Would Caroline want to move back to her home country soon?",
                "No information available",
                ["Caroline is in the process of adopting children and building a family here."],
            ),
            should_refine_open_domain_answer(
                "What card game is Deborah talking about?",
                "a card game about cats",
                ["Deborah played a card game about cats where players attack opponents with cards."],
            ),
            rescue.accepted and rescue.answer == "Likely no",
            any(skill.name == "open_domain_location_city_to_state" for skill in evolved),
            bool(open_domain_skills),
            "Frozen open-domain skills" in skill_prompt,
            runner_helpers.should_use_open_domain_cbi("open_domain"),
            not runner_helpers.should_use_open_domain_cbi(plan),
        ]
    )
    gated_bundle, gate_audit = runner_helpers.gate_temporal_memory_by_agreement(
        {
            "evidence_nodes": [
                {"source_ids": ["D3:15"]},
                {"source_ids": ["D26:2"]},
            ],
            "evidence_paths": [{"source_ids": ["D3:15"]}],
            "temporal_timeline": [{"source_ids": ["D26:2"]}],
        },
        ["D3:15", "D3:15", "S3"],
    )
    checks.extend(
        [
            gated_bundle["evidence_nodes"] == [{"source_ids": ["D3:15"]}],
            gated_bundle["evidence_paths"] == [{"source_ids": ["D3:15"]}],
            gated_bundle["temporal_timeline"] == [],
            gate_audit["dropped_items"] == 2,
        ]
    )
    if not all(checks):
        raise SystemExit("smoke test failed")
    print("smoke test passed")


if __name__ == "__main__":
    main()
