import argparse
from collections import Counter
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higraphmem2.answer.open_domain_cbi import answer_format_instruction


def add_source_imports(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--provider-config", default="configs/provider.yaml")
    parser.add_argument("--chat-model", default="", help="Override the chat model declared in --provider-config.")
    parser.add_argument("--data-file", default="data/locomo/data/locomo10.json")
    parser.add_argument("--output", default="experiments/results/locomo_v2_retrieval.json")
    parser.add_argument("--prediction-key", default="higraphmem_prediction")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSON and skip completed QA items.")
    parser.add_argument("--reuse-predictions-from", default="", help="Reuse model answers from a previous output JSON and rerun retrieval/post-processing.")
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--graph-top-k", default=4, type=int)
    parser.add_argument("--disable-graph", action="store_true")
    parser.add_argument(
        "--plain-text-rag",
        action="store_true",
        help="Baseline: fixed top-k raw-turn BM25 only; disable fact retrieval, routing, decomposition, temporal/open-domain post-processing, graph memory, and answer verification.",
    )
    parser.add_argument("--disable-bm25", action="store_true", help="Ablation: do not retrieve or include raw-turn BM25 evidence.")
    parser.add_argument("--disable-fact-memory", action="store_true", help="Ablation: do not retrieve or include Question-Focused Fact Memory evidence.")
    parser.add_argument("--enable-long-short-memory", action="store_true", help="Enable Grounded MemoryGraph plus Triggered Working Memory.")
    parser.add_argument("--disable-update", action="store_true", help="Ablation: initialize GMGraph from the first session and freeze it; do not ingest later sessions.")
    parser.add_argument("--disable-conflict-control", action="store_true", help="Ablation: retain Update but disable stale/conflicting-state inhibition.")
    parser.add_argument("--disable-twmem", action="store_true", help="Ablation: replace bounded, activated TWMem with equal-budget lexical retrieval over the full GMGraph.")
    parser.add_argument("--disable-gated-activation", action="store_true", help="Ablation: keep the bounded TWMem buffer but disable all query/intent activation and graph propagation.")
    parser.add_argument("--disable-relation-gate", action="store_true", help="Ablation: use equal graph-relation propagation coefficients.")
    parser.add_argument("--disable-router", action="store_true", help="Ablation: use one generic evidence-selection policy for all question types.")
    parser.add_argument("--disable-multihop-rerank", action="store_true", help="Ablation: do not reorder multi-hop evidence with the lexical chain heuristic.")
    parser.add_argument("--disable-temporal-resolver", action="store_true", help="Ablation: do not normalize temporal answers against retrieved dates after generation.")
    parser.add_argument("--disable-evidence-verification", action="store_true", help="Ablation: do not apply the post-generation evidence-support verifier.")
    parser.add_argument("--memory-run-dir", default="experiments/memory_runs", help="Directory for conversation-local long/short memory artifacts.")
    parser.add_argument("--long-short-top-k", default=8, type=int)
    parser.add_argument("--long-short-mode", choices=["augment", "only", "adaptive", "selective", "typed"], default="augment", help="Use Memory Evidence Bundle as augmentation, only evidence, question-adaptive activation, multi-hop-only selective activation, or Typed Memory Routing.")
    parser.add_argument(
        "--evidence-selection-mode",
        choices=["concat", "graph_guided_raw"],
        default="concat",
        help="concat preserves the reported multi-source prompt; graph_guided_raw uses graph/TWMem to select raw turns from a BM25 candidate pool.",
    )
    parser.add_argument("--graph-guided-candidate-pool", default=30, type=int, help="BM25 candidate-pool size for graph_guided_raw mode.")
    parser.add_argument("--graph-guided-max-turns", default=6, type=int, help="Maximum raw turns included by graph_guided_raw mode.")
    parser.add_argument("--graph-guided-max-chars", default=6000, type=int, help="Maximum raw-evidence characters included by graph_guided_raw mode.")
    parser.add_argument("--graph-only-update-ablation", action="store_true", help="Use only GMGraph evidence for a controlled Full-vs-w/o-Update comparison; excludes fact/BM25 evidence from answer context.")
    parser.add_argument("--enable-evidence-chain", action="store_true", help="Enable the previous heuristic evidence-chain module for ablation.")
    parser.add_argument("--enable-open-domain-state", action="store_true", help="Enable the previous open-domain profile module for ablation.")
    parser.add_argument("--disable-open-domain-cbi", action="store_true")
    parser.add_argument("--disable-open-domain-self-consistency", action="store_true")
    parser.add_argument(
        "--disable-temporal-memory-agreement-gate",
        action="store_true",
        help="Ablation: allow temporal long/short evidence even when it disagrees with base retrieval.",
    )
    parser.add_argument("--disable-dual-realization", action="store_true")
    parser.add_argument("--open-domain-expansion-mode", choices=["off", "weak", "always"], default="weak")
    parser.add_argument("--open-domain-expansion-threshold", default=0.24, type=float)
    parser.add_argument("--enable-specific-refiner", action="store_true", help="Enable narrow open-domain specific-name refinement for ablation.")
    parser.add_argument("--enable-targeted-open-ablation", action="store_true", help="Deprecated diagnostic ablation only; do not use for reported runs.")
    parser.add_argument("--skip-conversations", default=0, type=int)
    parser.add_argument("--max-conversations", default=0, type=int)
    parser.add_argument("--max-questions", default=0, type=int)
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated LoCoMo categories to evaluate, e.g. 3 for Open-Domain or 1,2,3,4 for the GAM protocol.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max-tokens", default=64, type=int)
    parser.add_argument(
        "--answer-prompt-profile",
        choices=["auto", "generic", "qwen_strict"],
        default="auto",
        help="Answer-realization prompt profile. auto selects qwen_strict for Qwen model names.",
    )
    parser.add_argument("--max-retries", default=5, type=int)
    parser.add_argument("--retry-sleep", default=2.0, type=float)
    parser.add_argument("--fail-on-api-error", action="store_true")
    parser.add_argument("--official-category5-options", action="store_true")
    parser.add_argument(
        "--skip-category5",
        action="store_true",
        help="Skip LoCoMo category 5 before retrieval and generation; use for GAM-style category 1-4 evaluation.",
    )
    args = parser.parse_args()
    selected_categories = parse_categories(args.categories, parser)
    if args.skip_category5 and args.official_category5_options:
        parser.error("--skip-category5 cannot be combined with --official-category5-options")
    if args.skip_category5 and "5" in selected_categories:
        parser.error("--skip-category5 cannot be combined with category 5 in --categories")
    if args.disable_twmem and args.disable_gated_activation:
        parser.error("--disable-twmem cannot be combined with --disable-gated-activation")
    if args.evidence_selection_mode == "graph_guided_raw":
        if args.plain_text_rag or args.disable_bm25 or not args.enable_long_short_memory:
            parser.error("--evidence-selection-mode graph_guided_raw requires BM25 and --enable-long-short-memory")
        if args.graph_guided_candidate_pool < 1 or args.graph_guided_max_turns < 1 or args.graph_guided_max_chars < 1:
            parser.error("graph-guided evidence limits must be positive")
    if args.graph_only_update_ablation:
        if not args.enable_long_short_memory:
            parser.error("--graph-only-update-ablation requires --enable-long-short-memory")
        if args.disable_twmem:
            parser.error("--graph-only-update-ablation cannot be combined with --disable-twmem")
        args.long_short_mode = "only"
        # Source agreement reads the text retrievers, which would reintroduce
        # the very bypass this diagnostic is intended to remove.
        args.disable_temporal_memory_agreement_gate = True

    source_root = Path(args.source_root)
    add_source_imports(source_root)

    from src.data.locomo import load_locomo
    from src.evaluation.locomo_metrics import score_qa
    from src.llm.openai_compatible import OpenAICompatibleModel
    from src.memory.locomo_memory import LoCoMoMemory, format_evidence
    from higraphmem2.answer.evidence_verifier import verify_answer
    from higraphmem2.answer.normalizer import normalize_answer_text
    from higraphmem2.answer.open_domain_cbi import (
        build_cbi_prompt,
        build_cbi_refine_prompt,
        build_cbi_rescue_prompt,
        build_cbi_self_consistency_prompts,
        evidence_relevance,
        select_cbi_answer,
        select_refine_answer,
        select_rescue_answer,
        select_self_consistent_answer,
        resolve_specific_open_domain_answer,
        should_refine_open_domain_answer,
        should_rescue_no_information,
    )
    from higraphmem2.answer.temporal import build_session_time_index, resolve_temporal_answer
    from higraphmem2.memory.fact_memory import QuestionFocusedFactMemory, format_fact_hits
    from higraphmem2.memory.open_domain_state import OpenDomainStateAggregator, format_open_domain_hits
    from higraphmem2.retrieval.evidence_budget import budget_for_plan
    from higraphmem2.retrieval.evidence_chain import EvidenceChainComposer, format_evidence_chains
    from higraphmem2.retrieval.multihop_reranker import rerank_multihop_evidence
    from higraphmem2.retrieval.open_domain_expansion import expand_open_domain_query
    from higraphmem2.retrieval.query_decomposition import decompose_question, format_decomposition
    from higraphmem2.retrieval.skill_router import QueryPlan, route_question

    samples = load_locomo(str(source_root / args.data_file))
    if args.skip_conversations > 0:
        samples = samples[args.skip_conversations :]
    if args.max_conversations > 0:
        samples = samples[: args.max_conversations]
    long_short_tools = load_long_short_memory_tools(ROOT) if args.enable_long_short_memory else None

    if args.dry_run:
        generator = None
        chat_model = args.chat_model
    else:
        provider = load_provider_config(source_root, args.provider_config)
        chat_model = args.chat_model or provider.get("chat_model", "gpt-3.5-turbo")
        generator = OpenAICompatibleModel(
            model=chat_model,
            api_key_env=provider.get("api_key_env", "OPENAI_API_KEY"),
            api_key=provider.get("api_key"),
            base_url=provider.get("base_url") or provider.get("api_base") or None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
    answer_prompt_profile = resolve_answer_prompt_profile(args.answer_prompt_profile, chat_model)

    resumed_samples = load_prediction_samples(args.output, args.prediction_key) if args.resume else {}
    reusable_samples = load_prediction_samples(args.reuse_predictions_from, args.prediction_key) if args.reuse_predictions_from else {}
    output_samples = []
    processed = 0
    skipped = 0
    skipped_category5 = 0
    skipped_unselected_category = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        bm25_memory = None if args.disable_bm25 else LoCoMoMemory(sample)
        fact_memory = None if args.plain_text_rag or args.disable_fact_memory else QuestionFocusedFactMemory(sample)
        open_domain_memory = OpenDomainStateAggregator(sample) if args.enable_open_domain_state and not args.plain_text_rag else None
        chain_composer = EvidenceChainComposer() if args.enable_evidence_chain and not args.plain_text_rag else None
        time_index = build_session_time_index(sample) if not args.plain_text_rag else None
        long_short_graph = None
        if long_short_tools:
            long_short_graph = long_short_tools["graph"].build_graph(
                sample,
                enable_update=not args.disable_update,
                enable_conflict_control=not args.disable_conflict_control,
            )
            if args.evidence_selection_mode == "graph_guided_raw":
                annotate_state_versions(long_short_graph["nodes"])
            write_long_short_graph_artifacts(Path(args.memory_run_dir) / sample_id / "long_term_graph", long_short_graph)
        resumed_qa = resumed_samples.get(sample_id, {})
        reusable_qa = reusable_samples.get(sample_id, {})
        out_sample = {"sample_id": sample_id, "qa": []}
        output_samples.append(out_sample)
        for qa in sample.get("qa", []):
            category = str(qa.get("category", ""))
            if selected_categories and category not in selected_categories:
                skipped_unselected_category += 1
                continue
            if args.skip_category5 and category == "5":
                skipped_category5 += 1
                continue
            if args.max_questions > 0 and processed >= args.max_questions:
                break
            resumed_item = resumed_qa.get(qa_resume_key(qa))
            if resumed_item and is_completed_prediction(resumed_item, args.prediction_key):
                out_sample["qa"].append(resumed_item)
                processed += 1
                skipped += 1
                continue
            reusable_item = reusable_qa.get(qa_resume_key(qa))
            visible_qa = build_visible_qa(qa, args.official_category5_options)
            out_qa = qa.copy()
            usage_events = []
            if generator is not None:
                # Current remote calls realize/refine answers. The phase field
                # leaves room for future LLM-backed build/update/retrieval calls.
                generator.set_usage_callback(usage_events.append)
            retrieval_question = visible_qa["question"]
            generation_question = visible_qa["generation_question"]
            semantic_plan = (
                QueryPlan(query_type="single_hop", answer_mode="span", confidence=1.0)
                if args.plain_text_rag
                else route_question(retrieval_question, [])
            )
            # Router ablation uses one fixed evidence policy. Semantic answer
            # formatting and the independent Time Gate still use the original
            # question interpretation, so disabling Router does not disable
            # either of those separate modules.
            query_plan = (
                QueryPlan(query_type="single_hop", answer_mode=semantic_plan.answer_mode, confidence=1.0)
                if args.disable_router and not args.plain_text_rag
                else semantic_plan
            )
            long_short_text = ""
            long_short_ids = []
            long_short_payload = None
            working_memory = None
            memory_dir = None
            if long_short_tools and long_short_graph:
                if args.disable_twmem:
                    working_memory = long_short_tools["stm"].induce_global_graph_memory(
                        long_short_graph["metadata"], retrieval_question
                    )
                else:
                    working_memory = long_short_tools["stm"].induce_working_memory(
                        long_short_graph["metadata"],
                        long_short_graph["nodes"],
                        long_short_graph["edges"],
                        retrieval_question,
                        enable_gated_activation=not args.disable_gated_activation,
                        enable_relation_gate=not args.disable_relation_gate,
                    )
                retrieve_top_k = args.long_short_top_k
                if args.long_short_mode == "typed" and query_plan.query_type == "temporal":
                    retrieve_top_k = max(retrieve_top_k, 12)
                long_short_payload = long_short_tools["retrieve"].retrieve(
                    retrieval_question,
                    long_short_graph["nodes"],
                    long_short_graph["edges"],
                    working_memory,
                    top_k=retrieve_top_k,
                )
                if args.long_short_mode in {"adaptive", "selective", "typed"}:
                    long_short_payload = adapt_long_short_bundle(long_short_payload, query_plan, mode=args.long_short_mode)
                memory_dir = Path(args.memory_run_dir) / sample_id / qa_memory_dir_name(visible_qa)
                long_short_text, long_short_ids = format_long_short_bundle(long_short_payload)
            budget = budget_for_plan(query_plan, default_bm25=args.top_k, default_graph=args.graph_top_k, graph_disabled=args.disable_graph)
            subqueries = [] if args.plain_text_rag else decompose_question(retrieval_question, query_plan.query_type)
            open_domain_expansions = []
            fact_hits = [] if fact_memory is None else fact_memory.retrieve(retrieval_question, top_k=budget.fact_top_k)
            decomposed_fact_hits = []
            decomposed_bm25_hits = []
            if subqueries:
                for subquery in subqueries:
                    if fact_memory is not None:
                        decomposed_fact_hits.extend(fact_memory.retrieve(subquery.subquestion, top_k=2))
                    if bm25_memory is not None:
                        decomposed_bm25_hits.extend(bm25_memory.retrieve(subquery.subquestion, top_k=2))
                fact_hits = merge_hits(fact_hits, decomposed_fact_hits, budget.fact_top_k + min(len(subqueries), 3))
            bm25_top_k = budget.bm25_top_k
            if args.evidence_selection_mode == "graph_guided_raw":
                # Session summaries and observations share the legacy BM25
                # index. Retrieve a wider local pool, then retain only raw
                # dialogue turns for the graph-guided final-evidence budget.
                bm25_top_k = max(bm25_top_k, args.graph_guided_candidate_pool * 3)
            bm25_hits = [] if bm25_memory is None else bm25_memory.retrieve(retrieval_question, top_k=bm25_top_k)
            if decomposed_bm25_hits:
                bm25_hits = merge_hits(bm25_hits, decomposed_bm25_hits, budget.bm25_top_k + min(len(subqueries), 3))
            base_fact_text, _ = format_fact_hits(fact_hits)
            base_bm25_text, _ = format_evidence(bm25_hits)
            open_domain_base_relevance = 0.0
            if query_plan.query_type == "open_domain" and not args.plain_text_rag:
                open_domain_base_relevance = evidence_relevance(
                    retrieval_question,
                    "\n".join(part for part in [base_fact_text, base_bm25_text] if part),
                )
                if should_expand_open_domain(args.open_domain_expansion_mode, open_domain_base_relevance, args.open_domain_expansion_threshold):
                    open_domain_expansions = expand_open_domain_query(retrieval_question)
            expanded_fact_hits = []
            expanded_bm25_hits = []
            if open_domain_expansions:
                for expansion in open_domain_expansions:
                    if fact_memory is not None:
                        expanded_fact_hits.extend(fact_memory.retrieve(expansion.query, top_k=3))
                    if bm25_memory is not None:
                        expanded_bm25_hits.extend(bm25_memory.retrieve(expansion.query, top_k=3))
                fact_hits = merge_hits(fact_hits, expanded_fact_hits, budget.fact_top_k + min(len(open_domain_expansions) * 2, 8))
            fact_text, fact_ids = format_fact_hits(fact_hits)
            # The reported system uses only raw-text retrieval and the typed
            # temporal long--short memory.  Legacy skill/state graph retrieval
            # is intentionally excluded to keep evaluation data-independent.
            state_hits = []
            graph_hits = []
            if expanded_bm25_hits:
                bm25_hits = merge_hits(bm25_hits, expanded_bm25_hits, budget.bm25_top_k + min(len(open_domain_expansions) * 2, 8))
            open_domain_hits = []
            if open_domain_memory and query_plan.query_type == "open_domain":
                open_domain_hits = open_domain_memory.retrieve(retrieval_question, top_k=3)
            evidence_chains = []
            if chain_composer and query_plan.query_type == "multi_hop":
                evidence_chains = chain_composer.compose(
                    retrieval_question,
                    [fact_hits, graph_hits, state_hits, bm25_hits, open_domain_hits],
                    top_k=3,
                )
            multihop_rerank = None
            if should_use_multihop_rerank(query_plan.query_type) and not args.plain_text_rag and not args.disable_multihop_rerank:
                multihop_rerank = rerank_multihop_evidence(
                    retrieval_question,
                    [fact_hits, graph_hits, state_hits, bm25_hits],
                    max_items=8,
                )
            graph_text, graph_ids = "", []
            state_text, state_ids = "", []
            bm25_text, bm25_ids = format_evidence(bm25_hits)
            open_domain_text, open_domain_ids = format_open_domain_hits(open_domain_hits)
            chain_text, chain_ids = format_evidence_chains(evidence_chains)
            multihop_text = multihop_rerank.evidence_text if multihop_rerank else ""
            multihop_ids = multihop_rerank.context_ids if multihop_rerank else []
            decomposition_text = format_decomposition(subqueries)
            temporal_memory_gate = {"applied": False, "kept_items": 0, "dropped_items": 0}
            if long_short_payload and semantic_plan.query_type == "temporal" and not args.disable_temporal_memory_agreement_gate:
                # A graph path is auxiliary evidence for dates.  It must point to a turn
                # independently retrieved by the raw-text retrievers; otherwise an event
                # with similar wording can overwrite the correct temporal anchor.
                agreement_ids = fact_ids + bm25_ids
                long_short_payload, temporal_memory_gate = gate_temporal_memory_by_agreement(
                    long_short_payload,
                    agreement_ids,
                )
                long_short_text, long_short_ids = format_long_short_bundle(long_short_payload)
            if memory_dir is not None:
                memory_dir.mkdir(parents=True, exist_ok=True)
                (memory_dir / "short_term_working_memory.json").write_text(
                    json.dumps(working_memory, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (memory_dir / "evidence_bundle.json").write_text(
                    json.dumps(long_short_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            graph_guided_selection = None
            if args.plain_text_rag:
                evidence_text = bm25_text
                context_ids = bm25_ids
            elif args.evidence_selection_mode == "graph_guided_raw":
                graph_guided_selection = select_graph_guided_raw_evidence(
                    bm25_hits,
                    long_short_payload or {},
                    retrieval_question,
                    candidate_pool=args.graph_guided_candidate_pool,
                    max_turns=args.graph_guided_max_turns,
                    max_chars=args.graph_guided_max_chars,
                    raw_turn_lookup={
                        str(node["memory_id"]): str(node["text"])
                        for node in getattr(bm25_memory, "nodes", [])
                        if node.get("source_type") == "turn"
                    },
                )
                evidence_text = graph_guided_selection["evidence_text"]
                context_ids = graph_guided_selection["selected_source_ids"]
            elif args.long_short_mode == "only" and long_short_tools:
                evidence_text = long_short_text
                context_ids = long_short_ids
            else:
                if query_plan.query_type == "temporal":
                    # Preserve the raw retrieved turn as the primary temporal anchor;
                    # graph/timeline text only disambiguates that same turn.
                    evidence_parts = [fact_text, bm25_text, long_short_text, graph_text, state_text, multihop_text, chain_text, open_domain_text]
                    context_ids = fact_ids + bm25_ids + long_short_ids + graph_ids + state_ids + multihop_ids + chain_ids + open_domain_ids
                else:
                    evidence_parts = [long_short_text, fact_text, graph_text, state_text, multihop_text, bm25_text, chain_text, open_domain_text]
                    context_ids = long_short_ids + multihop_ids + chain_ids + fact_ids + graph_ids + open_domain_ids + state_ids + bm25_ids
                evidence_text = "\n".join(part for part in evidence_parts if part)
            out_qa[args.prediction_key + "_context"] = context_ids
            out_qa[args.prediction_key + "_long_short_context"] = long_short_ids
            out_qa[args.prediction_key + "_multihop_rerank_context"] = multihop_ids
            out_qa[args.prediction_key + "_chain_context"] = chain_ids
            out_qa[args.prediction_key + "_fact_context"] = fact_ids
            out_qa[args.prediction_key + "_graph_context"] = graph_ids
            out_qa[args.prediction_key + "_open_domain_context"] = open_domain_ids
            out_qa[args.prediction_key + "_state_context"] = state_ids
            if graph_guided_selection is not None:
                out_qa[args.prediction_key + "_graph_guided_selection"] = graph_guided_selection
            out_qa[args.prediction_key + "_query_plan"] = {
                "query_type": semantic_plan.query_type,
                "evidence_policy_type": query_plan.query_type,
                "answer_mode": semantic_plan.answer_mode,
                "active_skills": query_plan.active_skills,
                "confidence": round(query_plan.confidence, 3),
                "alternative_types": query_plan.alternative_types,
                "multihop_rerank_source": multihop_rerank.source if multihop_rerank else "",
                "subquestions": [{"question": item.subquestion, "reason": item.reason} for item in subqueries],
                "open_domain_expansions": [{"query": item.query, "reason": item.reason} for item in open_domain_expansions],
                "open_domain_base_relevance": round(open_domain_base_relevance, 3),
                "open_domain_expansion_mode": args.open_domain_expansion_mode,
                "long_short_memory_enabled": bool(long_short_tools) and not args.disable_twmem,
                "graph_memory_evidence_enabled": bool(long_short_tools),
                "graph_only_update_ablation": args.graph_only_update_ablation,
                "evidence_selection_mode": args.evidence_selection_mode,
                "graph_guided_candidate_pool": args.graph_guided_candidate_pool if args.evidence_selection_mode == "graph_guided_raw" else None,
                "graph_guided_max_turns": args.graph_guided_max_turns if args.evidence_selection_mode == "graph_guided_raw" else None,
                "long_short_mode": args.long_short_mode,
                "graph_update_protocol": long_short_graph["metadata"]["consolidation"]["algorithm"] if long_short_graph else None,
                "graph_ingested_sessions": long_short_graph["metadata"]["ingested_sessions"] if long_short_graph else [],
                "graph_num_updates": long_short_graph["metadata"]["num_updates"] if long_short_graph else 0,
                "update_enabled": not args.disable_update,
                "conflict_control_enabled": not args.disable_update and not args.disable_conflict_control,
                "twmem_enabled": not args.disable_twmem,
                "gated_activation_enabled": not args.disable_twmem and not args.disable_gated_activation,
                "relation_gate_enabled": not args.disable_twmem and not args.disable_gated_activation and not args.disable_relation_gate,
                "router_enabled": not args.disable_router,
                "long_short_evidence_count": len(long_short_ids),
                "temporal_memory_agreement_gate": temporal_memory_gate,
                "budget": {
                    "fact_top_k": budget.fact_top_k,
                    "graph_top_k": budget.graph_top_k,
                    "state_top_k": budget.state_top_k,
                    "bm25_top_k": budget.bm25_top_k,
                },
            }
            if long_short_payload:
                out_qa[args.prediction_key + "_long_short_memory"] = {
                    "intent": long_short_payload.get("intent", ""),
                    "evidence_nodes": len(long_short_payload.get("evidence_nodes", [])),
                    "evidence_paths": len(long_short_payload.get("evidence_paths", [])),
                    "open_domain_clues": len(long_short_payload.get("open_domain_clues", [])),
                    "active_operations": long_short_payload.get("active_operations", []),
                    "adaptive_policy": long_short_payload.get("adaptive_policy", {}),
                    "audit": long_short_payload.get("audit", {}),
                }
            if args.dry_run:
                out_qa[args.prediction_key] = evidence_text.splitlines()[0][:160] if evidence_text else "No information available"
            elif reusable_item and is_completed_prediction(reusable_item, args.prediction_key):
                out_qa[args.prediction_key] = reusable_model_prediction(reusable_item, args.prediction_key)
                out_qa[args.prediction_key + "_reused_from"] = args.reuse_predictions_from
            else:
                try:
                    out_qa[args.prediction_key] = generator.generate(
                        build_prompt(
                            generation_question,
                            evidence_text,
                            semantic_plan.answer_mode,
                            profile=answer_prompt_profile,
                        )
                    )
                except Exception as exc:
                    if args.fail_on_api_error:
                        raise
                    out_qa[args.prediction_key] = "No information available"
                    out_qa[args.prediction_key + "_error"] = str(exc)
            normalized = normalize_answer_text(out_qa["question"], out_qa[args.prediction_key])
            normalized = map_category5_option(normalized, out_qa)
            if semantic_plan.query_type == "temporal" and not args.plain_text_rag and not args.disable_temporal_resolver:
                resolved, temporal_source = resolve_temporal_answer(
                    out_qa["question"],
                    normalized,
                    evidence_text.splitlines(),
                    time_index.dates_for_context_ids(context_ids),
                )
                if resolved != normalized:
                    out_qa[args.prediction_key + "_before_temporal_resolver"] = normalized
                    normalized = resolved
                out_qa[args.prediction_key + "_temporal_resolver_source"] = temporal_source
            if not args.plain_text_rag and should_use_open_domain_cbi(semantic_plan) and not args.disable_open_domain_cbi:
                open_domain_candidates = []
                if args.dry_run:
                    out_qa[args.prediction_key + "_cbi"] = {
                        "accepted": False,
                        "source": "dry_run_skipped",
                        "support_score": 0.0,
                        "candidate": "",
                    }
                else:
                    stored_cbi = reusable_item.get(args.prediction_key + "_cbi") if reusable_item else None
                    if stored_cbi and stored_cbi.get("candidate"):
                        cbi_candidate = stored_cbi.get("candidate")
                        out_qa[args.prediction_key + "_cbi_reused"] = True
                    elif reusable_item:
                        cbi_candidate = normalized
                        out_qa[args.prediction_key + "_cbi_reuse_skipped_missing_candidate"] = True
                    else:
                        try:
                            cbi_candidate = generator.generate(build_cbi_prompt(generation_question, evidence_text.splitlines()))
                        except Exception as exc:
                            cbi_candidate = normalized
                            out_qa[args.prediction_key + "_cbi_error"] = str(exc)
                    open_domain_candidates.append(cbi_candidate)
                    cbi = select_cbi_answer(out_qa["question"], normalized, cbi_candidate, evidence_text.splitlines())
                    out_qa[args.prediction_key + "_cbi"] = {
                        "accepted": cbi.accepted,
                        "source": cbi.source,
                        "support_score": round(cbi.support_score, 3),
                        "candidate": cbi_candidate,
                    }
                    if cbi.accepted:
                        out_qa[args.prediction_key + "_before_cbi"] = normalized
                        normalized = cbi.answer
                    if not args.disable_open_domain_self_consistency:
                        self_consistency_candidates = []
                        stored_sc = reusable_item.get(args.prediction_key + "_open_domain_self_consistency") if reusable_item else None
                        if stored_sc and stored_sc.get("candidates"):
                            self_consistency_candidates = list(stored_sc.get("candidates", []))
                            out_qa[args.prediction_key + "_open_domain_self_consistency_reused"] = True
                        elif reusable_item:
                            out_qa[args.prediction_key + "_open_domain_self_consistency_reuse_skipped_missing_candidates"] = True
                        else:
                            for prompt in build_cbi_self_consistency_prompts(generation_question, evidence_text.splitlines()):
                                try:
                                    self_consistency_candidates.append(generator.generate(prompt))
                                except Exception as exc:
                                    out_qa[args.prediction_key + "_open_domain_self_consistency_error"] = str(exc)
                        if self_consistency_candidates:
                            open_domain_candidates.extend(self_consistency_candidates)
                            sc = select_self_consistent_answer(
                                out_qa["question"],
                                normalized,
                                self_consistency_candidates,
                                evidence_text.splitlines(),
                            )
                            out_qa[args.prediction_key + "_open_domain_self_consistency"] = {
                                "accepted": sc.accepted,
                                "source": sc.source,
                                "support_score": round(sc.support_score, 3),
                                "candidate": sc.answer,
                                "candidates": self_consistency_candidates,
                            }
                            if sc.accepted:
                                out_qa[args.prediction_key + "_before_open_domain_self_consistency"] = normalized
                                normalized = sc.answer
                    if should_rescue_no_information(out_qa["question"], normalized, evidence_text.splitlines()):
                        stored_rescue = reusable_item.get(args.prediction_key + "_cbi_rescue") if reusable_item else None
                        if stored_rescue and stored_rescue.get("candidate"):
                            rescue_candidate = stored_rescue.get("candidate")
                            out_qa[args.prediction_key + "_cbi_rescue_reused"] = True
                        elif reusable_item:
                            rescue_candidate = normalized
                            out_qa[args.prediction_key + "_cbi_rescue_reuse_skipped_missing_candidate"] = True
                        else:
                            try:
                                rescue_candidate = generator.generate(build_cbi_rescue_prompt(generation_question, evidence_text.splitlines()))
                            except Exception as exc:
                                rescue_candidate = normalized
                                out_qa[args.prediction_key + "_cbi_rescue_error"] = str(exc)
                        open_domain_candidates.append(rescue_candidate)
                        rescue = select_rescue_answer(out_qa["question"], normalized, rescue_candidate, evidence_text.splitlines())
                        out_qa[args.prediction_key + "_cbi_rescue"] = {
                            "accepted": rescue.accepted,
                            "source": rescue.source,
                            "support_score": round(rescue.support_score, 3),
                            "candidate": rescue_candidate,
                        }
                        if rescue.accepted:
                            out_qa[args.prediction_key + "_before_cbi_rescue"] = normalized
                            normalized = rescue.answer
                    if args.enable_specific_refiner and should_refine_open_domain_answer(out_qa["question"], normalized, evidence_text.splitlines()):
                        stored_refine = reusable_item.get(args.prediction_key + "_cbi_refine") if reusable_item else None
                        if stored_refine and stored_refine.get("candidate"):
                            refine_candidate = stored_refine.get("candidate")
                            out_qa[args.prediction_key + "_cbi_refine_reused"] = True
                        elif reusable_item:
                            refine_candidate = normalized
                            out_qa[args.prediction_key + "_cbi_refine_reuse_skipped_missing_candidate"] = True
                        else:
                            try:
                                refine_candidate = generator.generate(build_cbi_refine_prompt(generation_question, normalized, evidence_text.splitlines()))
                            except Exception as exc:
                                refine_candidate = normalized
                                out_qa[args.prediction_key + "_cbi_refine_error"] = str(exc)
                        open_domain_candidates.append(refine_candidate)
                        refine = select_refine_answer(out_qa["question"], normalized, refine_candidate, evidence_text.splitlines())
                        out_qa[args.prediction_key + "_cbi_refine"] = {
                            "accepted": refine.accepted,
                            "source": refine.source,
                            "support_score": round(refine.support_score, 3),
                            "candidate": refine_candidate,
                        }
                        if refine.accepted:
                            out_qa[args.prediction_key + "_before_cbi_refine"] = normalized
                            normalized = refine.answer
                    if args.enable_targeted_open_ablation:
                        specific = resolve_specific_open_domain_answer(
                            out_qa["question"],
                            normalized,
                            evidence_text.splitlines(),
                            open_domain_candidates,
                        )
                        if specific and specific.accepted:
                            out_qa[args.prediction_key + "_before_specific_open_domain"] = normalized
                            out_qa[args.prediction_key + "_specific_open_domain"] = {
                                "accepted": specific.accepted,
                                "source": specific.source,
                                "support_score": round(specific.support_score, 3),
                                "candidate": specific.answer,
                            }
                            normalized = specific.answer
            if not args.plain_text_rag and should_use_dual_realization(semantic_plan, normalized) and not args.disable_dual_realization and not args.disable_open_domain_cbi:
                dual_result = run_dual_realization(
                    out_qa,
                    normalized,
                    generation_question,
                    evidence_text,
                    generator,
                    reusable_item,
                    args.prediction_key,
                    args.dry_run,
                )
                if dual_result:
                    dual_answer, dual_payload = dual_result
                    out_qa[args.prediction_key + "_dual_realization"] = dual_payload
                    if dual_answer != normalized:
                        out_qa[args.prediction_key + "_before_dual_realization"] = normalized
                        normalized = dual_answer
            if not args.plain_text_rag and not args.disable_evidence_verification:
                verification = verify_answer(out_qa["question"], normalized, evidence_text.splitlines(), semantic_plan.answer_mode)
                if verification.answer != normalized:
                    out_qa[args.prediction_key + "_before_evidence_verification"] = normalized
                    normalized = verification.answer
                out_qa[args.prediction_key + "_evidence_verification"] = {
                    "supported": verification.supported,
                    "source": verification.source,
                    "support_score": round(verification.support_score, 3),
                }
            out_qa[args.prediction_key] = normalized
            if usage_events:
                out_qa[args.prediction_key + "_token_usage"] = summarize_qa_token_usage(usage_events)
            if "answer" in out_qa:
                out_qa[args.prediction_key + "_f1"] = round(score_qa(normalized, out_qa), 3)
            out_sample["qa"].append(out_qa)
            processed += 1
            write_output(output_samples, args.prediction_key, args.output, score_qa)
        if args.max_questions > 0 and processed >= args.max_questions:
            break

    stats = write_output(output_samples, args.prediction_key, args.output, score_qa)
    run_meta = {
        "dataset": "locomo",
        "status": "completed",
        "skills": "disabled",
        "skip_conversations": args.skip_conversations,
        "skip_category5": args.skip_category5,
        "selected_categories": sorted(selected_categories),
        "num_skipped_category5_questions": skipped_category5,
        "num_skipped_unselected_category_questions": skipped_unselected_category,
        "official_category5_options": args.official_category5_options,
        "graph_enabled": not args.disable_graph,
        "plain_text_rag": args.plain_text_rag,
        "bm25_enabled": not args.disable_bm25,
        "fact_memory_enabled": not args.disable_fact_memory and not args.plain_text_rag,
        "long_short_memory_enabled": args.enable_long_short_memory,
        "graph_only_update_ablation": args.graph_only_update_ablation,
        "evidence_selection_mode": args.evidence_selection_mode,
        "graph_guided_candidate_pool": args.graph_guided_candidate_pool if args.evidence_selection_mode == "graph_guided_raw" else None,
        "graph_guided_max_turns": args.graph_guided_max_turns if args.evidence_selection_mode == "graph_guided_raw" else None,
        "twmem_replacement": "full_graph_lexical" if args.disable_twmem else "none",
        "long_short_mode": args.long_short_mode,
        "update_enabled": not args.disable_update,
        "conflict_control_enabled": not args.disable_update and not args.disable_conflict_control,
        "twmem_enabled": not args.disable_twmem,
        "gated_activation_enabled": not args.disable_twmem and not args.disable_gated_activation,
        "relation_gate_enabled": not args.disable_twmem and not args.disable_gated_activation and not args.disable_relation_gate,
        "router_enabled": not args.disable_router,
        "multihop_rerank_enabled": not args.disable_multihop_rerank,
        "temporal_resolver_enabled": not args.disable_temporal_resolver,
        "evidence_verification_enabled": not args.disable_evidence_verification,
        "memory_run_dir": args.memory_run_dir,
        "chat_model": chat_model,
        "answer_prompt_profile": answer_prompt_profile,
        "graph_top_k": args.graph_top_k,
        "evidence_chain_enabled": args.enable_evidence_chain,
        "open_domain_state_enabled": args.enable_open_domain_state,
        "open_domain_cbi_enabled": not args.disable_open_domain_cbi,
        "open_domain_self_consistency_enabled": not args.disable_open_domain_self_consistency,
        "open_domain_expansion_mode": args.open_domain_expansion_mode,
        "open_domain_expansion_threshold": args.open_domain_expansion_threshold,
        "dual_realization_enabled": not args.disable_dual_realization,
        "specific_refiner_enabled": args.enable_specific_refiner,
        "targeted_open_ablation_enabled": args.enable_targeted_open_ablation,
        "routing_source": "question_only",
        "retrieval_controller": "skill_router+evidence_budget+query_decomposition",
        "answer_controller": "answer_realization+open_domain_cbi+evidence_verification",
        "num_conversations": len(output_samples),
        "num_questions": processed,
        "num_resumed_questions": skipped,
        "category_counts": dict(Counter(str(qa.get("category", "unknown")) for sample in output_samples for qa in sample["qa"])),
        "stats": stats,
    }
    output = Path(args.output)
    output.with_name(output.stem + "_run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


def parse_categories(value: str, parser: argparse.ArgumentParser) -> set[str]:
    if not value.strip():
        return set()
    categories = {part.strip() for part in value.split(",") if part.strip()}
    invalid = sorted(categories - {"1", "2", "3", "4", "5"})
    if invalid:
        parser.error("--categories accepts only comma-separated values from 1,2,3,4,5; got: %s" % ",".join(invalid))
    return categories


def load_prediction_samples(path: str, prediction_key: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path:
        return {}
    output = Path(path)
    if not output.exists():
        return {}
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    resumed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sample in payload:
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            continue
        resumed[sample_id] = {}
        for qa in sample.get("qa", []):
            resumed[sample_id][qa_resume_key(qa)] = qa
    return resumed


def qa_resume_key(qa: Dict[str, Any]) -> str:
    return "%s\t%s" % (qa.get("category", ""), qa.get("question", ""))


def is_completed_prediction(qa: Dict[str, Any], prediction_key: str) -> bool:
    if qa.get(prediction_key + "_error"):
        return False
    return bool(str(qa.get(prediction_key, "")).strip())


def reusable_model_prediction(qa: Dict[str, Any], prediction_key: str) -> str:
    for suffix in [
        "_before_temporal_resolver",
        "_before_multi_hop_aggregation",
        "_before_single_hop_exact",
        "_before_cbi",
        "_before_cbi_rescue",
        "_before_cbi_refine",
        "_before_evidence_verification",
    ]:
        value = qa.get(prediction_key + suffix)
        if value:
            return str(value)
    return str(qa.get(prediction_key, "No information available"))


def merge_hits(primary, extra, limit: int):
    output = []
    seen = set()
    for hit in list(primary) + list(extra):
        memory_id = str(getattr(hit, "memory_id", ""))
        if memory_id in seen:
            continue
        output.append(hit)
        seen.add(memory_id)
    return sorted(output, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)[:limit]


def load_long_short_memory_tools(root: Path) -> Dict[str, Any]:
    base = root / "skills" / "memory_construction_subskills"
    return {
        "graph": load_module_from_path(base / "build_typed_temporal_graph" / "scripts" / "build_graph.py", "higraphmem2_eval_graph_skill"),
        "stm": load_module_from_path(base / "induce_short_term_working_memory" / "scripts" / "induce_memory.py", "higraphmem2_eval_stm_skill"),
        "retrieve": load_module_from_path(base / "retrieve_long_short_memory" / "scripts" / "retrieve.py", "higraphmem2_eval_retrieve_skill"),
    }


def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load memory skill script: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_long_short_graph_artifacts(output_dir: Path, graph: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "nodes.jsonl", graph.get("nodes", []))
    write_jsonl(output_dir / "edges.jsonl", graph.get("edges", []))
    (output_dir / "graph_metadata.json").write_text(
        json.dumps(graph.get("metadata", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def format_long_short_bundle(bundle: Dict[str, Any]):
    lines = []
    ids = []
    timeline_source_ids = set()
    for item in bundle.get("temporal_timeline", []):
        source_ids = compact_source_ids(item.get("source_ids", []))
        timeline_source_ids.update(source_ids)
        ids.extend(source_ids)
        lines.append(
            "[%s][temporal_timeline][S%s] %s"
            % (
                ",".join(source_ids) or "timeline",
                item.get("session", "?"),
                item.get("text", ""),
            )
        )
    for node in bundle.get("evidence_nodes", []):
        source_ids = compact_source_ids(node.get("source_ids", []))
        if timeline_source_ids and set(source_ids) & timeline_source_ids:
            continue
        ids.extend(source_ids)
        lines.append(
            "[%s][long_short_node][%.3f] %s: %s"
            % (
                ",".join(source_ids) or node.get("node_id", ""),
                float(node.get("score", 0.0)),
                node.get("node_type", ""),
                node.get("label", ""),
            )
        )
    for path in bundle.get("evidence_paths", []):
        source_ids = compact_source_ids(path.get("source_ids", []))
        if timeline_source_ids and set(source_ids) & timeline_source_ids:
            continue
        ids.extend(source_ids)
        lines.append(
            "[%s][long_short_path][%.3f] %s"
            % (
                ",".join(source_ids) or "graph_path",
                float(path.get("confidence", 0.0)),
                " -> ".join(str(item) for item in path.get("labels", [])),
            )
        )
    for clue in bundle.get("open_domain_clues", []):
        source_ids = compact_source_ids(clue.get("evidence_ids", []))
        ids.extend(source_ids)
        lines.append(
            "[%s][long_short_clue][%.3f] %s (%s)"
            % (
                ",".join(source_ids) or "open_domain_clue",
                float(clue.get("weight", 0.0)),
                clue.get("clue", ""),
                clue.get("bridge_type", ""),
            )
        )
    return "\n".join(lines), dedupe_preserve_order(ids)


def select_graph_guided_raw_evidence(
    bm25_hits,
    graph_bundle: Dict[str, Any],
    question: str = "",
    candidate_pool: int = 30,
    max_turns: int = 6,
    max_chars: int = 6000,
    raw_turn_lookup: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Use graph/TWMem evidence to choose raw turns from a BM25 candidate pool.

    BM25 supplies recall only.  A graph-supported candidate is always preferred
    to an unsupported BM25 candidate; BM25 fills any remaining slots so a
    sparse graph cannot leave the answer model without evidence.
    """
    graph_scores: Dict[str, float] = {}
    graph_reasons: Dict[str, list[str]] = {}
    stale_state_sources = set()
    current_state_query = bool(re.search(r"\b(currently|current|now|latest|as of)\b", question.lower()))

    def add_sources(items, field: str, base: float, score_key: str) -> None:
        for item in items:
            score = base + float(item.get(score_key, 0.0))
            attrs = item.get("attrs", {}) if isinstance(item.get("attrs", {}), dict) else {}
            is_stale_state = current_state_query and attrs.get("state_version_status") == "superseded"
            for source_id in compact_source_ids(item.get("source_ids", []), limit=20):
                if is_stale_state:
                    stale_state_sources.add(source_id)
                    continue
                graph_scores[source_id] = graph_scores.get(source_id, 0.0) + score
                graph_reasons.setdefault(source_id, []).append(field)

    add_sources(graph_bundle.get("evidence_nodes", []), "activated_node", 2.0, "score")
    add_sources(graph_bundle.get("evidence_paths", []), "graph_path", 1.0, "confidence")
    add_sources(graph_bundle.get("temporal_timeline", []), "temporal_timeline", 2.0, "score")
    add_sources(graph_bundle.get("open_domain_clues", []), "open_domain_clue", 1.5, "weight")

    # Preserve BM25 order for ties and deduplicate repeated source IDs emitted
    # by query decomposition or open-domain expansion.
    candidate_by_id = {}
    for rank, hit in enumerate(bm25_hits):
        source_type = str(getattr(hit, "source_type", "turn"))
        if source_type != "turn":
            continue
        source_id = str(getattr(hit, "memory_id", ""))
        if source_id and source_id not in candidate_by_id:
            candidate_by_id[source_id] = (rank, hit)
        if len(candidate_by_id) >= candidate_pool:
            break

    # Graph expansion may identify a relevant linked turn whose wording does
    # not overlap the question enough to survive BM25's lexical candidate set.
    # It can be added only when the source maps to an original dialogue turn.
    for source_id in graph_scores:
        if source_id in candidate_by_id or not raw_turn_lookup or source_id not in raw_turn_lookup:
            continue
        candidate_by_id[source_id] = (
            len(bm25_hits) + len(candidate_by_id),
            SimpleNamespace(memory_id=source_id, source_type="turn", text=raw_turn_lookup[source_id]),
        )

    ranked = sorted(
        candidate_by_id.items(),
        key=lambda pair: (
            pair[0] not in graph_scores,
            -graph_scores.get(pair[0], 0.0),
            pair[1][0],
        ),
    )
    selected = []
    used_chars = 0
    for source_id, (_, hit) in ranked:
        if current_state_query and source_id in stale_state_sources and source_id not in graph_scores:
            continue
        raw_text = str(getattr(hit, "text", "")).strip()
        if not raw_text:
            continue
        if selected and used_chars + len(raw_text) > max_chars:
            continue
        if not selected and len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars]
        reasons = dedupe_preserve_order(graph_reasons.get(source_id, ["bm25_fallback"]))
        selected.append({
            "source_id": source_id,
            "bm25_rank": int(candidate_by_id[source_id][0]) + 1,
            "graph_score": round(graph_scores.get(source_id, 0.0), 4),
            "selection_reason": reasons,
            "text": raw_text,
        })
        used_chars += len(raw_text)
        if len(selected) >= max_turns:
            break

    lines = [
        "[%s][graph_guided:%s] %s" % (
            item["source_id"],
            ",".join(item["selection_reason"]),
            item["text"],
        )
        for item in selected
    ]
    return {
        "mode": "graph_guided_raw_v1_1",
        "candidate_pool_size": len(candidate_by_id),
        "candidate_pool_limit": candidate_pool,
        "graph_expanded_turn_count": sum(source_id not in {str(getattr(hit, "memory_id", "")) for hit in bm25_hits} for source_id in candidate_by_id),
        "graph_supported_candidate_count": sum(source_id in graph_scores for source_id in candidate_by_id),
        "excluded_superseded_state_candidates": sum(source_id in stale_state_sources for source_id in candidate_by_id),
        "selected_source_ids": [item["source_id"] for item in selected],
        "selected_items": selected,
        "evidence_text": "\n".join(lines),
        "char_count": sum(len(item["text"]) for item in selected),
        "max_turns": max_turns,
        "max_chars": max_chars,
    }


def annotate_state_versions(nodes: list[dict]) -> None:
    """Add session-bounded state versions for the opt-in graph-guided path.

    Historical nodes remain in the graph.  Only current-state questions use
    this annotation to avoid selecting a source superseded in the same
    entity/slot chain.
    """
    groups: Dict[tuple[str, str], list[dict]] = {}
    for node in nodes:
        if node.get("type") not in {"State", "Preference", "Goal"}:
            continue
        attrs = node.get("attrs", {})
        subject = str(attrs.get("subject") or attrs.get("speaker") or "unknown").lower()
        slot = str(attrs.get("slot") or node.get("type", "")).lower()
        groups.setdefault((subject, slot), []).append(node)

    for (subject, slot), versions in groups.items():
        versions.sort(key=lambda node: (_latest_source_session(node.get("source_ids", [])), node.get("id", "")))
        for index, node in enumerate(versions):
            attrs = node.setdefault("attrs", {})
            start = _latest_source_session(node.get("source_ids", []))
            next_start = _latest_source_session(versions[index + 1].get("source_ids", [])) if index + 1 < len(versions) else None
            attrs.update({
                "state_version_chain": "%s:%s" % (subject, slot),
                "state_version": index + 1,
                "valid_from_session": start,
                "valid_to_session": next_start - 1 if next_start is not None else None,
                "state_version_status": "active" if next_start is None else "superseded",
            })


def _latest_source_session(source_ids) -> int:
    sessions = []
    for source_id in source_ids:
        match = re.match(r"D(\d+):", str(source_id))
        if match:
            sessions.append(int(match.group(1)))
    return max(sessions) if sessions else 0


def compact_source_ids(source_ids, limit: int = 3):
    return dedupe_preserve_order(str(item) for item in source_ids if str(item))[:limit]


def adapt_long_short_bundle(bundle: Dict[str, Any], query_plan, mode: str = "adaptive") -> Dict[str, Any]:
    query_type = getattr(query_plan, "query_type", "single_hop")
    output = dict(bundle)
    nodes = list(bundle.get("evidence_nodes", []))
    paths = list(bundle.get("evidence_paths", []))
    clues = list(bundle.get("open_domain_clues", []))

    if mode == "typed":
        output, policy = apply_typed_memory_routing(output, nodes, paths, clues, query_type)
    elif mode == "selective" and query_type != "multi_hop":
        output["evidence_nodes"] = []
        output["evidence_paths"] = []
        output["open_domain_clues"] = []
        policy = "selective_non_multihop_disabled"
    elif query_type == "multi_hop":
        output["evidence_nodes"] = filter_nodes(nodes, {"Person", "Event", "State", "Preference", "Goal", "EvidenceSpan"}, limit=4)
        output["evidence_paths"] = filter_paths(paths, {"PARTICIPATES_IN", "HAS_STATE", "HAS_PREFERENCE", "HAS_GOAL", "SUPPORTS", "SUMMARIZES"}, limit=4)
        output["open_domain_clues"] = []
        policy = "multi_hop_graph_path_activation"
    elif query_type == "temporal":
        output["evidence_nodes"] = filter_nodes(nodes, {"Time", "Event", "State", "EvidenceSpan"}, limit=2)
        output["evidence_paths"] = filter_paths(paths, {"HAPPENED_AT", "VALID_DURING", "TEMPORAL_NEXT", "SUPPORTS"}, limit=2)
        output["open_domain_clues"] = []
        policy = "temporal_time_grounded_activation"
    elif query_type == "open_domain":
        output["open_domain_clues"] = filter_question_matched_clues(clues, str(bundle.get("question", "")), limit=2)
        output["evidence_nodes"] = filter_nodes(nodes, {"Person", "OpenDomainClue", "Object", "EvidenceSpan"}, limit=2)
        output["evidence_paths"] = filter_paths(paths, {"BRIDGES_TO", "MENTIONS_OBJECT", "SUPPORTS"}, limit=2)
        policy = "open_domain_clue_gated_activation"
    else:
        output["evidence_nodes"] = filter_nodes(nodes, {"EvidenceSpan", "ObservationFact"}, limit=1)
        output["evidence_paths"] = []
        output["open_domain_clues"] = []
        policy = "single_hop_minimal_fallback"

    output["adaptive_policy"] = {
        "enabled": True,
        "query_type": query_type,
        "mode": mode,
        "policy": policy,
        "input_counts": {
            "nodes": len(nodes),
            "paths": len(paths),
            "clues": len(clues),
        },
        "output_counts": {
            "nodes": len(output.get("evidence_nodes", [])),
            "paths": len(output.get("evidence_paths", [])),
            "clues": len(output.get("open_domain_clues", [])),
            "timeline": len(output.get("temporal_timeline", [])),
        },
    }
    return output


def gate_temporal_memory_by_agreement(bundle: Dict[str, Any], base_context_ids) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep temporal graph evidence only when raw retrieval independently found it.

    Graph scores are useful for recall, but temporal answers are unusually sensitive
    to a same-entity event from the wrong session.  Agreement with fact/BM25 source
    ids makes the graph an annotator of a retrieved turn instead of a competing
    temporal anchor.
    """
    base_ids = {str(item) for item in base_context_ids if str(item).startswith("D")}
    output = dict(bundle)
    item_specs = {
        "evidence_nodes": "source_ids",
        "evidence_paths": "source_ids",
        "open_domain_clues": "evidence_ids",
        "temporal_timeline": "source_ids",
    }
    kept_counts = {}
    dropped = 0
    for field, source_field in item_specs.items():
        items = list(bundle.get(field, []))
        kept = []
        for item in items:
            source_ids = {str(value) for value in item.get(source_field, [])}
            if source_ids & base_ids:
                kept.append(item)
            else:
                dropped += 1
        output[field] = kept
        kept_counts[field] = len(kept)
    audit = {
        "applied": True,
        "base_turn_count": len(base_ids),
        "kept_items": sum(kept_counts.values()),
        "dropped_items": dropped,
        "kept_by_field": kept_counts,
    }
    output["temporal_agreement_gate"] = audit
    return output, audit


def apply_typed_memory_routing(output: Dict[str, Any], nodes, paths, clues, query_type: str):
    question = str(output.get("question", ""))
    if query_type == "single_hop":
        output.pop("temporal_timeline", None)
        output["evidence_nodes"] = filter_nodes(nodes, {"EvidenceSpan", "ObservationFact"}, limit=1)
        output["evidence_paths"] = []
        output["open_domain_clues"] = []
        return output, "typed_short_term_evidence_memory"

    if query_type == "temporal":
        temporal_nodes = filter_nodes(nodes, {"Time", "Event", "State", "EvidenceSpan", "ObservationFact"}, limit=5)
        temporal_paths = filter_paths(paths, {"HAPPENED_AT", "VALID_DURING", "TEMPORAL_NEXT", "SUPPORTS"}, limit=5)
        output["evidence_nodes"] = temporal_nodes
        output["evidence_paths"] = temporal_paths
        output["temporal_timeline"] = build_temporal_timeline(temporal_nodes, temporal_paths, limit=6)
        output["open_domain_clues"] = []
        return output, "typed_temporal_timeline_memory"

    if query_type == "open_domain":
        output.pop("temporal_timeline", None)
        matched_clues = filter_question_matched_clues(clues, question, limit=1)
        output["open_domain_clues"] = matched_clues
        evidence_source_ids = {sid for clue in matched_clues for sid in clue.get("evidence_ids", [])}
        output["evidence_nodes"] = filter_nodes_by_sources(nodes, {"EvidenceSpan", "ObservationFact", "OpenDomainClue"}, evidence_source_ids, limit=1)
        output["evidence_paths"] = filter_paths(paths, {"BRIDGES_TO", "SUPPORTS"}, limit=1) if matched_clues else []
        return output, "typed_short_term_clue_memory"

    if query_type == "multi_hop":
        output.pop("temporal_timeline", None)
        output["evidence_nodes"] = filter_nodes(nodes, {"Person", "Event", "State", "Preference", "Goal", "EvidenceSpan"}, limit=3)
        output["evidence_paths"] = filter_paths(paths, {"PARTICIPATES_IN", "HAS_STATE", "HAS_PREFERENCE", "HAS_GOAL", "SUPPORTS", "SUMMARIZES"}, limit=3)
        output["open_domain_clues"] = []
        return output, "typed_long_term_graph_plus_working_memory"

    output.pop("temporal_timeline", None)
    output["evidence_nodes"] = []
    output["evidence_paths"] = []
    output["open_domain_clues"] = []
    return output, "typed_unknown_disabled"


def filter_nodes(nodes, allowed_types, limit: int):
    filtered = [node for node in nodes if node.get("node_type") in allowed_types]
    return sorted(filtered, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:limit]


def filter_paths(paths, allowed_edge_types, limit: int):
    filtered = []
    for path in paths:
        labels = [str(item) for item in path.get("labels", [])]
        edge_type = labels[1] if len(labels) >= 3 else ""
        if edge_type in allowed_edge_types:
            filtered.append(path)
    return sorted(filtered, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)[:limit]


def build_temporal_timeline(nodes, paths, limit: int = 6):
    entries = []
    for node in nodes:
        source_ids = compact_source_ids(node.get("source_ids", []), limit=2)
        session = earliest_session_id(source_ids)
        if session is None:
            continue
        node_type = node.get("node_type", "Memory")
        priority = 2.0 if node_type in {"Time", "Event", "State"} else 1.0
        entries.append({
            "session": session,
            "source_ids": source_ids,
            "score": float(node.get("score", 0.0)),
            "rank": priority + float(node.get("score", 0.0)),
            "text": "%s: %s" % (node_type, node.get("label", "")),
        })
    for path in paths:
        source_ids = compact_source_ids(path.get("source_ids", []), limit=2)
        session = earliest_session_id(source_ids)
        if session is None:
            continue
        labels = [str(item) for item in path.get("labels", [])]
        edge_type = labels[1] if len(labels) >= 3 else ""
        priority = 3.0 if edge_type in {"HAPPENED_AT", "VALID_DURING", "TEMPORAL_NEXT"} else 0.5
        if edge_type == "SUPPORTS" and len(labels) >= 3 and len(tokenize_simple(labels[2])) <= 2:
            priority = 0.1
        entries.append({
            "session": session,
            "source_ids": source_ids,
            "score": float(path.get("confidence", 0.0)),
            "rank": priority + float(path.get("confidence", 0.0)),
            "text": " -> ".join(labels),
        })
    best_by_source = {}
    for item in entries:
        key = tuple(item.get("source_ids", []))
        if not key:
            continue
        if key not in best_by_source or float(item.get("rank", 0.0)) > float(best_by_source[key].get("rank", 0.0)):
            best_by_source[key] = item
    output = sorted(best_by_source.values(), key=lambda item: (int(item["session"]), -float(item.get("rank", 0.0))))
    for item in output:
        item.pop("rank", None)
    return output[:limit]


def earliest_session_id(source_ids) -> int | None:
    sessions = []
    for source_id in source_ids:
        parsed = session_id_from_source_id(str(source_id))
        if parsed is not None:
            sessions.append(parsed)
    return min(sessions) if sessions else None


def session_id_from_source_id(source_id: str) -> int | None:
    if source_id.startswith("D"):
        head = source_id[1:].split(":", 1)[0]
    elif source_id.startswith("S"):
        head = source_id[1:]
    else:
        return None
    return int(head) if head.isdigit() else None


def filter_nodes_by_sources(nodes, allowed_types, source_ids, limit: int):
    if not source_ids:
        return filter_nodes(nodes, allowed_types, limit)
    filtered = []
    for node in nodes:
        if node.get("node_type") not in allowed_types:
            continue
        if set(str(item) for item in node.get("source_ids", [])) & set(str(item) for item in source_ids):
            filtered.append(node)
    return sorted(filtered, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:limit]


def filter_question_matched_clues(clues, question: str, limit: int):
    allowed = bridge_types_for_question(question)
    filtered = [
        clue
        for clue in clues
        if not allowed or clue.get("bridge_type") in allowed or clue_has_question_overlap(clue, question)
    ]
    return sorted(filtered, key=lambda item: float(item.get("weight", 0.0)), reverse=True)[:limit]


def bridge_types_for_question(question: str):
    lowered = question.lower()
    if any(token in lowered for token in ["console", "game", "card game", "board game"]):
        return {"entertainment_clue_to_named_item"}
    if any(token in lowered for token in ["state", "country", "holiday", "beach", "mountain", "national park", "where"]):
        return {"location_clue_to_place"}
    if any(token in lowered for token in ["degree", "field", "career", "job", "work"]):
        return {"profile_clue_to_career_or_field", "work_clue_to_industry_or_organization"}
    if any(token in lowered for token in ["book", "bookshelf", "read", "author"]):
        return {"reading_clue_to_likely_book_or_author"}
    if any(token in lowered for token in ["condition", "technique", "method", "kind of"]):
        return {"descriptive_clue_to_concept"}
    return set()


def clue_has_question_overlap(clue: Dict[str, Any], question: str) -> bool:
    question_tokens = set(tokenize_simple(question))
    clue_tokens = set(tokenize_simple(str(clue.get("clue", ""))))
    return bool(question_tokens & clue_tokens)


def tokenize_simple(text: str):
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(token) > 2]


def dedupe_preserve_order(items):
    output = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def qa_memory_dir_name(qa: Dict[str, Any]) -> str:
    text = str(qa.get("question", ""))[:80].lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in text).strip("-")
    category = str(qa.get("category", "unknown"))
    return "qa_%s_%s" % (category, safe or "question")


def build_visible_qa(qa: Dict[str, Any], official_category5_options: bool) -> Dict[str, Any]:
    visible = {
        "question": str(qa.get("question", "")),
        "category": str(qa.get("category", "unknown")),
        "generation_question": str(qa.get("question", "")),
    }
    if official_category5_options and str(qa.get("category")) == "5":
        adversarial_answer = qa.get("adversarial_answer")
        if adversarial_answer:
            visible["category5_options"] = {
                "a": "Not mentioned in the conversation",
                "b": str(adversarial_answer),
            }
            visible["generation_question"] = (
                visible["question"]
                + " Select the correct answer: (a) %s (b) %s."
                % (visible["category5_options"]["a"], visible["category5_options"]["b"])
            )
    return visible


def should_use_open_domain_cbi(query_plan) -> bool:
    query_type = getattr(query_plan, "query_type", query_plan)
    # Alternative routes are retrieval fallbacks, not authorization to run an
    # inference-style answer rewriter.  Rewriting multi-hop or single-hop
    # answers here can replace answer-critical facts with generic paraphrases.
    return query_type == "open_domain"


def should_use_multihop_rerank(query_type: str) -> bool:
    return query_type == "multi_hop"


def should_expand_open_domain(mode: str, base_relevance: float, threshold: float) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return True
    return base_relevance < threshold


def should_use_dual_realization(query_plan, current_answer: str) -> bool:
    if query_plan.query_type == "open_domain":
        return False
    if "open_domain" not in getattr(query_plan, "alternative_types", []):
        return False
    if "no information available" in str(current_answer).lower():
        return True
    return float(getattr(query_plan, "confidence", 1.0)) < 0.64


def run_dual_realization(
    qa: Dict[str, Any],
    current_answer: str,
    generation_question: str,
    evidence_text: str,
    generator,
    reusable_item: Dict[str, Any],
    prediction_key: str,
    dry_run: bool,
):
    from higraphmem2.answer.evidence_verifier import support_score
    from higraphmem2.answer.open_domain_cbi import build_cbi_prompt, normalize_open_domain_answer, select_cbi_answer

    if dry_run:
        return None
    stored = reusable_item.get(prediction_key + "_dual_realization") if reusable_item else None
    if stored and stored.get("candidate"):
        candidate = stored.get("candidate")
        reused = True
    elif reusable_item:
        return None
    else:
        try:
            candidate = generator.generate(build_cbi_prompt(generation_question, evidence_text.splitlines()))
        except Exception as exc:
            return (
                current_answer,
                {
                    "accepted": False,
                    "source": "dual_api_error",
                    "candidate": "",
                    "error": str(exc),
                },
            )
        reused = False

    cbi = select_cbi_answer(qa["question"], current_answer, candidate, evidence_text.splitlines())
    current_clean = normalize_open_domain_answer(qa["question"], current_answer)
    candidate_clean = cbi.answer
    current_support = support_score(qa["question"], current_clean, evidence_text)
    candidate_support = support_score(qa["question"], candidate_clean, evidence_text)
    accepted = False
    source = "dual_kept_current"
    answer = current_clean
    if cbi.accepted and _dual_candidate_better(current_clean, candidate_clean, current_support, candidate_support):
        accepted = True
        source = "dual_open_domain_selected"
        answer = candidate_clean
    payload = {
        "accepted": accepted,
        "source": source,
        "candidate": candidate,
        "normalized_candidate": candidate_clean,
        "current_support": round(current_support, 3),
        "candidate_support": round(candidate_support, 3),
        "reused": reused,
    }
    return answer, payload


def _dual_candidate_better(current_answer: str, candidate: str, current_support: float, candidate_support: float) -> bool:
    current_no_info = "no information available" in current_answer.lower()
    candidate_no_info = "no information available" in candidate.lower()
    if candidate_no_info:
        return False
    if current_no_info:
        return True
    if candidate_support >= current_support + 0.10 and len(candidate.split()) <= 10:
        return True
    return False


def load_provider_config(source_root: Path, path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-3.5-turbo",
        }
    config_path = source_root / path
    local_path = config_path.with_name("provider.local.yaml")
    if local_path.exists():
        config_path = local_path
    if not config_path.exists():
        return {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-3.5-turbo",
        }
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload.get("provider", payload)


def resolve_answer_prompt_profile(requested_profile: str, chat_model: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    return "qwen_strict" if "qwen" in (chat_model or "").lower() else "generic"


def build_prompt(
    question: str,
    evidence_text: str,
    answer_mode: str = "span",
    profile: str = "generic",
) -> str:
    mode_instruction = {
        "date": "Return only the shortest date or duration phrase.",
        "span": "Return only the exact short answer phrase.",
        "list": "Return only a concise comma-separated list.",
        "inference": "Return only the shortest evidence-grounded inference phrase; include yes/no only when the question asks yes/no.",
    }.get(answer_mode, "Return only the short answer.")
    target_instruction = answer_format_instruction(question)
    strict_instruction = ""
    if profile == "qwen_strict":
        strict_instruction = (
            " Treat this as an evidence extraction task, not a conversation. Find the evidence line that directly answers the question "
            "and copy only its answer value. Do not answer with a word copied from the question. "
            "For a date question, return the date or duration stated in the evidence; for a person, place, or object question, return that entity. "
            "Do not default to absence phrases such as No, No evidence, or No information available. "
            "Output No only when a yes/no question is explicitly negated by the evidence; otherwise use No information available only when no supplied evidence states an answer."
        )
    return (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n\n"
        "Use only the memory evidence. %s %s Do not write a full sentence or explanation. "
        "For multiple answers, return a comma-separated list with no duplicates.%s "
        "If the evidence is insufficient, answer: No information available."
    ) % (evidence_text or "No retrieved evidence.", question, mode_instruction, target_instruction, strict_instruction)


def build_generation_question(qa: Dict[str, Any], official_category5_options: bool) -> str:
    question = qa["question"]
    if not official_category5_options or qa.get("category") != 5:
        return question
    adversarial_answer = qa.get("answer", qa.get("adversarial_answer"))
    if not adversarial_answer:
        return question
    options = {
        "a": "Not mentioned in the conversation",
        "b": str(adversarial_answer),
    }
    qa["answer"] = str(adversarial_answer)
    qa["category5_options"] = options
    return question + " Select the correct answer: (a) %s (b) %s." % (options["a"], options["b"])


def map_category5_option(prediction: str, qa: Dict[str, Any]) -> str:
    options = qa.get("category5_options")
    if not options:
        return prediction
    lowered = prediction.strip().lower()
    if lowered in ["a", "(a)", "option a", "answer a"]:
        return options["a"]
    if lowered in ["b", "(b)", "option b", "answer b"]:
        return options["b"]
    if "(a)" in lowered and "(b)" not in lowered:
        return options["a"]
    if "(b)" in lowered and "(a)" not in lowered:
        return options["b"]
    return prediction


def write_output(samples, prediction_key: str, output_path: str, score_qa_fn) -> Dict[str, Any]:
    category_counts = Counter()
    category_scores = Counter()
    total_count = 0
    total_score = 0.0
    unlabeled_count = 0
    token_events = []
    token_qas = 0

    for sample in samples:
        for qa in sample.get("qa", []):
            if "answer" not in qa:
                unlabeled_count += 1
                continue
            score_key = prediction_key + "_f1"
            if score_key not in qa:
                qa[score_key] = round(score_qa_fn(qa.get(prediction_key, ""), qa), 3)
            category = str(qa.get("category", "unknown"))
            category_counts[category] += 1
            category_scores[category] += qa[score_key]
            total_count += 1
            total_score += qa[score_key]
            usage = qa.get(prediction_key + "_token_usage")
            if usage:
                token_qas += 1
                token_events.extend(usage.get("events", []))

    stats = {
        "prediction_key": prediction_key,
        "note": "Debug stats only. Use eval/official_locomo_eval.py for reported LoCoMo scores.",
        "num_questions": total_count,
        "num_labeled_questions": total_count,
        "num_unlabeled_questions": unlabeled_count,
        "overall_f1": round(total_score / total_count, 4) if total_count else 0.0,
        "category_counts": dict(category_counts),
        "category_f1": {
            category: round(category_scores[category] / count, 4)
            for category, count in category_counts.items()
        },
        "token_usage": summarize_token_events(token_events, token_qas),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = output.with_name(output.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def summarize_qa_token_usage(events):
    """Store raw provider usage per QA so interrupted shards remain resumable."""
    return {"events": events, "summary": summarize_token_events(events, 1)}


def summarize_token_events(events, qa_count: int):
    phases = {}
    for event in events:
        phase = event.get("phase", "answer")
        bucket = phases.setdefault(phase, {
            "calls": 0, "calls_with_usage": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "latency_seconds": 0.0,
        })
        bucket["calls"] += 1
        bucket["latency_seconds"] += float(event.get("latency_seconds") or 0.0)
        if event.get("usage_available"):
            bucket["calls_with_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                bucket[key] += int(event.get(key) or 0)
    def combine(phase_names):
        return {
            key: sum(phases.get(phase, {}).get(key, 0) for phase in phase_names)
            for key in ("calls", "calls_with_usage", "prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")
        }
    online = combine(["answer"])
    end_to_end = combine(list(phases))
    denominator = qa_count or 1
    return {
        "qa_with_recorded_calls": qa_count,
        "phases": phases,
        "online": online,
        "end_to_end": end_to_end,
        "mean_online_total_tokens_per_qa": round(online["total_tokens"] / denominator, 3),
        "mean_end_to_end_total_tokens_per_qa": round(end_to_end["total_tokens"] / denominator, 3),
        "mean_online_api_latency_seconds_per_call": round(online["latency_seconds"] / online["calls"], 3) if online["calls"] else None,
        "mean_online_api_latency_seconds_per_qa": round(online["latency_seconds"] / denominator, 3),
        "online_completion_tokens_per_second": round(online["completion_tokens"] / online["latency_seconds"], 3) if online["latency_seconds"] else None,
        "online_total_tokens_per_second": round(online["total_tokens"] / online["latency_seconds"], 3) if online["latency_seconds"] else None,
        "usage_source": "provider_response_usage; latency is successful provider API wall time only",
    }


if __name__ == "__main__":
    main()
