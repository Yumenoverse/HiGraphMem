#!/usr/bin/env python3
"""Unit checks for the opt-in graph-guided raw-evidence variant."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graph_guided_runner", ROOT / "eval" / "run_locomo_v2.py")
RUNNER = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(RUNNER)


class GraphGuidedRawEvidenceTest(unittest.TestCase):
    def test_graph_supported_turn_precedes_higher_ranked_bm25_turn(self):
        hits = [
            SimpleNamespace(memory_id="D1:1", text="Alice enjoys jazz."),
            SimpleNamespace(memory_id="D4:2", text="Alice moved to Paris last June."),
            SimpleNamespace(memory_id="D5:1", text="Alice changed jobs."),
        ]
        bundle = {
            "evidence_nodes": [{"source_ids": ["D4:2"], "score": 1.2}],
            "evidence_paths": [{"source_ids": ["D4:2"], "confidence": 0.8}],
        }
        result = RUNNER.select_graph_guided_raw_evidence(hits, bundle, max_turns=2, max_chars=500)
        self.assertEqual(result["selected_source_ids"], ["D4:2", "D1:1"])
        self.assertEqual(result["selected_items"][0]["selection_reason"], ["activated_node", "graph_path"])
        self.assertIn("Alice moved to Paris last June.", result["evidence_text"])

    def test_budget_limits_raw_turns_and_characters(self):
        hits = [
            SimpleNamespace(memory_id="D1:1", text="A" * 80),
            SimpleNamespace(memory_id="D1:2", text="B" * 80),
        ]
        result = RUNNER.select_graph_guided_raw_evidence(hits, {}, max_turns=6, max_chars=100)
        self.assertEqual(result["selected_source_ids"], ["D1:1"])
        self.assertLessEqual(result["char_count"], 100)

    def test_session_summaries_do_not_consume_raw_turn_budget(self):
        hits = [
            SimpleNamespace(memory_id="S1", source_type="session_summary", text="summary"),
            SimpleNamespace(memory_id="OBS1", source_type="observation", text="observation"),
            SimpleNamespace(memory_id="D1:1", source_type="turn", text="raw dialogue turn"),
        ]
        result = RUNNER.select_graph_guided_raw_evidence(hits, {}, max_turns=6)
        self.assertEqual(result["selected_source_ids"], ["D1:1"])

    def test_graph_can_expand_to_a_raw_turn_outside_bm25_candidates(self):
        hits = [SimpleNamespace(memory_id="D1:1", source_type="turn", text="Alice likes jazz.")]
        bundle = {"evidence_nodes": [{"source_ids": ["D4:2"], "score": 2.0}]}
        result = RUNNER.select_graph_guided_raw_evidence(
            hits, bundle, raw_turn_lookup={"D4:2": "Alice moved to Paris last June."}, max_turns=1
        )
        self.assertEqual(result["selected_source_ids"], ["D4:2"])
        self.assertEqual(result["graph_expanded_turn_count"], 1)

    def test_current_state_query_excludes_superseded_state_source(self):
        nodes = [
            {"id": "old", "type": "State", "source_ids": ["D2:1"], "attrs": {"subject": "Alice", "slot": "career_status"}},
            {"id": "new", "type": "State", "source_ids": ["D5:1"], "attrs": {"subject": "Alice", "slot": "career_status"}},
        ]
        RUNNER.annotate_state_versions(nodes)
        hits = [
            SimpleNamespace(memory_id="D2:1", text="Alice was a designer."),
            SimpleNamespace(memory_id="D5:1", text="Alice is now a manager."),
        ]
        bundle = {"evidence_nodes": [
            {"source_ids": ["D2:1"], "score": 2.0, "attrs": nodes[0]["attrs"]},
            {"source_ids": ["D5:1"], "score": 1.0, "attrs": nodes[1]["attrs"]},
        ]}
        result = RUNNER.select_graph_guided_raw_evidence(hits, bundle, "What is Alice currently doing?", max_turns=2)
        self.assertEqual(result["selected_source_ids"], ["D5:1"])
        self.assertEqual(nodes[0]["attrs"]["valid_to_session"], 4)


if __name__ == "__main__":
    unittest.main()
