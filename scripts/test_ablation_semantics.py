#!/usr/bin/env python3
"""Regression tests for the reported Update and Gated Activation ablations."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCRIPT = ROOT / "skills/memory_construction_subskills/build_typed_temporal_graph/scripts/build_graph.py"
STM_SCRIPT = ROOT / "skills/memory_construction_subskills/induce_short_term_working_memory/scripts/induce_memory.py"
RETRIEVE_SCRIPT = ROOT / "skills/memory_construction_subskills/retrieve_long_short_memory/scripts/retrieve.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


GRAPH = load_module(GRAPH_SCRIPT, "test_ablation_graph")
STM = load_module(STM_SCRIPT, "test_ablation_stm")
RETRIEVE = load_module(RETRIEVE_SCRIPT, "test_ablation_retrieve")


SAMPLE = {
    "sample_id": "ablation-test",
    "conversation": {
        "session_1_date_time": "1 January 2024",
        "session_1": [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I moved to Paris yesterday."},
            {"dia_id": "D1:2", "speaker": "Bob", "text": "I love jazz music."},
        ],
        "session_2_date_time": "2 January 2024",
        "session_2": [
            {"dia_id": "D2:1", "speaker": "Alice", "text": "I started a design job today."},
        ],
    },
    "observation": {},
}


class AblationSemanticsTest(unittest.TestCase):
    def test_disable_update_freezes_identical_initial_graph(self):
        import copy
        prefix = copy.deepcopy(SAMPLE)
        del prefix["conversation"]["session_2"]
        del prefix["conversation"]["session_2_date_time"]
        initial = GRAPH.build_graph(prefix)
        frozen = GRAPH.build_graph(SAMPLE, enable_update=False)
        self.assertEqual(initial["nodes"], frozen["nodes"])
        self.assertEqual(initial["edges"], frozen["edges"])
        self.assertEqual(frozen["metadata"]["ingested_sessions"], [1])
        self.assertEqual(frozen["metadata"]["num_updates"], 0)
        self.assertFalse(frozen["metadata"]["consolidation"]["update_enabled"])

    def test_updates_use_previous_graph_without_future_knowledge(self):
        from unittest.mock import patch
        import copy
        snapshots = []
        original = GRAPH._apply_streaming_update_step

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            snapshots.append(copy.deepcopy((args[0], args[1])))
            return result

        with patch.object(GRAPH, "_apply_streaming_update_step", side_effect=capture):
            full = GRAPH.build_graph(SAMPLE)
        frozen = GRAPH.build_graph(SAMPLE, enable_update=False)
        self.assertEqual(sorted(snapshots[0][0].values(), key=lambda n: n["id"]), frozen["nodes"])
        self.assertNotIn("evidence:D2:1", snapshots[0][0])
        self.assertIn("evidence:D2:1", snapshots[1][0])
        old = snapshots[0][0]["evidence:D1:1"]["weight"]
        self.assertAlmostEqual(snapshots[1][0]["evidence:D1:1"]["weight"], round(old * 0.92, 4))
        trace = full["metadata"]["consolidation"]["trace"]
        self.assertEqual([s["phase"] for s in trace["steps"]], ["initialize", "update"])
        self.assertEqual(full["metadata"]["num_updates"], 1)

    def test_sessions_are_sorted_numerically_without_99_limit(self):
        sample = {"conversation": {
            f"session_{i}": [{"dia_id": f"D{i}:1", "speaker": "Alice", "text": "Hello."}]
            for i in (101, 10, 2)
        }}
        graph = GRAPH.build_graph(sample)
        self.assertEqual(graph["metadata"]["ingested_sessions"], [2, 10, 101])
        self.assertEqual(graph["metadata"]["num_updates"], 2)
        frozen = GRAPH.build_graph(sample, enable_update=False)
        self.assertEqual(frozen["metadata"]["ingested_sessions"], [2])

    def test_empty_conversation_has_no_updates(self):
        for enabled in (True, False):
            graph = GRAPH.build_graph({}, enable_update=enabled)
            self.assertEqual(graph["nodes"], [])
            self.assertEqual(graph["metadata"]["num_updates"], 0)

    def test_full_update_still_writes_consolidation_weights(self):
        graph = GRAPH.build_graph(SAMPLE, enable_update=True)
        self.assertTrue(graph["metadata"]["consolidation"]["update_enabled"])
        self.assertTrue(all("weight" in item for item in graph["nodes"] + graph["edges"]))

    def test_disable_gated_activation_keeps_only_static_bounded_buffer(self):
        graph = GRAPH.build_graph(SAMPLE)
        memory = STM.induce_working_memory(
            graph["metadata"],
            graph["nodes"],
            graph["edges"],
            "When did Alice move to Paris?",
            enable_gated_activation=False,
        )

        self.assertFalse(memory["trigger_model"]["gated_activation_enabled"])
        self.assertEqual(memory["activation_model"]["propagation_steps"], 0)
        self.assertEqual(memory["activated_nodes"], [])
        self.assertGreater(len(memory["static_buffer_nodes"]), 0)
        self.assertLessEqual(
            len(memory["static_buffer_nodes"]),
            memory["trigger_model"]["capacity_limit"],
        )

        bundle = RETRIEVE.retrieve(
            "When did Alice move to Paris?",
            graph["nodes"],
            graph["edges"],
            memory,
            top_k=10,
        )
        static_ids = {item["node_id"] for item in memory["static_buffer_nodes"]}
        retrieved_ids = {item["node_id"] for item in bundle["evidence_nodes"]}
        self.assertFalse(bundle["gated_activation_enabled"])
        self.assertTrue(retrieved_ids <= static_ids)

    def test_enabled_gated_activation_remains_question_conditioned(self):
        graph = GRAPH.build_graph(SAMPLE)
        memory = STM.induce_working_memory(
            graph["metadata"],
            graph["nodes"],
            graph["edges"],
            "When did Alice move to Paris?",
        )
        self.assertTrue(memory["trigger_model"]["gated_activation_enabled"])
        self.assertEqual(memory["activation_model"]["propagation_steps"], 2)
        self.assertGreater(len(memory["activated_nodes"]), 0)
        self.assertEqual(memory["static_buffer_nodes"], [])

    def test_twmem_ablation_keeps_equal_budget_global_graph_retrieval(self):
        graph = GRAPH.build_graph(SAMPLE)
        memory = STM.induce_global_graph_memory(graph["metadata"], "When did Alice move to Paris?")
        bundle = RETRIEVE.retrieve("When did Alice move to Paris?", graph["nodes"], graph["edges"], memory, top_k=10)
        self.assertEqual(memory["memory_type"], "global_graph_retrieval_baseline")
        self.assertEqual(bundle["selection_mode"], "full_graph_lexical")
        self.assertIn("evidence:D1:1", {item["node_id"] for item in bundle["evidence_nodes"]})


if __name__ == "__main__":
    unittest.main()
