#!/usr/bin/env python3
"""Verify answer support against a memory evidence bundle without gold labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-bundle", required=True)
    parser.add_argument("--candidate-answer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = json.loads(Path(args.evidence_bundle).read_text(encoding="utf-8"))
    result = verify(args.candidate_answer, bundle)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def verify(answer: str, bundle: dict) -> dict:
    normalized = _normalize(answer)
    evidence_text = " ".join(
        item.get("label", "") for item in bundle.get("evidence_nodes", [])
    ) + " " + " ".join(
        " ".join(item.get("labels", [])) for item in bundle.get("evidence_paths", [])
    ) + " " + " ".join(
        item.get("clue", "") for item in bundle.get("open_domain_clues", [])
    )
    answer_tokens = set(_tokens(normalized))
    evidence_tokens = set(_tokens(evidence_text))
    overlap = len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1)
    licensed_bridge = bool(bundle.get("open_domain_clues")) and bundle.get("intent") == "open_domain"
    accepted = overlap >= 0.5 or (licensed_bridge and overlap > 0)
    if normalized.lower() == "no information available":
        accepted = True
    return {
        "candidate_answer": answer,
        "normalized_answer": normalized,
        "accepted": accepted,
        "support_score": round(overlap + (0.2 if licensed_bridge else 0.0), 3),
        "failure_reason": "" if accepted else "answer_not_supported_by_memory_bundle",
        "audit": {
            "gold_answer_visible": False,
            "gold_evidence_visible": False,
            "test_error_visible": False,
        },
    }


def _normalize(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer or "No information available"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


if __name__ == "__main__":
    main()
