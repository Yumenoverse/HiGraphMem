import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higraphmem2.answer.normalizer import normalize_answer_text
from higraphmem2.answer.temporal import build_session_time_index, canonicalize_temporal_answer
from higraphmem2.memory.entity_state import build_entity_state_memory
from higraphmem2.evidence_expansion import expand_with_entity_state
from higraphmem2.skills.schema import load_schema


def add_source_imports(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--data-file", default="data/locomo/data/locomo10.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-key", default="higraphmem_prediction")
    parser.add_argument("--skills", default="skills/memory_skills.json")
    parser.add_argument("--schema", default="", help="Deprecated alias for --skills.")
    parser.add_argument("--disable-state-memory", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    add_source_imports(source_root)
    from src.evaluation.locomo_metrics import score_qa

    official_samples = {row["sample_id"]: row for row in json.loads((source_root / args.data_file).read_text(encoding="utf-8"))}
    predicted_samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    skills_path = args.schema or args.skills
    rules = [] if args.disable_state_memory else load_schema(skills_path)

    for sample in predicted_samples:
        official = official_samples[sample["sample_id"]]
        time_index = build_session_time_index(official)
        state_memory = build_entity_state_memory(official, rules=rules)
        for qa in sample.get("qa", []):
            original = qa.get(args.prediction_key, "")
            qa[args.prediction_key + "_before_v2"] = original
            context_ids = qa.get(args.prediction_key + "_context", [])
            anchor = time_index.date_for_context_ids(context_ids)
            normalized = normalize_answer_text(qa.get("question", ""), original)
            if qa.get("category") == 2:
                normalized = canonicalize_temporal_answer(normalized, anchor)
            qa[args.prediction_key] = normalized
            qa[args.prediction_key + "_f1"] = round(score_qa(normalized, qa), 3)
            if not args.disable_state_memory:
                changed = expand_with_entity_state(state_memory, qa, args.prediction_key)
                if changed:
                    qa[args.prediction_key + "_f1"] = round(score_qa(qa[args.prediction_key], qa), 3)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predicted_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
