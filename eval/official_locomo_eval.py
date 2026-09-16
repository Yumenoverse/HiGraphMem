import argparse
from collections import defaultdict
import json
import re
import sys
from pathlib import Path
import string
import types
from typing import Dict, Iterable, List


CATEGORY_NAMES = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
}
CATEGORY5_NAME = "Adversarial"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="GPT-4o-mini", help="Display name only; this script does not call a model.")
    parser.add_argument("--model-name", default="", help="Display name alias for --model.")
    parser.add_argument("--method", default="HiGraphMem-Retrieval")
    parser.add_argument("--prediction-key", default="higraphmem_prediction")
    parser.add_argument("--include-category-5", action="store_true")
    parser.add_argument("--include-local-bleu1", action="store_true")
    parser.add_argument(
        "--bleu1-mode",
        default="precision",
        choices=["precision", "sentence_bleu"],
        help="precision is unigram precision without brevity penalty; sentence_bleu adds BLEU brevity penalty.",
    )
    args = parser.parse_args()
    model_name = args.model_name or args.model

    source_root = Path(args.source_root).resolve()
    locomo_root = source_root / "data" / "locomo"
    sys.path.insert(0, str(locomo_root))
    install_optional_official_eval_stubs()

    try:
        from task_eval.evaluation import eval_question_answering
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency required by LoCoMo official evaluator: %s. "
            "Install HiGraphMem requirements, then rerun this command." % exc.name
        ) from exc

    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    labeled_samples, labeled_qas, skipped = collect_labeled(samples, args.prediction_key, args.include_category_5)
    exact_matches, _lengths, recall = eval_question_answering(labeled_qas, args.prediction_key)

    idx = 0
    for sample in labeled_samples:
        for qa in sample.get("qa", []):
            qa[args.prediction_key + "_official_f1"] = round(exact_matches[idx], 3)
            if recall:
                qa[args.prediction_key + "_official_recall"] = round(recall[idx], 3)
            idx += 1

    summary = summarize(labeled_qas, exact_matches, args.include_category_5, args.include_local_bleu1, args.prediction_key, args.bleu1_mode)
    result = {
        "model": model_name,
        "method": args.method,
        "prediction_key": args.prediction_key,
        "protocol": "GAM-main-categories-1-4" if not args.include_category_5 else "LoCoMo-categories-1-5-with-adversarial",
        "metric_sources": {
            "f1": "LoCoMo official task_eval/evaluation.py::eval_question_answering",
            "bleu1": ("local " + args.bleu1_mode) if args.include_local_bleu1 else "not_computed",
        },
        "official_evaluator": str(locomo_root / "task_eval" / "evaluation.py"),
        "input": args.input,
        "num_labeled_questions": len(labeled_qas),
        "num_unlabeled_questions": skipped["unlabeled"],
        "num_excluded_category5_questions": skipped["category5_excluded"],
        "summary": summary,
        "markdown_table": markdown_table(model_name, args.method, summary),
    }

    output = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + "_official_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    labeled_output = output.with_name(output.stem + "_labeled_predictions.json")
    labeled_output.write_text(json.dumps(labeled_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(result["markdown_table"])
    print()
    count_names = ["Multi-Hop", "Temporal", "Open-Domain", "Single-Hop"]
    if CATEGORY5_NAME in summary:
        count_names.append(CATEGORY5_NAME)
    count_names.extend(["Avg", "Weighted Avg"])
    print("Counts:", {name: summary[name]["count"] for name in count_names})
    print("Official eval JSON:", output)
    print("Labeled predictions:", labeled_output)


def collect_labeled(samples: List[dict], prediction_key: str, include_category_5: bool):
    labeled_samples = []
    labeled_qas = []
    skipped = {"unlabeled": 0, "category5_excluded": 0}
    for sample in samples:
        out_sample = {"sample_id": sample.get("sample_id"), "qa": []}
        for qa in sample.get("qa", []):
            if qa.get("category") == 5 and not include_category_5:
                skipped["category5_excluded"] += 1
                continue
            if "answer" not in qa and qa.get("category") == 5 and "adversarial_answer" in qa:
                qa = qa.copy()
                qa["answer"] = qa["adversarial_answer"]
            if "answer" not in qa:
                skipped["unlabeled"] += 1
                continue
            if prediction_key not in qa:
                raise KeyError("Missing prediction key %s for question: %s" % (prediction_key, qa.get("question", "")))
            out_qa = qa.copy()
            out_sample["qa"].append(out_qa)
            labeled_qas.append(out_qa)
        if out_sample["qa"]:
            labeled_samples.append(out_sample)
    return labeled_samples, labeled_qas, skipped


def summarize(
    qas: Iterable[dict],
    scores: List[float],
    include_category_5: bool,
    include_local_bleu1: bool,
    prediction_key: str,
    bleu1_mode: str,
) -> Dict[str, Dict[str, float]]:
    category_names = dict(CATEGORY_NAMES)
    if include_category_5:
        category_names[5] = CATEGORY5_NAME
    counts = defaultdict(int)
    score_sums = defaultdict(float)
    bleu_sums = defaultdict(float)
    for qa, score in zip(qas, scores):
        category = int(qa.get("category"))
        if category not in category_names:
            continue
        counts[category] += 1
        score_sums[category] += score
        if include_local_bleu1:
            bleu_sums[category] += bleu1(qa.get(prediction_key, ""), qa.get("answer", ""), mode=bleu1_mode)

    rows: Dict[str, Dict[str, float]] = {}
    for category, name in category_names.items():
        count = counts[category]
        rows[name] = {
            "count": count,
            "f1": 100 * score_sums[category] / count if count else None,
            "bleu1": 100 * bleu_sums[category] / count if include_local_bleu1 and count else None,
        }
    populated_rows = [rows[name] for name in category_names.values() if rows[name]["count"]]
    total_count = sum(row["count"] for row in populated_rows)
    total_score = sum(score_sums.values())
    total_bleu = sum(bleu_sums.values())
    rows["Avg"] = {
        "count": total_count,
        "f1": sum(row["f1"] for row in populated_rows if row["f1"] is not None) / len(populated_rows) if populated_rows else None,
        "bleu1": (
            sum(row["bleu1"] for row in populated_rows if row["bleu1"] is not None) / len(populated_rows)
            if include_local_bleu1 and populated_rows
            else None
        ),
    }
    rows["Weighted Avg"] = {
        "count": total_count,
        "f1": 100 * total_score / total_count if total_count else None,
        "bleu1": 100 * total_bleu / total_count if include_local_bleu1 and total_count else None,
    }
    return rows


def markdown_table(model: str, method: str, rows: Dict[str, Dict[str, float]]) -> str:
    columns = ["Multi-Hop", "Temporal", "Open-Domain", "Single-Hop"]
    if CATEGORY5_NAME in rows:
        columns.append(CATEGORY5_NAME)
    columns.append("Avg")
    columns.append("Weighted Avg")

    header = ["Model", "Method"]
    values = [model, method]
    for column in columns:
        header.extend([column + " F1", column + " BLEU-1"])
        values.extend([fmt(rows[column]["f1"]), fmt(rows[column]["bleu1"])])
    align = ["---", "---:"] + ["---:"] * (len(header) - 2)
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(align) + " |",
            "| " + " | ".join(values) + " |",
        ]
    )


def fmt(value: float) -> str:
    if value is None:
        return "-"
    return "%.2f" % value


def normalize_for_bleu(text: object) -> List[str]:
    text = str(text).lower().replace(",", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return text.split()


def bleu1(prediction: object, answer: object, mode: str = "precision") -> float:
    pred_tokens = normalize_for_bleu(prediction)
    answer_tokens = normalize_for_bleu(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    pred_counts = defaultdict(int)
    answer_counts = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in answer_tokens:
        answer_counts[token] += 1
    overlap = sum(min(pred_counts[token], answer_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    if mode == "sentence_bleu":
        brevity_penalty = 1.0 if len(pred_tokens) > len(answer_tokens) else pow(2.718281828459045, 1.0 - (len(answer_tokens) / len(pred_tokens)))
        return brevity_penalty * precision
    return precision


def install_optional_official_eval_stubs() -> None:
    if "bert_score" in sys.modules:
        return
    module = types.ModuleType("bert_score")

    def _unused_score(*_args, **_kwargs):
        raise RuntimeError("bert_score.score is not used for LoCoMo QA F1 evaluation.")

    module.score = _unused_score
    sys.modules["bert_score"] = module


if __name__ == "__main__":
    main()
