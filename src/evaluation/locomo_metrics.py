import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_answer(text: object) -> str:
    text = str(text).lower().replace(",", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def f1_score(prediction: object, ground_truth: object) -> float:
    pred_tokens = [_stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: object, ground_truth: object) -> float:
    predictions = [part.strip() for part in str(prediction).split(",")]
    ground_truths = [part.strip() for part in str(ground_truth).split(",")]
    return sum(max(f1_score(pred, gold) for pred in predictions) for gold in ground_truths) / len(ground_truths)


def score_qa(prediction: object, qa: dict) -> float:
    answer = str(qa["answer"])
    if qa.get("category") == 3:
        answer = answer.split(";")[0].strip()
    category = qa.get("category")
    if category in [2, 3, 4]:
        return f1_score(prediction, answer)
    if category == 1:
        return multi_answer_f1(prediction, answer)
    if category == 5:
        lowered = str(prediction).lower()
        return 1.0 if "no information available" in lowered or "not mentioned" in lowered else 0.0
    raise ValueError("Unknown LoCoMo category: %s" % category)


def score_output(samples: List[dict], prediction_key: str) -> Tuple[List[dict], Dict[str, object]]:
    category_counts = defaultdict(int)
    category_scores = defaultdict(float)
    total_count = 0
    total_score = 0.0
    for sample in samples:
        for qa in sample["qa"]:
            score_key = prediction_key + "_f1"
            qa[score_key] = round(score_qa(qa.get(prediction_key, ""), qa), 3)
            category = str(qa.get("category"))
            category_counts[category] += 1
            category_scores[category] += qa[score_key]
            total_count += 1
            total_score += qa[score_key]
    stats = {
        "prediction_key": prediction_key,
        "num_questions": total_count,
        "overall_f1": round(total_score / total_count, 4) if total_count else 0.0,
        "category_counts": dict(category_counts),
        "category_f1": {
            category: round(category_scores[category] / count, 4)
            for category, count in category_counts.items()
        },
    }
    return samples, stats


def write_scored_output(samples: List[dict], prediction_key: str, output_path: str) -> Dict[str, object]:
    scored, stats = score_output(samples, prediction_key)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = output.with_name(output.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats

