def exact_match(prediction: str, answer: str) -> float:
    return float(prediction.strip().lower() == answer.strip().lower())


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = prediction.lower().split()
    answer_tokens = answer.lower().split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = set(pred_tokens).intersection(answer_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)

