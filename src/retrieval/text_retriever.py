from typing import List


class TextRetriever:
    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        terms = set(query.lower().split())
        scored = []
        for text in self.texts:
            score = len(terms.intersection(text.lower().split()))
            scored.append((score, text))
        return [text for score, text in sorted(scored, reverse=True)[:top_k] if score > 0]
