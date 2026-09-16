from typing import Iterable, Protocol


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class FullContextBaseline:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def answer(self, question: str, sessions: Iterable[object]) -> str:
        context = "\n\n".join(str(session) for session in sessions)
        prompt = f"Answer the question using the conversation history.\n\nHistory:\n{context}\n\nQuestion: {question}"
        return self.generator.generate(prompt)

