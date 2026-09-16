from .base import BaseModel


class LocalModel(BaseModel):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LocalModel is a placeholder. Wire vLLM or transformers here.")

