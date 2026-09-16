from .base import BaseModel
from typing import Optional


class APIModel(BaseModel):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        raise RuntimeError("APIModel is a placeholder. Wire your provider client here.")
