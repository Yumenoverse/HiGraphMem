import os
import time
from typing import Any, Callable, Dict, Optional

from .base import BaseModel


class OpenAICompatibleModel(BaseModel):
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        max_retries: int = 5,
        retry_sleep: float = 2.0,
    ) -> None:
        api_key = api_key or os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError("Missing API key environment variable: %s" % api_key_env)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.api_key = api_key
        self.base_url = _normalize_base_url(base_url)
        self.client = self._build_client()
        self.usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_usage_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Register a per-call usage sink without changing generation behavior."""
        self.usage_callback = callback

    def generate(self, prompt: str, phase: str = "answer") -> str:
        messages = [
            {
                "role": "system",
                "content": "Answer questions using only the supplied memory evidence. Keep answers short.",
            },
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate_once(messages, phase=phase)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_for = self.retry_sleep * (2 ** attempt)
                print("API call failed (%s). Retrying in %.1fs [%s/%s]." % (exc, sleep_for, attempt + 1, self.max_retries))
                time.sleep(sleep_for)
        raise last_error

    def _generate_once(self, messages, phase: str = "answer") -> str:
        if self.client is not None:
            started_at = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self._record_usage(
                getattr(response, "usage", None),
                phase,
                latency_seconds=time.perf_counter() - started_at,
            )
            return response.choices[0].message.content.strip()

        import openai

        openai.api_key = self.api_key
        if self.base_url:
            openai.api_base = self.base_url
        started_at = time.perf_counter()
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._record_usage(
            response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None),
            phase,
            latency_seconds=time.perf_counter() - started_at,
        )
        return response.choices[0].message.content.strip()

    def _record_usage(self, usage: Any, phase: str, latency_seconds: float) -> None:
        if self.usage_callback is None:
            return

        def get(name: str):
            if usage is None:
                return None
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)

        prompt_tokens = get("prompt_tokens")
        completion_tokens = get("completion_tokens")
        total_tokens = get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        self.usage_callback({
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_available": total_tokens is not None,
            "latency_seconds": round(latency_seconds, 6),
        })

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = normalized + "/v1"
    return normalized
