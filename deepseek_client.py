"""Modular DeepSeek-via-OpenRouter client wrapper.

Targets OpenRouter's chat-completions API using the ``deepseek/deepseek-v4-flash``
model. Handles transient errors with bounded exponential backoff and always
sends the ``HTTP-Referer`` / ``X-Title`` headers OpenRouter expects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

from config import DEEPSEEK_MODEL, OPENROUTER_BASE_URL, Settings


class ModelError(RuntimeError):
    """Raised when the model call fails after exhausting retries."""


@dataclass
class DeepSeekClient:
    """Thin, reusable wrapper around one OpenRouter model.

    A single :class:`requests.Session` is reused across calls so connection
    pooling and the standing headers are set up once.
    """

    settings: Settings
    model: str = DEEPSEEK_MODEL
    max_retries: int = 4
    base_backoff: float = 1.5
    request_timeout: float = 45.0
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.openrouter_api_key:
            raise ModelError(
                "OPENROUTER_API_KEY is not set; the model engine cannot run. "
                "Copy .env.example to .env and fill it in."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
                # OpenRouter attribution headers (required for ranking credit).
                "HTTP-Referer": self.settings.openrouter_referer,
                "X-Title": self.settings.openrouter_title,
            }
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant text for a chat-completion request.

        Retries on network errors and 429/5xx responses with exponential
        backoff. Raises :class:`ModelError` once retries are exhausted.
        """
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self.request_timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"] or ""
                # Retry only on transient statuses.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = ModelError(
                        f"transient HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    raise ModelError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                time.sleep(self.base_backoff * (2 ** attempt))

        raise ModelError(f"model call failed after {self.max_retries} attempts: {last_error}")
