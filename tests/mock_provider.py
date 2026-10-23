"""Deterministic, $0 provider for tests and offline demos.

Implements the same ``complete()`` surface as DeepSeekClient. It looks up an
answer for the task prompt embedded in the user message. Unknown prompts come
back as ``"unsure"`` so escalation/consensus paths are exercised.
"""

from __future__ import annotations


class MockProvider:
    def __init__(self, answer_map: dict[str, str] | None = None):
        # Maps the task prompt text -> canned answer.
        self.answer_map = answer_map or {}
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self.calls += 1
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # The prompt may be wrapped (e.g. critic pass); match by containment.
        for prompt, answer in self.answer_map.items():
            if prompt in user:
                return answer
        return "unsure"
