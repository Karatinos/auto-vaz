"""Multi-agent prototypes.

Each prototype is an isolated execution sub-graph over one shared model
provider. They differ in routing, sampling, and reflection so their
processing efficiencies can be compared head-to-head on identical tasks.

A "provider" is any object exposing::

    complete(messages: list[dict], *, temperature: float, max_tokens: int) -> str

so :class:`deepseek_client.DeepSeekClient` and the test MockProvider are
interchangeable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol


class Provider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> str: ...


@dataclass
class Task:
    """A single evaluation task."""

    task_id: str
    prompt: str
    expected: str | None = None  # gold answer, used only for local grading


def _ask(provider: Provider, system: str, user: str, *, temperature: float) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return provider.complete(messages, temperature=temperature, max_tokens=1024).strip()


class Prototype:
    """Base class: unique name + a solve() that returns a candidate answer."""

    name: str = "base"
    description: str = ""

    def solve(self, provider: Provider, task: Task) -> str:  # pragma: no cover
        raise NotImplementedError


class P1SinglePass(Prototype):
    """Control: one direct call, no ensemble, no reflection."""

    name = "p1"
    description = "baseline single-pass"

    def solve(self, provider: Provider, task: Task) -> str:
        return _ask(provider, "Answer concisely.", task.prompt, temperature=0.1)


class P3EnsembleVote(Prototype):
    """Three independent attempts, majority vote on the answer."""

    name = "p3"
    description = "ensemble + majority vote"

    def solve(self, provider: Provider, task: Task) -> str:
        answers = [
            _ask(provider, "Answer concisely.", task.prompt, temperature=0.7)
            for _ in range(3)
        ]
        return Counter(answers).most_common(1)[0][0]


class P4ReflectCritic(Prototype):
    """Draft, then a critic pass that may revise the draft."""

    name = "p4"
    description = "reflect / self-critique"

    def solve(self, provider: Provider, task: Task) -> str:
        draft = _ask(provider, "Answer concisely.", task.prompt, temperature=0.3)
        critique = _ask(
            provider,
            "You are a strict reviewer. If the answer is correct, repeat it "
            "verbatim. Otherwise reply only with the corrected answer.",
            f"Task: {task.prompt}\nProposed answer: {draft}",
            temperature=0.0,
        )
        return critique or draft


class P5WideConsensus(Prototype):
    """Five low-temperature samples, self-consistency majority."""

    name = "p5"
    description = "wide self-consistency"

    def solve(self, provider: Provider, task: Task) -> str:
        answers = [
            _ask(provider, "Answer concisely.", task.prompt, temperature=0.5)
            for _ in range(5)
        ]
        return Counter(answers).most_common(1)[0][0]


class P7CascadeEscalate(Prototype):
    """Cheap first pass; escalate to a careful pass only on low confidence."""

    name = "p7"
    description = "cascade with escalation"

    def solve(self, provider: Provider, task: Task) -> str:
        first = _ask(provider, "Answer concisely.", task.prompt, temperature=0.2)
        # Escalate when the cheap pass looks unsure or empty.
        if not first or "unsure" in first.lower() or "i don't know" in first.lower():
            return _ask(
                provider,
                "Think step by step, then give only the final answer.",
                task.prompt,
                temperature=0.0,
            )
        return first


def all_prototypes() -> list[Prototype]:
    """The registered prototype line-up used for A/B comparison."""
    return [P1SinglePass(), P3EnsembleVote(), P4ReflectCritic(), P5WideConsensus(), P7CascadeEscalate()]
