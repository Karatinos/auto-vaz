"""Benchmark connector.

Defines an abstract task source plus a fully-working offline/local source.

The live network adapter against ``EVAL_TARGET_URL`` is intentionally left
unimplemented in this file: it is the account-registration + live-submission
path, and it is only appropriate to run against infrastructure the operator
owns or is explicitly authorized to test. Wire it up deliberately and locally
rather than having it ship enabled by default. See ``LiveBenchmarkSource``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from prototypes import Task


class BenchmarkSource(ABC):
    """Abstract lifecycle a prototype runner depends on."""

    @abstractmethod
    def fetch_tasks(self, limit: int | None = None) -> list[Task]:
        """Return evaluation tasks (bounded by ``limit`` when given)."""

    @abstractmethod
    def grade(self, task: Task, answer: str) -> bool:
        """Return True when ``answer`` solves ``task``."""


class LocalBenchmarkSource(BenchmarkSource):
    """Offline source backed by a local JSON file.

    Expected JSON shape::

        [{"task_id": "t1", "prompt": "...", "expected": "..."}, ...]

    Grading is a canonicalized exact match against the local gold answer,
    which never leaves this process and is never shown to a prototype.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"local task file not found: {self.path}")
        raw = json.loads(self.path.read_text())
        self._tasks = [
            Task(task_id=item["task_id"], prompt=item["prompt"], expected=item.get("expected"))
            for item in raw
        ]

    def fetch_tasks(self, limit: int | None = None) -> list[Task]:
        return self._tasks[:limit] if limit else list(self._tasks)

    @staticmethod
    def _canon(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def grade(self, task: Task, answer: str) -> bool:
        if task.expected is None:
            return False
        return self._canon(answer) == self._canon(task.expected)


class LiveBenchmarkSource(BenchmarkSource):
    """Live adapter against ``EVAL_TARGET_URL``.

    Registers one ``farrrm-{run_id}`` tracking id per run, fetches assignments,
    and grades by submitting each solution to the benchmark scoreboard. All
    requests carry the ``x-benchmark`` token via :class:`rest_connector`.

    Only appropriate against an endpoint you own or are explicitly authorized
    to test. Requires network reachability to the target, so it cannot run from
    the egress-restricted cloud sandbox — launch it from your own terminal.
    """

    def __init__(self, settings, run_id: str):
        # Imported lazily so the offline path never needs requests configured.
        from rest_connector import BenchmarkConnector

        self.connector = BenchmarkConnector(settings)
        self.track_id = self.connector.register(run_id)

    def fetch_tasks(self, limit: int | None = None) -> list[Task]:
        tasks: list[Task] = []
        for item in self.connector.fetch_assignments(limit=limit):
            tasks.append(
                Task(
                    task_id=str(item.get("id")),
                    prompt=item.get("prompt") or item.get("content") or "",
                    expected=None,  # graded remotely; no local gold
                )
            )
        return tasks

    def grade(self, task: Task, answer: str) -> bool:
        return self.connector.submit(task.task_id, answer)
