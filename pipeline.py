"""Two-phase validation pipeline and telemetry.

Phase 1 (baseline): run every prototype over the baseline task set, track
per-prototype success rate. A prototype is "stable" once its rate > 60%.

Phase 2 (advanced): only prototypes whose Phase-1 rate strictly exceeds 90%
are advanced to the comprehensive block.

All metrics are dumped to ``metrics_report.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmark_client import BenchmarkSource
from config import (
    PHASE1_STABLE_THRESHOLD,
    PHASE1_TASK_COUNT,
    PHASE2_UNLOCK_THRESHOLD,
)
from prototypes import Prototype, Provider, Task


@dataclass
class PrototypeResult:
    name: str
    description: str
    attempted: int = 0
    solved: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    stable: bool = False
    advanced_to_phase2: bool = False
    phase2_attempted: int = 0
    phase2_solved: int = 0

    @property
    def success_rate(self) -> float:
        return self.solved / self.attempted if self.attempted else 0.0


@dataclass
class Report:
    phase1_task_count: int
    phase1_stable_threshold: float
    phase2_unlock_threshold: float
    prototypes: list[PrototypeResult] = field(default_factory=list)


def _run_phase(
    prototype: Prototype,
    provider: Provider,
    source: BenchmarkSource,
    tasks: list[Task],
) -> tuple[int, int, int, float]:
    """Return (attempted, solved, errors, elapsed_seconds)."""
    attempted = solved = errors = 0
    start = time.monotonic()
    for task in tasks:
        attempted += 1
        try:
            answer = prototype.solve(provider, task)
        except Exception:  # a failed trajectory counts, it must not abort the run
            errors += 1
            continue
        if source.grade(task, answer):
            solved += 1
    return attempted, solved, errors, time.monotonic() - start


def run_pipeline(
    prototypes: list[Prototype],
    provider: Provider,
    source: BenchmarkSource,
    *,
    output_path: str | Path = "metrics_report.json",
) -> Report:
    """Execute Phase 1 for all prototypes, Phase 2 for those that qualify."""
    phase1_tasks = source.fetch_tasks(limit=PHASE1_TASK_COUNT)
    report = Report(
        phase1_task_count=len(phase1_tasks),
        phase1_stable_threshold=PHASE1_STABLE_THRESHOLD,
        phase2_unlock_threshold=PHASE2_UNLOCK_THRESHOLD,
    )

    for proto in prototypes:
        res = PrototypeResult(name=proto.name, description=proto.description)
        res.attempted, res.solved, res.errors, res.elapsed_s = _run_phase(
            proto, provider, source, phase1_tasks
        )
        res.stable = res.success_rate > PHASE1_STABLE_THRESHOLD

        # Phase 2 unlocks only on a strictly-better-than-90% baseline.
        if res.success_rate > PHASE2_UNLOCK_THRESHOLD:
            res.advanced_to_phase2 = True
            phase2_tasks = source.fetch_tasks()  # full/comprehensive block
            p2a, p2s, _, _ = _run_phase(proto, provider, source, phase2_tasks)
            res.phase2_attempted, res.phase2_solved = p2a, p2s

        report.prototypes.append(res)

    _dump(report, output_path)
    return report


def _dump(report: Report, output_path: str | Path) -> None:
    payload = asdict(report)
    # Enrich with derived rates for downstream mathematical evaluation.
    for proto in payload["prototypes"]:
        attempted = proto["attempted"] or 1
        proto["success_rate"] = proto["solved"] / attempted
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
