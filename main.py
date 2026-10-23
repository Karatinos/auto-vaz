"""Core orchestration entry point.

Initializes the model provider and the benchmark source, runs the two-phase
validation pipeline across all prototypes, and writes ``metrics_report.json``.

Offline (default): grade against a local task file with a deterministic mock
or the real DeepSeek provider.

    python main.py --source local --tasks data/local_tasks/sample.json

The live source against EVAL_TARGET_URL is intentionally not selectable here;
see benchmark_client.LiveBenchmarkSource.
"""

from __future__ import annotations

import argparse
import sys

from config import load_settings
from pipeline import run_pipeline
from prototypes import all_prototypes


def _build_provider(kind: str, source=None):
    if kind == "deepseek":
        from deepseek_client import DeepSeekClient

        return DeepSeekClient(load_settings())
    if kind == "mock":
        # Build a deterministic answer map from the local gold answers so the
        # offline demo actually solves tasks (no network, no keys, $0).
        from tests.mock_provider import MockProvider

        answer_map = {t.prompt: (t.expected or "") for t in source.fetch_tasks()}
        return MockProvider(answer_map)
    raise SystemExit(f"unknown provider: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="auto-vaz evaluation harness")
    parser.add_argument("--source", choices=["local", "live"], default="local")
    parser.add_argument("--tasks", default="data/local_tasks/sample.json")
    parser.add_argument("--provider", choices=["deepseek", "mock"], default="deepseek")
    parser.add_argument("--label", default="run", help="run id for live tracking (farrrm-<label>)")
    parser.add_argument("--out", default="metrics_report.json")
    args = parser.parse_args(argv)

    if args.source == "live":
        # Live, networked, paid path. Requires reachability to EVAL_TARGET_URL
        # and OpenRouter — run from a terminal with network access, not the
        # cloud sandbox.
        from benchmark_client import LiveBenchmarkSource

        source = LiveBenchmarkSource(load_settings(), run_id=args.label)
        provider = _build_provider("deepseek")
    else:
        from benchmark_client import LocalBenchmarkSource

        source = LocalBenchmarkSource(args.tasks)
        provider = _build_provider(args.provider, source)

    report = run_pipeline(all_prototypes(), provider, source, output_path=args.out)

    print(f"Wrote {args.out}")
    for proto in report.prototypes:
        flag = "stable" if proto.stable else "below-threshold"
        p2 = " -> phase2" if proto.advanced_to_phase2 else ""
        print(
            f"  {proto.name:4} rate={proto.success_rate:5.1%} "
            f"({proto.solved}/{proto.attempted}) [{flag}]{p2}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
