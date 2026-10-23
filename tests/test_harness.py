"""Offline test suite. Runs with the deterministic mock provider — $0, no network."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmark_client import LiveBenchmarkSource, LocalBenchmarkSource
from config import Settings
from pipeline import run_pipeline
from prototypes import Task, all_prototypes
from tests.mock_provider import MockProvider

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "local_tasks" / "sample.json"


def _answer_map(source: LocalBenchmarkSource) -> dict[str, str]:
    return {t.prompt: (t.expected or "") for t in source.fetch_tasks()}


class TestPrototypes(unittest.TestCase):
    def test_five_distinct_prototypes(self):
        protos = all_prototypes()
        names = [p.name for p in protos]
        self.assertEqual(len(names), len(set(names)), "prototype names must be unique")
        self.assertGreaterEqual(len(protos), 4)
        self.assertLessEqual(len(protos), 5)

    def test_prototype_solves_known_task(self):
        task = Task("x", "Capital of France?", "Paris")
        provider = MockProvider({"Capital of France?": "Paris"})
        for proto in all_prototypes():
            self.assertEqual(proto.solve(provider, task), "Paris", proto.name)


class TestLocalSource(unittest.TestCase):
    def test_fetch_and_grade(self):
        src = LocalBenchmarkSource(SAMPLE)
        tasks = src.fetch_tasks()
        self.assertEqual(len(tasks), 5)
        self.assertTrue(src.grade(tasks[0], "  paris "))   # canonicalized match
        self.assertFalse(src.grade(tasks[0], "London"))

    def test_limit(self):
        src = LocalBenchmarkSource(SAMPLE)
        self.assertEqual(len(src.fetch_tasks(limit=2)), 2)


class TestLiveSourceGuarded(unittest.TestCase):
    def test_live_source_requires_credentials(self):
        # No EVAL_TARGET_URL / EVAL_BENCHMARK_HEADER -> connector refuses.
        blank = Settings(
            openrouter_api_key="",
            openrouter_referer="",
            openrouter_title="",
            eval_target_url="",
            eval_benchmark_header="",
        )
        from rest_connector import ConnectorError

        with self.assertRaises(ConnectorError):
            LiveBenchmarkSource(blank, run_id="test")


class TestPipeline(unittest.TestCase):
    def test_full_pipeline_all_correct(self):
        src = LocalBenchmarkSource(SAMPLE)
        provider = MockProvider(_answer_map(src))
        out = Path(__file__).resolve().parent / "_metrics_tmp.json"
        report = run_pipeline(all_prototypes(), provider, src, output_path=out)
        try:
            for res in report.prototypes:
                self.assertEqual(res.attempted, 5)
                self.assertEqual(res.solved, 5, res.name)
                self.assertTrue(res.stable)
                self.assertTrue(res.advanced_to_phase2)  # 100% > 90%
            # Report file is valid JSON with derived rates.
            data = json.loads(out.read_text())
            self.assertIn("prototypes", data)
            self.assertAlmostEqual(data["prototypes"][0]["success_rate"], 1.0)
        finally:
            out.unlink(missing_ok=True)

    def test_phase2_locked_when_below_threshold(self):
        src = LocalBenchmarkSource(SAMPLE)
        # Only one of five prompts is answerable -> 20% success.
        provider = MockProvider({"Capital of France?": "Paris"})
        out = Path(__file__).resolve().parent / "_metrics_tmp2.json"
        report = run_pipeline(all_prototypes(), provider, src, output_path=out)
        try:
            for res in report.prototypes:
                self.assertFalse(res.stable)            # 20% < 60%
                self.assertFalse(res.advanced_to_phase2)  # 20% < 90%
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
