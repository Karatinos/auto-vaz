"""Runtime configuration.

Every sensitive value is pulled at runtime from the environment via
``os.getenv()`` (loaded from a local ``.env`` by python-dotenv). Nothing
sensitive is hardcoded here or anywhere else in the tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from the repo root if present. Values already in the real
# environment take precedence (override=False).
load_dotenv(override=False)

# Fixed, non-sensitive engine parameters.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"

# Benchmark connector sends its token in this static header name.
BENCHMARK_HEADER_NAME = "x-benchmark"

# Validation pipeline thresholds (see README / migration guide).
PHASE1_TASK_COUNT = 28
PHASE1_STABLE_THRESHOLD = 0.60   # prototype "stable" once success rate > 60%
PHASE2_UNLOCK_THRESHOLD = 0.90   # advance to Phase 2 only if rate > 90%


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime settings."""

    openrouter_api_key: str
    openrouter_referer: str
    openrouter_title: str
    eval_target_url: str
    eval_benchmark_header: str

    @property
    def has_model_credentials(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_benchmark_credentials(self) -> bool:
        return bool(self.eval_target_url and self.eval_benchmark_header)


def load_settings() -> Settings:
    """Read all configuration from the environment.

    Missing values come back as empty strings rather than raising, so the
    harness can run its offline/local path with no credentials at all. The
    components that need a given value check for it and fail loudly there.
    """
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_referer=os.getenv("OPENROUTER_REFERER", "http://localhost"),
        openrouter_title=os.getenv("OPENROUTER_TITLE", "auto-vaz-eval-harness"),
        eval_target_url=os.getenv("EVAL_TARGET_URL", ""),
        eval_benchmark_header=os.getenv("EVAL_BENCHMARK_HEADER", ""),
    )
