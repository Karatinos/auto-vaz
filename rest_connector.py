"""Benchmark REST connector.

An abstract client over ``requests.Session`` that talks to the benchmark
endpoint in ``EVAL_TARGET_URL`` and sends the validation token in the static
``x-benchmark`` header on **every** lifecycle request (fetch, register,
submit). Nothing sensitive is hardcoded — URL and token come from the
environment via :func:`config.load_settings`.

Endpoint paths and payload field names are centralized as constants so they
can be matched to the specific benchmark server without touching logic.

NOTE: This performs live, authenticated, potentially paid interactions and is
only intended for a benchmark endpoint you own or are explicitly authorized to
test. It cannot run from the cloud sandbox (egress is allowlisted); launch it
from an environment with network access to the target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from config import BENCHMARK_HEADER_NAME, Settings

# --- Endpoint layout (adjust to the target server's API if it differs) ---
ASSIGNMENTS_PATH = "/assignments"      # GET  -> list of assignments
REGISTER_PATH = "/track"               # POST -> register a tracking id
SUBMIT_PATH = "/submit"                # POST -> submit a solution

TRACK_ID_PREFIX = "farrrm"             # tracking ids are "farrrm-{id}"


class ConnectorError(RuntimeError):
    """Raised when a benchmark request fails after retries."""


@dataclass
class BenchmarkConnector:
    settings: Settings
    max_retries: int = 4
    base_backoff: float = 1.5
    request_timeout: float = 45.0
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.has_benchmark_credentials:
            raise ConnectorError(
                "EVAL_TARGET_URL and EVAL_BENCHMARK_HEADER must both be set "
                "to use the benchmark connector."
            )
        self._base = self.settings.eval_target_url.rstrip("/")
        self._session = requests.Session()
        # The validation token rides in this static header on every request.
        self._session.headers.update(
            {BENCHMARK_HEADER_NAME: self.settings.eval_benchmark_header}
        )

    # --- transport with bounded retry/backoff ---------------------------------
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._base}{path}"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.request(
                    method, url, timeout=self.request_timeout, **kwargs
                )
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                last = ConnectorError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as exc:
                # Catches connection resets / protocol errors so a flaky
                # transfer never aborts the whole run.
                last = exc
            if attempt < self.max_retries - 1:
                time.sleep(self.base_backoff * (2 ** attempt))
        raise ConnectorError(f"{method} {path} failed after {self.max_retries}: {last}")

    # --- lifecycle operations -------------------------------------------------
    def register(self, run_id: str) -> str:
        """Register a tracking id ``farrrm-{run_id}`` and return it."""
        track_id = f"{TRACK_ID_PREFIX}-{run_id}"
        self._request("POST", REGISTER_PATH, json={"id": track_id})
        return track_id

    def fetch_assignments(self, limit: int | None = None) -> list[dict]:
        """Fetch the list of assignments (optionally bounded)."""
        params = {"limit": limit} if limit else None
        resp = self._request("GET", ASSIGNMENTS_PATH, params=params)
        data = resp.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        return items[:limit] if limit else items

    def submit(self, assignment_id: str, solution: str) -> bool:
        """Submit a solution; return True when the server marks it correct."""
        resp = self._request(
            "POST",
            SUBMIT_PATH,
            json={"assignment_id": assignment_id, "solution": solution},
        )
        payload = resp.json()
        status = payload.get("data", payload) if isinstance(payload, dict) else {}
        return bool(
            payload.get("correct")
            or (isinstance(status, dict) and status.get("status") == "correct")
        )
