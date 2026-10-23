# auto-vaz — multi-agent evaluation harness

A small, modular harness for A/B-comparing multi-agent LLM prototypes on a
shared engine, using DeepSeek V4 Flash via OpenRouter, with a two-phase
validation pipeline and a machine-readable metrics report.

## Layout

| File | Role |
|------|------|
| `config.py` | Runtime settings, all via `os.getenv()` / `.env` (no hardcoded secrets) |
| `deepseek_client.py` | OpenRouter client for `deepseek/deepseek-v4-flash` (retries, backoff, `HTTP-Referer`/`X-Title`) |
| `prototypes.py` | 5 distinct multi-agent sub-graphs (p1/p3/p4/p5/p7) over one provider |
| `benchmark_client.py` | Abstract `BenchmarkSource` + `LocalBenchmarkSource` (offline) + `LiveBenchmarkSource` |
| `rest_connector.py` | `requests.Session` REST client; sends the token in the `x-benchmark` header on every request |
| `pipeline.py` | Phase 1 (baseline, >60% = stable) / Phase 2 (unlock on >90%) + `metrics_report.json` |
| `main.py` | CLI entry point |
| `tests/` | Offline unittest suite (deterministic mock provider, $0, no network) |

## Prototypes

- **p1** — baseline single-pass (control)
- **p3** — ensemble + majority vote
- **p4** — reflect / self-critique
- **p5** — wide self-consistency (5 samples)
- **p7** — cascade with escalation on low confidence

## Setup

```bash
cp .env.example .env     # fill in OPENROUTER_API_KEY, EVAL_TARGET_URL, EVAL_BENCHMARK_HEADER
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Tests — offline, free, no keys
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Offline demo — deterministic mock provider, writes metrics_report.json
.venv/bin/python main.py --provider mock --tasks data/local_tasks/sample.json

# Offline, real model (needs OPENROUTER_API_KEY)
.venv/bin/python main.py --source local --provider deepseek --tasks data/local_tasks/sample.json
```

## Live / paid runs

The live path (`--source live`) registers a `farrrm-<label>` tracking id,
fetches assignments from `EVAL_TARGET_URL`, and grades by submitting solutions —
all with the `x-benchmark` token. It makes **paid** OpenRouter calls and talks to
your benchmark endpoint, so it only runs against infrastructure you own or are
authorized to test, and it needs real network access.

It **cannot** run from a cloud sandbox (egress is restricted to pip/npm/GitHub).
Launch it from your own machine:

```bash
.venv/bin/python main.py --source live --provider deepseek --label compare-01
```

or double-click `START_LIVE_COMPARE.command` in Terminal.app (prompts before
spending, sets up the venv, writes `metrics_report.json`).

## Notes

- Endpoint paths / payload field names in `rest_connector.py` are centralized
  constants — adjust them to your benchmark server's API.
- No secret ever lives in the repo; everything sensitive comes from `.env`,
  which is git-ignored.
