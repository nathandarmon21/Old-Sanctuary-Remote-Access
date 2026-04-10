# The Sanctuary

A multi-agent LLM market simulation. Four seller firms and four buyer firms trade
widgets over 30 simulated days. Agents use a dual-tier architecture: a weekly
strategic planning call (larger model) and daily tactical decision calls (smaller
model). The simulation logs every LLM call, transaction, message, and state change
to JSONL files and generates a PDF report at the end of each run.

## Prerequisites

- Python 3.9+
- One of the two inference backends below

## Inference backends

### Option A — Anthropic API (fast dev path, recommended for Phase 1)

Uses Claude Haiku 4.5 via the Anthropic API. Fast enough for tight
prompt-tuning iteration loops (~3–5 min per 3-day smoke test).

**Cost:** ~$0.15 per 3-day smoke test, ~$1.50 per 30-day run.

```bash
pip install -e .

# Add your key to .env at the project root
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3-day smoke test (~3-5 min)
python scripts/run_simulation.py --config configs/smoke_haiku.yaml --seed 42

# Full 30-day run (~25-40 min)
python scripts/run_simulation.py --config configs/dev_haiku.yaml --seed 42
```

### Option B — Local Ollama (slow but free, for offline/production path)

Uses qwen2.5 models served by [Ollama](https://ollama.com).
Suitable for runs on SEAS compute where API egress is restricted.
Expect 10–20× slower iteration than the Anthropic path on a laptop GPU.

```bash
pip install -e .
ollama pull qwen2.5:7b && ollama pull qwen2.5:3b
```

**Required Ollama configuration** (run once per machine):

```bash
launchctl setenv OLLAMA_NUM_PARALLEL 8   # macOS only
launchctl setenv OLLAMA_MAX_LOADED_MODELS 2
# Quit Ollama from menu bar, then reopen
launchctl getenv OLLAMA_NUM_PARALLEL     # should print: 8
```

Verify parallelism is working:

```bash
python3 scripts/check_ollama_parallelism.py   # expect ≥ 3× speedup
```

**Performance expectations (Ollama):**  
A 3-day smoke test should complete in **5–8 minutes** on a modern laptop
with `OLLAMA_NUM_PARALLEL=8`. A full 30-day run: **30–50 minutes**.  
If meaningfully slower, check `max_parallel_llm_calls: 8` in your config,
verify the env var is set, and confirm Ollama was restarted after setting it.

```bash
# 3-day smoke test
python scripts/run_simulation.py --config configs/smoke_3day.yaml --seed 42

# Full 30-day run
python scripts/run_simulation.py --config configs/dev_available.yaml --seed 42
```

## Running the Simulation

The system supports three operational modes sharing the same simulation core.

### Mode 1: Batch (no dashboard)

Runs the simulation as a standalone process, writes a complete run directory, exits.

```bash
# 3-day smoke test
python -m sanctuary.run --config configs/smoke_3day.yaml --seed 42

# Full 30-day run
python -m sanctuary.run --config configs/dev_local.yaml --seed 42

# Custom output directory
python -m sanctuary.run --config configs/dev_local.yaml --seed 42 --output runs/my_run/
```

### Mode 2: Live Dashboard

Starts FastAPI server, runs the simulation as an asyncio task, broadcasts state via WebSocket.

```bash
python -m sanctuary.dev --config configs/dev_local.yaml --port 8090
# Open http://localhost:8090 in your browser
```

Controls: pause, resume, speed adjustment, fast-forward. Per-agent drill-down shows
tactical decisions and CEO strategic reviews in separate tabs.

### Mode 3: Replay

Loads a completed run directory and serves it through the dashboard with timeline scrubbing.

```bash
python -m sanctuary.replay --run runs/<run_id>/ --port 8090
```

### Resume from Checkpoint

Simulations checkpoint every 5 days. To resume an interrupted run:

```bash
python -m sanctuary.run --config configs/dev_local.yaml --seed 42 --resume runs/<run_id>/
```

## Project Layout

```
sanctuary/              Core simulation package
  engine.py             Main simulation loop with all subsystem integration
  agent.py              Dual-tier agent (strategic + tactical LLM calls)
  market.py             Market state, offers, transactions
  economics.py          Production costs (lookup table), revenue, bankruptcy
  revelation.py         Deterministic 5-day quality revelation
  context_manager.py    Tier-specific context assembly + market digest
  events.py             Structured events.jsonl writer
  transcripts.py        Per-agent transcript storage
  run_directory.py      Run directory schema manager
  messaging.py          Intra-day message routing
  config.py             Config loading and validation (Pydantic v2)
  run.py / dev.py / replay.py   Three mode entry points
  prompts/              Dual-tier prompt templates
    tactical.py         Daily operations prompts
    strategic.py        Weekly CEO review prompts
    common.py           Shared formatters
    sub_round.py        Accept/decline only prompts
  protocols/            Market governance regimes
    base.py             Protocol base class with lifecycle hooks
    factory.py          Protocol registry and dispatch
    no_protocol.py      Baseline (no reputation, no auditing)
  metrics/              Post-run metric computation
    misrepresentation.py  Misrepresentation Rate
    allocative_efficiency.py  AE + Price-Cost Margin (Lerner Index)
    market_integrity.py   PPI, Markup Correlation, Exploitation Rate, Trust Persistence
    aggregate.py        Compute all metrics from events.jsonl
  analytics/            In-run lightweight analytics
    scanner.py          Chain-of-thought keyword scanner (7 categories)
    series.py           Daily time series tracker (CSV export)
  checkpointing/        Save/restore at day boundaries
    checkpoint.py       Serialize/deserialize engine state + RNG
  dashboard/            FastAPI + WebSocket dashboard
    app.py              Backend (live + replay modes)
    static/index.html   Single-file SPA (8 agents, dual-tier drill-down)
  providers/            LLM backend adapters
    base.py             Abstract ModelProvider
    ollama.py / vllm.py / anthropic.py
  logs.py               Legacy JSONL logging (kept for backward compat)
  report.py             PDF report generator (legacy, untouched)
  style.py              Visual style constants

configs/                YAML config files
  dev_local.yaml        Qwen 2.5 via Ollama (spec baseline)
  dev_haiku.yaml        Claude Haiku via Anthropic API
  production.yaml       vLLM on GPU cluster (Phase 2+)
  smoke_*.yaml          3-day smoke tests

tests/                  pytest suite (383 tests)
runs/                   Simulation output directories (gitignored)
docs/                   Specification document
```

## Run Directory Schema

Each simulation produces a directory with fixed schema:

```
runs/<run_id>/
  manifest.json               Run metadata + status (running/complete/crashed)
  events.jsonl                 Canonical chronological event log
  metrics.json                 Computed scalar metrics
  series.csv                   Daily time series
  final_state.json             End-of-run engine snapshot
  config_used.yaml             Exact config that ran
  agents/
    <agent_id>/
      tactical_transcript.jsonl   Every tactical LLM call (full prompt + response)
      strategic_transcript.jsonl  Every strategic LLM call (full prompt + response)
      reasoning_log.jsonl         Parsed reasoning for fast access
  checkpoints/                 Saved every 5 days
```

The events log is the source of truth. Metrics can be recomputed from it without re-running.

## Reproducibility

Two runs with the same `--seed` produce identical transaction logs
(excluding wall-clock timestamps). Parallelism is applied to LLM calls only;
all market state mutations happen in deterministic alphabetical order.
