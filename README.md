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

## Quick start (Anthropic path)

```bash
pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3-day smoke test (~3-5 min, ~$0.15)
python scripts/run_simulation.py --config configs/smoke_haiku.yaml --seed 42

# Generate report for any run
python scripts/generate_report.py runs/<run_id>
```

## Project layout

```
sanctuary/          Core simulation package
  agent.py          Dual-tier agent (strategic + tactical LLM calls)
  simulation.py     Main 30-day loop with intra-day parallelism
  market.py         Market state, offers, transactions
  economics.py      Production costs, revenue, bankruptcy logic
  revelation.py     Stochastic quality revelation scheduler
  messaging.py      Intra-day message routing
  logs.py           JSONL structured logging
  report.py         PDF report generator
  prompts.py        All LLM prompt templates
  style.py          Visual style constants (colours, fonts)
  providers/        LLM backend adapters (Ollama, vLLM)
  config.py         Config loading and validation

configs/            YAML config files
  dev_local.yaml    Canonical dev config (qwen2.5, 30 days)
  dev_available.yaml  Same but with verified model names
  smoke_3day.yaml   Fast 3-day smoke test

scripts/
  run_simulation.py  Main CLI entry point
  generate_report.py Standalone report generator
  check_ollama_parallelism.py  Parallelism diagnostic
  run_background.sh  Background run launcher

tests/              pytest test suite (106 tests)
runs/               Simulation output directories (gitignored)
```

## Run output

Each run produces a directory `runs/{run_id}/` containing:

- `config.json` — full config snapshot
- `transactions.jsonl` — every completed transaction
- `revelations.jsonl` — quality revelations with cash adjustments
- `messages.jsonl` — inter-agent messages
- `market_state.jsonl` — daily market snapshots
- `events.jsonl` — discrete events (bankruptcy, factory builds, parse errors)
- `heartbeat.txt` — last updated every simulated day (for monitoring live runs)
- `agents/{name}/` — per-agent strategic calls, tactical calls, policy history
- `report.pdf` — generated at run completion

## Reproducibility

Two runs with the same `--seed` produce byte-identical transaction logs
(excluding wall-clock timestamps). Parallelism is applied to LLM calls only;
all market state mutations happen in deterministic alphabetical order.
