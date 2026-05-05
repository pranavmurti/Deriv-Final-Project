# Replayable Strategy Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic staged pipeline that formalizes informal strategy text, backtests reproducibly, computes metrics in code, critiques robustness, and validates all required artifacts.

**Architecture:** A single orchestrator script (`pipeline.py`) enforces stage order and writes artifacts after each stage. Deterministic strategy runners produce ledgers and metrics from historical/synthetic data. LLM-facing steps are isolated to formalization and critique stubs with call logging in `llm_calls.jsonl`.

**Tech Stack:** Python, pandas, numpy, yfinance, json/csv stdlib.

---

### Task 1: Pipeline skeleton and staged state machine
- [ ] Define stage enum and transition checks.
- [ ] Load `strategies.json` and write initial assumptions artifact.
- [ ] Implement stage guards preventing report generation before validation gates.

### Task 2: Deterministic data layer
- [ ] Fetch market data for known instruments via yfinance.
- [ ] Build deterministic GBM simulator for synthetic strategy data.
- [ ] Emit `data_manifest.json` with source metadata and simulation parameters.

### Task 3: Formalization and schema validation
- [ ] Implement Stage 1 LLM call wrapper with artifact logging (`llm_calls.jsonl`).
- [ ] Produce `specs/{strategy_id}.json` with required schema and ambiguities.
- [ ] Validate each spec and enforce >=3 substantive ambiguities.

### Task 4: Backtest engine and ledgers
- [ ] Implement strategy-specific deterministic execution with shared assumptions.
- [ ] Handle partial entries, stop/target, session/day filters, martingale, max DD/trade-count stops.
- [ ] Write per-trade ledgers to `ledgers/{strategy_id}.csv`.

### Task 5: Deterministic metrics and robustness outputs
- [ ] Compute metrics from ledgers/equity (no LLM math) and write `metrics.json`.
- [ ] Implement walk-forward windows and parameter sensitivity for A/B.
- [ ] Run Stage 2 critique calls and save `critiques.json`.

### Task 6: Reporting and validation
- [ ] Generate `report.md` and optional `comparative_brief.md`.
- [ ] Implement `validate.py` checking artifacts, schema, reconciliation, and high-risk martingale flag.
- [ ] Run pipeline + validator from clean outputs and fix any failures.
