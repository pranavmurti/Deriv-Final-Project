# Replayable Strategy Pipeline

Run from a clean checkout:

1. `python3 -m pip install -r requirements.txt`
2. Optional OpenRouter setup for real LLM calls in Stage 1 and Stage 2:
   - `export OPENROUTER_API_KEY="your_key_here"`
   - `export OPENROUTER_MODEL="gpt-oss-20b"`
   - `export STRICT_LLM_MODE="true"` (optional: fail run if any Stage 1/2 LLM call falls back)
3. `python3 pipeline.py`
4. `python3 validate.py`

If `OPENROUTER_API_KEY` is unset, the pipeline falls back to deterministic local templates for formalization and critique.
If `STRICT_LLM_MODE=true`, fallback is disabled for Stage 1 and Stage 2 and the run fails fast on any OpenRouter failure.

Or:

- `make run`
- `make validate`

Outputs include `specs/`, `ledgers/`, `metrics.json`, `critiques.json`, `report.md`, optional robustness artifacts, and `llm_calls.jsonl`.
