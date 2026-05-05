import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _required_artifacts(root: Path) -> List[Path]:
    return [
        root / "strategies.json",
        root / "data_manifest.json",
        root / "metrics.json",
        root / "critiques.json",
        root / "report.md",
        root / "llm_calls.jsonl",
    ]


def run_validation(root: Path) -> None:
    for p in _required_artifacts(root):
        _require(p.exists(), f"Missing required artifact: {p.name}")
    strategies = _load_json(root / "strategies.json")
    metrics: Dict[str, Dict] = _load_json(root / "metrics.json")
    critiques: Dict[str, Dict] = _load_json(root / "critiques.json")
    _load_json(root / "data_manifest.json")
    for st in strategies:
        sid = st["id"]
        spec_path = root / "specs" / f"{sid}.json"
        ledger_path = root / "ledgers" / f"{sid}.csv"
        _require(spec_path.exists(), f"Missing spec for {sid}")
        _require(ledger_path.exists(), f"Missing ledger for {sid}")
        spec = _load_json(spec_path)
        _require("explicit_ambiguities" in spec, f"No explicit_ambiguities for {sid}")
        _require(len(spec["explicit_ambiguities"]) >= 3, f"Need >=3 ambiguities for {sid}")
        for k in ["strategy_id", "instrument", "timeframe", "data_source", "entry_conditions", "exit_conditions", "position_sizing_rule", "stop_loss_rule", "take_profit_rule", "session_filters", "risk_controls", "explicit_ambiguities"]:
            _require(k in spec, f"Spec schema missing {k} for {sid}")
        ledger = pd.read_csv(ledger_path)
        _require(set(["strategy_id", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size", "pnl", "return_pct", "exit_reason"]).issubset(ledger.columns), f"Ledger columns invalid for {sid}")
        metric_total = float(metrics[sid]["total_return"])
        ledger_total = float(ledger["pnl"].sum()) if not ledger.empty else 0.0
        _require(abs(metric_total - ledger_total) < 1e-6, f"Ledger/metrics mismatch for {sid}")
    # Ensure LLM logs include formalisation and critique per strategy
    logs = [json.loads(line) for line in (root / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed_llm_stages = {
        "STRATEGIES_FORMALISED",
        "STRATEGIES_CRITIQUED",
        "PARAMETER_SENSITIVITY_INTERPRETATION",
        "ADVERSARIAL_SCENARIO_GENERATION",
    }
    for rec in logs:
        _require(rec["stage"] in allowed_llm_stages, f"Unexpected LLM stage in logs: {rec['stage']}")
    for st in strategies:
        sid = st["id"]
        _require(any(l["stage"] == "STRATEGIES_FORMALISED" and l["strategy_id"] == sid for l in logs), f"Missing formalization log for {sid}")
        _require(any(l["stage"] == "STRATEGIES_CRITIQUED" and l["strategy_id"] == sid for l in logs), f"Missing critique log for {sid}")
    _require(any("intrabar ordering assumption" in (root / "report.md").read_text(encoding="utf-8").lower() for _ in [0]), "Backtest assumption not documented")
    strategy_ids = {x["id"] for x in strategies}
    if "C" in strategy_ids:
        _require(critiques["C"]["risk_flag"] == "high_risk", "Strategy C must be flagged high risk")


if __name__ == "__main__":
    run_validation(Path(__file__).resolve().parent)
    print("Validation passed.")
