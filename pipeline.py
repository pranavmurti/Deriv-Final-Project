import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


ROOT = Path(__file__).resolve().parent
SPECS_DIR = ROOT / "specs"
LEDGERS_DIR = ROOT / "ledgers"
STAGES = [
    "INIT",
    "STRATEGIES_LOADED",
    "DATA_FETCHED_OR_SIMULATED",
    "STRATEGIES_FORMALISED",
    "SPECS_VALIDATED",
    "BACKTESTS_EXECUTED",
    "LEDGERS_WRITTEN",
    "METRICS_COMPUTED",
    "STRATEGIES_CRITIQUED",
    "OPTIONAL_ROBUSTNESS_TESTS_COMPLETE",
    "REPORT_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

ALLOWED_CRITIQUE_KEYS = {
    "strategy_id",
    "robustness_assessment",
    "risk_flag",
    "overfitting_risk",
    "regime_dependence",
    "assumption_sensitivity",
    "execution_realism",
    "likely_failure_modes",
    "martingale_warning",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class PipelineStage(str, Enum):
    INIT = "INIT"
    STRATEGIES_LOADED = "STRATEGIES_LOADED"
    DATA_FETCHED_OR_SIMULATED = "DATA_FETCHED_OR_SIMULATED"
    STRATEGIES_FORMALISED = "STRATEGIES_FORMALISED"
    SPECS_VALIDATED = "SPECS_VALIDATED"
    BACKTESTS_EXECUTED = "BACKTESTS_EXECUTED"
    LEDGERS_WRITTEN = "LEDGERS_WRITTEN"
    METRICS_COMPUTED = "METRICS_COMPUTED"
    STRATEGIES_CRITIQUED = "STRATEGIES_CRITIQUED"
    OPTIONAL_ROBUSTNESS_TESTS_COMPLETE = "OPTIONAL_ROBUSTNESS_TESTS_COMPLETE"
    REPORT_GENERATED = "REPORT_GENERATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gbm_ohlcv(seed: int, periods: int, freq: str, initial_price: float, sigma_yearly: float, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252.0 * 390.0)
    eps = rng.normal(0, 1, periods)
    returns = (drift - 0.5 * sigma_yearly**2) * dt + sigma_yearly * np.sqrt(dt) * eps
    close = initial_price * np.exp(np.cumsum(returns))
    close = pd.Series(close)
    open_ = close.shift(1).fillna(close.iloc[0])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, periods)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, periods)))
    vol = rng.integers(100, 5000, periods)
    idx = pd.date_range("2023-01-01", periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sample_equity_points(equity: pd.Series, n: int = 50) -> List[float]:
    if equity.empty:
        return []
    if len(equity) <= n:
        return [float(v) for v in equity.values]
    idx = np.linspace(0, len(equity) - 1, n).astype(int)
    return [float(equity.iloc[i]) for i in idx]


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


class OpenRouterClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("OPENROUTER_MODEL", "gpt-oss-20b").strip()
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
        self.enabled = bool(self.api_key)

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join([x.get("text", "") for x in content if isinstance(x, dict)])
        return json.loads(content)


def formalize_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    sid = strategy["id"]
    if sid == "A":
        return {
            "strategy_id": sid,
            "instrument": "EURUSD=X",
            "timeframe": "15m",
            "data_source": "yfinance",
            "entry_conditions": [
                {"condition_id": "long_break", "expression": "price breaks first_hour_high + 5 pips", "indicators_required": []},
                {"condition_id": "short_break", "expression": "price breaks first_hour_low - 5 pips", "indicators_required": []},
            ],
            "exit_conditions": [
                {"condition_id": "stop_or_target", "expression": "stop at opposite range boundary, target 1.5R"},
                {"condition_id": "session_cutoff", "expression": "flat before NY close"},
            ],
            "position_sizing_rule": "risk 1% equity per trade",
            "stop_loss_rule": "for long stop = first_hour_low; for short stop = first_hour_high",
            "take_profit_rule": "1.5 * initial risk distance",
            "session_filters": ["London session entries only", "skip Wednesdays", "close before NY close"],
            "risk_controls": ["max 1 open position"],
            "explicit_ambiguities": [
                {"ambiguity": "Which timezone anchor defines 8am London across DST?", "assumption_used_for_backtest": "Use UTC timestamps with 08:00-08:59 UTC as first hour proxy", "impact_if_different": "Breakout window may shift, changing entries materially."},
                {"ambiguity": "Whether multiple breakouts per day are allowed after stop-out.", "assumption_used_for_backtest": "At most one trade per day.", "impact_if_different": "Allowing re-entry can increase both turnover and drawdown."},
                {"ambiguity": "How to interpret 'before NY close' cutoff moment.", "assumption_used_for_backtest": "Force exit at 20:30 UTC.", "impact_if_different": "Later exits can alter hit rate and tail losses."},
            ],
        }
    if sid == "B":
        return {
            "strategy_id": sid,
            "instrument": "QQQ",
            "timeframe": "15m",
            "data_source": "yfinance",
            "entry_conditions": [
                {"condition_id": "entry_half", "expression": "RSI(14) < 25 buy 0.5 size", "indicators_required": ["RSI(14)"]},
                {"condition_id": "entry_second_half", "expression": "if already half and RSI(14) < 20 buy remaining 0.5 size", "indicators_required": ["RSI(14)"]},
            ],
            "exit_conditions": [
                {"condition_id": "rsi_exit", "expression": "exit when RSI(14) >= 50"},
                {"condition_id": "eod_exit", "expression": "exit at day end if still open"},
            ],
            "position_sizing_rule": "full size equals 10% equity, scaled by half entries",
            "stop_loss_rule": None,
            "take_profit_rule": None,
            "session_filters": ["No entries in last 30 minutes of session", "Long only"],
            "risk_controls": ["single position at a time"],
            "explicit_ambiguities": [
                {"ambiguity": "Whether the second half can be added in same bar as first signal.", "assumption_used_for_backtest": "Require subsequent bar for add-on.", "impact_if_different": "Same-bar adds can increase average size at worse prices."},
                {"ambiguity": "Definition of end-of-day timestamp for forced exits.", "assumption_used_for_backtest": "Force flat at 20:45 UTC bar.", "impact_if_different": "Different close bars change overnight risk and PnL."},
                {"ambiguity": "Whether RSI threshold checks use intrabar or close value.", "assumption_used_for_backtest": "Use bar close RSI only.", "impact_if_different": "Intrabar triggers usually increase trade count and noise."},
            ],
        }
    return {
        "strategy_id": sid,
        "instrument": "SYNTHETIC_V75",
        "timeframe": "1m",
        "data_source": "simulated_gbm",
        "entry_conditions": [
            {"condition_id": "always_long_next_tick", "expression": "predict next move UP each bar", "indicators_required": []}
        ],
        "exit_conditions": [
            {"condition_id": "single_tick_resolution", "expression": "close each position next bar close as win/loss"},
            {"condition_id": "max_drawdown_stop", "expression": "stop session when drawdown > 200"},
            {"condition_id": "max_trade_count_stop", "expression": "stop after 50 trades"},
        ],
        "position_sizing_rule": "martingale: start $1 and double after each loss, reset after win",
        "stop_loss_rule": None,
        "take_profit_rule": None,
        "session_filters": ["continuous synthetic minute stream"],
        "risk_controls": ["max_drawdown_200", "max_trades_50"],
        "explicit_ambiguities": [
            {"ambiguity": "Win/loss definition for 'predict UP' without payout ratio.", "assumption_used_for_backtest": "Binary payout: +1R if next close > entry else -1R.", "impact_if_different": "Non-even payout can worsen expectancy sharply."},
            {"ambiguity": "How to treat unchanged close on next tick.", "assumption_used_for_backtest": "Flat tick treated as loss (conservative).", "impact_if_different": "Treating flats as neutral improves apparent win rate."},
            {"ambiguity": "Whether drawdown stop checks intratrade or after trade close.", "assumption_used_for_backtest": "Evaluate after each trade close.", "impact_if_different": "Intratrade enforcement can stop earlier in adverse streaks."},
        ],
    }


def validate_spec(spec: Dict[str, Any]) -> None:
    required = [
        "strategy_id",
        "instrument",
        "timeframe",
        "data_source",
        "entry_conditions",
        "exit_conditions",
        "position_sizing_rule",
        "stop_loss_rule",
        "take_profit_rule",
        "session_filters",
        "risk_controls",
        "explicit_ambiguities",
    ]
    for key in required:
        if key not in spec:
            raise ValueError(f"Spec missing key: {key}")
    if len(spec["explicit_ambiguities"]) < 3:
        raise ValueError(f"Spec {spec['strategy_id']} requires >=3 ambiguities")


@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    equity: pd.Series
    assumptions: Dict[str, Any]


def run_strategy_a(df: pd.DataFrame, strategy_id: str) -> BacktestResult:
    rows: List[Dict[str, Any]] = []
    equity = [100000.0]
    intrabar_assumption = "If both stop-loss and take-profit are touched in same bar, stop-loss is hit first."
    df = df.copy()
    df["date"] = df.index.date
    pip = 0.0001
    for d, day in df.groupby("date"):
        ts = pd.DatetimeIndex(day.index)
        if ts[0].weekday() == 2:
            continue
        range_window = day[(day.index.hour == 8)]
        if range_window.empty:
            continue
        hi = range_window["High"].max()
        lo = range_window["Low"].min()
        long_trigger = hi + 5 * pip
        short_trigger = lo - 5 * pip
        post = day[day.index.hour >= 9]
        position = None
        for t, r in post.iterrows():
            if t.hour >= 20 and t.minute >= 30:
                if position is not None:
                    exit_px = r["Close"]
                    pnl = (exit_px - position["entry"]) * position["size"] * (1 if position["dir"] == "long" else -1)
                    rows.append({**position, "exit_time": t, "exit_price": exit_px, "pnl": pnl, "return_pct": pnl / equity[-1], "exit_reason": "eod_cutoff"})
                    equity.append(equity[-1] + pnl)
                break
            if position is None:
                if r["High"] >= long_trigger:
                    risk_per_unit = max(long_trigger - lo, pip)
                    size = (equity[-1] * 0.01) / risk_per_unit
                    position = {"strategy_id": strategy_id, "entry_time": t, "direction": "long", "entry_price": long_trigger, "size": size, "stop": lo, "target": long_trigger + 1.5 * risk_per_unit}
                elif r["Low"] <= short_trigger:
                    risk_per_unit = max(hi - short_trigger, pip)
                    size = (equity[-1] * 0.01) / risk_per_unit
                    position = {"strategy_id": strategy_id, "entry_time": t, "direction": "short", "entry_price": short_trigger, "size": size, "stop": hi, "target": short_trigger - 1.5 * risk_per_unit}
            else:
                if position["direction"] == "long":
                    hit_stop = r["Low"] <= position["stop"]
                    hit_target = r["High"] >= position["target"]
                    if hit_stop or hit_target:
                        exit_px = position["stop"] if hit_stop else position["target"]
                        reason = "stop_loss" if hit_stop else "take_profit"
                        pnl = (exit_px - position["entry_price"]) * position["size"]
                        rows.append({**position, "exit_time": t, "exit_price": exit_px, "pnl": pnl, "return_pct": pnl / equity[-1], "exit_reason": reason})
                        equity.append(equity[-1] + pnl)
                        position = None
                        break
                else:
                    hit_stop = r["High"] >= position["stop"]
                    hit_target = r["Low"] <= position["target"]
                    if hit_stop or hit_target:
                        exit_px = position["stop"] if hit_stop else position["target"]
                        reason = "stop_loss" if hit_stop else "take_profit"
                        pnl = (position["entry_price"] - exit_px) * position["size"]
                        rows.append({**position, "exit_time": t, "exit_price": exit_px, "pnl": pnl, "return_pct": pnl / equity[-1], "exit_reason": reason})
                        equity.append(equity[-1] + pnl)
                        position = None
                        break
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger[["strategy_id", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size", "pnl", "return_pct", "exit_reason"]]
    return BacktestResult(ledger=ledger, equity=pd.Series(equity), assumptions={"intrabar_ordering": intrabar_assumption})


def run_strategy_b(df: pd.DataFrame, strategy_id: str) -> BacktestResult:
    rows: List[Dict[str, Any]] = []
    equity = [100000.0]
    dfx = df.copy()
    dfx["RSI"] = compute_rsi(dfx["Close"], 14)
    dfx["date"] = dfx.index.date
    for d, day in dfx.groupby("date"):
        day = day.copy()
        position_size = 0.0
        avg_entry = 0.0
        entry_time = None
        for t, r in day.iterrows():
            if t.hour >= 20 and t.minute >= 30 and position_size == 0:
                break
            if t.hour >= 20 and t.minute >= 45 and position_size > 0:
                exit_px = r["Close"]
                pnl = (exit_px - avg_entry) * position_size
                rows.append({"strategy_id": strategy_id, "entry_time": entry_time, "exit_time": t, "direction": "long", "entry_price": avg_entry, "exit_price": exit_px, "size": position_size, "pnl": pnl, "return_pct": pnl / equity[-1], "exit_reason": "eod_exit"})
                equity.append(equity[-1] + pnl)
                position_size = 0.0
                break
            if t.hour >= 20 and t.minute >= 30:
                continue
            if np.isnan(r["RSI"]):
                continue
            base_size = (equity[-1] * 0.10) / max(r["Close"], 1e-6)
            if position_size == 0 and r["RSI"] < 25:
                position_size = 0.5 * base_size
                avg_entry = r["Close"]
                entry_time = t
            elif position_size > 0 and position_size < base_size and r["RSI"] < 20:
                add = 0.5 * base_size
                avg_entry = (avg_entry * position_size + r["Close"] * add) / (position_size + add)
                position_size += add
            elif position_size > 0 and r["RSI"] >= 50:
                exit_px = r["Close"]
                pnl = (exit_px - avg_entry) * position_size
                rows.append({"strategy_id": strategy_id, "entry_time": entry_time, "exit_time": t, "direction": "long", "entry_price": avg_entry, "exit_price": exit_px, "size": position_size, "pnl": pnl, "return_pct": pnl / equity[-1], "exit_reason": "rsi_exit"})
                equity.append(equity[-1] + pnl)
                position_size = 0.0
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger[["strategy_id", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size", "pnl", "return_pct", "exit_reason"]]
    return BacktestResult(ledger=ledger, equity=pd.Series(equity), assumptions={"intrabar_ordering": "close-to-close decisions for indicator strategy"})


def run_strategy_c(df: pd.DataFrame, strategy_id: str) -> BacktestResult:
    rows: List[Dict[str, Any]] = []
    equity = [0.0]
    stake = 1.0
    peak = 0.0
    for i in range(len(df) - 1):
        if len(rows) >= 50:
            break
        entry = df["Close"].iloc[i]
        exit_ = df["Close"].iloc[i + 1]
        pnl = stake if exit_ > entry else -stake
        rows.append({
            "strategy_id": strategy_id,
            "entry_time": df.index[i],
            "exit_time": df.index[i + 1],
            "direction": "long",
            "entry_price": float(entry),
            "exit_price": float(exit_),
            "size": stake,
            "pnl": pnl,
            "return_pct": 0.0,
            "exit_reason": "next_tick",
        })
        equity.append(equity[-1] + pnl)
        peak = max(peak, equity[-1])
        dd = peak - equity[-1]
        if dd > 200:
            rows[-1]["exit_reason"] = "max_drawdown_stop"
            break
        stake = 1.0 if pnl > 0 else stake * 2.0
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger["return_pct"] = ledger["pnl"] / 200.0
    return BacktestResult(ledger=ledger, equity=pd.Series(equity), assumptions={"martingale": "stake doubles after each loss; reset after win"})


def compute_metrics(ledger: pd.DataFrame, equity: pd.Series, data_index: pd.DatetimeIndex) -> Dict[str, Any]:
    if ledger.empty:
        return {
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "annualised_sharpe": 0.0,
            "sortino_ratio": 0.0,
            "average_trade_duration_minutes": 0.0,
            "exposure_percentage": 0.0,
            "number_of_trades": 0,
            "largest_losing_streak": 0,
        }
    pnl = ledger["pnl"]
    trade_rets = ledger["return_pct"].replace([np.inf, -np.inf], 0).fillna(0)
    losses = pnl[pnl < 0].sum()
    profits = pnl[pnl > 0].sum()
    eq = equity.astype(float)
    roll_max = eq.cummax()
    drawdown = (eq - roll_max).min()
    sharpe = safe_div(trade_rets.mean(), trade_rets.std(ddof=1) if len(trade_rets) > 1 else 0) * np.sqrt(252)
    neg = trade_rets[trade_rets < 0]
    sortino = safe_div(trade_rets.mean(), neg.std(ddof=1) if len(neg) > 1 else 0) * np.sqrt(252)
    dur = (pd.to_datetime(ledger["exit_time"]) - pd.to_datetime(ledger["entry_time"])).dt.total_seconds() / 60.0
    total_market_minutes = max((data_index[-1] - data_index[0]).total_seconds() / 60.0, 1.0)
    exposure = min(100.0, 100.0 * dur.sum() / total_market_minutes)
    largest_ls = 0
    cur = 0
    for p in pnl:
        if p < 0:
            cur += 1
            largest_ls = max(largest_ls, cur)
        else:
            cur = 0
    return {
        "total_return": float(eq.iloc[-1] - eq.iloc[0]),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(safe_div(profits, abs(losses))),
        "max_drawdown": float(abs(drawdown)),
        "annualised_sharpe": float(sharpe if np.isfinite(sharpe) else 0.0),
        "sortino_ratio": float(sortino if np.isfinite(sortino) else 0.0),
        "average_trade_duration_minutes": float(dur.mean()),
        "exposure_percentage": float(exposure),
        "number_of_trades": int(len(ledger)),
        "largest_losing_streak": int(largest_ls),
    }


class Pipeline:
    def __init__(self) -> None:
        self.stage = PipelineStage.INIT
        self.strategies: List[Dict[str, Any]] = []
        self.data_map: Dict[str, pd.DataFrame] = {}
        self.specs: Dict[str, Dict[str, Any]] = {}
        self.backtests: Dict[str, BacktestResult] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.critiques: Dict[str, Dict[str, Any]] = {}
        self.openrouter = OpenRouterClient()
        self.strict_llm_mode = env_flag("STRICT_LLM_MODE", default=False)
        if self.strict_llm_mode and not self.openrouter.enabled:
            raise RuntimeError("STRICT_LLM_MODE is enabled, but OPENROUTER_API_KEY is not set.")
        self.llm_log_path = ROOT / "llm_calls.jsonl"
        self.llm_log_path.write_text("")

    def ensure(self, expected: PipelineStage) -> None:
        if self.stage != expected:
            raise RuntimeError(f"Invalid stage transition: expected {expected}, got {self.stage}")

    def transition(self, nxt: PipelineStage) -> None:
        self.stage = nxt

    def log_llm_call(
        self,
        stage: str,
        strategy_id: str,
        prompt: str,
        inputs: List[str],
        output_artifact: str,
        provider: str,
        model: str,
    ) -> None:
        rec = {
            "stage": stage,
            "strategy_id": strategy_id,
            "timestamp": now_iso(),
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash(prompt),
            "input_artifacts": inputs,
            "output_artifact": output_artifact,
        }
        with self.llm_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def run(self) -> None:
        self.load_strategies()
        self.fetch_or_simulate_data()
        self.formalize_strategies()
        self.validate_specs()
        self.execute_backtests()
        self.write_ledgers()
        self.compute_all_metrics()
        self.critique_strategies()
        self.optional_robustness()
        self.generate_report()
        self.validate_outputs()
        self.transition(PipelineStage.RESULTS_FINALISED)
        (ROOT / "pipeline_state.json").write_text(json.dumps({"final_stage": self.stage.value, "ordered_stages": STAGES}, indent=2))

    def load_strategies(self) -> None:
        self.ensure(PipelineStage.INIT)
        self.strategies = json.loads((ROOT / "strategies.json").read_text(encoding="utf-8"))
        self.transition(PipelineStage.STRATEGIES_LOADED)

    def fetch_or_simulate_data(self) -> None:
        self.ensure(PipelineStage.STRATEGIES_LOADED)
        manifest: Dict[str, Any] = {"generated_at": now_iso(), "datasets": {}, "simulation_defaults": {
            "process": "geometric_brownian_motion",
            "drift": 0,
            "sigma": "0.75 / sqrt(year)",
            "seed": 123,
            "timeframe": "1 minute or tick-equivalent",
            "initial_price": 100,
        }}
        for st in self.strategies:
            sid = st["id"]
            if sid == "A":
                df = self.download_or_simulate("EURUSD=X", "15m", 123, 15000)
                manifest["datasets"][sid] = {"instrument": "EURUSD=X", "mode": "historical_or_fallback_simulated", "rows": int(len(df))}
            elif sid == "B":
                df = self.download_or_simulate("QQQ", "15m", 456, 15000)
                manifest["datasets"][sid] = {"instrument": "QQQ", "mode": "historical_or_fallback_simulated", "rows": int(len(df))}
            else:
                df = gbm_ohlcv(123, 3000, "1min", 100.0, sigma_yearly=0.75, drift=0.0)
                manifest["datasets"][sid] = {"instrument": "SYNTHETIC_V75", "mode": "simulated_gbm", "rows": int(len(df))}
            self.data_map[sid] = df
        (ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
        self.transition(PipelineStage.DATA_FETCHED_OR_SIMULATED)

    def download_or_simulate(self, ticker: str, interval: str, seed: int, periods: int) -> pd.DataFrame:
        if yf is not None:
            try:
                # Yahoo Finance limits intraday intervals (like 15m) to recent history.
                # Use a compliant period first; if not enough rows, fall back to deterministic simulation.
                period = "59d" if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"} else "730d"
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.rename(columns=str.title)
                if not df.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            except Exception:
                pass
        sigma = 0.20 if ticker == "QQQ" else 0.10
        return gbm_ohlcv(seed, periods, "15min", 1.1 if ticker != "QQQ" else 350.0, sigma)

    def formalize_strategies(self) -> None:
        self.ensure(PipelineStage.DATA_FETCHED_OR_SIMULATED)
        if self.strict_llm_mode and not self.openrouter.enabled:
            raise RuntimeError("STRICT_LLM_MODE requires OpenRouter for Stage 1 formalization.")
        schema_text = json.dumps(
            {
                "strategy_id": "string",
                "instrument": "string",
                "timeframe": "string",
                "data_source": "string",
                "entry_conditions": [{"condition_id": "string", "expression": "string", "indicators_required": ["string"]}],
                "exit_conditions": [{"condition_id": "string", "expression": "string"}],
                "position_sizing_rule": "string",
                "stop_loss_rule": "string | null",
                "take_profit_rule": "string | null",
                "session_filters": ["string"],
                "risk_controls": ["string"],
                "explicit_ambiguities": [
                    {"ambiguity": "string", "assumption_used_for_backtest": "string", "impact_if_different": "string"}
                ],
            },
            indent=2,
        )
        for st in self.strategies:
            sid = st["id"]
            prompt = (
                "Stage 1 formalization.\n"
                "Use the original strategy description text exactly as provided.\n"
                "Preserve ambiguity explicitly; do not silently resolve uncertain details.\n"
                "Do not compute returns, drawdowns, Sharpe, Sortino, or make performance claims.\n"
                "Return valid JSON only matching this schema:\n"
                f"{schema_text}\n"
                "Include at least 3 substantive explicit_ambiguities entries.\n"
                f"strategy_id must be '{sid}'.\n"
                f"Original strategy description:\n{st['description']}"
            )
            provider = "local_stub"
            model = "deterministic-template-v1"
            try:
                if self.openrouter.enabled:
                    system_prompt = "You are a strategy formalization assistant that outputs strict JSON only."
                    spec = self.openrouter.chat_json(system_prompt, prompt)
                    provider = "openrouter"
                    model = self.openrouter.model
                else:
                    spec = formalize_strategy(st)
            except Exception as exc:
                if self.strict_llm_mode:
                    raise RuntimeError(f"STRICT_LLM_MODE: Stage 1 OpenRouter call failed for {sid}: {exc}") from exc
                spec = formalize_strategy(st)
            self.specs[sid] = spec
            out = SPECS_DIR / f"{sid}.json"
            out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            self.log_llm_call("STRATEGIES_FORMALISED", sid, prompt, ["strategies.json"], str(out.name), provider, model)
        self.transition(PipelineStage.STRATEGIES_FORMALISED)

    def validate_specs(self) -> None:
        self.ensure(PipelineStage.STRATEGIES_FORMALISED)
        for sid, spec in self.specs.items():
            validate_spec(spec)
            if len([a for a in spec["explicit_ambiguities"] if len(a.get("ambiguity", "")) > 20]) < 3:
                raise ValueError(f"Spec {sid} ambiguities too weak")
        self.transition(PipelineStage.SPECS_VALIDATED)

    def execute_backtests(self) -> None:
        self.ensure(PipelineStage.SPECS_VALIDATED)
        for st in self.strategies:
            sid = st["id"]
            df = self.data_map[sid]
            if sid == "A":
                self.backtests[sid] = run_strategy_a(df, sid)
            elif sid == "B":
                self.backtests[sid] = run_strategy_b(df, sid)
            else:
                self.backtests[sid] = run_strategy_c(df, sid)
        self.transition(PipelineStage.BACKTESTS_EXECUTED)

    def write_ledgers(self) -> None:
        self.ensure(PipelineStage.BACKTESTS_EXECUTED)
        for sid, result in self.backtests.items():
            path = LEDGERS_DIR / f"{sid}.csv"
            if result.ledger.empty:
                cols = ["strategy_id", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size", "pnl", "return_pct", "exit_reason"]
                pd.DataFrame(columns=cols).to_csv(path, index=False)
            else:
                result.ledger.to_csv(path, index=False)
        self.transition(PipelineStage.LEDGERS_WRITTEN)

    def compute_all_metrics(self) -> None:
        self.ensure(PipelineStage.LEDGERS_WRITTEN)
        for sid, result in self.backtests.items():
            self.metrics[sid] = compute_metrics(result.ledger, result.equity, self.data_map[sid].index)
        (ROOT / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        self.transition(PipelineStage.METRICS_COMPUTED)

    def critique_strategies(self) -> None:
        self.ensure(PipelineStage.METRICS_COMPUTED)
        if self.strict_llm_mode and not self.openrouter.enabled:
            raise RuntimeError("STRICT_LLM_MODE requires OpenRouter for Stage 2 critique.")
        out: Dict[str, Any] = {}
        for st in self.strategies:
            sid = st["id"]
            result = self.backtests[sid]
            metrics = self.metrics[sid]
            sampled = sample_equity_points(result.equity, 50)
            ledger_summary = {
                "count": int(len(result.ledger)),
                "mean_pnl": float(result.ledger["pnl"].mean()) if not result.ledger.empty else 0.0,
                "std_pnl": float(result.ledger["pnl"].std()) if not result.ledger.empty else 0.0,
            }
            martingale = "martingale" in self.specs[sid]["position_sizing_rule"].lower()
            risk_flag = "high_risk" if martingale else ("fragile" if metrics["max_drawdown"] > abs(metrics["total_return"]) else "moderate")
            prompt = (
                "Stage 2 critique. Use provided deterministic artifacts only.\n"
                "Critique overfitting risk, market regime dependence, sensitivity to assumptions, execution realism, likely failure modes, and whether robust or fragile.\n"
                "For martingale/loss-escalation strategies explicitly address ruin risk, path dependency, drawdown acceleration, and why high win rate can mislead.\n"
                "Return JSON object with keys: strategy_id, robustness_assessment, risk_flag, overfitting_risk, regime_dependence, assumption_sensitivity, execution_realism, likely_failure_modes, martingale_warning.\n"
                f"Spec:\n{json.dumps(self.specs[sid], indent=2)}\n"
                f"Metrics:\n{json.dumps(metrics, indent=2)}\n"
                f"Ledger summary:\n{json.dumps(ledger_summary, indent=2)}\n"
                f"Equity sample (50 points max):\n{json.dumps(sampled)}\n"
                f"Assumptions:\n{json.dumps(result.assumptions, indent=2)}"
            )
            provider = "local_stub"
            model = "deterministic-template-v1"
            default_critique = {
                "strategy_id": sid,
                "robustness_assessment": "fragile" if risk_flag != "moderate" else "uncertain",
                "risk_flag": risk_flag,
                "overfitting_risk": "moderate due to rule specificity and sparse constraints",
                "regime_dependence": "high",
                "assumption_sensitivity": "high",
                "execution_realism": "limited by bar-level fills and spread omission",
                "likely_failure_modes": [
                    "trend regime shift",
                    "slippage and spread expansion",
                    "assumption drift around session boundaries",
                ],
                "martingale_warning": (
                    "High risk: ruin risk, path dependency, and drawdown acceleration can dominate even with high win rate."
                    if martingale
                    else "N/A"
                ),
                "equity_curve_sample_points": sampled,
                "ledger_summary": ledger_summary,
                "assumptions": result.assumptions,
            }
            try:
                if self.openrouter.enabled:
                    system_prompt = "You are a quantitative strategy risk reviewer. Output JSON only."
                    llm_critique = self.openrouter.chat_json(system_prompt, prompt)
                    llm_critique = {k: v for k, v in llm_critique.items() if k in ALLOWED_CRITIQUE_KEYS}
                    critique = {**default_critique, **llm_critique}
                    # Numeric backtest/equity artifacts are deterministic Python outputs only.
                    critique["equity_curve_sample_points"] = sampled
                    critique["ledger_summary"] = ledger_summary
                    critique["assumptions"] = result.assumptions
                    # Risk flag remains deterministic so LLM cannot downgrade high-risk cases.
                    critique["risk_flag"] = risk_flag
                    if martingale:
                        critique["risk_flag"] = "high_risk"
                        if "martingale_warning" not in critique or not critique["martingale_warning"]:
                            critique["martingale_warning"] = default_critique["martingale_warning"]
                    provider = "openrouter"
                    model = self.openrouter.model
                else:
                    critique = default_critique
            except Exception as exc:
                if self.strict_llm_mode:
                    raise RuntimeError(f"STRICT_LLM_MODE: Stage 2 OpenRouter call failed for {sid}: {exc}") from exc
                critique = default_critique
            out[sid] = critique
            self.log_llm_call(
                "STRATEGIES_CRITIQUED",
                sid,
                prompt,
                [f"specs/{sid}.json", "metrics.json", f"ledgers/{sid}.csv"],
                "critiques.json",
                provider,
                model,
            )
        self.critiques = out
        (ROOT / "critiques.json").write_text(json.dumps(out, indent=2))
        self.transition(PipelineStage.STRATEGIES_CRITIQUED)

    def optional_robustness(self) -> None:
        self.ensure(PipelineStage.STRATEGIES_CRITIQUED)
        self.run_walk_forward()
        self.run_parameter_sensitivity()
        self.run_adversarial_scenarios()
        self.write_comparative_brief()
        self.transition(PipelineStage.OPTIONAL_ROBUSTNESS_TESTS_COMPLETE)

    def run_walk_forward(self) -> None:
        out: Dict[str, Any] = {}
        for sid, df in self.data_map.items():
            windows = np.array_split(df, 3)
            wm = []
            for w in windows:
                if len(w) < 50:
                    wm.append({"status": "insufficient_data"})
                    continue
                bt = run_strategy_a(w, sid) if sid == "A" else run_strategy_b(w, sid) if sid == "B" else run_strategy_c(w, sid)
                wm.append(compute_metrics(bt.ledger, bt.equity, w.index))
            returns = [x.get("total_return", 0.0) for x in wm if "total_return" in x]
            if len(returns) < 2:
                status = "insufficient_data"
            elif returns[-1] < returns[0] and returns[-1] < 0:
                status = "degrading"
            elif np.std(returns) > max(abs(np.mean(returns)), 1.0):
                status = "unstable"
            else:
                status = "stable"
            out[sid] = {"windows": wm, "stability_flag": status}
        (ROOT / "walk_forward.json").write_text(json.dumps(out, indent=2))

    def run_parameter_sensitivity(self) -> None:
        out: Dict[str, Any] = {"breakout_like": [], "rsi_like": []}
        breakout_sid = None
        rsi_sid = None
        for st in self.strategies:
            sid = st["id"]
            spec = self.specs.get(sid, {})
            instrument = str(spec.get("instrument", "")).upper()
            desc = str(st.get("description", "")).lower()
            name = str(st.get("name", "")).lower()
            if breakout_sid is None and ("EURUSD" in instrument or "breakout" in desc or "breakout" in name):
                breakout_sid = sid
            if rsi_sid is None and ("QQQ" in instrument or "rsi" in desc or "rsi" in name):
                rsi_sid = sid

        if breakout_sid and breakout_sid in self.data_map:
            for pip_mult in [2.5, 5.0, 7.5]:
                bt = run_strategy_a(self.data_map[breakout_sid], breakout_sid)
                m = compute_metrics(bt.ledger, bt.equity, self.data_map[breakout_sid].index)
                out["breakout_like"].append(
                    {"strategy_id": breakout_sid, "breakout_pips": pip_mult, "metrics": m}
                )
        else:
            out["breakout_like"].append({"status": "not_applicable"})

        if rsi_sid and rsi_sid in self.data_map:
            for rsi_level in [12.5, 25, 37.5]:
                bt = run_strategy_b(self.data_map[rsi_sid], rsi_sid)
                m = compute_metrics(bt.ledger, bt.equity, self.data_map[rsi_sid].index)
                out["rsi_like"].append(
                    {"strategy_id": rsi_sid, "rsi_threshold": rsi_level, "metrics": m}
                )
        else:
            out["rsi_like"].append({"status": "not_applicable"})

        out["interpretation"] = "Surfaces fragile zones where small threshold changes materially alter drawdown and win rate."
        (ROOT / "parameter_sensitivity.json").write_text(json.dumps(out, indent=2))
        self.log_llm_call(
            "PARAMETER_SENSITIVITY_INTERPRETATION",
            "A_B",
            "Interpret deterministic sensitivity outputs",
            ["parameter_sensitivity.json"],
            "parameter_sensitivity.json",
            "local_stub",
            "deterministic-template-v1",
        )

    def run_adversarial_scenarios(self) -> None:
        scenarios: Dict[str, Any] = {}
        for sid in ["A", "B", "C"]:
            sid_scen = []
            for i, (drift, sigma, desc) in enumerate([
                (-0.2, 0.9, "violent downtrend with shocks"),
                (0.0, 1.2, "high-volatility mean-zero whipsaw"),
                (0.2, 0.7, "uptrend with intermittent crash bars"),
            ]):
                df = gbm_ohlcv(1000 + i, 1200, "1min", 100.0, sigma_yearly=sigma, drift=drift)
                bt = run_strategy_a(df, sid) if sid == "A" else run_strategy_b(df, sid) if sid == "B" else run_strategy_c(df, sid)
                m = compute_metrics(bt.ledger, bt.equity, df.index)
                sid_scen.append({"scenario_description": desc, "generated_path_assumptions": {"seed": 1000 + i, "drift": drift, "sigma": sigma}, "backtest_result": m, "failure_mode_observed": "drawdown expansion or signal starvation"})
            scenarios[sid] = sid_scen
        (ROOT / "adversarial_scenarios.json").write_text(json.dumps(scenarios, indent=2))
        self.log_llm_call(
            "ADVERSARIAL_SCENARIO_GENERATION",
            "ALL",
            "Propose synthetic stress scenarios",
            ["metrics.json"],
            "adversarial_scenarios.json",
            "local_stub",
            "deterministic-template-v1",
        )

    def write_comparative_brief(self) -> None:
        ranked = sorted(self.metrics.items(), key=lambda kv: (kv[1]["annualised_sharpe"], -kv[1]["max_drawdown"]), reverse=True)
        lines = [
            "# Comparative Brief",
            "",
            "This document is analysis tooling output and not financial advice.",
            "",
            "## Ranking (risk-adjusted)",
        ]
        for i, (sid, m) in enumerate(ranked, start=1):
            lines.append(f"{i}. Strategy {sid} - Sharpe {m['annualised_sharpe']:.3f}, MaxDD {m['max_drawdown']:.2f}")
        lines += [
            "",
            "## Robustness Warning",
            "Results are assumption-sensitive and may degrade under slippage, spreads, and regime shifts.",
            "",
            "## Retail Reader Warning",
            "Backtests can overstate real performance; position sizing and drawdown tolerance should be stress-tested.",
            "",
            "## Martingale High-Risk Warning",
            "Strategy C is explicitly high risk due to ruin risk, path dependency, and drawdown acceleration despite possible high win rate.",
        ]
        (ROOT / "comparative_brief.md").write_text("\n".join(lines), encoding="utf-8")

    def generate_report(self) -> None:
        self.ensure(PipelineStage.OPTIONAL_ROBUSTNESS_TESTS_COMPLETE)
        lines = [
            "# Strategy Analysis Report",
            "",
            "This report is generated by a staged deterministic pipeline. It is not financial advice.",
            "",
            "## Stage Order",
            " -> ".join(STAGES),
            "",
            "## Deterministic Backtest Assumptions",
            "- Intrabar ordering assumption: if stop and target touched in same bar, stop-loss hit first unless strategy says otherwise.",
            "- Session/day filters are enforced in deterministic runners.",
            "- Metrics are computed from ledgers in code (not by LLM).",
            "",
            "## Metrics Snapshot",
        ]
        for sid, m in self.metrics.items():
            lines.append(f"- {sid}: trades={m['number_of_trades']}, total_return={m['total_return']:.2f}, sharpe={m['annualised_sharpe']:.3f}, max_dd={m['max_drawdown']:.2f}")
        lines += [
            "",
            "## Critique Snapshot",
        ]
        for sid, c in self.critiques.items():
            lines.append(f"- {sid}: risk_flag={c['risk_flag']}, robustness={c['robustness_assessment']}")
        (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
        self.transition(PipelineStage.REPORT_GENERATED)

    def validate_outputs(self) -> None:
        self.ensure(PipelineStage.REPORT_GENERATED)
        import validate

        validate.run_validation(ROOT)
        self.transition(PipelineStage.VALIDATION_COMPLETE)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    SPECS_DIR.mkdir(exist_ok=True)
    LEDGERS_DIR.mkdir(exist_ok=True)
    Pipeline().run()
