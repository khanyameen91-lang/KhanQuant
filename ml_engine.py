"""
ml_engine.py — Machine Learning Feedback Loop

Learns from historical trade performance to:
  1. Rank strategies by risk-adjusted returns (Sharpe ratio)
  2. Adjust scoring weights dynamically
  3. Disable consistently underperforming strategies
  4. Train a RandomForest classifier to predict trade outcomes
  5. Track performance metrics: win rate, profit factor, Sharpe, Sortino

Retrains nightly when bot is idle. Uses scikit-learn (free, local).
All data comes from the trade journal (executor.py writes every close).
"""

import json
import math
import time
from datetime import datetime, date, timedelta
from pathlib import Path

LOG_DIR         = Path("logs")
JOURNAL_FILE    = LOG_DIR / "trade_journal.json"
ML_WEIGHTS_FILE = LOG_DIR / "ml_weights.json"
ML_STATS_FILE   = LOG_DIR / "ml_stats.json"
ML_MODEL_FILE   = LOG_DIR / "ml_model.pkl"

MIN_TRADES_TO_TRAIN = 50  # Need at least 50 trades before training (walk-forward valid)

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. ML features limited.")

# Regime types for one-hot encoding
_REGIME_TYPES = [
    "bull_trend", "bear_trend", "high_volatility", "low_volatility",
    "range_bound", "fomc_day", "cpi_day", "opex_day", "triple_witching",
    "gap_and_go", "reversal_day", "vol_expansion", "vol_compression", "unknown",
]


# ── Journal Loading ────────────────────────────────────────────────────────────

def _load_journal() -> list:
    if not JOURNAL_FILE.exists():
        return []
    try:
        return json.loads(JOURNAL_FILE.read_text())
    except Exception:
        return []


# ── Performance Metrics ────────────────────────────────────────────────────────

def _normalize_strategy_key(display_name: str) -> str:
    """
    Map a trade's free-text strategy display name (e.g. "Bull Put Credit
    Spread", "0DTE C Scalp (Bullish)", "Iron Condor") to the canonical,
    underscored key used everywhere else in the risk/ML pipeline
    (credit_spread, 0dte_scalp, iron_condor, spread, strangle, long_option).

    This used to be inlined separately in two places: here (implicitly, via
    a raw `.find(canonical_key)` substring search against the display name)
    and in compute_all_strategy_stats (a real if/elif classifier against the
    display name). The two never agreed — "iron condor".find("iron_condor")
    is -1 because of the underscore, so compute_strategy_stats("iron_condor")
    always returned zero trades no matter how many were actually traded.
    Same for credit_spread, 0dte_scalp, and long_option; and "spread" as a
    canonical key also happened to substring-match credit spread trades,
    so a losing credit-spread streak could end up disabling debit spreads
    instead. Both call sites now go through this single classifier so a
    trade is always bucketed the same way it's later looked up.
    """
    s = (display_name or "").lower()
    if "condor" in s:
        return "iron_condor"
    if "credit" in s or "bull put" in s or "bear call" in s:
        return "credit_spread"
    if "spread" in s:
        return "spread"
    if "0dte" in s or "scalp" in s:
        return "0dte_scalp"
    if "strangle" in s:
        return "strangle"
    if "long" in s:
        return "long_option"
    return "unknown"


def compute_strategy_stats(trades: list, strategy: str = None) -> dict:
    """
    Compute comprehensive performance statistics for a set of trades.
    Optional: filter by strategy type (a canonical key like "credit_spread",
    matched via _normalize_strategy_key — not a raw substring search).
    """
    filtered = [t for t in trades if t.get("pnl") is not None]
    if strategy:
        filtered = [t for t in filtered if _normalize_strategy_key(t.get("strategy", "")) == strategy.lower()]

    if not filtered:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0, "sharpe": 0,
                "avg_pnl": 0, "total_pnl": 0, "max_drawdown": 0}

    pnls = [t["pnl"] for t in filtered]
    wins  = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate  = len(wins) / len(pnls)
    avg_pnl   = total_pnl / len(pnls)

    # Profit factor
    gross_profit = sum(wins)
    gross_loss   = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (daily, annualized roughly)
    if NUMPY_AVAILABLE and len(pnls) >= 5:
        pnl_array = np.array(pnls)
        avg = np.mean(pnl_array)
        std = np.std(pnl_array)
        sharpe = (avg / std * math.sqrt(252)) if std > 0 else 0.0

        # Sortino ratio (downside deviation only)
        neg_returns = pnl_array[pnl_array < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 0.001
        sortino = (avg / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

        # Max drawdown
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    else:
        sharpe   = 0.0
        sortino  = 0.0
        max_drawdown = 0.0

    return {
        "trades":         len(filtered),
        "win_rate":       round(win_rate * 100, 1),
        "wins":           len(wins),
        "losses":         len(losses),
        "profit_factor":  round(profit_factor, 2),
        "sharpe":         round(sharpe, 2),
        "sortino":        round(sortino, 2),
        "avg_pnl":        round(avg_pnl, 2),
        "total_pnl":      round(total_pnl, 2),
        "max_drawdown":   round(max_drawdown, 2),
        "avg_win":        round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss":       round(sum(losses) / len(losses), 2) if losses else 0,
    }


def compute_all_strategy_stats(trades: list) -> dict:
    """Compute performance stats broken down by strategy type."""
    strategies = {_normalize_strategy_key(t.get("strategy", "unknown")) for t in trades}
    strategies.discard("unknown")

    result = {"overall": compute_strategy_stats(trades)}
    for strat in strategies:
        result[strat] = compute_strategy_stats(trades, strat)
    return result


# ── Weight Adjustment ─────────────────────────────────────────────────────────

def _compute_optimal_weights(strategy_stats: dict) -> dict:
    """
    Adjust scoring weights based on what factors predicted wins.
    This is a simple heuristic: strategies with higher Sharpe get more
    weight in the composite score toward their typical score profile.
    """
    # We adjust the scoring component weights based on overall performance
    # This is stored and used by scoring.py
    overall = strategy_stats.get("overall", {})

    win_rate = overall.get("win_rate", 50) / 100.0
    sharpe   = overall.get("sharpe", 0)

    # If win rate is very high, technical/momentum signals work → up tech weight
    # If win rate is low, reduce overconfidence → reduce all deviations
    from scoring import BASE_WEIGHTS
    weights = dict(BASE_WEIGHTS)

    if sharpe > 1.5:
        # System is performing well — keep weights as-is
        pass
    elif sharpe > 0.5:
        # Decent — slight adjustment
        weights["technical"] = min(0.45, weights["technical"] * 1.1)
        weights["flow"]      = min(0.35, weights["flow"] * 1.05)
    elif sharpe < 0:
        # Underperforming — increase regime weight, reduce technical
        weights["regime"]    = min(0.20, weights["regime"] * 1.5)
        weights["technical"] = max(0.20, weights["technical"] * 0.85)

    # Normalize
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


def _should_disable_strategy(stats: dict) -> list:
    """
    Return list of strategies that should be disabled based on performance.
    Criteria (all-time stats, not a rolling window): at least 10 trades,
    Sharpe < 0, AND win rate < 35%.
    """
    disabled = []
    for strat, s in stats.items():
        if strat == "overall":
            continue
        if s["trades"] >= 10 and s["sharpe"] < 0 and s["win_rate"] < 35:
            disabled.append(strat)
            print(f"ML: Disabling {strat} — Sharpe {s['sharpe']:.2f}, "
                  f"Win rate {s['win_rate']:.0f}%, {s['trades']} trades")
    return disabled


# ── RandomForest Trade Predictor ──────────────────────────────────────────────

def _extract_features(trade: dict) -> list:
    """
    Extract numerical features from a trade record for ML training.
    Features: 10-factor sub-scores + regime one-hot (14 types) + trade params.
    All sub-scores must be logged at trade entry for this to work.
    """
    # Regime one-hot encoding
    regime = (trade.get("regime_type") or "unknown").lower().replace(" ", "_")
    regime_onehot = [1.0 if r == regime else 0.0 for r in _REGIME_TYPES]

    base_features = [
        float(trade.get("confidence", 50)),
        float(trade.get("technical_score", 50)),
        float(trade.get("flow_score", 50)),
        float(trade.get("sentiment_score", 50)),
        float(trade.get("volatility_score", 50)),
        float(trade.get("regime_score", 50)),
        float(trade.get("institutional_score", 50)),
        # Phase 1 new scores
        float(trade.get("ev_score", 50)),
        float(trade.get("liquidity_score", 50)),
        float(trade.get("rs_score", 50)),
        # Trade params
        float(trade.get("iv_rank", 50)),
        float(trade.get("dte", 14)),
        1.0 if trade.get("direction") == "bullish" else (
            0.0 if trade.get("direction") == "bearish" else 0.5
        ),
        float(trade.get("max_loss", 100)),
        # Time features
        float(trade.get("hour_of_day", 10)),   # 9-16 EST
        1.0 if trade.get("day_of_week", 2) in (0, 4) else 0.0,  # Mon/Fri flag
    ]
    return base_features + regime_onehot


_FEATURE_NAMES = [
    "confidence", "technical", "flow", "sentiment", "volatility",
    "regime_score", "institutional", "ev_score", "liquidity_score", "rs_score",
    "iv_rank", "dte", "direction", "max_loss", "hour_of_day", "mon_fri_flag",
] + [f"regime_{r}" for r in _REGIME_TYPES]


def train_model(trades: list) -> dict:
    """
    Train a RandomForest classifier on trade history.
    Features: all sub-scores + trade parameters.
    Label: 1 = winner, 0 = loser.
    Returns accuracy metrics.
    """
    if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
        return {"status": "sklearn_not_available"}

    # Filter to trades that have sub-scores logged
    scored_trades = [t for t in trades if t.get("confidence") is not None]
    if len(scored_trades) < MIN_TRADES_TO_TRAIN:
        return {
            "status": "insufficient_data",
            "trades_available": len(scored_trades),
            "trades_needed": MIN_TRADES_TO_TRAIN,
        }

    X = np.array([_extract_features(t) for t in scored_trades])
    y = np.array([1 if t.get("pnl", 0) > 0 else 0 for t in scored_trades])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Walk-forward cross-validation (respects temporal order — no lookahead bias)
    wf_cv_acc = 0.0
    wf_details = []
    n_splits = min(5, len(scored_trades) // 20)  # at least 20 trades per fold
    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accs = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            m = RandomForestClassifier(n_estimators=100, max_depth=5,
                                       min_samples_leaf=3, random_state=42,
                                       class_weight="balanced")
            m.fit(X_tr, y_tr)
            acc = float(m.score(X_te, y_te))
            fold_accs.append(acc)
            wf_details.append({"fold": fold + 1, "train_n": len(train_idx),
                                "test_n": len(test_idx), "accuracy": round(acc * 100, 1)})
        wf_cv_acc = float(np.mean(fold_accs)) if fold_accs else 0.0

    model.fit(X_scaled, y)

    # Feature importances (top 10)
    importances = {name: round(float(imp), 4)
                   for name, imp in zip(_FEATURE_NAMES, model.feature_importances_)}
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    # Log top features
    print("ML: Top predictive features:")
    for fname, fimp in top_features.items():
        print(f"     {fname}: {fimp:.4f}")

    # Save model + scaler
    try:
        with open(ML_MODEL_FILE, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler,
                         "feature_names": _FEATURE_NAMES}, f)
    except Exception as e:
        print(f"Warning: Could not save ML model: {e}")

    return {
        "status":              "trained",
        "trades_used":         len(scored_trades),
        "walk_forward_cv_acc": round(wf_cv_acc * 100, 1),
        "wf_fold_details":     wf_details,
        "feature_importance":  top_features,
        "win_rate_actual":     round(float(y.mean()) * 100, 1),
        "n_features":          len(_FEATURE_NAMES),
    }


def predict_win_probability(trade_features: dict) -> float:
    """
    Use trained model to predict P(win) for a new trade.
    Returns probability 0-1, or 0.5 if model not available.
    """
    if not SKLEARN_AVAILABLE or not ML_MODEL_FILE.exists():
        return 0.5

    try:
        with open(ML_MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        model  = bundle["model"]
        scaler = bundle["scaler"]

        X = np.array([_extract_features(trade_features)])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]  # P(win)
        return float(prob)
    except Exception as e:
        print(f"Warning: ML prediction error: {e}")
        return 0.5


# ── Nightly Retraining ────────────────────────────────────────────────────────

def nightly_retrain() -> dict:
    """
    Full nightly ML pipeline:
    1. Load journal
    2. Compute strategy stats
    3. Adjust weights
    4. Identify disabled strategies
    5. Retrain model
    6. Save everything
    """
    print("ML: Starting nightly retraining...")
    trades = _load_journal()

    if len(trades) < 5:
        return {"status": "insufficient_data", "trades": len(trades)}

    # Strategy performance stats
    stats = compute_all_strategy_stats(trades)

    # Compute new weights
    weights = _compute_optimal_weights(stats)

    # Disabled strategies
    disabled = _should_disable_strategy(stats)

    # Train ML model
    model_result = train_model(trades)

    # Save weights
    ML_WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    ML_WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))

    # Save full stats
    ml_stats = {
        "last_retrain": datetime.now().isoformat(),
        "trades_analyzed": len(trades),
        "strategy_stats": stats,
        "weights": weights,
        "disabled_strategies": disabled,
        "model": model_result,
    }
    ML_STATS_FILE.write_text(json.dumps(ml_stats, indent=2, default=str))

    print(f"ML: Retrain complete — {len(trades)} trades analyzed")
    print(f"ML: Overall win rate: {stats['overall'].get('win_rate', 0):.0f}%")
    print(f"ML: Sharpe ratio: {stats['overall'].get('sharpe', 0):.2f}")
    if disabled:
        print(f"ML: Disabled strategies: {disabled}")
    if model_result.get("status") == "trained":
        print(f"ML: Walk-forward CV accuracy: {model_result.get('walk_forward_cv_acc', 0):.0f}%")

    return ml_stats


def get_ml_stats() -> dict:
    """Return latest ML stats for dashboard."""
    if ML_STATS_FILE.exists():
        try:
            return json.loads(ML_STATS_FILE.read_text())
        except Exception:
            pass
    return {
        "last_retrain": None,
        "trades_analyzed": 0,
        "strategy_stats": {},
        "weights": {},
        "disabled_strategies": [],
        "model": {"status": "not_trained"},
    }


def is_strategy_disabled(strategy_type: str) -> bool:
    """Check if a strategy has been disabled by ML."""
    stats = get_ml_stats()
    return strategy_type in stats.get("disabled_strategies", [])


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running ML nightly retrain...")
    result = nightly_retrain()
    print(f"\nStatus: {result.get('status', 'complete')}")
    print(f"Trades analyzed: {result.get('trades_analyzed', 0)}")
    overall = result.get("strategy_stats", {}).get("overall", {})
    if overall:
        print(f"Win rate: {overall.get('win_rate', 0):.0f}%")
        print(f"Sharpe: {overall.get('sharpe', 0):.2f}")
        print(f"Profit factor: {overall.get('profit_factor', 0):.2f}")
