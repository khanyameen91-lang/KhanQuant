"""
backtester.py — Walk-Forward Backtesting & Monte Carlo Simulation Engine

Validates strategy performance on historical trade data before deploying
any changes to the live (paper) system.

Features:
  1. Walk-Forward Testing — rolling out-of-sample validation
  2. Performance Metrics — Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor
  3. Monte Carlo Simulation — distribution of possible outcomes
  4. Strategy-level breakdown — identify which strategies drive returns
  5. Time-of-day analysis — optimal entry windows
  6. Regime analysis — performance by market condition

Usage:
  python backtester.py             — full report on trade journal
  python backtester.py --monte-carlo — run Monte Carlo simulation
  python backtester.py --strategy credit_spread — filter by strategy
"""

import json
import math
import random
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

LOG_DIR      = Path("logs")
JOURNAL_FILE = LOG_DIR / "trade_journal.json"

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

ACCOUNT_SIZE = 5000.0   # For % return calculations


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_trades(strategy_filter: str = None, min_trades: int = 5) -> list:
    """Load and validate trade journal entries."""
    if not JOURNAL_FILE.exists():
        print(f"Trade journal not found: {JOURNAL_FILE}")
        return []

    try:
        trades = json.loads(JOURNAL_FILE.read_text())
    except Exception as e:
        print(f"Error loading journal: {e}")
        return []

    # Filter to closed trades with P&L
    trades = [t for t in trades if t.get("pnl") is not None and t.get("exit_date")]

    if strategy_filter:
        trades = [t for t in trades
                  if strategy_filter.lower() in t.get("strategy", "").lower()]

    if len(trades) < min_trades:
        print(f"Only {len(trades)} closed trades found (minimum {min_trades} required)")
        return []

    # Sort chronologically (oldest first for walk-forward)
    trades.sort(key=lambda t: t.get("entry_date", ""), reverse=False)
    return trades


# ── Performance Metrics ────────────────────────────────────────────────────────

def compute_metrics(trades: list, rf_rate: float = 0.05) -> dict:
    """
    Compute full performance metrics on a list of trades.

    Args:
        trades: List of trade dicts with 'pnl' field
        rf_rate: Risk-free rate (annualized, e.g., 0.05 = 5%)

    Returns dict with all key metrics.
    """
    if not trades:
        return {}

    pnls = [float(t.get("pnl", 0)) for t in trades]
    n    = len(pnls)

    total_pnl     = sum(pnls)
    winners       = [p for p in pnls if p > 0]
    losers        = [p for p in pnls if p <= 0]

    win_rate      = len(winners) / n if n > 0 else 0
    avg_win       = sum(winners) / len(winners) if winners else 0
    avg_loss      = abs(sum(losers) / len(losers)) if losers else 0
    profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else float('inf')

    # Risk-adjusted returns
    daily_rf = rf_rate / 252
    excess_pnls = [p - daily_rf * ACCOUNT_SIZE for p in pnls]
    mean_excess = sum(excess_pnls) / n if n > 0 else 0
    std_pnl     = _std(pnls)
    std_neg     = _std([p for p in pnls if p < 0]) if losers else 0.0001

    sharpe  = (mean_excess / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0
    sortino = (mean_excess / std_neg * math.sqrt(252)) if std_neg > 0 else 0

    # Max drawdown
    equity_curve = [0.0]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)

    peak     = 0.0
    max_dd   = 0.0
    max_dd_dollars = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    max_dd_dollars = max_dd
    max_dd_pct     = (max_dd / ACCOUNT_SIZE * 100) if ACCOUNT_SIZE > 0 else 0

    # Calmar ratio: return / max drawdown
    annualized_return = total_pnl / ACCOUNT_SIZE * 100 * (252 / max(n, 1))
    calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0

    # Expectancy per trade
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Longest losing streak
    max_streak = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return {
        "trade_count":        n,
        "total_pnl":          round(total_pnl, 2),
        "total_return_pct":   round(total_pnl / ACCOUNT_SIZE * 100, 2),
        "win_rate":           round(win_rate * 100, 1),
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "profit_factor":      round(profit_factor, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "sortino_ratio":      round(sortino, 2),
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "max_drawdown_pct":   round(max_dd_pct, 2),
        "calmar_ratio":       round(calmar, 2),
        "max_losing_streak":  max_streak,
        "annualized_return_pct": round(annualized_return, 2),
        "equity_curve":       [round(v, 2) for v in equity_curve],
    }


def _std(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0001
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) if variance > 0 else 0.0001


# ── Walk-Forward Testing ───────────────────────────────────────────────────────

def walk_forward_test(trades: list, train_pct: float = 0.6) -> dict:
    """
    Walk-forward validation: train on first X% of data, test on rest.
    Simulates how the bot would have performed on unseen data.

    Uses 3 windows for robustness.
    """
    if len(trades) < 10:
        return {"error": "Not enough trades for walk-forward validation"}

    n      = len(trades)
    result = {
        "windows": [],
        "in_sample_avg_sharpe":  0,
        "out_sample_avg_sharpe": 0,
        "degradation_pct":       0,
    }

    # 3 overlapping windows
    window_size = n // 3
    train_size  = int(window_size * train_pct)

    windows_sharpe_in  = []
    windows_sharpe_out = []

    for i in range(3):
        start = i * (n // 3)
        end   = start + window_size

        train = trades[start : start + train_size]
        test  = trades[start + train_size : end]

        if len(train) < 3 or len(test) < 2:
            continue

        train_metrics = compute_metrics(train)
        test_metrics  = compute_metrics(test)

        window_result = {
            "window": i + 1,
            "train_trades": len(train),
            "test_trades":  len(test),
            "train_win_rate":    train_metrics.get("win_rate", 0),
            "test_win_rate":     test_metrics.get("win_rate", 0),
            "train_sharpe":      train_metrics.get("sharpe_ratio", 0),
            "test_sharpe":       test_metrics.get("sharpe_ratio", 0),
            "train_pnl":         train_metrics.get("total_pnl", 0),
            "test_pnl":          test_metrics.get("total_pnl", 0),
            "degradation_pct":   0,
        }

        if train_metrics.get("sharpe_ratio", 0) > 0:
            deg = (1 - test_metrics.get("sharpe_ratio", 0) /
                   train_metrics["sharpe_ratio"]) * 100
            window_result["degradation_pct"] = round(deg, 1)

        result["windows"].append(window_result)
        windows_sharpe_in.append(train_metrics.get("sharpe_ratio", 0))
        windows_sharpe_out.append(test_metrics.get("sharpe_ratio", 0))

    if windows_sharpe_in:
        result["in_sample_avg_sharpe"]  = round(sum(windows_sharpe_in) / len(windows_sharpe_in), 2)
        result["out_sample_avg_sharpe"] = round(sum(windows_sharpe_out) / len(windows_sharpe_out), 2)
        if result["in_sample_avg_sharpe"] > 0:
            result["degradation_pct"] = round(
                (1 - result["out_sample_avg_sharpe"] / result["in_sample_avg_sharpe"]) * 100, 1
            )

    return result


# ── Monte Carlo Simulation ─────────────────────────────────────────────────────

def monte_carlo_simulation(
    trades: list,
    simulations: int = 1000,
    forward_trades: int = 50,
) -> dict:
    """
    Monte Carlo simulation: randomly resample trade P&Ls to estimate
    distribution of possible future outcomes.

    Args:
        trades: Historical closed trades
        simulations: Number of simulation paths
        forward_trades: How many future trades to simulate per path

    Returns percentile distributions and risk metrics.
    """
    if len(trades) < 5:
        return {"error": "Need at least 5 trades for Monte Carlo"}

    pnls  = [float(t.get("pnl", 0)) for t in trades]
    paths = []

    for _ in range(simulations):
        # Resample trades with replacement (bootstrap)
        sampled = random.choices(pnls, k=forward_trades)
        cumulative = []
        running = 0
        for p in sampled:
            running += p
            cumulative.append(running)
        paths.append(cumulative)

    # Final values across all simulations
    final_values    = [path[-1] for path in paths]
    final_values.sort()

    # Drawdown distribution
    max_drawdowns = []
    for path in paths:
        peak = 0
        max_dd = 0
        running = 0
        for p_change in pnls[:forward_trades] if len(pnls) >= forward_trades else pnls:
            running += p_change
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        max_drawdowns.append(max_dd)
    max_drawdowns.sort()

    n = len(final_values)

    return {
        "simulations":    simulations,
        "forward_trades": forward_trades,
        "final_pnl": {
            "p5":    round(final_values[int(n * 0.05)], 2),
            "p25":   round(final_values[int(n * 0.25)], 2),
            "p50":   round(final_values[int(n * 0.50)], 2),
            "p75":   round(final_values[int(n * 0.75)], 2),
            "p95":   round(final_values[int(n * 0.95)], 2),
            "mean":  round(sum(final_values) / n, 2),
        },
        "max_drawdown": {
            "p50":  round(max_drawdowns[int(len(max_drawdowns) * 0.50)], 2),
            "p90":  round(max_drawdowns[int(len(max_drawdowns) * 0.90)], 2),
            "p95":  round(max_drawdowns[int(len(max_drawdowns) * 0.95)], 2),
        },
        "probability_of_profit":   round(sum(1 for v in final_values if v > 0) / n * 100, 1),
        "probability_ruin":        round(sum(1 for v in final_values if v < -ACCOUNT_SIZE * 0.20) / n * 100, 1),
        "expected_outcome":        round(sum(final_values) / n, 2),
    }


# ── Strategy Breakdown ────────────────────────────────────────────────────────

def strategy_breakdown(trades: list) -> dict:
    """Break down performance by strategy type."""
    strategies = {}
    for t in trades:
        stype = t.get("strategy", "unknown")
        if stype not in strategies:
            strategies[stype] = []
        strategies[stype].append(t)

    result = {}
    for stype, strades in strategies.items():
        if len(strades) >= 2:
            m = compute_metrics(strades)
            result[stype] = {
                "trades":      m["trade_count"],
                "win_rate":    m["win_rate"],
                "total_pnl":   m["total_pnl"],
                "profit_factor": m["profit_factor"],
                "sharpe":      m["sharpe_ratio"],
                "expectancy":  m["expectancy_per_trade"],
            }
    return result


# ── Time-of-Day Analysis ──────────────────────────────────────────────────────

def time_of_day_analysis(trades: list) -> dict:
    """Analyze which hours have best/worst performance."""
    hours = {}
    for t in trades:
        entry = t.get("entry_date") or t.get("entry_time", "")
        try:
            if "T" in entry:
                dt = datetime.fromisoformat(entry)
            else:
                dt = datetime.strptime(entry, "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
        except Exception:
            continue

        if hour not in hours:
            hours[hour] = []
        hours[hour].append(float(t.get("pnl", 0)))

    result = {}
    for hour, pnls in hours.items():
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            result[f"{hour:02d}:00"] = {
                "trades":   len(pnls),
                "win_rate": round(wins / len(pnls) * 100, 1),
                "avg_pnl":  round(sum(pnls) / len(pnls), 2),
                "total_pnl": round(sum(pnls), 2),
            }

    return dict(sorted(result.items()))


# ── Full Report ────────────────────────────────────────────────────────────────

def run_full_report(strategy_filter: str = None, run_mc: bool = True):
    """Print the complete backtesting report."""
    print("\n" + "="*65)
    print("  INSTITUTIONAL BACKTESTING REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    if strategy_filter:
        print(f"  Strategy filter: {strategy_filter}")
    print("="*65)

    trades = load_trades(strategy_filter)
    if not trades:
        print("No trade data available. Close some trades and try again.")
        return

    # ── Overall Metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(trades)
    print(f"\n{'─'*65}")
    print(f"  OVERALL PERFORMANCE ({metrics['trade_count']} trades)")
    print(f"{'─'*65}")
    print(f"  Total P&L:          ${metrics['total_pnl']:+.2f} ({metrics['total_return_pct']:+.1f}%)")
    print(f"  Win Rate:           {metrics['win_rate']:.1f}%")
    print(f"  Avg Win:            ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss:           ${metrics['avg_loss']:.2f}")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}  (>1.5 = good, >2.0 = excellent)")
    print(f"  Expectancy/Trade:   ${metrics['expectancy_per_trade']:+.2f}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}  (>1.0 = good, >2.0 = excellent)")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:       ${metrics['max_drawdown_dollars']:.2f} ({metrics['max_drawdown_pct']:.1f}%)")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}  (>1.0 = good)")
    print(f"  Max Losing Streak:  {metrics['max_losing_streak']} trades")
    print(f"  Annualized Return:  {metrics['annualized_return_pct']:.1f}%")

    # ── Institutional Grade Assessment ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  INSTITUTIONAL GRADE ASSESSMENT")
    print(f"{'─'*65}")

    grades = []
    if metrics["sharpe_ratio"] >= 2.0:
        grades.append("✓ Excellent Sharpe (≥2.0)")
    elif metrics["sharpe_ratio"] >= 1.0:
        grades.append("~ Good Sharpe (≥1.0)")
    else:
        grades.append("✗ Poor Sharpe (<1.0) — needs improvement")

    if metrics["profit_factor"] >= 2.0:
        grades.append("✓ Excellent Profit Factor (≥2.0)")
    elif metrics["profit_factor"] >= 1.5:
        grades.append("~ Good Profit Factor (≥1.5)")
    else:
        grades.append("✗ Poor Profit Factor (<1.5)")

    if metrics["max_drawdown_pct"] <= 5.0:
        grades.append("✓ Controlled Drawdown (≤5%)")
    elif metrics["max_drawdown_pct"] <= 10.0:
        grades.append("~ Moderate Drawdown (≤10%)")
    else:
        grades.append("✗ High Drawdown (>10%) — risk management needed")

    if metrics["win_rate"] >= 60:
        grades.append("✓ High Win Rate (≥60%)")
    elif metrics["win_rate"] >= 50:
        grades.append("~ Acceptable Win Rate (≥50%)")
    else:
        grades.append("~ Low Win Rate — ensure profit factor compensates")

    for g in grades:
        print(f"  {g}")

    # ── Walk-Forward ───────────────────────────────────────────────────────────
    if len(trades) >= 10:
        print(f"\n{'─'*65}")
        print("  WALK-FORWARD VALIDATION")
        print(f"{'─'*65}")
        wf = walk_forward_test(trades)
        if "error" not in wf:
            print(f"  In-Sample Avg Sharpe:  {wf['in_sample_avg_sharpe']:.2f}")
            print(f"  Out-Sample Avg Sharpe: {wf['out_sample_avg_sharpe']:.2f}")
            if wf["degradation_pct"] < 30:
                print(f"  ✓ Degradation: {wf['degradation_pct']:.1f}% (good — strategy generalizes)")
            elif wf["degradation_pct"] < 60:
                print(f"  ~ Degradation: {wf['degradation_pct']:.1f}% (moderate — monitor for overfitting)")
            else:
                print(f"  ✗ Degradation: {wf['degradation_pct']:.1f}% (high — possible overfitting)")

            for w in wf["windows"]:
                print(f"  Window {w['window']}: train WR={w['train_win_rate']:.0f}% → "
                      f"test WR={w['test_win_rate']:.0f}% | "
                      f"train Sharpe={w['train_sharpe']:.2f} → "
                      f"test Sharpe={w['test_sharpe']:.2f}")

    # ── Strategy Breakdown ─────────────────────────────────────────────────────
    breakdown = strategy_breakdown(trades)
    if breakdown:
        print(f"\n{'─'*65}")
        print("  STRATEGY BREAKDOWN")
        print(f"{'─'*65}")
        sorted_strats = sorted(breakdown.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        for stype, m in sorted_strats:
            print(f"  {stype[:25]:<25} "
                  f"WR:{m['win_rate']:.0f}% "
                  f"PF:{m['profit_factor']:.2f} "
                  f"Exp:${m['expectancy']:+.0f} "
                  f"P&L:${m['total_pnl']:+.0f} "
                  f"({m['trades']} trades)")

    # ── Time-of-Day ────────────────────────────────────────────────────────────
    tod = time_of_day_analysis(trades)
    if tod:
        print(f"\n{'─'*65}")
        print("  TIME-OF-DAY ANALYSIS")
        print(f"{'─'*65}")
        for hour, data in tod.items():
            if data["trades"] >= 2:
                indicator = "✓" if data["avg_pnl"] > 0 else "✗"
                print(f"  {hour}: {indicator} WR={data['win_rate']:.0f}% "
                      f"Avg P&L=${data['avg_pnl']:+.2f} "
                      f"({data['trades']} trades)")

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    if run_mc and len(trades) >= 5:
        print(f"\n{'─'*65}")
        print("  MONTE CARLO SIMULATION (1,000 paths × 50 trades)")
        print(f"{'─'*65}")
        mc = monte_carlo_simulation(trades, simulations=1000, forward_trades=50)
        if "error" not in mc:
            print(f"  P&L Distribution (next 50 trades):")
            print(f"    Worst 5%:  ${mc['final_pnl']['p5']:+.2f}")
            print(f"    Worst 25%: ${mc['final_pnl']['p25']:+.2f}")
            print(f"    Median:    ${mc['final_pnl']['p50']:+.2f}")
            print(f"    Best 75%:  ${mc['final_pnl']['p75']:+.2f}")
            print(f"    Best 95%:  ${mc['final_pnl']['p95']:+.2f}")
            print(f"  Probability of Profit:  {mc['probability_of_profit']:.1f}%")
            print(f"  Probability of Ruin:    {mc['probability_ruin']:.1f}%")
            print(f"  Median Max Drawdown:    ${mc['max_drawdown']['p50']:.2f}")
            print(f"  90th Pct Drawdown:      ${mc['max_drawdown']['p90']:.2f}")
            print(f"  Expected Outcome:       ${mc['expected_outcome']:+.2f}")

    print(f"\n{'='*65}")
    print("  END OF REPORT")
    print(f"{'='*65}\n")

    return metrics


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting & Monte Carlo Analysis")
    parser.add_argument("--strategy",    type=str, default=None,  help="Filter by strategy type")
    parser.add_argument("--no-mc",       action="store_true",     help="Skip Monte Carlo simulation")
    parser.add_argument("--monte-carlo-only", action="store_true",help="Run only Monte Carlo")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    run_full_report(
        strategy_filter=args.strategy,
        run_mc=not args.no_mc,
    )
