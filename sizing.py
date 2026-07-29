"""
sizing.py — Dynamic Position Sizing Engine

Calculates optimal position size for each trade based on:
  - Composite confidence score (higher confidence = larger size)
  - Current volatility environment (VIX-adjusted)
  - Account drawdown state (reduce after losses)
  - Strategy type risk profile
  - Kelly criterion approximation
  - Available risk budget

Output: recommended max_loss in dollars for this specific trade.
"""

import os
import json
from datetime import date
from pathlib import Path

import state_store

JOURNAL_FILE      = state_store.JOURNAL_FILE

# Account parameters
ACCOUNT_SIZE      = float(os.environ.get("ACCOUNT_SIZE", 5000))
DAILY_LOSS_LIMIT  = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
MAX_POSITION_SIZE = float(os.environ.get("MAX_POSITION_SIZE", 150))
MIN_POSITION_SIZE = 25.0   # never risk less than $25


# ── Helper data loading ────────────────────────────────────────────────────────

_load_daily_stats = state_store.load_daily_stats


def _load_recent_trades(lookback: int = 20) -> list:
    """Get recent trades from journal for win rate calculation."""
    if not JOURNAL_FILE.exists():
        return []
    try:
        journal = json.loads(JOURNAL_FILE.read_text())
        return journal[:lookback]
    except Exception:
        return []


def _recent_win_rate(lookback: int = 20) -> float:
    """Calculate win rate over last N trades."""
    trades = _load_recent_trades(lookback)
    if not trades:
        return 0.50   # assume 50% until data exists
    winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winners / len(trades)


def _recent_avg_win_loss(lookback: int = 20) -> tuple[float, float]:
    """Return (avg_win, avg_loss) in dollars over last N trades."""
    trades = _load_recent_trades(lookback)
    wins   = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
    losses = [abs(t["pnl"]) for t in trades if t.get("pnl", 0) < 0]
    avg_win  = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else MAX_POSITION_SIZE * 0.75
    return avg_win, avg_loss


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion: f* = (p * b - q) / b
    where p = win rate, q = 1-p, b = avg_win / avg_loss

    Returns fraction of account to risk (capped at 5% for safety).
    """
    if avg_loss <= 0:
        return 0.01
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b if b > 0 else 0
    # Use half-Kelly for safety (classic risk management)
    half_kelly = kelly * 0.5
    # Cap at 5% of account
    return max(0.005, min(0.05, half_kelly))


# ── Confidence-Based Sizing ───────────────────────────────────────────────────

def confidence_to_size_pct(confidence: float) -> float:
    """
    Map confidence score (0-100) to fraction of max position size.

    < 70:  No trade
    70-75: 40% of max
    76-80: 55% of max
    81-85: 70% of max
    86-90: 85% of max
    91+:   100% of max
    """
    if confidence < 70:
        return 0.0
    elif confidence < 76:
        return 0.40
    elif confidence < 81:
        return 0.55
    elif confidence < 86:
        return 0.70
    elif confidence < 91:
        return 0.85
    else:
        return 1.00


# ── Volatility Adjustment ─────────────────────────────────────────────────────

def vix_size_adjustment(vix: float) -> float:
    """
    Reduce position size as VIX rises — options are more expensive
    and markets are more unpredictable.

    VIX <= 15: 1.10 (slight increase, cheap options)
    VIX 15-20: 1.00 (no adjustment)
    VIX 20-25: 0.85
    VIX 25-30: 0.70
    VIX 30-35: 0.55
    VIX > 35:  0.40 (extreme fear, small size only)
    """
    if vix <= 15:
        return 1.10
    elif vix <= 20:
        return 1.00
    elif vix <= 25:
        return 0.85
    elif vix <= 30:
        return 0.70
    elif vix <= 35:
        return 0.55
    else:
        return 0.40


# ── Drawdown Adjustment ───────────────────────────────────────────────────────

def drawdown_size_adjustment(daily_pnl: float) -> float:
    """
    Reduce size proportionally to how much of the daily budget is consumed.

    0% loss:   1.00 (full size)
    25% loss:  0.85
    50% loss:  0.65
    75% loss:  0.40 (near limit)
    """
    if daily_pnl >= 0:
        return 1.00
    loss_fraction = abs(daily_pnl) / DAILY_LOSS_LIMIT
    if loss_fraction < 0.25:
        return 1.00
    elif loss_fraction < 0.50:
        return 0.85
    elif loss_fraction < 0.75:
        return 0.65
    else:
        return 0.40


# ── Strategy Risk Profiles ────────────────────────────────────────────────────

_STRATEGY_BASE_RISK = {
    "long_option":   0.90,  # Can lose full premium — be slightly more conservative
    "0dte_scalp":    0.60,  # 0DTE = high gamma risk, smaller size
    "spread":        1.00,  # Defined risk, full max size allowed
    "debit_spread":  1.00,
    "credit_spread": 1.00,
    "iron_condor":   1.00,
    "strangle":      0.80,  # Can lose full premium on both sides
}


# ── Main Sizing Function ───────────────────────────────────────────────────────

def calculate_position_size(
    confidence: float,
    strategy_type: str,
    vix: float = 20.0,
    use_kelly: bool = True,
) -> dict:
    """
    Calculate the recommended maximum dollar risk for this trade.

    Args:
        confidence:    Trade confidence score (0-100)
        strategy_type: 'long_option', 'spread', 'credit_spread', etc.
        vix:           Current VIX level
        use_kelly:     Whether to incorporate Kelly criterion

    Returns:
        dict with 'max_loss' (dollar risk) and breakdown of adjustments
    """
    stats = _load_daily_stats()
    daily_pnl = stats.get("pnl", 0.0)

    # 1. Confidence-based fraction
    conf_pct = confidence_to_size_pct(confidence)
    if conf_pct == 0:
        return {
            "max_loss": 0,
            "reason": f"Confidence {confidence:.0f} below minimum threshold",
            "approved": False,
        }

    # 2. Strategy base risk multiplier
    strat_mult = _STRATEGY_BASE_RISK.get(strategy_type, 1.0)

    # 3. VIX adjustment
    vix_mult = vix_size_adjustment(vix)

    # 4. Drawdown adjustment
    dd_mult = drawdown_size_adjustment(daily_pnl)

    # 5. Kelly criterion (uses historical win rate)
    kelly_mult = 1.0
    if use_kelly:
        win_rate = _recent_win_rate(lookback=20)
        avg_win, avg_loss = _recent_avg_win_loss(lookback=20)
        kelly_frac = kelly_fraction(win_rate, avg_win, avg_loss)
        # Kelly as a fraction of account, mapped to our max position
        kelly_amount = ACCOUNT_SIZE * kelly_frac
        kelly_mult = min(1.0, kelly_amount / MAX_POSITION_SIZE)
        kelly_mult = max(0.3, kelly_mult)  # floor at 30%

    # 6. Remaining budget check
    remaining_budget = DAILY_LOSS_LIMIT - abs(min(daily_pnl, 0))

    # Compute raw size
    raw_size = MAX_POSITION_SIZE * conf_pct * strat_mult * vix_mult * dd_mult * kelly_mult

    # Clip to remaining budget and absolute limits
    final_size = min(raw_size, remaining_budget, MAX_POSITION_SIZE)
    final_size = max(MIN_POSITION_SIZE, final_size)

    # Round to nearest $5
    final_size = round(final_size / 5) * 5

    return {
        "max_loss":         final_size,
        "approved":         True,
        "base_max":         MAX_POSITION_SIZE,
        "confidence_pct":   round(conf_pct * 100),
        "strategy_mult":    round(strat_mult, 2),
        "vix_mult":         round(vix_mult, 2),
        "drawdown_mult":    round(dd_mult, 2),
        "kelly_mult":       round(kelly_mult, 2),
        "remaining_budget": round(remaining_budget, 2),
        "daily_pnl":        daily_pnl,
    }


def filter_setups_by_max_loss(setups: list, max_loss_limit: float) -> list:
    """Filter a list of setups to only those within the max loss limit."""
    return [s for s in setups if s.get("max_loss", 9999) <= max_loss_limit]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Position Sizing Examples:")
    print()

    for conf, strat, vix in [
        (92, "long_option", 18),
        (85, "credit_spread", 22),
        (78, "iron_condor", 28),
        (72, "0dte_scalp", 20),
        (65, "spread", 20),
    ]:
        result = calculate_position_size(conf, strat, vix)
        if result["approved"]:
            print(f"  Confidence {conf}% | {strat} | VIX {vix} → Max Loss: ${result['max_loss']:.0f}")
            print(f"    Adjustments: conf={result['confidence_pct']}% × "
                  f"strat={result['strategy_mult']} × "
                  f"vix={result['vix_mult']} × "
                  f"drawdown={result['drawdown_mult']} × "
                  f"kelly={result['kelly_mult']}")
        else:
            print(f"  Confidence {conf}% | {strat} → SKIP ({result['reason']})")
        print()
