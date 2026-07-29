"""
expected_value.py — Expected Value (EV) Engine

Calculates the mathematical expected value of every trade before execution.
A trade with negative expected value is ALWAYS rejected, regardless of
how good the technical setup looks.

EV Formula:
  EV = (P_win × avg_win) − (P_loss × avg_loss)

Where:
  P_win     = ML probability estimate (or historical win rate for strategy)
  avg_win   = max_profit × profit_take_pct (we don't always capture full profit)
  P_loss    = 1 − P_win
  avg_loss  = max_loss × stop_loss_pct (we exit before max loss in practice)

EV Score (0-100):
  EV > $50  → 100
  EV > $20  → 80-99
  EV > $5   → 60-79
  EV > $0   → 40-59 (positive but marginal)
  EV ≤ $0   → 0 (REJECT)
"""

import json
from pathlib import Path
from typing import Optional

JOURNAL_FILE = Path("logs/trade_journal.json")

# ── Profit/Loss Target Assumptions by Strategy ─────────────────────────────────
# These reflect realistic take-profit and stop-loss behavior (not theoretical max)
STRATEGY_PARAMS = {
    "credit_spread": {
        "profit_take_pct": 0.50,   # Close at 50% of max credit (industry standard)
        "stop_loss_pct":   0.75,   # Stop at 3x credit received
        "min_p_win":       0.55,   # Must have at least 55% win probability
    },
    "debit_spread": {
        "profit_take_pct": 0.65,   # Capture 65% of max profit
        "stop_loss_pct":   0.60,   # Stop at 60% of debit paid
        "min_p_win":       0.50,
    },
    "long_option": {
        "profit_take_pct": 1.00,   # Long options: target 2:1 or better
        "stop_loss_pct":   0.50,   # Stop at 50% of premium paid
        "min_p_win":       0.45,   # Lower bar — higher upside compensates
    },
    "iron_condor": {
        "profit_take_pct": 0.50,   # Take at 50% of max credit
        "stop_loss_pct":   1.50,   # Iron condors: 2x credit received is stop
        "min_p_win":       0.60,   # Must have 60%+ win rate
    },
    "strangle": {
        "profit_take_pct": 0.50,
        "stop_loss_pct":   0.75,
        "min_p_win":       0.50,
    },
    "0dte_scalp": {
        "profit_take_pct": 0.80,   # Capture 80% of move on 0DTE
        "stop_loss_pct":   0.40,   # Tight stop on 0DTE
        "min_p_win":       0.52,   # Coin-flip-plus on 0DTE
    },
}

# Historical win rates by strategy (fallback when no ML data)
HISTORICAL_WIN_RATES = {
    "credit_spread": 0.68,    # Credit spreads win ~68% of time
    "debit_spread":  0.52,    # Debit spreads ~52%
    "long_option":   0.48,    # Long options ~48%
    "iron_condor":   0.72,    # Iron condors ~72%
    "strangle":      0.50,
    "0dte_scalp":    0.52,
}


def _load_strategy_win_rate(strategy_type: str, lookback: int = 30) -> Optional[float]:
    """Load historical win rate for a specific strategy from the journal."""
    if not JOURNAL_FILE.exists():
        return None
    try:
        journal = json.loads(JOURNAL_FILE.read_text())
        relevant = [t for t in journal[:lookback]
                    if strategy_type in t.get("strategy", "").lower()
                    and t.get("pnl") is not None]
        if len(relevant) < 5:
            return None
        wins = sum(1 for t in relevant if t["pnl"] > 0)
        return wins / len(relevant)
    except Exception:
        return None


def compute_ev(
    setup: dict,
    ml_win_prob: Optional[float] = None,
) -> dict:
    """
    Compute expected value for a trade setup.

    Args:
        setup: Trade setup dict with max_loss, max_profit, type, net_credit/net_debit
        ml_win_prob: ML-estimated win probability (0-1), or None to use historical

    Returns dict with:
        ev_dollars: Expected value in dollars
        ev_score: 0-100 score
        p_win: win probability used
        avg_win: expected win amount
        avg_loss: expected loss amount
        approved: bool (False = negative EV, reject)
        reasoning: explanation string
    """
    stype       = _normalize_strategy_type(setup.get("type", "long_option"))
    max_loss    = abs(float(setup.get("max_loss", 0)))
    max_profit  = float(setup.get("max_profit") or 0)
    net_credit  = float(setup.get("net_credit", 0))

    # For credit spreads, max_profit = net_credit * 100 if not set
    if max_profit == 0 and net_credit > 0:
        max_profit = net_credit * 100

    # Fallback: if max_profit still unknown, use 1:1 ratio
    if max_profit == 0:
        max_profit = max_loss

    params = STRATEGY_PARAMS.get(stype, STRATEGY_PARAMS["long_option"])

    # Win probability: ML → historical → default
    if ml_win_prob is not None and 0.0 < ml_win_prob < 1.0:
        p_win = ml_win_prob
        source = "ML model"
    else:
        historical = _load_strategy_win_rate(stype)
        if historical is not None:
            p_win  = historical
            source = f"historical ({int(p_win*100)}%)"
        else:
            p_win  = HISTORICAL_WIN_RATES.get(stype, 0.50)
            source = f"baseline ({int(p_win*100)}%)"

    p_loss   = 1.0 - p_win
    avg_win  = max_profit * params["profit_take_pct"]
    avg_loss = max_loss   * params["stop_loss_pct"]

    ev_dollars = (p_win * avg_win) - (p_loss * avg_loss)

    # Score: 0-100
    if ev_dollars <= 0:
        ev_score = 0
    elif ev_dollars >= 50:
        ev_score = 100
    elif ev_dollars >= 20:
        ev_score = round(80 + (ev_dollars - 20) / 30 * 20)
    elif ev_dollars >= 5:
        ev_score = round(60 + (ev_dollars - 5) / 15 * 20)
    else:
        ev_score = round(40 + ev_dollars / 5 * 20)

    # Approval: positive EV AND meets minimum win probability
    approved = ev_dollars > 0 and p_win >= params["min_p_win"]

    if not approved:
        if ev_dollars <= 0:
            reasoning = (
                f"NEGATIVE EV: ${ev_dollars:.2f} — "
                f"P(win)={p_win:.0%} × ${avg_win:.0f} = ${p_win*avg_win:.0f} vs "
                f"P(loss)={p_loss:.0%} × ${avg_loss:.0f} = ${p_loss*avg_loss:.0f}"
            )
        else:
            reasoning = (
                f"LOW WIN PROB: {p_win:.0%} < minimum {params['min_p_win']:.0%} "
                f"for {stype}"
            )
    else:
        reasoning = (
            f"EV: +${ev_dollars:.2f} | "
            f"P(win)={p_win:.0%}({source}) × avg_win=${avg_win:.0f} | "
            f"P(loss)={p_loss:.0%} × avg_loss=${avg_loss:.0f}"
        )

    return {
        "ev_dollars":  round(ev_dollars, 2),
        "ev_score":    ev_score,
        "p_win":       round(p_win, 4),
        "p_loss":      round(p_loss, 4),
        "avg_win":     round(avg_win, 2),
        "avg_loss":    round(avg_loss, 2),
        "approved":    approved,
        "source":      source,
        "reasoning":   reasoning,
        "strategy":    stype,
    }


def compute_ev_score(setup: dict, ml_win_prob: Optional[float] = None) -> float:
    """Convenience function: returns just the 0-100 EV score."""
    return compute_ev(setup, ml_win_prob)["ev_score"]


def _normalize_strategy_type(raw: str) -> str:
    """Map raw strategy type strings to canonical names."""
    raw = raw.lower().replace(" ", "_").replace("-", "_")
    if "iron_condor" in raw or "condor" in raw:
        return "iron_condor"
    if "credit" in raw or "bull_put" in raw or "bear_call" in raw:
        return "credit_spread"
    if "debit" in raw or "bull_call" in raw or "bear_put" in raw:
        return "debit_spread"
    if "strangle" in raw or "straddle" in raw:
        return "strangle"
    if "0dte" in raw or "zero_dte" in raw or "scalp" in raw:
        return "0dte_scalp"
    return "long_option"


def get_ev_summary(ev_result: dict) -> str:
    """Return a one-line summary for logging / Claude context."""
    status = "✓ APPROVED" if ev_result["approved"] else "✗ REJECTED"
    return (
        f"EV {status}: ${ev_result['ev_dollars']:+.2f} "
        f"(score={ev_result['ev_score']}/100) — {ev_result['reasoning']}"
    )


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_setups = [
        {
            "name": "TSLA Bull Put Credit Spread",
            "type": "credit_spread",
            "max_loss": 130, "net_credit": 0.70,
        },
        {
            "name": "NVDA Long Call",
            "type": "long_option",
            "max_loss": 85, "max_profit": 0,
        },
        {
            "name": "SPY Iron Condor",
            "type": "iron_condor",
            "max_loss": 200, "net_credit": 1.50,
        },
        {
            "name": "QQQ Bear Put Debit Spread",
            "type": "debit_spread",
            "max_loss": 100, "max_profit": 200,
        },
    ]

    for setup in test_setups:
        ev = compute_ev(setup, ml_win_prob=0.62)
        print(f"\n{setup['name']}:")
        print(f"  {get_ev_summary(ev)}")
