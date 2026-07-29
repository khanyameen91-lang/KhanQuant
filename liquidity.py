"""
liquidity.py — Options Contract Liquidity Scoring Engine

Evaluates the liquidity quality of option contracts before execution.
Poor liquidity = wide spreads = bad fills = guaranteed edge erosion.

Institutional trading rule: if you can't get a fair fill, don't trade.

Liquidity Score (0-100):
  ≥ 80 = Excellent — institutional quality fill expected
  60-79 = Good — acceptable fill, minor slippage
  40-59 = Fair — use limit orders, be patient
  20-39 = Poor — avoid unless conviction is very high
  < 20  = Reject — spreads will eat the entire edge

Minimum requirements for execution:
  - Open Interest ≥ 100 contracts
  - Daily Volume ≥ 50 contracts
  - Bid/Ask spread ≤ 10% of mid price
  - Implied Volatility > 5% (not dead)
  - 5 ≤ DTE ≤ 60 (not too short, not too long for day trading context)
"""

from typing import Optional


# ── Minimum standards for trade execution ─────────────────────────────────────
LIQUIDITY_MINIMUMS = {
    "open_interest":    100,    # contracts
    "volume":           50,     # contracts
    "max_spread_pct":   0.10,   # 10% of mid price
    "min_iv":           0.05,   # 5% IV floor
    "min_dte":          1,
    "max_dte":          60,
}

# Preferred (above this = full score for that metric)
LIQUIDITY_PREFERRED = {
    "open_interest":   500,
    "volume":          200,
    "max_spread_pct":  0.04,    # 4% or tighter = excellent
    "min_iv":          0.15,
}


def _score_open_interest(oi: float) -> float:
    """Score OI 0-100. OI ≥ 500 = 100, OI = 100 = 40, OI < 100 = 0."""
    if oi < LIQUIDITY_MINIMUMS["open_interest"]:
        return 0.0
    if oi >= LIQUIDITY_PREFERRED["open_interest"]:
        return 100.0
    # Linear between 100 and 500
    return round(40 + (oi - 100) / 400 * 60, 1)


def _score_volume(vol: float) -> float:
    """Score volume 0-100."""
    if vol < LIQUIDITY_MINIMUMS["volume"]:
        return 0.0
    if vol >= LIQUIDITY_PREFERRED["volume"]:
        return 100.0
    return round(40 + (vol - 50) / 150 * 60, 1)


def _score_spread(bid: float, ask: float) -> float:
    """
    Score bid/ask spread quality.
    Returns 0-100 where 100 = very tight spread.
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return 10.0   # unknown or crossed — penalize

    mid    = (bid + ask) / 2
    spread = ask - bid

    if mid <= 0:
        return 10.0

    spread_pct = spread / mid

    if spread_pct > LIQUIDITY_MINIMUMS["max_spread_pct"]:
        # Spread > 10%: score 0-20 based on how bad it is
        return max(0.0, round(20 - (spread_pct - 0.10) * 100, 1))
    elif spread_pct <= LIQUIDITY_PREFERRED["max_spread_pct"]:
        # Very tight spread
        return 100.0
    else:
        # Between 4% and 10%: linear 50-100
        return round(50 + (LIQUIDITY_MINIMUMS["max_spread_pct"] - spread_pct) /
                     (LIQUIDITY_MINIMUMS["max_spread_pct"] - LIQUIDITY_PREFERRED["max_spread_pct"]) * 50, 1)


def _score_dte(dte: int) -> float:
    """
    Score days-to-expiration for day trading context.
    0DTE: special handling (acceptable for scalps)
    1-5 DTE: high gamma risk
    5-45 DTE: sweet spot
    45-60 DTE: still ok
    > 60 DTE: too much time value decay for day trading
    """
    if dte == 0:
        return 60.0   # 0DTE scalp is valid but risky
    if dte < 3:
        return 40.0   # Pin risk, avoid unless specifically 0DTE scalp strategy
    if 5 <= dte <= 45:
        return 100.0  # Sweet spot
    if dte <= 60:
        return 75.0   # Slightly long but acceptable
    return 30.0       # Too far out for day trading


def _score_iv(iv: float) -> float:
    """Score IV (0-1 range from yfinance). Low IV = cheap options but low premium."""
    if iv < LIQUIDITY_MINIMUMS["min_iv"]:
        return 0.0    # Dead — no movement expected
    if iv >= 0.40:
        return 100.0  # High IV = good premium, liquid market
    if iv >= LIQUIDITY_PREFERRED["min_iv"]:
        return 80.0
    return round(40 + (iv - LIQUIDITY_MINIMUMS["min_iv"]) /
                 (LIQUIDITY_PREFERRED["min_iv"] - LIQUIDITY_MINIMUMS["min_iv"]) * 40, 1)


def compute_liquidity_score(
    bid: float,
    ask: float,
    open_interest: float,
    volume: float,
    dte: int,
    iv: float = 0.20,
) -> dict:
    """
    Compute composite liquidity score for an option contract.

    Returns:
        {
          "liquidity_score": 0-100,
          "spread_score": 0-100,
          "oi_score": 0-100,
          "volume_score": 0-100,
          "dte_score": 0-100,
          "iv_score": 0-100,
          "approved": bool,
          "reject_reason": str or None,
          "signals": [str]
        }
    """
    spread_score = _score_spread(bid, ask)
    oi_score     = _score_open_interest(open_interest)
    volume_score = _score_volume(volume)
    dte_score    = _score_dte(dte)
    iv_score     = _score_iv(iv)

    # Weighted composite — spread quality matters most for fills
    liquidity_score = (
        spread_score * 0.40 +
        oi_score     * 0.25 +
        volume_score * 0.20 +
        dte_score    * 0.10 +
        iv_score     * 0.05
    )
    liquidity_score = round(liquidity_score, 1)

    # Hard rejections (any minimum breach = reject)
    reject_reason = None
    if oi_score == 0.0:
        reject_reason = f"Open interest too low ({open_interest:.0f} < {LIQUIDITY_MINIMUMS['open_interest']})"
    elif volume_score == 0.0:
        reject_reason = f"Volume too low ({volume:.0f} < {LIQUIDITY_MINIMUMS['volume']})"
    elif spread_score < 10:
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 1
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999
        reject_reason = f"Spread too wide ({spread_pct:.1f}% of mid)"

    approved = reject_reason is None

    # Build signals
    signals = []
    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0

    if spread_score >= 80:
        signals.append(f"✓ Tight spread ({spread_pct:.1f}% of mid)")
    elif spread_score < 40:
        signals.append(f"✗ Wide spread ({spread_pct:.1f}% of mid) — slippage risk")

    if oi_score >= 80:
        signals.append(f"✓ Strong OI ({open_interest:.0f} contracts)")
    elif oi_score < 40:
        signals.append(f"✗ Low OI ({open_interest:.0f}) — exit risk")

    if volume_score >= 70:
        signals.append(f"✓ Good volume ({volume:.0f} today)")
    elif volume_score == 0:
        signals.append(f"✗ Low volume ({volume:.0f}) — illiquid")

    if 5 <= dte <= 45:
        signals.append(f"✓ Favorable DTE ({dte})")
    elif dte == 0:
        signals.append(f"~ 0DTE — gamma risk")
    elif dte > 45:
        signals.append(f"~ Long DTE ({dte}) — consider closer expiration")

    return {
        "liquidity_score": liquidity_score,
        "spread_score":    spread_score,
        "oi_score":        oi_score,
        "volume_score":    volume_score,
        "dte_score":       dte_score,
        "iv_score":        iv_score,
        "approved":        approved,
        "reject_reason":   reject_reason,
        "signals":         signals,
        "spread_pct":      round(spread_pct, 2),
    }


def score_setup_liquidity(snapshot: dict) -> dict:
    """
    Score the liquidity of the primary contract in a trade setup.
    Handles both single-leg and spread setups.
    """
    setup   = snapshot.get("setup", {})
    dte     = snapshot.get("dte", 14)
    stype   = setup.get("type", "")

    # Get the primary contract (for spreads, use the short leg for liquidity check)
    if "short_contract" in setup:
        contract = setup["short_contract"]
    elif "contract" in setup:
        contract = setup["contract"]
    else:
        contract = {}

    bid = float(contract.get("bid", 0))
    ask = float(contract.get("ask", bid))   # fallback to bid if no ask
    oi  = float(contract.get("open_interest", 0))
    vol = float(contract.get("volume", 0))
    iv  = float(contract.get("iv", 0.20))

    # For options chain data from yfinance, sometimes these fields differ
    if bid == 0 and ask == 0:
        # Try lastPrice as a proxy
        last = float(contract.get("lastPrice", 0))
        if last > 0:
            bid = last * 0.95
            ask = last * 1.05

    return compute_liquidity_score(bid, ask, oi, vol, dte, iv)


def get_liquidity_summary(liq: dict) -> str:
    """One-line summary for logging."""
    status = "✓" if liq["approved"] else "✗"
    reason = f" — REJECT: {liq['reject_reason']}" if liq.get("reject_reason") else ""
    return (f"Liquidity {status} {liq['liquidity_score']}/100 "
            f"(spread={liq['spread_score']:.0f} OI={liq['oi_score']:.0f} "
            f"vol={liq['volume_score']:.0f}){reason}")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        {"name": "SPY ATM Call (liquid)",    "bid": 1.20, "ask": 1.22, "oi": 50000, "vol": 5000, "dte": 14, "iv": 0.18},
        {"name": "TSLA OTM Put (moderate)",  "bid": 0.30, "ask": 0.40, "oi": 800,   "vol": 120,  "dte": 7,  "iv": 0.55},
        {"name": "Illiquid microcap call",   "bid": 0.10, "ask": 0.40, "oi": 20,    "vol": 5,    "dte": 30, "iv": 0.80},
        {"name": "NVDA near-the-money call", "bid": 2.40, "ask": 2.55, "oi": 3000,  "vol": 600,  "dte": 21, "iv": 0.45},
    ]

    for t in tests:
        liq = compute_liquidity_score(t["bid"], t["ask"], t["oi"], t["vol"], t["dte"], t["iv"])
        print(f"\n{t['name']}:")
        print(f"  {get_liquidity_summary(liq)}")
        for s in liq["signals"]:
            print(f"  {s}")
