"""
options_pricing.py — Black-Scholes option pricing and Greeks.

market_data.py used to hardcode every contract's gamma to a flat 0.02,
theta to -mid*0.01, and vega to mid*0.1 — identical numbers regardless of
strike, expiration, or moneyness — and derived delta from a linear
moneyness formula with no volatility or time-to-expiry term at all. Every
downstream consumer (strike selection targeting a specific delta, portfolio
Greek exposure limits in risk_manager.py, the GEX/gamma-exposure signal in
flow_analyzer.py) was therefore working off numbers that didn't actually
move the way real Greeks do.

This module computes real Black-Scholes Greeks using each contract's own
strike, the underlying spot price, time to expiration, and its actual
implied volatility — all of which (except the risk-free rate) were already
being fetched from the option chain. No new data source is required; this
replaces fabricated math with real math on data that was already there.
"""

import os
import math
from datetime import date

# 10yr treasury-ish default; override via .env if you want to track the
# actual risk-free rate more precisely (intelligence.py already fetches
# ^TNX for macro signals, but wiring that in here isn't necessary for the
# Greeks to be meaningfully more correct than a fixed reasonable constant).
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", 0.045))

_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def days_to_expiration(expiration: str, as_of: date = None) -> int:
    """Calendar days between `as_of` (default today) and an ISO expiration
    date string. Returns 0 for same-day (0DTE) expirations, never negative."""
    as_of = as_of or date.today()
    try:
        exp_date = date.fromisoformat(expiration)
    except Exception:
        return 0
    return max(0, (exp_date - as_of).days)


def black_scholes_greeks(
    spot: float,
    strike: float,
    dte_days: float,
    iv_pct: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> dict:
    """
    Real Black-Scholes price + Greeks for a single option contract.

    Args:
        spot: underlying price
        strike: option strike
        dte_days: calendar days to expiration (0 is valid — 0DTE)
        iv_pct: implied volatility as a percentage (e.g. 28.5 for 28.5%),
                matching how market_data.py already stores `iv`
        option_type: "C" or "P"
        r: annualized risk-free rate as a decimal (e.g. 0.045)

    Returns:
        {"price": float, "delta": float, "gamma": float,
         "theta": float, "vega": float}

        delta: call in [0, 1], put in [-1, 0] — standard convention.
        gamma: positive, same for calls and puts.
        theta: dollars of decay per share PER DAY (negative for a long
               option) — matches the unit the rest of the codebase expects
               (risk_manager.py multiplies by 100 for per-contract dollars).
        vega: dollars of sensitivity per share PER 1 POINT of IV (e.g. IV
              moving from 28% to 29%) — same per-share convention as theta.
    """
    is_call = (option_type or "C").upper().startswith("C")

    if spot <= 0 or strike <= 0:
        return {"price": 0.0, "delta": 0.5 if is_call else -0.5,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    # Floor both T and sigma so d1/d2 never divide by zero — a 0DTE contract
    # still has a few hours of real time value, not literally none.
    T = max(dte_days, 0.25) / 365.0
    sigma = max(iv_pct, 1.0) / 100.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    pdf_d1 = _norm_pdf(d1)
    disc = math.exp(-r * T)

    if is_call:
        price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = (-(spot * pdf_d1 * sigma) / (2 * sqrtT)
                        - r * strike * disc * _norm_cdf(d2))
    else:
        price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (-(spot * pdf_d1 * sigma) / (2 * sqrtT)
                         + r * strike * disc * _norm_cdf(-d2))

    gamma = pdf_d1 / (spot * sigma * sqrtT)
    vega_full = spot * pdf_d1 * sqrtT   # per 1.0 (100 percentage points) of IV

    return {
        "price": round(max(price, 0.0), 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_annual / 365.0, 4),   # per day
        "vega":  round(vega_full / 100.0, 4),      # per 1 point of IV
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("ATM call, 30 DTE, 25% IV",  100, 100, 30, 25.0, "C"),
        ("ATM put, 30 DTE, 25% IV",   100, 100, 30, 25.0, "P"),
        ("OTM call, 7 DTE, 40% IV",   100, 110, 7,  40.0, "C"),
        ("ITM put, 14 DTE, 30% IV",   100, 110, 14, 30.0, "P"),
        ("0DTE call, 20% IV",         100, 100, 0,  20.0, "C"),
    ]
    for name, spot, strike, dte, iv, typ in cases:
        g = black_scholes_greeks(spot, strike, dte, iv, typ)
        print(f"{name}: {g}")
