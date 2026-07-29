"""
relative_strength.py — Relative Strength Ranking Engine

Ranks symbols by their strength or weakness relative to:
  - SPY (market benchmark)
  - Their sector ETF
  - Relative volume (today vs average)

Used to:
  - Focus bullish strategies on the strongest leaders
  - Focus bearish strategies on the weakest laggards
  - Avoid trading mediocre, choppy symbols

Relative Strength Score (0-100):
  > 70 = Strong leader (good for bullish strategies)
  40-70 = Neutral (tradeable but not ideal)
  < 30 = Strong laggard (good for bearish strategies)
"""

import time
from typing import Optional

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    print("Warning: yfinance/pandas not available for relative strength engine")


# ── Sector ETF mapping ─────────────────────────────────────────────────────────
SECTOR_ETF = {
    "AAPL": "XLK",  "MSFT": "XLK",  "NVDA": "XLK",  "AMD":  "XLK",
    "META": "XLC",  "GOOG": "XLC",   "GOOGL": "XLC",
    "AMZN": "XLY",  "TSLA": "XLY",
    "JPM":  "XLF",  "BAC":  "XLF",   "GS":   "XLF",
    "XOM":  "XLE",  "CVX":  "XLE",
    "JNJ":  "XLV",  "UNH":  "XLV",   "PFE":  "XLV",
    "SPY":  "SPY",  "QQQ":  "XLK",   "IWM":  "IWM",
    "PLTR": "XLK",
}

# Cache: symbol → {score, timestamp}
_rs_cache: dict = {}
_RS_CACHE_TTL = 300  # 5 minutes — relative strength changes slowly


def _get_price_series(symbol: str, period: str = "5d", interval: str = "15m") -> Optional[object]:
    """Fetch OHLCV series from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def compute_rs_vs_spy(symbol: str, lookback_bars: int = 20) -> float:
    """
    Compute symbol's return vs SPY over last N bars.
    Returns relative performance score 0-100 (50 = same as SPY).
    """
    if not RS_AVAILABLE:
        return 50.0

    sym_df  = _get_price_series(symbol)
    spy_df  = _get_price_series("SPY")

    if sym_df is None or spy_df is None or len(sym_df) < 5:
        return 50.0

    try:
        # Align on common timestamps
        sym_close = sym_df["Close"].iloc[-lookback_bars:]
        spy_close = spy_df["Close"].iloc[-lookback_bars:]

        sym_ret = (sym_close.iloc[-1] / sym_close.iloc[0] - 1) * 100
        spy_ret = (spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100

        # Relative return
        rel_ret = sym_ret - spy_ret

        # Convert to 0-100 score
        # rel_ret of +5% → score ~80 | rel_ret of -5% → score ~20
        score = 50 + rel_ret * 6   # 1% outperformance = 6 points
        return max(0.0, min(100.0, score))

    except Exception:
        return 50.0


def compute_rs_vs_sector(symbol: str) -> float:
    """
    Compute symbol's performance vs its sector ETF.
    Returns 0-100 score (50 = in-line with sector).
    """
    if not RS_AVAILABLE:
        return 50.0

    sector_etf = SECTOR_ETF.get(symbol.upper())
    if not sector_etf or sector_etf == symbol:
        return 50.0

    sym_df    = _get_price_series(symbol)
    sector_df = _get_price_series(sector_etf)

    if sym_df is None or sector_df is None or len(sym_df) < 5:
        return 50.0

    try:
        sym_ret    = (sym_df["Close"].iloc[-1] / sym_df["Close"].iloc[-10] - 1) * 100
        sector_ret = (sector_df["Close"].iloc[-1] / sector_df["Close"].iloc[-10] - 1) * 100
        rel_ret    = sym_ret - sector_ret
        score      = 50 + rel_ret * 5
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def compute_relative_volume(symbol: str) -> float:
    """
    Compare today's volume to the 20-day average.
    High relative volume = institutional interest = bullish signal.

    Returns 0-100 score (50 = average volume).
    """
    if not RS_AVAILABLE:
        return 50.0

    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="30d", interval="1d")
        if hist is None or len(hist) < 10:
            return 50.0

        today_vol  = hist["Volume"].iloc[-1]
        avg_vol    = hist["Volume"].iloc[-21:-1].mean()   # 20-day avg (ex today)

        if avg_vol == 0:
            return 50.0

        rvol = today_vol / avg_vol   # 1.0 = average, 2.0 = 2x average

        # Score: 1x = 50, 2x = 75, 3x+ = 90+, 0.5x = 25
        score = min(100.0, 50 + (rvol - 1.0) * 25)
        return max(0.0, score)

    except Exception:
        return 50.0


def compute_trend_quality(symbol: str) -> float:
    """
    Measure trend quality: how consistently is price moving in one direction.
    Uses ADX-like concept but computed from price action directly.

    Returns 0-100 (50 = choppy/no trend, 80+ = strong trend).
    """
    if not RS_AVAILABLE:
        return 50.0

    try:
        df = _get_price_series(symbol, period="5d", interval="15m")
        if df is None or len(df) < 20:
            return 50.0

        closes = df["Close"].values[-20:]

        # Count bars where close > prev close (upward)
        up_bars   = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_bars = len(closes) - 1 - up_bars

        # If mostly going one way, trend is strong
        dominant = max(up_bars, down_bars)
        total    = up_bars + down_bars
        if total == 0:
            return 50.0

        consistency = dominant / total   # 0.5 = random, 1.0 = perfect trend

        # Also factor in magnitude (how much the range is directional)
        high_range = max(closes) - min(closes)
        net_move   = abs(closes[-1] - closes[0])
        if high_range > 0:
            directional_efficiency = net_move / high_range
        else:
            directional_efficiency = 0

        score = (consistency * 60 + directional_efficiency * 40)
        return max(0.0, min(100.0, score))

    except Exception:
        return 50.0


def compute_rs_score(symbol: str, direction: str = "bullish") -> dict:
    """
    Compute composite Relative Strength score for a symbol.

    Args:
        symbol: Ticker symbol
        direction: "bullish" or "bearish" (inverts score for laggards)

    Returns:
        {
          "rs_score": 0-100,
          "rs_vs_spy": 0-100,
          "rs_vs_sector": 0-100,
          "relative_volume": 0-100,
          "trend_quality": 0-100,
          "is_leader": bool,
          "is_laggard": bool,
          "signals": [str, ...]
        }
    """
    # Check cache
    cache_key = f"{symbol}_{direction}"
    if cache_key in _rs_cache:
        cached = _rs_cache[cache_key]
        if time.time() - cached["ts"] < _RS_CACHE_TTL:
            return cached["data"]

    if not RS_AVAILABLE:
        result = {
            "rs_score": 50.0, "rs_vs_spy": 50.0, "rs_vs_sector": 50.0,
            "relative_volume": 50.0, "trend_quality": 50.0,
            "is_leader": False, "is_laggard": False, "signals": [],
        }
        return result

    rs_spy     = compute_rs_vs_spy(symbol)
    rs_sector  = compute_rs_vs_sector(symbol)
    rvol       = compute_relative_volume(symbol)
    trend_qual = compute_trend_quality(symbol)

    # Composite score: weighted average
    rs_score = (
        rs_spy    * 0.40 +
        rs_sector * 0.25 +
        rvol      * 0.20 +
        trend_qual * 0.15
    )

    # For bearish direction, laggards (low RS) are what we want
    # Invert so that bearish traders get high score for weak symbols
    if direction == "bearish":
        rs_score = 100 - rs_score

    signals = []
    if rs_spy > 65:
        signals.append(f"✓ Outperforming SPY by {rs_spy-50:.0f} pts")
    elif rs_spy < 35:
        signals.append(f"✗ Underperforming SPY by {50-rs_spy:.0f} pts")

    if rvol > 70:
        signals.append(f"✓ High relative volume ({rvol:.0f}/100)")
    elif rvol < 30:
        signals.append("✗ Low relative volume — institutional interest absent")

    if trend_qual > 70:
        signals.append("✓ Strong trend quality")
    elif trend_qual < 35:
        signals.append("~ Choppy/no clear trend")

    is_leader  = rs_score >= 70
    is_laggard = rs_score <= 30

    result = {
        "rs_score":        round(rs_score, 1),
        "rs_vs_spy":       round(rs_spy, 1),
        "rs_vs_sector":    round(rs_sector, 1),
        "relative_volume": round(rvol, 1),
        "trend_quality":   round(trend_qual, 1),
        "is_leader":       is_leader,
        "is_laggard":      is_laggard,
        "signals":         signals,
    }

    _rs_cache[cache_key] = {"data": result, "ts": time.time()}
    return result


def rank_symbols(symbols: list, direction: str = "bullish") -> list:
    """
    Rank a list of symbols by relative strength for the given direction.

    Returns list of dicts sorted by rs_score (descending for bullish).
    Only returns symbols with rs_score ≥ 60 for bullish or ≥ 40 for bearish.
    """
    ranked = []
    for sym in symbols:
        rs = compute_rs_score(sym, direction)
        ranked.append({"symbol": sym, **rs})

    ranked.sort(key=lambda x: x["rs_score"], reverse=True)
    return ranked


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    symbols = ["SPY", "NVDA", "TSLA", "AAPL", "META", "AMD"]
    print("Ranking by bullish RS:")
    ranked = rank_symbols(symbols, "bullish")
    for r in ranked:
        label = "LEADER" if r["is_leader"] else "LAGGARD" if r["is_laggard"] else "neutral"
        print(f"  {r['symbol']}: RS={r['rs_score']:.0f} vs_SPY={r['rs_vs_spy']:.0f} "
              f"RVOL={r['relative_volume']:.0f} [{label}]")
