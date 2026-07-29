"""
regime.py — Market Regime Detection Engine (v3)

Automatically identifies the current market environment
and recommends optimal strategy types.

14 Regime Types:
  Trend:         TRENDING_BULL | TRENDING_BEAR | RANGING | CHOPPY
  Volatility:    HIGH_VOL | NORMAL_VOL | LOW_VOL | VOL_EXPANDING | VOL_CONTRACTING
  Composite:     STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
  Calendar:      FOMC_DAY | CPI_DAY | OPEX_DAY | TRIPLE_WITCHING
  Special:       GAP_AND_GO | REVERSAL_DAY | EARNINGS_DAY
  Vol Events:    VOL_EXPANSION | VOL_COMPRESSION

Output feeds strategy selection, score weighting, and position sizing.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import date
import json

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from indicators import get_full_analysis, _get_ohlcv, _compute_atr, _compute_adx, _compute_ema, _compute_rsi


# ── Regime Constants ───────────────────────────────────────────────────────────
VIX_LOW       = 15   # Low vol regime
VIX_NORMAL    = 20
VIX_HIGH      = 25   # High vol regime
VIX_EXTREME   = 35   # Extreme fear

ADX_CHOPPY    = 20   # Below this = choppy / no trend
ADX_TRENDING  = 25
ADX_STRONG    = 40

REGIME_CACHE_FILE = Path("logs/regime_cache.json")
REGIME_CACHE_TTL  = 300  # 5 minutes


@dataclass
class MarketRegime:
    # Trend regime
    trend: str = "RANGING"          # TRENDING_BULL | TRENDING_BEAR | RANGING | CHOPPY
    trend_strength: float = 0.0     # 0-100, ADX-derived

    # Volatility regime
    vol: str = "NORMAL_VOL"         # HIGH_VOL | NORMAL_VOL | LOW_VOL | EXTREME_VOL
    vol_direction: str = "STABLE"   # EXPANDING | CONTRACTING | STABLE
    vix: float = 20.0
    vix_pct: float = 50.0           # VIX percentile (0-100)

    # Composite market condition
    condition: str = "NEUTRAL"      # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR

    # Breadth
    spy_vs_ema50: float = 0.0       # SPY % above/below EMA50
    qqq_vs_ema50: float = 0.0

    # Calendar/event regimes
    is_fomc_day:         bool = False
    is_cpi_day:          bool = False
    is_opex_day:         bool = False
    is_triple_witching:  bool = False
    is_gap_and_go:       bool = False
    is_reversal_day:     bool = False
    is_vol_expansion:    bool = False
    is_vol_compression:  bool = False
    calendar_event:      str  = ""   # Human-readable description of calendar event

    # Opening gap tracking
    spy_gap_pct: float = 0.0        # Today's gap up/down vs yesterday close

    # Size multiplier based on regime risk
    size_multiplier: float = 1.0    # 0.0 (FOMC=no trades) to 1.0 (normal)

    # Best strategies for this regime
    preferred_strategies: list = field(default_factory=list)
    avoid_strategies: list = field(default_factory=list)

    # Scoring adjustments
    score_adjustments: dict = field(default_factory=dict)

    # Summary
    summary: str = "Neutral market regime"
    confidence: float = 0.0


def _get_vix() -> tuple[float, float]:
    """Get current VIX level and 52-week percentile."""
    if not YF_AVAILABLE:
        return 20.0, 50.0
    try:
        vix_ticker = yf.Ticker("^VIX")
        hist = vix_ticker.history(period="1y")
        if hist.empty:
            return 20.0, 50.0
        current_vix = float(hist["Close"].iloc[-1])
        pct = float((hist["Close"] < current_vix).mean() * 100)
        return current_vix, pct
    except Exception:
        return 20.0, 50.0


def _get_index_analysis(symbol: str) -> dict:
    """Get key metrics for SPY/QQQ for breadth analysis."""
    analysis = get_full_analysis(symbol, primary_tf="1d")
    if not analysis:
        return {}
    primary = analysis.get("primary", {})
    return {
        "above_ema50": primary.get("above_ema50", True),
        "above_ema200": primary.get("above_ema200", True),
        "adx": primary.get("adx", 20),
        "rsi": primary.get("rsi", 50),
        "macd_bullish": primary.get("macd_bullish", False),
        "macd_bearish": primary.get("macd_bearish", False),
        "ema_aligned_bull": primary.get("ema_aligned_bull", False),
        "ema_aligned_bear": primary.get("ema_aligned_bear", False),
        "price": primary.get("price", 0),
        "ema50": primary.get("ema50", 0),
        "ema200": primary.get("ema200", 0),
    }


def _detect_calendar_events(regime: MarketRegime) -> MarketRegime:
    """
    Detect calendar-based regime events:
    FOMC Day, CPI Day, OPEX Day, Triple Witching.
    Uses Finnhub economic calendar + date math.
    """
    today = date.today()

    # ── OPEX Day: 3rd Friday of each month ────────────────────────────────────
    # Find 3rd Friday
    first_day = today.replace(day=1)
    first_friday_offset = (4 - first_day.weekday()) % 7   # days to first Friday
    third_friday = first_day.replace(day=1 + first_friday_offset + 14)
    if today == third_friday:
        regime.is_opex_day    = True
        regime.calendar_event = "OPEX Day (3rd Friday)"
        regime.size_multiplier = min(regime.size_multiplier, 0.5)

    # ── Triple Witching: 3rd Friday of Mar, Jun, Sep, Dec ────────────────────
    if today == third_friday and today.month in (3, 6, 9, 12):
        regime.is_triple_witching = True
        regime.calendar_event     = "Triple Witching"
        regime.size_multiplier    = min(regime.size_multiplier, 0.25)

    # ── FOMC / CPI / PPI detection via intelligence cache ────────────────────
    try:
        intel_cache = Path("logs/intelligence_cache.json")
        if intel_cache.exists():
            data = json.loads(intel_cache.read_text())
            macro = data.get("macro", {})
            upcoming = [e.lower() for e in macro.get("upcoming_events", [])]
            next_hrs  = macro.get("next_event_hours", 999)

            if any("fomc" in e or "fed rate" in e or "powell" in e for e in upcoming):
                if next_hrs <= 8:
                    regime.is_fomc_day     = True
                    regime.calendar_event  = "FOMC Meeting Day"
                    regime.size_multiplier = 0.0  # No new trades on FOMC day

            if any("cpi" in e or "inflation" in e or "consumer price" in e for e in upcoming):
                if next_hrs <= 6:
                    regime.is_cpi_day      = True
                    regime.calendar_event  = "CPI Release Day"
                    regime.size_multiplier = min(regime.size_multiplier, 0.5)

            if any("ppi" in e or "producer price" in e for e in upcoming):
                if next_hrs <= 6:
                    regime.calendar_event  = "PPI Release Day"
                    regime.size_multiplier = min(regime.size_multiplier, 0.5)

            if any("nonfarm" in e or "jobs" in e or "unemployment" in e for e in upcoming):
                if next_hrs <= 4:
                    regime.calendar_event  = "Jobs Report Day"
                    regime.size_multiplier = min(regime.size_multiplier, 0.5)

    except Exception:
        pass

    return regime


def _detect_intraday_events(regime: MarketRegime) -> MarketRegime:
    """
    Detect intraday patterns: gap-and-go, reversal day,
    volatility expansion/compression.
    """
    if not YF_AVAILABLE:
        return regime

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="5d", interval="1d")
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            today_open = float(hist["Open"].iloc[-1])
            gap_pct    = (today_open - prev_close) / prev_close * 100
            regime.spy_gap_pct = round(gap_pct, 2)

            # Gap-and-Go: gap > 0.5% that is continuing (not fading)
            today_close = float(hist["Close"].iloc[-1])
            gap_continued = (
                (gap_pct > 0.5 and today_close > today_open) or
                (gap_pct < -0.5 and today_close < today_open)
            )
            if abs(gap_pct) >= 0.5 and gap_continued:
                regime.is_gap_and_go = True

            # Reversal Day: gap that faded back
            gap_faded = (
                (gap_pct > 0.5 and today_close < prev_close) or
                (gap_pct < -0.5 and today_close > prev_close)
            )
            if abs(gap_pct) >= 0.3 and gap_faded:
                regime.is_reversal_day = True

    except Exception:
        pass

    # Vol expansion/compression from vol_direction
    if regime.vol_direction == "EXPANDING" and regime.vix > 20:
        regime.is_vol_expansion = True
    elif regime.vol_direction == "CONTRACTING" and regime.vix < 20:
        regime.is_vol_compression = True

    return regime


def detect_regime() -> MarketRegime:
    """
    Detect the current market regime from VIX, SPY, QQQ data.
    Cached for REGIME_CACHE_TTL seconds to avoid redundant fetches.
    """
    # Try cache first
    if REGIME_CACHE_FILE.exists():
        try:
            cached = json.loads(REGIME_CACHE_FILE.read_text())
            if time.time() - cached.get("ts", 0) < REGIME_CACHE_TTL:
                return _dict_to_regime(cached["regime"])
        except Exception:
            pass

    regime = _compute_regime()

    # Cache result
    try:
        REGIME_CACHE_FILE.parent.mkdir(exist_ok=True)
        REGIME_CACHE_FILE.write_text(json.dumps({
            "ts": time.time(),
            "regime": _regime_to_dict(regime)
        }, indent=2))
    except Exception:
        pass

    return regime


def _compute_regime() -> MarketRegime:
    """Core regime computation logic."""
    regime = MarketRegime()

    # ── VIX Analysis ──────────────────────────────────────────────────────────
    vix, vix_pct = _get_vix()
    regime.vix = vix
    regime.vix_pct = vix_pct

    if vix >= VIX_EXTREME:
        regime.vol = "EXTREME_VOL"
    elif vix >= VIX_HIGH:
        regime.vol = "HIGH_VOL"
    elif vix <= VIX_LOW:
        regime.vol = "LOW_VOL"
    else:
        regime.vol = "NORMAL_VOL"

    # VIX direction (is vol expanding or contracting?)
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="5d")
        if not vix_hist.empty and len(vix_hist) >= 3:
            vix_5d_ago = float(vix_hist["Close"].iloc[-3])
            if vix > vix_5d_ago * 1.10:
                regime.vol_direction = "EXPANDING"
            elif vix < vix_5d_ago * 0.90:
                regime.vol_direction = "CONTRACTING"
            else:
                regime.vol_direction = "STABLE"
    except Exception:
        pass

    # ── SPY / QQQ Breadth ────────────────────────────────────────────────────
    spy = _get_index_analysis("SPY")
    qqq = _get_index_analysis("QQQ")

    spy_price = spy.get("price", 0)
    spy_ema50 = spy.get("ema50", spy_price)
    spy_pct_above_ema50 = ((spy_price / spy_ema50) - 1) * 100 if spy_ema50 else 0
    regime.spy_vs_ema50 = spy_pct_above_ema50

    qqq_price = qqq.get("price", 0)
    qqq_ema50 = qqq.get("ema50", qqq_price)
    qqq_pct_above_ema50 = ((qqq_price / qqq_ema50) - 1) * 100 if qqq_ema50 else 0
    regime.qqq_vs_ema50 = qqq_pct_above_ema50

    # ── Trend Detection ───────────────────────────────────────────────────────
    spy_adx = spy.get("adx", 20)
    regime.trend_strength = spy_adx

    # Bullish signals
    bull_signals = sum([
        spy.get("above_ema50", False),
        spy.get("above_ema200", False),
        spy.get("ema_aligned_bull", False),
        spy.get("macd_bullish", False),
        qqq.get("above_ema50", False),
        qqq.get("ema_aligned_bull", False),
        spy.get("rsi", 50) > 50,
    ])

    bear_signals = sum([
        not spy.get("above_ema50", True),
        not spy.get("above_ema200", True),
        spy.get("ema_aligned_bear", False),
        spy.get("macd_bearish", False),
        not qqq.get("above_ema50", True),
        qqq.get("ema_aligned_bear", False),
        spy.get("rsi", 50) < 50,
    ])

    total_signals = 7
    bull_pct = bull_signals / total_signals
    bear_pct = bear_signals / total_signals

    if spy_adx < ADX_CHOPPY:
        regime.trend = "CHOPPY"
    elif spy_adx < ADX_TRENDING:
        regime.trend = "RANGING"
    elif bull_pct >= 0.65:
        regime.trend = "TRENDING_BULL"
    elif bear_pct >= 0.65:
        regime.trend = "TRENDING_BEAR"
    else:
        regime.trend = "RANGING"

    # ── Composite Condition ───────────────────────────────────────────────────
    if bull_pct >= 0.75 and vix < VIX_HIGH and regime.trend == "TRENDING_BULL":
        regime.condition = "STRONG_BULL"
    elif bull_pct >= 0.55 and vix < VIX_EXTREME:
        regime.condition = "BULL"
    elif bear_pct >= 0.75 and regime.trend == "TRENDING_BEAR":
        regime.condition = "STRONG_BEAR"
    elif bear_pct >= 0.55:
        regime.condition = "BEAR"
    else:
        regime.condition = "NEUTRAL"

    # ── Calendar event detection ──────────────────────────────────────────────
    regime = _detect_calendar_events(regime)
    regime = _detect_intraday_events(regime)

    # ── Strategy Preferences ──────────────────────────────────────────────────
    preferred = []
    avoid     = []

    # Override everything on high-risk calendar days
    if regime.is_fomc_day:
        preferred = []
        avoid     = ["long_option", "debit_spread", "credit_spread",
                     "iron_condor", "strangle", "0dte_scalp"]
    elif regime.is_triple_witching:
        preferred = ["credit_spread"]
        avoid     = ["long_option", "0dte_scalp", "strangle", "iron_condor"]
    elif regime.is_opex_day:
        preferred = ["credit_spread", "iron_condor"]
        avoid     = ["0dte_scalp", "strangle"]
    elif regime.is_cpi_day:
        preferred = []   # Wait until 30min after CPI release
        avoid     = ["long_option", "debit_spread"]
    elif regime.is_gap_and_go:
        preferred = ["long_option", "debit_spread", "0dte_scalp"]
        avoid     = ["iron_condor", "credit_spread"]
    elif regime.is_reversal_day:
        preferred = ["long_option", "debit_spread"]   # Fades only
        avoid     = ["iron_condor"]
    elif regime.is_vol_expansion:
        preferred = ["long_option", "strangle", "debit_spread"]
        avoid     = ["credit_spread", "iron_condor"]
    elif regime.is_vol_compression:
        preferred = ["credit_spread", "iron_condor"]
        avoid     = ["long_option", "strangle"]
    else:
        # Standard regime-based selection
        if regime.trend == "TRENDING_BULL":
            preferred += ["long_option", "debit_spread", "credit_spread"]
            avoid     += ["iron_condor"]
        elif regime.trend == "TRENDING_BEAR":
            preferred += ["long_option", "debit_spread", "credit_spread"]
            avoid     += ["iron_condor"]
        elif regime.trend in ("RANGING", "CHOPPY"):
            preferred += ["iron_condor", "credit_spread"]
            avoid     += ["0dte_scalp", "strangle"]

        if regime.vol in ("HIGH_VOL", "EXTREME_VOL"):
            preferred += ["credit_spread", "iron_condor"]
            avoid     += ["long_option", "strangle"]
        elif regime.vol == "LOW_VOL":
            preferred += ["long_option", "strangle", "0dte_scalp"]
            avoid     += ["iron_condor"]

    # Remove duplicates
    regime.preferred_strategies = list(dict.fromkeys(preferred))
    regime.avoid_strategies     = list(dict.fromkeys(avoid))

    # ── Score Adjustments ────────────────────────────────────────────────────
    adjustments = {}
    if regime.vol in ("HIGH_VOL", "EXTREME_VOL"):
        adjustments["flow_weight"]      = 1.3
        adjustments["tech_weight"]      = 0.8
        adjustments["sentiment_weight"] = 1.2
    elif regime.vol == "LOW_VOL":
        adjustments["tech_weight"]  = 1.3
        adjustments["flow_weight"]  = 0.8

    if regime.trend in ("TRENDING_BULL", "TRENDING_BEAR"):
        adjustments["tech_weight"] = adjustments.get("tech_weight", 1.0) * 1.2
    elif regime.trend == "CHOPPY":
        adjustments["tech_weight"] = adjustments.get("tech_weight", 1.0) * 0.7

    if regime.is_fomc_day or regime.is_triple_witching:
        adjustments["sentiment_weight"] = 2.0   # News dominates on these days

    regime.score_adjustments = adjustments

    # ── Summary ───────────────────────────────────────────────────────────────
    regime.confidence = max(bull_pct, bear_pct) * 100

    calendar_str = f" | {regime.calendar_event}" if regime.calendar_event else ""
    gap_str = f" | GAP {regime.spy_gap_pct:+.1f}%" if abs(regime.spy_gap_pct) > 0.3 else ""
    size_str = f" | SIZE {regime.size_multiplier:.0%}" if regime.size_multiplier < 1.0 else ""

    regime.summary = (
        f"{regime.condition} | {regime.trend} | "
        f"VIX {vix:.1f} ({regime.vol})"
        f"{calendar_str}{gap_str}{size_str}"
    )

    print(f"Regime: {regime.summary}")
    return regime


def _regime_to_dict(r: MarketRegime) -> dict:
    return {
        "trend": r.trend, "trend_strength": r.trend_strength,
        "vol": r.vol, "vol_direction": r.vol_direction,
        "vix": r.vix, "vix_pct": r.vix_pct,
        "condition": r.condition,
        "spy_vs_ema50": r.spy_vs_ema50, "qqq_vs_ema50": r.qqq_vs_ema50,
        "preferred_strategies": r.preferred_strategies,
        "avoid_strategies": r.avoid_strategies,
        "score_adjustments": r.score_adjustments,
        "summary": r.summary, "confidence": r.confidence,
    }


def _dict_to_regime(d: dict) -> MarketRegime:
    r = MarketRegime()
    for k, v in d.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r


def regime_strategy_score(setup_type: str, regime: MarketRegime) -> float:
    """
    Return a 0-1 multiplier for how well a strategy fits the current regime.
    Used to adjust confidence scores.
    """
    # Hard blocks for extreme calendar events
    if regime.is_fomc_day:
        return 0.0   # No trades on FOMC day
    if regime.is_triple_witching:
        return 0.25  # Near-zero on triple witching
    if regime.size_multiplier == 0.0:
        return 0.0

    if setup_type in regime.avoid_strategies:
        return 0.4   # Heavy penalty
    if setup_type in regime.preferred_strategies:
        return 1.3   # Bonus
    return 1.0       # Neutral


def get_regime_size_multiplier(regime: MarketRegime) -> float:
    """Return the position size multiplier for this regime (0.0-1.0)."""
    return regime.size_multiplier


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Detecting market regime...")
    r = detect_regime()
    print(f"\nRegime: {r.summary}")
    print(f"Trend: {r.trend} (strength: {r.trend_strength:.0f})")
    print(f"Volatility: {r.vol} (direction: {r.vol_direction})")
    print(f"VIX: {r.vix:.1f} ({r.vix_pct:.0f}th pct)")
    print(f"Condition: {r.condition} (confidence: {r.confidence:.0f}%)")
    print(f"Preferred: {r.preferred_strategies}")
    print(f"Avoid: {r.avoid_strategies}")
