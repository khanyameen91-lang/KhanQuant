"""
indicators.py — Multi-Timeframe Technical Analysis Engine

Fetches OHLCV data via yfinance and computes a comprehensive
suite of technical indicators across multiple timeframes.
Used to generate the Technical Score (0-100) for the composite
confidence engine.

Indicators computed:
  - VWAP (intraday anchored)
  - EMA 9, 20, 50, 200
  - RSI (14)
  - MACD (12/26/9)
  - Bollinger Bands (20, 2σ)
  - ADX + DMI (14)
  - ATR (14)
  - Stochastic RSI
  - OBV
  - Relative volume
"""

import time
import warnings
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance --break-system-packages")

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    # pandas-ta removed from PyPI — all indicators computed manually, no action needed


# Timeframe configs: (yfinance interval, yfinance period)
TIMEFRAMES = {
    "1m":  ("1m",  "1d"),
    "5m":  ("5m",  "5d"),
    "15m": ("15m", "5d"),
    "30m": ("30m", "30d"),
    "1h":  ("1h",  "60d"),
    "1d":  ("1d",  "2y"),
}

# Cache to avoid hammering yfinance on every scan
_price_cache: dict = {}
_cache_ttl = 60  # seconds


def _get_ohlcv(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from yfinance with caching."""
    if not YF_AVAILABLE:
        return None

    cache_key = f"{symbol}_{interval}"
    cached = _price_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _cache_ttl:
        return cached["df"]

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        _price_cache[cache_key] = {"df": df, "ts": time.time()}
        return df
    except Exception as e:
        print(f"⚠️  yfinance error {symbol} {interval}: {e}")
        return None


# ── Core Indicator Computations ────────────────────────────────────────────────

def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series) -> tuple:
    ema12 = _compute_ema(series, 12)
    ema26 = _compute_ema(series, 26)
    macd = ema12 - ema26
    signal = _compute_ema(macd, 9)
    histogram = macd - signal
    return macd, signal, histogram


def _compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> tuple:
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    pct_b = (series - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / sma
    return upper, sma, lower, pct_b, bandwidth


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Return (ADX, +DI, -DI)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm.abs()] = 0
    minus_dm[minus_dm < plus_dm.abs()] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx, plus_di, minus_di


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP (resets each session for intraday frames)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap


def _compute_anchored_vwap(df: pd.DataFrame, anchor: str = "session") -> dict:
    """
    Anchored VWAP — reset at session open (9:30 ET).
    Returns {vwap, vwap_std_1, vwap_std_2} for current bar.
    Standard deviation bands give key S/R levels.

    anchor: 'session' = today's open, 'week' = Monday open.
    """
    if df is None or len(df) < 5:
        return {"anchored_vwap": None, "avwap_upper1": None,
                "avwap_lower1": None, "avwap_upper2": None, "avwap_lower2": None}

    # Find anchor point
    try:
        if df.index.tz is not None:
            df_et = df.tz_convert("America/New_York")
        else:
            df_et = df
        today = pd.Timestamp.now(tz="America/New_York").date()
        if anchor == "week":
            # Anchor to Monday of current week
            today_dt = pd.Timestamp.now(tz="America/New_York")
            monday = today_dt - pd.Timedelta(days=today_dt.weekday())
            mask = df_et.index >= monday.normalize()
        else:
            # Anchor to today's session open
            mask = df_et.index.date >= today
        session_df = df[mask] if mask.any() else df.tail(78)  # ~6.5h of 5m bars
    except Exception:
        session_df = df.tail(78)

    if len(session_df) < 2:
        session_df = df.tail(20)

    typical = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
    cum_vol  = session_df["volume"].cumsum()
    cum_tpv  = (typical * session_df["volume"]).cumsum()

    avwap = cum_tpv / cum_vol.replace(0, np.nan)
    current_avwap = float(avwap.iloc[-1]) if not avwap.empty else None

    # Standard deviation bands
    if current_avwap and len(session_df) >= 5:
        dev = (typical - avwap) ** 2
        cum_dev = (dev * session_df["volume"]).cumsum()
        vwap_std = (cum_dev / cum_vol.replace(0, np.nan)) ** 0.5
        std_val = float(vwap_std.iloc[-1]) if not vwap_std.empty else 0
        return {
            "anchored_vwap": round(current_avwap, 2),
            "avwap_upper1":  round(current_avwap + std_val, 2),
            "avwap_lower1":  round(current_avwap - std_val, 2),
            "avwap_upper2":  round(current_avwap + 2 * std_val, 2),
            "avwap_lower2":  round(current_avwap - 2 * std_val, 2),
            "avwap_std":     round(std_val, 2),
        }

    return {"anchored_vwap": current_avwap, "avwap_upper1": None,
            "avwap_lower1": None, "avwap_upper2": None, "avwap_lower2": None}


def _compute_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """
    Volume Profile: distributes volume across price levels to find:
    - POC  (Point of Control) — price level with most volume traded
    - VAH  (Value Area High) — top of 70% volume range
    - VAL  (Value Area Low)  — bottom of 70% volume range

    Uses the last session's 5m or 15m data.
    """
    if df is None or len(df) < 10:
        return {"poc": None, "vah": None, "val": None}

    # Use recent session bars
    session_bars = df.tail(78)  # ~6.5h at 5m interval

    price_min = float(session_bars["low"].min())
    price_max = float(session_bars["high"].max())
    if price_max <= price_min:
        return {"poc": None, "vah": None, "val": None}

    bin_size = (price_max - price_min) / bins
    price_levels = [price_min + i * bin_size for i in range(bins)]
    volume_bins  = [0.0] * bins

    for _, row in session_bars.iterrows():
        bar_low  = float(row["low"])
        bar_high = float(row["high"])
        bar_vol  = float(row["volume"])
        # Distribute volume proportionally across the bar's range
        for i, level in enumerate(price_levels):
            level_high = level + bin_size
            overlap_lo = max(level, bar_low)
            overlap_hi = min(level_high, bar_high)
            if overlap_hi > overlap_lo and (bar_high - bar_low) > 0:
                fraction = (overlap_hi - overlap_lo) / (bar_high - bar_low)
                volume_bins[i] += bar_vol * fraction

    total_vol = sum(volume_bins)
    if total_vol == 0:
        return {"poc": None, "vah": None, "val": None}

    # POC = bin with most volume
    poc_idx = volume_bins.index(max(volume_bins))
    poc = round(price_levels[poc_idx] + bin_size / 2, 2)

    # Value Area (70% of total volume, centered at POC)
    target_vol  = total_vol * 0.70
    acc_vol     = volume_bins[poc_idx]
    lo_idx      = poc_idx
    hi_idx      = poc_idx

    while acc_vol < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
        can_go_lo = lo_idx > 0
        can_go_hi = hi_idx < bins - 1
        add_lo = volume_bins[lo_idx - 1] if can_go_lo else 0
        add_hi = volume_bins[hi_idx + 1] if can_go_hi else 0
        if add_lo >= add_hi and can_go_lo:
            lo_idx -= 1
            acc_vol += volume_bins[lo_idx]
        elif can_go_hi:
            hi_idx += 1
            acc_vol += volume_bins[hi_idx]
        else:
            break

    vah = round(price_levels[hi_idx] + bin_size, 2)
    val = round(price_levels[lo_idx], 2)

    return {
        "poc":            poc,
        "vah":            vah,
        "val":            val,
        "volume_nodes":   len([v for v in volume_bins if v > total_vol / bins * 1.5]),
        "total_volume":   round(total_vol, 0),
    }


def _compute_stoch_rsi(rsi: pd.Series, period: int = 14) -> tuple:
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)
    k = stoch_rsi.rolling(3).mean() * 100
    d = k.rolling(3).mean()
    return k, d


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def _relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    """Current volume vs average volume of last N bars."""
    if len(df) < lookback + 1:
        return 1.0
    avg = df["volume"].iloc[-(lookback + 1):-1].mean()
    current = df["volume"].iloc[-1]
    return current / avg if avg > 0 else 1.0


# ── Full Timeframe Analysis ────────────────────────────────────────────────────

def analyze_timeframe(symbol: str, tf: str) -> Optional[dict]:
    """
    Compute all indicators for one timeframe.
    Returns a dict of indicator values (most recent bar).
    """
    interval, period = TIMEFRAMES.get(tf, ("1d", "2y"))
    df = _get_ohlcv(symbol, interval, period)
    if df is None or len(df) < 50:
        return None

    close = df["close"]
    price = float(close.iloc[-1])

    # EMAs
    ema9   = float(_compute_ema(close, 9).iloc[-1])
    ema20  = float(_compute_ema(close, 20).iloc[-1])
    ema50  = float(_compute_ema(close, 50).iloc[-1])
    ema200 = float(_compute_ema(close, 200).iloc[-1]) if len(df) >= 200 else ema50

    # RSI
    rsi_series = _compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])

    # MACD
    macd_line, macd_signal, macd_hist = _compute_macd(close)
    macd_val  = float(macd_line.iloc[-1])
    macd_sig  = float(macd_signal.iloc[-1])
    macd_h    = float(macd_hist.iloc[-1])
    macd_prev = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else macd_h

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bw = _compute_bollinger(close)
    bb_upper_v = float(bb_upper.iloc[-1])
    bb_lower_v = float(bb_lower.iloc[-1])
    bb_pct_b_v = float(bb_pct_b.iloc[-1])
    bb_bw_v    = float(bb_bw.iloc[-1])

    # ATR
    atr = float(_compute_atr(df).iloc[-1])
    atr_pct = (atr / price) * 100 if price > 0 else 0

    # ADX
    adx_series, plus_di, minus_di = _compute_adx(df)
    adx     = float(adx_series.iloc[-1])
    plus_di_v  = float(plus_di.iloc[-1])
    minus_di_v = float(minus_di.iloc[-1])

    # VWAP (intraday only meaningful for 1m/5m/15m)
    vwap = None
    anchored_vwap_data = {}
    volume_profile_data = {}
    if tf in ("1m", "5m", "15m", "30m"):
        vwap = float(_compute_vwap(df).iloc[-1])
        # Anchored VWAP + volume profile
        anchored_vwap_data  = _compute_anchored_vwap(df, anchor="session")
        volume_profile_data = _compute_volume_profile(df)

    # Stochastic RSI
    stoch_k, stoch_d = _compute_stoch_rsi(rsi_series)
    stoch_k_v = float(stoch_k.iloc[-1])
    stoch_d_v = float(stoch_d.iloc[-1])

    # OBV trend (is OBV rising or falling?)
    obv = _compute_obv(df)
    obv_trend = float(obv.iloc[-1]) - float(obv.iloc[-5]) if len(obv) > 5 else 0.0
    obv_rising = obv_trend > 0

    # Relative volume
    rel_vol = _relative_volume(df)

    # Trend direction
    above_ema9   = price > ema9
    above_ema20  = price > ema20
    above_ema50  = price > ema50
    above_ema200 = price > ema200
    ema_aligned_bull = ema9 > ema20 > ema50  # bullish alignment
    ema_aligned_bear = ema9 < ema20 < ema50  # bearish alignment

    return {
        "timeframe": tf,
        "price": price,
        "ema9": ema9, "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "above_ema9": above_ema9, "above_ema20": above_ema20,
        "above_ema50": above_ema50, "above_ema200": above_ema200,
        "ema_aligned_bull": ema_aligned_bull, "ema_aligned_bear": ema_aligned_bear,
        "rsi": rsi,
        "rsi_overbought": rsi > 70, "rsi_oversold": rsi < 30,
        "macd": macd_val, "macd_signal": macd_sig, "macd_hist": macd_h,
        "macd_bullish": macd_h > 0 and macd_h > macd_prev,
        "macd_bearish": macd_h < 0 and macd_h < macd_prev,
        "bb_upper": bb_upper_v, "bb_lower": bb_lower_v,
        "bb_pct_b": bb_pct_b_v, "bb_bandwidth": bb_bw_v,
        "bb_overbought": bb_pct_b_v > 0.9, "bb_oversold": bb_pct_b_v < 0.1,
        "bb_squeeze": bb_bw_v < 0.02,
        "atr": atr, "atr_pct": atr_pct,
        "adx": adx, "adx_trending": adx > 25, "adx_strong": adx > 40,
        "plus_di": plus_di_v, "minus_di": minus_di_v,
        "di_bullish": plus_di_v > minus_di_v,
        "vwap": vwap,
        "above_vwap": (price > vwap) if vwap else None,
        # Anchored VWAP
        "anchored_vwap":  anchored_vwap_data.get("anchored_vwap"),
        "avwap_upper1":   anchored_vwap_data.get("avwap_upper1"),
        "avwap_lower1":   anchored_vwap_data.get("avwap_lower1"),
        "avwap_upper2":   anchored_vwap_data.get("avwap_upper2"),
        "avwap_lower2":   anchored_vwap_data.get("avwap_lower2"),
        "above_avwap": (
            (price > anchored_vwap_data["anchored_vwap"])
            if anchored_vwap_data.get("anchored_vwap") else None
        ),
        # Volume Profile
        "vp_poc":  volume_profile_data.get("poc"),
        "vp_vah":  volume_profile_data.get("vah"),
        "vp_val":  volume_profile_data.get("val"),
        "price_in_value_area": (
            volume_profile_data.get("val", 0) <= price <= volume_profile_data.get("vah", float("inf"))
            if volume_profile_data.get("val") and volume_profile_data.get("vah") else None
        ),
        "stoch_k": stoch_k_v, "stoch_d": stoch_d_v,
        "stoch_overbought": stoch_k_v > 80, "stoch_oversold": stoch_k_v < 20,
        "obv_rising": obv_rising,
        "rel_volume": rel_vol, "high_rel_volume": rel_vol > 1.5,
    }


def get_full_analysis(symbol: str, primary_tf: str = "15m") -> dict:
    """
    Run analysis across all relevant timeframes and return
    a unified indicator snapshot with multi-timeframe alignment scores.
    """
    results = {}
    for tf in ["1m", "5m", "15m", "1h", "1d"]:
        data = analyze_timeframe(symbol, tf)
        if data:
            results[tf] = data

    if not results:
        return {}

    primary = results.get(primary_tf) or results.get("15m") or list(results.values())[-1]

    # Multi-timeframe confluence
    bull_count = sum(1 for tf_data in results.values() if tf_data.get("ema_aligned_bull"))
    bear_count = sum(1 for tf_data in results.values() if tf_data.get("ema_aligned_bear"))
    total_tfs  = len(results)

    macd_bull_count = sum(1 for tf_data in results.values() if tf_data.get("macd_bullish"))
    macd_bear_count = sum(1 for tf_data in results.values() if tf_data.get("macd_bearish"))

    above_vwap_count = sum(1 for tf_data in results.values()
                           if tf_data.get("above_vwap") is True)

    return {
        "symbol": symbol,
        "primary": primary,
        "timeframes": results,
        "mtf_bull_alignment": bull_count / total_tfs if total_tfs else 0,
        "mtf_bear_alignment": bear_count / total_tfs if total_tfs else 0,
        "mtf_macd_bull": macd_bull_count / total_tfs if total_tfs else 0,
        "mtf_macd_bear": macd_bear_count / total_tfs if total_tfs else 0,
        "above_vwap_pct": above_vwap_count / total_tfs if total_tfs else 0.5,
        "total_timeframes": total_tfs,
    }


# ── Technical Score (0-100) ────────────────────────────────────────────────────

def compute_technical_score(symbol: str, direction: str = "bullish") -> dict:
    """
    Compute a 0-100 technical score for a given symbol and trade direction.
    Higher = stronger setup from a technical perspective.
    Returns score + all supporting data.
    """
    analysis = get_full_analysis(symbol)
    if not analysis:
        return {"score": 50, "analysis": {}, "signals": [], "direction": direction}

    primary = analysis.get("primary", {})
    signals = []
    score_points = 0.0
    max_points   = 0.0

    def add(condition: bool, points: float, label: str):
        nonlocal score_points, max_points
        max_points += points
        if condition:
            score_points += points
            signals.append(f"✓ {label}")
        else:
            signals.append(f"✗ {label}")

    is_bull = direction == "bullish"

    # ── Trend alignment (30 pts)
    add(primary.get("ema_aligned_bull") if is_bull else primary.get("ema_aligned_bear"),
        10, "EMA stack aligned (9>20>50)" if is_bull else "EMA stack aligned (9<20<50)")
    add(primary.get("above_ema50") if is_bull else not primary.get("above_ema50", True),
        10, "Price above EMA50" if is_bull else "Price below EMA50")
    add(primary.get("above_ema200") if is_bull else not primary.get("above_ema200", True),
        10, "Price above EMA200" if is_bull else "Price below EMA200")

    # ── VWAP (10 pts) — regular + anchored
    vwap_check = primary.get("above_vwap")
    if vwap_check is not None:
        add(vwap_check if is_bull else not vwap_check,
            6, "Above VWAP" if is_bull else "Below VWAP")

    avwap_check = primary.get("above_avwap")
    if avwap_check is not None:
        add(avwap_check if is_bull else not avwap_check,
            4, "Above Anchored VWAP" if is_bull else "Below Anchored VWAP")

    # ── Volume Profile (10 pts bonus)
    poc = primary.get("vp_poc")
    vah = primary.get("vp_vah")
    val = primary.get("vp_val")
    price_v = primary.get("price", 0)
    in_value_area = primary.get("price_in_value_area")
    if poc and is_bull:
        add(price_v >= poc, 5, f"Price at/above POC ({poc:.2f})")
    elif poc and not is_bull:
        add(price_v <= poc, 5, f"Price at/below POC ({poc:.2f})")
    if vah and val:
        # Breakout above VAH is bullish; breakdown below VAL is bearish
        if is_bull:
            add(price_v > vah, 5, f"Breaking above Value Area High ({vah:.2f})")
        else:
            add(price_v < val, 5, f"Breaking below Value Area Low ({val:.2f})")

    # ── Momentum (25 pts)
    rsi = primary.get("rsi", 50)
    rsi_bullish = 45 < rsi < 70
    rsi_bearish = 30 < rsi < 55
    add(rsi_bullish if is_bull else rsi_bearish,
        10, f"RSI healthy ({rsi:.0f})")

    add(primary.get("macd_bullish") if is_bull else primary.get("macd_bearish"),
        10, "MACD histogram expanding in direction")

    add(primary.get("di_bullish") if is_bull else not primary.get("di_bullish", True),
        5, "+DI > -DI" if is_bull else "-DI > +DI")

    # ── Trend strength (15 pts)
    add(primary.get("adx_trending"),
        10, f"ADX trending ({primary.get('adx', 0):.0f} > 25)")
    add(primary.get("high_rel_volume"),
        5, f"Relative volume elevated ({primary.get('rel_volume', 1):.1f}x)")

    # ── Multi-timeframe confluence (20 pts)
    mtf_bull = analysis.get("mtf_bull_alignment", 0)
    mtf_bear = analysis.get("mtf_bear_alignment", 0)
    mtf_aligned = mtf_bull >= 0.6 if is_bull else mtf_bear >= 0.6
    add(mtf_aligned, 10, f"MTF EMA alignment ({(mtf_bull if is_bull else mtf_bear)*100:.0f}%)")

    mtf_macd = analysis.get("mtf_macd_bull", 0) if is_bull else analysis.get("mtf_macd_bear", 0)
    add(mtf_macd >= 0.6, 10, f"MTF MACD confluence ({mtf_macd*100:.0f}%)")

    # ── OBV confirmation
    add(primary.get("obv_rising") if is_bull else not primary.get("obv_rising", True),
        5, "OBV confirms direction")

    # Normalize to 0-100
    score = (score_points / max_points * 100) if max_points > 0 else 50
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "direction": direction,
        "signals": signals,
        "primary_tf": primary.get("timeframe"),
        "rsi": primary.get("rsi"),
        "adx": primary.get("adx"),
        "atr": primary.get("atr"),
        "atr_pct": primary.get("atr_pct"),
        "vwap": primary.get("vwap"),
        "anchored_vwap": primary.get("anchored_vwap"),
        "avwap_upper1": primary.get("avwap_upper1"),
        "avwap_lower1": primary.get("avwap_lower1"),
        "vp_poc": primary.get("vp_poc"),
        "vp_vah": primary.get("vp_vah"),
        "vp_val": primary.get("vp_val"),
        "rel_volume": primary.get("rel_volume"),
        "bb_squeeze": primary.get("bb_squeeze"),
        "analysis": analysis,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing indicators on SPY (bullish)...")
    result = compute_technical_score("SPY", "bullish")
    print(f"\nTechnical Score: {result['score']}/100")
    print(f"RSI: {result.get('rsi', 'N/A'):.1f}" if result.get('rsi') else "RSI: N/A")
    print(f"ADX: {result.get('adx', 'N/A'):.1f}" if result.get('adx') else "ADX: N/A")
    print(f"Rel Volume: {result.get('rel_volume', 'N/A'):.2f}x" if result.get('rel_volume') else "Rel Vol: N/A")
    print("\nSignals:")
    for s in result["signals"]:
        print(f"  {s}")
