"""
flow_analyzer.py — Options Flow Intelligence Engine

Analyzes options chain data to detect unusual activity,
sweep orders, block trades, gamma positioning, and
put/call dynamics.

Detects:
  - Unusual volume vs open interest (sweeps / blocks)
  - Put/call volume ratio (directional bias)
  - Put/call OI ratio (dealer positioning)
  - Net gamma exposure (dealer hedging pressure)
  - OI changes between scans (accumulation detection)
  - Strike clustering (magnet levels)

All data is sourced from Tastytrade API — no additional
paid data feeds required.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from auth import session

# Persistent OI snapshot for change detection (compared each scan)
OI_SNAPSHOT_FILE = Path("logs/oi_snapshot.json")
FLOW_LOG_FILE    = Path("logs/flow_log.json")


# ── Low-level data fetching ────────────────────────────────────────────────────

def _get_full_chain(symbol: str, expiration: str) -> list:
    """Fetch full option chain with all available data."""
    try:
        params = {"expiration-date": expiration}
        resp = session.get(f"/option-chains/{symbol}/nested", params=params)
        expirations = resp.get("data", {}).get("items", [])

        contracts = []
        for exp in expirations:
            if exp.get("expiration-date") != expiration:
                continue
            for strike_data in exp.get("strikes", []):
                strike = float(strike_data.get("strike-price", 0))
                for opt_type, key in [("C", "call"), ("P", "put")]:
                    c = strike_data.get(key, {})
                    if not c:
                        continue
                    contracts.append({
                        "symbol":        c.get("symbol", ""),
                        "underlying":    symbol,
                        "expiration":    expiration,
                        "strike":        strike,
                        "type":          opt_type,
                        "bid":           float(c.get("bid", 0)),
                        "ask":           float(c.get("ask", 0)),
                        "last":          float(c.get("last", 0)),
                        "volume":        int(c.get("volume", 0)),
                        "open_interest": int(c.get("open-interest", 0)),
                        "delta":         float(c.get("delta", 0)),
                        "gamma":         float(c.get("gamma", 0)),
                        "theta":         float(c.get("theta", 0)),
                        "vega":          float(c.get("vega", 0)),
                        "iv":            float(c.get("implied-volatility", 0)) * 100,
                    })
        return contracts
    except Exception as e:
        print(f"⚠️  Chain fetch error {symbol}: {e}")
        return []


def _get_near_expirations(symbol: str, max_dte: int = 30) -> list:
    """Get all expirations within max_dte days."""
    try:
        resp = session.get(f"/option-chains/{symbol}/expirations")
        exps = resp.get("data", {}).get("items", [])
        today = date.today()
        result = []
        for exp in exps:
            exp_str = exp.get("expiration-date")
            if exp_str:
                dte = (date.fromisoformat(exp_str) - today).days
                if 0 <= dte <= max_dte:
                    result.append(exp_str)
        return sorted(result)
    except Exception:
        return []


# ── Unusual Activity Detection ────────────────────────────────────────────────

def detect_unusual_activity(contracts: list, threshold: float = 3.0) -> list:
    """
    Find contracts where volume / open_interest > threshold.
    This indicates unusual buying activity (potential sweeps or institutional buying).
    """
    unusual = []
    for c in contracts:
        oi = c["open_interest"]
        vol = c["volume"]
        if oi <= 0 or vol <= 0:
            continue
        ratio = vol / oi
        if ratio >= threshold and vol >= 100:  # min 100 contracts
            premium = c["ask"] * vol * 100
            unusual.append({
                **c,
                "vol_oi_ratio": round(ratio, 2),
                "premium_moved": round(premium, 0),
                "signal": "UNUSUAL_VOLUME",
            })
    return sorted(unusual, key=lambda x: x["vol_oi_ratio"], reverse=True)


def detect_large_trades(contracts: list, min_premium: float = 50000) -> list:
    """
    Detect block trades — single options orders worth $50K+ in premium.
    Approximated as high volume * midpoint * 100.
    """
    blocks = []
    for c in contracts:
        if c["volume"] <= 0:
            continue
        mid = (c["bid"] + c["ask"]) / 2
        premium = mid * c["volume"] * 100
        if premium >= min_premium:
            blocks.append({
                **c,
                "estimated_premium": round(premium, 0),
                "signal": "BLOCK_TRADE",
            })
    return sorted(blocks, key=lambda x: x["estimated_premium"], reverse=True)


# ── Put/Call Analysis ─────────────────────────────────────────────────────────

def compute_put_call_ratios(contracts: list) -> dict:
    """
    Compute volume and OI put/call ratios.
    PC ratio > 1.2 = bearish sentiment / fear
    PC ratio < 0.7 = bullish sentiment / greed
    """
    call_vol = sum(c["volume"] for c in contracts if c["type"] == "C")
    put_vol  = sum(c["volume"] for c in contracts if c["type"] == "P")
    call_oi  = sum(c["open_interest"] for c in contracts if c["type"] == "C")
    put_oi   = sum(c["open_interest"] for c in contracts if c["type"] == "P")

    vol_pc_ratio = put_vol / call_vol if call_vol > 0 else 1.0
    oi_pc_ratio  = put_oi / call_oi if call_oi > 0 else 1.0

    # Interpret
    if vol_pc_ratio > 1.2:
        vol_sentiment = "BEARISH"
    elif vol_pc_ratio < 0.7:
        vol_sentiment = "BULLISH"
    else:
        vol_sentiment = "NEUTRAL"

    if oi_pc_ratio > 1.3:
        oi_sentiment = "BEARISH"
    elif oi_pc_ratio < 0.7:
        oi_sentiment = "BULLISH"
    else:
        oi_sentiment = "NEUTRAL"

    return {
        "call_volume": call_vol,
        "put_volume":  put_vol,
        "call_oi":     call_oi,
        "put_oi":      put_oi,
        "vol_pc_ratio": round(vol_pc_ratio, 3),
        "oi_pc_ratio":  round(oi_pc_ratio, 3),
        "vol_sentiment": vol_sentiment,
        "oi_sentiment":  oi_sentiment,
    }


# ── Gamma Exposure ────────────────────────────────────────────────────────────

def compute_gamma_exposure(contracts: list, price: float) -> dict:
    """
    Estimate net dealer gamma exposure (GEX).
    Dealers are SHORT calls and LONG puts by convention.
    Positive GEX = dealers are long gamma (stabilizing)
    Negative GEX = dealers are short gamma (amplifying moves)

    GEX per strike = gamma * OI * 100 * price^2 * 0.01
    """
    gex_by_strike = {}
    for c in contracts:
        strike = c["strike"]
        gamma  = c["gamma"]
        oi     = c["open_interest"]
        # Dealer gamma: short calls = -gamma, long puts = +gamma
        if c["type"] == "C":
            gex = -gamma * oi * 100 * (price ** 2) * 0.01
        else:
            gex = gamma * oi * 100 * (price ** 2) * 0.01

        if strike not in gex_by_strike:
            gex_by_strike[strike] = 0.0
        gex_by_strike[strike] += gex

    total_gex = sum(gex_by_strike.values())

    # Find gamma flip point (where GEX changes sign)
    sorted_strikes = sorted(gex_by_strike.keys())
    gamma_flip = None
    for i in range(len(sorted_strikes) - 1):
        s1, s2 = sorted_strikes[i], sorted_strikes[i + 1]
        g1, g2 = gex_by_strike[s1], gex_by_strike[s2]
        if g1 * g2 < 0:  # sign change
            gamma_flip = (s1 + s2) / 2
            break

    # Key resistance/support from max gamma strikes
    top_gex_strikes = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "total_gex": round(total_gex, 0),
        "gex_positive": total_gex > 0,
        "gamma_flip": gamma_flip,
        "top_gamma_strikes": [{"strike": s, "gex": round(g, 0)} for s, g in top_gex_strikes],
        "regime": "POSITIVE_GEX" if total_gex > 0 else "NEGATIVE_GEX",
    }


# ── OI Change Detection ───────────────────────────────────────────────────────

def _load_oi_snapshot() -> dict:
    if OI_SNAPSHOT_FILE.exists():
        try:
            return json.loads(OI_SNAPSHOT_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_oi_snapshot(snapshot: dict):
    OI_SNAPSHOT_FILE.parent.mkdir(exist_ok=True)
    OI_SNAPSHOT_FILE.write_text(json.dumps(snapshot))


def detect_oi_changes(symbol: str, contracts: list, expiration: str) -> list:
    """
    Compare current OI to last scan's OI to find new accumulation.
    Rising OI + rising volume = new positions (directional signal).
    """
    snap = _load_oi_snapshot()
    key = f"{symbol}_{expiration}"
    prev = snap.get(key, {})

    changes = []
    current_snap = {}

    for c in contracts:
        sym = c["symbol"]
        current_oi = c["open_interest"]
        prev_oi    = prev.get(sym, current_oi)
        oi_change  = current_oi - prev_oi
        current_snap[sym] = current_oi

        if abs(oi_change) >= 100 and c["volume"] > 0:
            pct = (oi_change / prev_oi * 100) if prev_oi > 0 else 0
            changes.append({
                **c,
                "oi_change": oi_change,
                "oi_change_pct": round(pct, 1),
                "signal": "OI_RISING" if oi_change > 0 else "OI_FALLING",
            })

    # Persist new snapshot
    snap[key] = current_snap
    _save_oi_snapshot(snap)

    return sorted(changes, key=lambda x: abs(x["oi_change"]), reverse=True)[:10]


# ── Strike Clustering / Max Pain ─────────────────────────────────────────────

def compute_max_pain(contracts: list) -> float:
    """
    Max pain: the strike where total option premium expires worthless.
    Strong gravitational pull on price as expiration approaches.
    """
    strikes = sorted(set(c["strike"] for c in contracts))
    if not strikes:
        return 0.0

    pain = {}
    for test_price in strikes:
        total_pain = 0
        for c in contracts:
            if c["type"] == "C":
                intrinsic = max(0, test_price - c["strike"])
            else:
                intrinsic = max(0, c["strike"] - test_price)
            total_pain += intrinsic * c["open_interest"] * 100
        pain[test_price] = total_pain

    return min(pain, key=pain.get) if pain else 0.0


# ── Full Flow Analysis ────────────────────────────────────────────────────────

def analyze_flow(symbol: str, price: float) -> dict:
    """
    Run complete options flow analysis for a symbol.
    Returns a scored flow report.
    """
    expirations = _get_near_expirations(symbol, max_dte=30)
    if not expirations:
        return _empty_flow(symbol)

    all_contracts = []
    nearest_exp = expirations[0]

    for exp in expirations[:3]:  # analyze nearest 3 expirations
        chain = _get_full_chain(symbol, exp)
        all_contracts.extend(chain)

    if not all_contracts:
        return _empty_flow(symbol)

    # Core analyses
    pc_ratios    = compute_put_call_ratios(all_contracts)
    unusual      = detect_unusual_activity(all_contracts, threshold=2.5)
    blocks       = detect_large_trades(all_contracts, min_premium=25000)
    gex          = compute_gamma_exposure(all_contracts, price)
    oi_changes   = detect_oi_changes(symbol, all_contracts, nearest_exp)
    max_pain_val = compute_max_pain(all_contracts)

    # Flow direction: what is the smart money doing?
    call_unusual = [u for u in unusual if u["type"] == "C"]
    put_unusual  = [u for u in unusual if u["type"] == "P"]
    call_blocks  = [b for b in blocks if b["type"] == "C"]
    put_blocks   = [b for b in blocks if b["type"] == "P"]

    call_premium = sum(b["estimated_premium"] for b in call_blocks)
    put_premium  = sum(b["estimated_premium"] for b in put_blocks)

    if call_premium > put_premium * 2:
        flow_direction = "BULLISH"
    elif put_premium > call_premium * 2:
        flow_direction = "BEARISH"
    elif len(call_unusual) > len(put_unusual) * 1.5:
        flow_direction = "BULLISH"
    elif len(put_unusual) > len(call_unusual) * 1.5:
        flow_direction = "BEARISH"
    else:
        flow_direction = "NEUTRAL"

    # Rising OI on calls vs puts
    call_oi_rising = sum(1 for c in oi_changes if c["type"] == "C" and c["oi_change"] > 0)
    put_oi_rising  = sum(1 for c in oi_changes if c["type"] == "P" and c["oi_change"] > 0)

    return {
        "symbol":            symbol,
        "price":             price,
        "flow_direction":    flow_direction,
        "pc_ratios":         pc_ratios,
        "unusual_activity":  unusual[:5],
        "block_trades":      blocks[:5],
        "gamma_exposure":    gex,
        "oi_changes":        oi_changes[:5],
        "max_pain":          max_pain_val,
        "price_vs_max_pain": round(price - max_pain_val, 2),
        "call_premium_moved": call_premium,
        "put_premium_moved":  put_premium,
        "unusual_call_count": len(call_unusual),
        "unusual_put_count":  len(put_unusual),
        "call_oi_rising":    call_oi_rising,
        "put_oi_rising":     put_oi_rising,
        "has_unusual_flow":  len(unusual) > 0,
        "has_block_trades":  len(blocks) > 0,
    }


def _empty_flow(symbol: str) -> dict:
    return {
        "symbol": symbol, "flow_direction": "NEUTRAL",
        "pc_ratios": {"vol_pc_ratio": 1.0, "oi_pc_ratio": 1.0,
                      "vol_sentiment": "NEUTRAL", "oi_sentiment": "NEUTRAL"},
        "unusual_activity": [], "block_trades": [],
        "gamma_exposure": {"total_gex": 0, "gex_positive": True, "regime": "POSITIVE_GEX"},
        "oi_changes": [], "max_pain": 0, "price_vs_max_pain": 0,
        "call_premium_moved": 0, "put_premium_moved": 0,
        "unusual_call_count": 0, "unusual_put_count": 0,
        "call_oi_rising": 0, "put_oi_rising": 0,
        "has_unusual_flow": False, "has_block_trades": False,
    }


# ── Flow Score (0-100) ────────────────────────────────────────────────────────

def compute_flow_score(flow: dict, direction: str) -> dict:
    """
    Convert raw flow data into a 0-100 flow score for the given trade direction.
    """
    score = 50.0  # Start neutral
    signals = []
    is_bull = direction == "bullish"
    fd = flow.get("flow_direction", "NEUTRAL")

    # Direction alignment (±20)
    if (is_bull and fd == "BULLISH") or (not is_bull and fd == "BEARISH"):
        score += 20
        signals.append(f"✓ Flow direction aligns ({fd})")
    elif (is_bull and fd == "BEARISH") or (not is_bull and fd == "BULLISH"):
        score -= 20
        signals.append(f"✗ Flow direction opposes ({fd})")

    # Put/call ratio (±15)
    vol_pc = flow.get("pc_ratios", {}).get("vol_pc_ratio", 1.0)
    if is_bull and vol_pc < 0.8:
        score += 15
        signals.append(f"✓ PC ratio bullish ({vol_pc:.2f})")
    elif not is_bull and vol_pc > 1.2:
        score += 15
        signals.append(f"✓ PC ratio bearish ({vol_pc:.2f})")
    elif (is_bull and vol_pc > 1.2) or (not is_bull and vol_pc < 0.8):
        score -= 10
        signals.append(f"✗ PC ratio opposes ({vol_pc:.2f})")

    # Unusual activity (±10)
    call_unusual = flow.get("unusual_call_count", 0)
    put_unusual  = flow.get("unusual_put_count", 0)
    if is_bull and call_unusual > put_unusual:
        score += 10
        signals.append(f"✓ Unusual call activity ({call_unusual} calls vs {put_unusual} puts)")
    elif not is_bull and put_unusual > call_unusual:
        score += 10
        signals.append(f"✓ Unusual put activity ({put_unusual} puts vs {call_unusual} calls)")

    # Block trades (±10)
    call_prem = flow.get("call_premium_moved", 0)
    put_prem  = flow.get("put_premium_moved", 0)
    if is_bull and call_prem > 10000:
        score += 10
        signals.append(f"✓ Call blocks detected (${call_prem:,.0f} premium)")
    elif not is_bull and put_prem > 10000:
        score += 10
        signals.append(f"✓ Put blocks detected (${put_prem:,.0f} premium)")

    # Gamma exposure (±5)
    gex_positive = flow.get("gamma_exposure", {}).get("gex_positive", True)
    if is_bull and gex_positive:
        score += 5
        signals.append("✓ Positive GEX (stabilizing environment)")
    elif not is_bull and not gex_positive:
        score += 5
        signals.append("✓ Negative GEX (amplified moves expected)")

    # OI rising in direction (±5)
    call_oi_up = flow.get("call_oi_rising", 0)
    put_oi_up  = flow.get("put_oi_rising", 0)
    if is_bull and call_oi_up > put_oi_up:
        score += 5
        signals.append(f"✓ Call OI accumulating ({call_oi_up} strikes)")
    elif not is_bull and put_oi_up > call_oi_up:
        score += 5
        signals.append(f"✓ Put OI accumulating ({put_oi_up} strikes)")

    score = max(0, min(100, score))
    return {"score": round(score, 1), "direction": direction, "signals": signals, "flow": flow}


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from auth import initialize
    initialize()

    print("Analyzing SPY options flow...")
    flow = analyze_flow("SPY", 540.0)
    print(f"\nFlow direction: {flow['flow_direction']}")
    print(f"PC ratio (volume): {flow['pc_ratios']['vol_pc_ratio']:.2f}")
    print(f"GEX: {flow['gamma_exposure']['regime']}")
    print(f"Max pain: ${flow['max_pain']:.0f}")
    print(f"Unusual activity: {len(flow['unusual_activity'])} contracts")
    print(f"Block trades: {len(flow['block_trades'])} contracts")

    score = compute_flow_score(flow, "bullish")
    print(f"\nFlow Score (bullish): {score['score']}/100")
    for s in score["signals"]:
        print(f"  {s}")
