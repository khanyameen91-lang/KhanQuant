"""
risk_manager.py — Portfolio Risk Management Engine

Enforces all risk rules at the portfolio level:
  - Daily loss limits
  - Consecutive loss protection (trading halt)
  - Portfolio delta / gamma exposure caps
  - ATR-based dynamic stop losses
  - Sector concentration limits
  - Max drawdown tracking

Reads from and writes to the trade journal and positions file.
"""

import os
import json
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from calendar import month_abbr

LOG_DIR            = Path("logs")
POSITIONS_FILE     = LOG_DIR / "positions.json"
def _daily_stats_file() -> Path:
    return LOG_DIR / f"stats_{date.today().isoformat()}.json"
JOURNAL_FILE       = LOG_DIR / "trade_journal.json"
RISK_STATE_FILE    = LOG_DIR / "risk_state.json"
MONTHLY_STATS_FILE = LOG_DIR / f"monthly_{date.today().strftime('%Y_%m')}.json"

# ── Risk Limits (from env or defaults) ────────────────────────────────────────
DAILY_LOSS_LIMIT       = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
MAX_POSITION_SIZE      = float(os.environ.get("MAX_POSITION_SIZE", 150))
MAX_OPEN_POSITIONS     = int(os.environ.get("MAX_OPEN_POSITIONS", 3))
CONSECUTIVE_LOSS_HALT  = int(os.environ.get("CONSECUTIVE_LOSS_HALT", 3))   # halt after N losses
MAX_PORTFOLIO_DELTA    = float(os.environ.get("MAX_PORTFOLIO_DELTA", 150)) # net delta exposure
MAX_DRAWDOWN_PCT       = float(os.environ.get("MAX_DRAWDOWN_PCT", 5.0))    # % of account
ACCOUNT_SIZE           = float(os.environ.get("ACCOUNT_SIZE", 5000))
MONTHLY_LOSS_LIMIT     = float(os.environ.get("MONTHLY_LOSS_LIMIT", 1200)) # 2× weekly limit

# Per-strategy monthly drawdown limits (max loss per strategy in a calendar month)
STRATEGY_MONTHLY_LIMITS = {
    "long_option":   300.0,   # High-risk/high-reward, limit exposure
    "0dte_scalp":    400.0,   # Most volatile, tighter limit
    "credit_spread": 500.0,   # Defined risk, more room
    "spread":        400.0,
    "iron_condor":   500.0,
    "strangle":      300.0,
    "default":       400.0,
}

# Max concurrent positions per strategy type
STRATEGY_MAX_CONCURRENT = {
    "0dte_scalp":    2,
    "long_option":   2,
    "credit_spread": 2,
    "iron_condor":   1,
    "spread":        2,
    "strangle":      1,
}


@dataclass
class RiskState:
    consecutive_losses: int  = 0
    trading_halted:     bool = False
    halt_reason:        str  = ""
    peak_daily_pnl:     float = 0.0
    current_drawdown:   float = 0.0
    last_updated:       str  = ""


def _load_risk_state() -> RiskState:
    if RISK_STATE_FILE.exists():
        try:
            d = json.loads(RISK_STATE_FILE.read_text())
            return RiskState(**d)
        except Exception:
            pass
    return RiskState()


def _save_risk_state(rs: RiskState):
    LOG_DIR.mkdir(exist_ok=True)
    rs.last_updated = datetime.now().isoformat()
    RISK_STATE_FILE.write_text(json.dumps(rs.__dict__, indent=2))


def _load_positions() -> list:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except Exception:
            return []
    return []


def _load_daily_stats() -> dict:
    f = _daily_stats_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {"date": date.today().isoformat(), "pnl": 0.0, "trade_count": 0,
            "winners": 0, "losers": 0}


def _load_journal() -> list:
    if JOURNAL_FILE.exists():
        try:
            return json.loads(JOURNAL_FILE.read_text())
        except Exception:
            return []
    return []


def _load_monthly_stats() -> dict:
    if MONTHLY_STATS_FILE.exists():
        try:
            return json.loads(MONTHLY_STATS_FILE.read_text())
        except Exception:
            pass
    return {
        "month": date.today().strftime("%Y-%m"),
        "pnl": 0.0,
        "trade_count": 0,
        "strategy_pnl": {},
        "winners": 0,
        "losers": 0,
    }


def update_monthly_stats(trade_pnl: float, strategy_type: str = "default"):
    """Call after each trade closes to update monthly tracking."""
    stats = _load_monthly_stats()
    stats["pnl"] = round(stats.get("pnl", 0.0) + trade_pnl, 2)
    stats["trade_count"] = stats.get("trade_count", 0) + 1
    if trade_pnl > 0:
        stats["winners"] = stats.get("winners", 0) + 1
    else:
        stats["losers"] = stats.get("losers", 0) + 1

    by_strat = stats.get("strategy_pnl", {})
    key = strategy_type.lower()
    by_strat[key] = round(by_strat.get(key, 0.0) + trade_pnl, 2)
    stats["strategy_pnl"] = by_strat

    LOG_DIR.mkdir(exist_ok=True)
    MONTHLY_STATS_FILE.write_text(json.dumps(stats, indent=2))
    return stats


def check_strategy_drawdown(strategy_type: str) -> tuple:
    """
    Check if a strategy has exceeded its monthly loss limit.
    Returns (allowed: bool, reason: str).
    """
    stats    = _load_monthly_stats()
    by_strat = stats.get("strategy_pnl", {})
    key      = strategy_type.lower().replace(" ", "_")
    # normalize key
    for k in STRATEGY_MONTHLY_LIMITS:
        if k in key:
            key = k
            break

    strat_pnl   = by_strat.get(key, 0.0)
    limit       = STRATEGY_MONTHLY_LIMITS.get(key, STRATEGY_MONTHLY_LIMITS["default"])
    strat_loss  = abs(min(strat_pnl, 0))

    if strat_loss >= limit:
        return False, (f"Monthly drawdown limit hit for {key}: "
                       f"${strat_loss:.0f} >= ${limit:.0f} limit")
    return True, f"{key} monthly loss: ${strat_loss:.0f}/${limit:.0f}"


def check_strategy_concentration(strategy_type: str) -> tuple:
    """
    Enforce max concurrent positions per strategy type.
    Returns (allowed: bool, reason: str).
    """
    positions = _load_positions()
    key = strategy_type.lower().replace(" ", "_")
    for k in STRATEGY_MAX_CONCURRENT:
        if k in key:
            key = k
            break

    limit = STRATEGY_MAX_CONCURRENT.get(key, 2)
    count = sum(1 for p in positions
                if key in p.get("setup", {}).get("type", "").lower())
    if count >= limit:
        return False, f"Max concurrent {key} positions ({limit}) reached"
    return True, f"{key} positions: {count}/{limit}"


def get_monthly_summary() -> dict:
    """Return monthly performance summary for dashboard."""
    stats = _load_monthly_stats()
    by_strat = stats.get("strategy_pnl", {})
    month_pnl = stats.get("pnl", 0.0)
    monthly_loss_pct = abs(min(month_pnl, 0)) / MONTHLY_LOSS_LIMIT * 100

    strat_summary = {}
    for k, pnl in by_strat.items():
        limit = STRATEGY_MONTHLY_LIMITS.get(k, STRATEGY_MONTHLY_LIMITS["default"])
        strat_summary[k] = {
            "pnl":       round(pnl, 2),
            "limit":     limit,
            "used_pct":  round(abs(min(pnl, 0)) / limit * 100, 1),
            "ok":        abs(min(pnl, 0)) < limit,
        }

    return {
        "month":             stats.get("month", ""),
        "monthly_pnl":       month_pnl,
        "monthly_loss_limit": MONTHLY_LOSS_LIMIT,
        "monthly_loss_pct":  round(monthly_loss_pct, 1),
        "trade_count":       stats.get("trade_count", 0),
        "winners":           stats.get("winners", 0),
        "losers":            stats.get("losers", 0),
        "win_rate":          round(stats.get("winners", 0) /
                                   max(1, stats.get("trade_count", 1)) * 100, 1),
        "strategy_breakdown": strat_summary,
    }


# ── Portfolio Greeks ───────────────────────────────────────────────────────────

def get_portfolio_greeks() -> dict:
    """
    Estimate portfolio-level Greeks from open positions.
    Delta is the most important for directional exposure management.
    """
    positions = _load_positions()
    total_delta  = 0.0
    total_gamma  = 0.0
    total_theta  = 0.0
    total_vega   = 0.0

    for pos in positions:
        setup = pos.get("setup", {})
        t = setup.get("type", "")

        if t in ("long_option", "0dte_scalp"):
            c = setup.get("contract", {})
            mult = 1 if setup.get("direction") == "bullish" else -1
            total_delta += c.get("delta", 0) * 100 * mult
            total_gamma += c.get("gamma", 0) * 100
            total_theta += c.get("theta", 0) * 100
            total_vega  += c.get("vega", 0) * 100

        elif t in ("spread",):
            ll = setup.get("long_leg", {})
            sl = setup.get("short_leg", {})
            net_delta = ll.get("delta", 0) - sl.get("delta", 0)
            total_delta += net_delta * 100
            total_gamma += (ll.get("gamma", 0) - sl.get("gamma", 0)) * 100
            total_theta += (ll.get("theta", 0) - sl.get("theta", 0)) * 100
            total_vega  += (ll.get("vega", 0) - sl.get("vega", 0)) * 100

        elif t == "credit_spread":
            sl = setup.get("short_leg", {})
            ll = setup.get("long_leg", {})
            net_delta = sl.get("delta", 0) - ll.get("delta", 0)  # short first
            total_delta -= net_delta * 100  # short = negative delta for calls
            total_theta += (sl.get("theta", 0) - ll.get("theta", 0)) * 100

        elif t == "iron_condor":
            # Net delta should be near 0 for symmetric condors
            sp = setup.get("short_put", {})
            lp = setup.get("long_put", {})
            sc = setup.get("short_call", {})
            lc = setup.get("long_call", {})
            ic_delta = (
                -sp.get("delta", 0) + lp.get("delta", 0)
                -sc.get("delta", 0) + lc.get("delta", 0)
            ) * 100
            total_delta += ic_delta
            total_theta += (sp.get("theta", 0) + sc.get("theta", 0)) * 100  # theta positive for condors

        elif t == "strangle":
            lc = setup.get("long_call", {})
            lp = setup.get("long_put", {})
            total_delta += (lc.get("delta", 0) + lp.get("delta", 0)) * 100
            total_vega  += (lc.get("vega", 0) + lp.get("vega", 0)) * 100

    return {
        "net_delta": round(total_delta, 2),
        "net_gamma": round(total_gamma, 4),
        "net_theta": round(total_theta, 2),
        "net_vega":  round(total_vega, 2),
        "delta_dollars": round(total_delta * 1.0, 2),  # approx $/point
    }


# ── Consecutive Loss Tracking ─────────────────────────────────────────────────

def update_consecutive_losses(trade_won: bool) -> RiskState:
    """Update consecutive loss counter after a trade closes."""
    rs = _load_risk_state()

    if trade_won:
        rs.consecutive_losses = 0
        if rs.halt_reason == "CONSECUTIVE_LOSSES":
            rs.trading_halted = False
            rs.halt_reason = ""
            print("Risk: Consecutive loss counter reset — trading resumed")
    else:
        rs.consecutive_losses += 1
        print(f"Risk: Consecutive loss #{rs.consecutive_losses}")
        if rs.consecutive_losses >= CONSECUTIVE_LOSS_HALT:
            rs.trading_halted = True
            rs.halt_reason = "CONSECUTIVE_LOSSES"
            print(f"🛑 Risk: {CONSECUTIVE_LOSS_HALT} consecutive losses — TRADING HALTED")

    _save_risk_state(rs)
    return rs


# ── Pre-Trade Risk Checks ─────────────────────────────────────────────────────

def check_all_risk(setup: dict) -> tuple[bool, str]:
    """
    Run all pre-trade risk checks.
    Returns (approved: bool, reason: str).

    Call this BEFORE any order placement.
    """
    positions  = _load_positions()
    stats      = _load_daily_stats()
    rs         = _load_risk_state()
    greeks     = get_portfolio_greeks()

    # ── Hard stops ────────────────────────────────────────────────────────────

    # 1. Trading halt
    if rs.trading_halted:
        return False, f"Trading halted: {rs.halt_reason}"

    # 2. Max open positions
    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions ({MAX_OPEN_POSITIONS}) reached"

    # 3. Daily loss limit
    if stats["pnl"] <= -DAILY_LOSS_LIMIT:
        return False, f"Daily loss limit hit (${abs(stats['pnl']):.0f}/${DAILY_LOSS_LIMIT:.0f})"

    # 4. Max loss per trade
    if setup.get("max_loss", 0) > MAX_POSITION_SIZE:
        return False, f"Trade max loss ${setup['max_loss']:.0f} > limit ${MAX_POSITION_SIZE:.0f}"

    # 5. Same symbol check
    open_symbols = [p["symbol"] for p in positions]
    if setup.get("symbol") in open_symbols:
        return False, f"Already have position in {setup['symbol']}"

    # 6. Remaining daily budget
    remaining = DAILY_LOSS_LIMIT - abs(min(stats["pnl"], 0))
    if setup.get("max_loss", 0) > remaining:
        return False, f"Trade risk ${setup['max_loss']:.0f} > remaining budget ${remaining:.0f}"

    # ── Portfolio-level checks ────────────────────────────────────────────────

    # 7. Portfolio delta exposure
    new_delta = 0.0
    direction = setup.get("direction", "neutral")
    if direction == "bullish":
        new_delta = 50.0   # approximate
    elif direction == "bearish":
        new_delta = -50.0

    if abs(greeks["net_delta"] + new_delta) > MAX_PORTFOLIO_DELTA:
        return False, f"Portfolio delta exposure too high ({greeks['net_delta']:.0f} + {new_delta:.0f})"

    # 8. Max drawdown for the day
    if stats.get("pnl", 0) < 0:
        drawdown_pct = abs(stats["pnl"]) / ACCOUNT_SIZE * 100
        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            return False, f"Max daily drawdown reached ({drawdown_pct:.1f}%)"

    # 9. Monthly loss limit
    monthly = _load_monthly_stats()
    monthly_pnl = monthly.get("pnl", 0.0)
    if monthly_pnl <= -MONTHLY_LOSS_LIMIT:
        return False, f"Monthly loss limit hit (${abs(monthly_pnl):.0f}/${MONTHLY_LOSS_LIMIT:.0f})"

    # 10. Per-strategy monthly drawdown
    strategy_type = setup.get("type", "default")
    dd_ok, dd_reason = check_strategy_drawdown(strategy_type)
    if not dd_ok:
        return False, dd_reason

    # 11. Per-strategy concentration
    conc_ok, conc_reason = check_strategy_concentration(strategy_type)
    if not conc_ok:
        return False, conc_reason

    return True, "All risk checks passed"


# ── Dynamic Stop Loss Calculator ──────────────────────────────────────────────

def calculate_dynamic_stop(position: dict, atr: float, current_price: float) -> dict:
    """
    Calculate ATR-based dynamic stop loss levels.
    Returns recommended stop and target levels.
    """
    setup     = position.get("setup", {})
    direction = position.get("direction", "bullish")
    max_loss  = position.get("max_loss", MAX_POSITION_SIZE)
    max_profit = position.get("max_profit") or max_loss * 2

    # ATR-based price stop
    atr_multiplier = 1.5
    if direction == "bullish":
        price_stop   = current_price - (atr * atr_multiplier)
        price_target = current_price + (atr * atr_multiplier * 2)
    else:
        price_stop   = current_price + (atr * atr_multiplier)
        price_target = current_price - (atr * atr_multiplier * 2)

    # Option P&L based exits
    pnl_stop_pct   = 0.75   # Close at -75% of max loss
    pnl_target_pct = 0.50   # Take 50% of max profit

    return {
        "price_stop":      round(price_stop, 2),
        "price_target":    round(price_target, 2),
        "pnl_stop":        -round(max_loss * pnl_stop_pct, 2),
        "pnl_target_50":   round(max_profit * 0.50, 2),
        "pnl_target_25":   round(max_profit * 0.25, 2),  # partial exit 1
        "pnl_target_75":   round(max_profit * 0.75, 2),  # partial exit 2
        "atr":             round(atr, 2),
        "atr_pct":         round(atr / current_price * 100, 2),
    }


# ── Sector Concentration ──────────────────────────────────────────────────────

_SECTOR_MAP = {
    "SPY": "BROAD", "QQQ": "TECH", "IWM": "BROAD",
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH",
    "AMD": "TECH", "META": "TECH", "GOOGL": "TECH",
    "AMZN": "TECH", "TSLA": "CONS_DISC",
    "JPM": "FINANCE", "BAC": "FINANCE", "GS": "FINANCE",
    "XLE": "ENERGY", "XLF": "FINANCE", "XLV": "HEALTH",
}

def check_sector_concentration(symbol: str) -> tuple[bool, str]:
    """
    Ensure we're not overconcentrated in one sector.
    Max 2 positions in same sector.
    """
    positions = _load_positions()
    sector = _SECTOR_MAP.get(symbol, "OTHER")
    same_sector = [p for p in positions
                   if _SECTOR_MAP.get(p.get("symbol", ""), "OTHER") == sector]
    if len(same_sector) >= 2:
        return False, f"Sector concentration limit: already {len(same_sector)} {sector} positions"
    return True, f"Sector OK ({sector}: {len(same_sector)}/2)"


# ── Risk Dashboard Data ────────────────────────────────────────────────────────

def get_risk_summary() -> dict:
    """
    Return a full risk summary for the dashboard.
    """
    stats    = _load_daily_stats()
    rs       = _load_risk_state()
    greeks   = get_portfolio_greeks()
    positions = _load_positions()

    daily_loss_pct = abs(min(stats["pnl"], 0)) / DAILY_LOSS_LIMIT * 100
    remaining_risk = DAILY_LOSS_LIMIT - abs(min(stats["pnl"], 0))

    monthly = get_monthly_summary()

    return {
        "daily_pnl":           round(stats["pnl"], 2),
        "daily_loss_limit":    DAILY_LOSS_LIMIT,
        "daily_loss_used_pct": round(daily_loss_pct, 1),
        "remaining_risk_budget": round(remaining_risk, 2),
        "open_positions":      len(positions),
        "max_positions":       MAX_OPEN_POSITIONS,
        "consecutive_losses":  rs.consecutive_losses,
        "consecutive_loss_limit": CONSECUTIVE_LOSS_HALT,
        "trading_halted":      rs.trading_halted,
        "halt_reason":         rs.halt_reason,
        "portfolio_delta":     greeks["net_delta"],
        "portfolio_theta":     greeks["net_theta"],
        "portfolio_vega":      greeks["net_vega"],
        "max_portfolio_delta": MAX_PORTFOLIO_DELTA,
        "winners":             stats.get("winners", 0),
        "losers":              stats.get("losers", 0),
        "win_rate":            round(
            stats.get("winners", 0) / max(1, stats.get("trade_count", 1)) * 100, 1
        ),
        # Monthly
        "monthly_pnl":         monthly["monthly_pnl"],
        "monthly_loss_limit":  MONTHLY_LOSS_LIMIT,
        "monthly_loss_pct":    monthly["monthly_loss_pct"],
        "monthly_trades":      monthly["trade_count"],
        "strategy_breakdown":  monthly.get("strategy_breakdown", {}),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Risk Manager Status:")
    summary = get_risk_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    test_setup = {"symbol": "SPY", "max_loss": 120, "direction": "bullish"}
    approved, reason = check_all_risk(test_setup)
    print(f"\nTrade check: {'APPROVED' if approved else 'BLOCKED'} — {reason}")
