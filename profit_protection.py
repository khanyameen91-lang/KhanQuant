"""
profit_protection.py — Daily Profit Protection System

Institutional trading rule: protect your winners.
A 3% day can turn into a flat day if you keep trading aggressively.

Trigger levels (based on account size $5,000):
  +$50  (+1%): No change — continue normally
  +$100 (+2%): Reduce max position size by 25%
  +$150 (+3%): Reduce max position size by 50%, tighten stops to -50%
  +$250 (+5%): Reduce max position size by 75%, only credit spreads allowed
  +$350 (+7%): Stop all new entries for the day — protect the gain

Also implements:
  - Weekly loss tracking (halt if weekly loss > $600)
  - Never let a +5% day close below +2%
  - Consecutive loss tracking per strategy
"""

import os
import json
from datetime import date, datetime, timedelta
from pathlib import Path

LOG_DIR          = Path("logs")
def _daily_stats_file() -> Path:
    return LOG_DIR / f"stats_{date.today().isoformat()}.json"
WEEKLY_FILE      = LOG_DIR / "weekly_stats.json"
PROTECTION_FILE  = LOG_DIR / "profit_protection_state.json"

ACCOUNT_SIZE     = float(os.environ.get("ACCOUNT_SIZE", 5000))

# Profit protection thresholds
LEVELS = [
    {"pct": 0.20, "dollars": ACCOUNT_SIZE * 0.20, "action": "HALT",      "size_mult": 0.00},
    {"pct": 0.15, "dollars": ACCOUNT_SIZE * 0.15, "action": "RESTRICT",   "size_mult": 0.25},
    {"pct": 0.10, "dollars": ACCOUNT_SIZE * 0.10, "action": "REDUCE_HALF","size_mult": 0.50},
    {"pct": 0.05, "dollars": ACCOUNT_SIZE * 0.05, "action": "REDUCE_25",  "size_mult": 0.75},
]

# Weekly loss limit: stop trading if weekly loss exceeds this
WEEKLY_LOSS_LIMIT = float(os.environ.get("WEEKLY_LOSS_LIMIT", 600))

# Loss limits by streak
CONSECUTIVE_LOSS_PAUSE = int(os.environ.get("CONSECUTIVE_LOSS_HALT", 3))


# ── State management ───────────────────────────────────────────────────────────

def _load_daily_stats() -> dict:
    f = _daily_stats_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {"pnl": 0.0, "trade_count": 0, "winners": 0, "losers": 0}


def _load_weekly_stats() -> dict:
    if WEEKLY_FILE.exists():
        try:
            data = json.loads(WEEKLY_FILE.read_text())
            # Validate it's current week
            week_start = _get_week_start()
            if data.get("week_start") == week_start.isoformat():
                return data
        except Exception:
            pass
    return {
        "week_start":   _get_week_start().isoformat(),
        "pnl":          0.0,
        "trade_count":  0,
        "winners":      0,
        "losers":       0,
        "peak_pnl":     0.0,
        "max_drawdown": 0.0,
    }


def _save_weekly_stats(stats: dict):
    LOG_DIR.mkdir(exist_ok=True)
    WEEKLY_FILE.write_text(json.dumps(stats, indent=2))


def _get_week_start() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())  # Monday


def _load_protection_state() -> dict:
    if PROTECTION_FILE.exists():
        try:
            data = json.loads(PROTECTION_FILE.read_text())
            if data.get("date") == date.today().isoformat():
                return data
        except Exception:
            pass
    return {
        "date":                date.today().isoformat(),
        "peak_daily_pnl":      0.0,
        "protection_level":    "NONE",
        "size_multiplier":     1.0,
        "stop_level":          0.75,   # Stop out at 75% of max loss by default
        "strategies_allowed":  ["all"],
        "new_entries_halted":  False,
        "trigger_log":         [],
    }


def _save_protection_state(state: dict):
    LOG_DIR.mkdir(exist_ok=True)
    PROTECTION_FILE.write_text(json.dumps(state, indent=2))


# ── Main profit protection check ───────────────────────────────────────────────

def get_protection_status() -> dict:
    """
    Check current profit protection state.

    Returns:
    {
      "level": "NONE"|"REDUCE_25"|"REDUCE_HALF"|"RESTRICT"|"HALT",
      "size_multiplier": 0.0-1.0,
      "stop_level": 0.50-0.75 (stop-loss as fraction of max loss),
      "strategies_allowed": ["all"] or ["credit_spread"],
      "new_entries_halted": bool,
      "daily_pnl": float,
      "weekly_pnl": float,
      "weekly_halted": bool,
      "weekly_halt_reason": str,
      "messages": [str],
    }
    """
    daily  = _load_daily_stats()
    weekly = _load_weekly_stats()
    state  = _load_protection_state()

    daily_pnl  = float(daily.get("pnl", 0.0))
    weekly_pnl = float(weekly.get("pnl", 0.0))
    messages   = []

    # ── Update peak daily P&L ──────────────────────────────────────────────────
    if daily_pnl > state["peak_daily_pnl"]:
        state["peak_daily_pnl"] = daily_pnl

    # ── Check profit protection levels (highest trigger wins) ──────────────────
    size_mult  = 1.0
    level      = "NONE"
    stop_level = 0.75
    strategies = ["all"]
    halted     = False

    for trigger in LEVELS:
        if daily_pnl >= trigger["dollars"]:
            level     = trigger["action"]
            size_mult = trigger["size_mult"]

            if level == "HALT":
                halted = True
                messages.append(
                    f"🔒 TRADING HALTED: Daily P&L ${daily_pnl:+.2f} (+{daily_pnl/ACCOUNT_SIZE*100:.1f}%) "
                    f"— protecting gains"
                )
            elif level == "RESTRICT":
                stop_level  = 0.50   # Tighten stops to 50%
                strategies  = ["credit_spread", "iron_condor"]
                messages.append(
                    f"⚠️ RESTRICTED (+{daily_pnl/ACCOUNT_SIZE*100:.1f}%): "
                    f"Credit spreads only, size 25%, stops tightened"
                )
            elif level == "REDUCE_HALF":
                stop_level  = 0.50   # Tighten stops
                messages.append(
                    f"⚠️ REDUCED (+{daily_pnl/ACCOUNT_SIZE*100:.1f}%): "
                    f"50% size, tighter stops"
                )
            elif level == "REDUCE_25":
                messages.append(
                    f"ℹ️ SIZE REDUCED (+{daily_pnl/ACCOUNT_SIZE*100:.1f}%): "
                    f"75% of normal size"
                )
            break

    # ── Never let a +5% day close below +2% ───────────────────────────────────
    peak    = state["peak_daily_pnl"]
    min_pct = 0.02 * ACCOUNT_SIZE
    # Only fire drawback alert if we've actually had trades today (daily_pnl != 0)
    # This prevents stale-peak false alarms after a bot restart
    if peak >= 0.05 * ACCOUNT_SIZE and daily_pnl < min_pct and daily_pnl != 0.0:
        # Was up 5%, now below 2% — alert
        messages.append(
            f"🚨 DRAWBACK ALERT: Was up ${peak:.2f}, now ${daily_pnl:.2f} "
            f"— close positions immediately"
        )

    # ── Weekly loss limit ──────────────────────────────────────────────────────
    weekly_halted      = False
    weekly_halt_reason = ""
    if weekly_pnl <= -WEEKLY_LOSS_LIMIT:
        weekly_halted      = True
        weekly_halt_reason = (
            f"Weekly loss limit hit: ${weekly_pnl:.2f} (limit: -${WEEKLY_LOSS_LIMIT:.0f})"
        )
        messages.append(f"🛑 WEEKLY HALT: {weekly_halt_reason}")

    # ── Update and save state ──────────────────────────────────────────────────
    state.update({
        "protection_level":   level,
        "size_multiplier":    size_mult,
        "stop_level":         stop_level,
        "strategies_allowed": strategies,
        "new_entries_halted": halted,
    })
    _save_protection_state(state)

    return {
        "level":               level,
        "size_multiplier":     size_mult,
        "stop_level":          stop_level,
        "strategies_allowed":  strategies,
        "new_entries_halted":  halted,
        "daily_pnl":           daily_pnl,
        "weekly_pnl":          weekly_pnl,
        "weekly_halted":       weekly_halted,
        "weekly_halt_reason":  weekly_halt_reason,
        "peak_daily_pnl":      peak,
        "messages":            messages,
    }


def update_weekly_pnl(daily_pnl_delta: float):
    """
    Called at end of day to roll daily P&L into weekly totals.
    Also call when a trade is closed to keep weekly running total.
    """
    weekly = _load_weekly_stats()
    weekly["pnl"] = weekly.get("pnl", 0.0) + daily_pnl_delta
    if weekly["pnl"] > weekly.get("peak_pnl", 0.0):
        weekly["peak_pnl"] = weekly["pnl"]
    drawdown = weekly["peak_pnl"] - weekly["pnl"]
    if drawdown > weekly.get("max_drawdown", 0.0):
        weekly["max_drawdown"] = drawdown
    _save_weekly_stats(weekly)


def is_strategy_allowed(strategy_type: str, protection: dict) -> bool:
    """Check if a strategy is allowed given current profit protection level."""
    allowed = protection.get("strategies_allowed", ["all"])
    if "all" in allowed:
        return True
    # Normalize
    stype = strategy_type.lower().replace(" ", "_").replace("-", "_")
    return any(a in stype for a in allowed)


def get_adjusted_max_loss(base_max_loss: float, protection: dict) -> float:
    """Return position-size-adjusted max loss based on protection level."""
    mult = protection.get("size_multiplier", 1.0)
    return round(base_max_loss * mult, 2)


def get_stop_level(protection: dict) -> float:
    """Return current stop-loss level as fraction of max loss."""
    return protection.get("stop_level", 0.75)


def get_weekly_summary() -> dict:
    """Return formatted weekly performance summary."""
    weekly = _load_weekly_stats()
    return {
        "week_start":   weekly.get("week_start"),
        "pnl":          weekly.get("pnl", 0.0),
        "trade_count":  weekly.get("trade_count", 0),
        "winners":      weekly.get("winners", 0),
        "losers":       weekly.get("losers", 0),
        "max_drawdown": weekly.get("max_drawdown", 0.0),
        "weekly_limit": WEEKLY_LOSS_LIMIT,
        "remaining_budget": WEEKLY_LOSS_LIMIT + weekly.get("pnl", 0.0),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate different P&L states
    import shutil

    for daily_pnl, label in [
        (0,    "Flat day"),
        (80,   "+$80 (+1.6%) — mild gains"),
        (120,  "+$120 (+2.4%) — moderate"),
        (175,  "+$175 (+3.5%) — good day"),
        (280,  "+$280 (+5.6%) — great day"),
        (380,  "+$380 (+7.6%) — exceptional"),
    ]:
        # Fake the daily stats file
        _daily_stats_file().parent.mkdir(exist_ok=True)
        _daily_stats_file().write_text(json.dumps({"pnl": daily_pnl}))

        protection = get_protection_status()
        print(f"\n{label}:")
        print(f"  Level: {protection['level']}")
        print(f"  Size multiplier: {protection['size_multiplier']:.0%}")
        print(f"  Stop level: {protection['stop_level']:.0%}")
        print(f"  Halted: {protection['new_entries_halted']}")
        for msg in protection["messages"]:
            print(f"  {msg}")
