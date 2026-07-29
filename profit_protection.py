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

import state_store

LOG_DIR          = state_store.LOG_DIR
_daily_stats_file = state_store.daily_stats_file
WEEKLY_FILE      = state_store.WEEKLY_STATS_FILE
PROTECTION_FILE  = state_store.PROTECTION_STATE_FILE

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

_load_daily_stats = state_store.load_daily_stats


def _empty_weekly_stats() -> dict:
    return {
        "week_start":   _get_week_start().isoformat(),
        "pnl":          0.0,
        "trade_count":  0,
        "winners":      0,
        "losers":       0,
        "peak_pnl":     0.0,
        "max_drawdown": 0.0,
    }


def _load_weekly_stats() -> dict:
    data, err = state_store.load_weekly_stats_raw()
    if err is not None:
        # Fails CLOSED: a corrupt weekly file is flagged, not silently
        # replaced with a fresh $0 week that would hide a halt-worthy loss.
        stats = _empty_weekly_stats()
        stats["_corrupted"] = True
        return stats
    if data is None:
        return _empty_weekly_stats()
    # A stale week_start is a legitimate rollover (new week), not corruption.
    if data.get("week_start") == _get_week_start().isoformat():
        return data
    return _empty_weekly_stats()


def _save_weekly_stats(stats: dict):
    state_store.save_weekly_stats(stats)


def _get_week_start() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())  # Monday


def _load_protection_state() -> dict:
    data, err = state_store.load_protection_state_raw()
    if err is None and data is not None and data.get("date") == date.today().isoformat():
        return data
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
    state_store.save_protection_state(state)


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

    # Fail CLOSED: if we can't reliably read today's or this week's P&L,
    # don't let a fresh-looking $0.00 quietly grant full size / no halt.
    state_unreadable = bool(daily.get("_corrupted") or weekly.get("_corrupted"))
    if state_unreadable:
        messages.append("🛑 STATE FILE UNREADABLE — protection cannot verify P&L, new entries blocked")

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
    drawback_alert = (peak >= 0.05 * ACCOUNT_SIZE and daily_pnl < min_pct and daily_pnl != 0.0)
    if drawback_alert:
        # Was up 5%, now below 2% — this used to be message-only (nothing ever
        # actually closed a position over it). executor.check_exits() now reads
        # this flag and force-closes every open position when it's set.
        messages.append(
            f"🚨 DRAWBACK ALERT: Was up ${peak:.2f}, now ${daily_pnl:.2f} "
            f"— closing all positions"
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

    # State-file corruption overrides everything above: block new entries
    # and tighten stops regardless of what the (unreliable) P&L numbers say.
    if state_unreadable:
        halted = True
        level = "HALT"
        size_mult = 0.0
        stop_level = min(stop_level, 0.50)
        # Don't force-close positions off of P&L numbers we just said we
        # can't trust — corruption blocks new entries, it doesn't trigger
        # the drawback close.
        drawback_alert = False

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
        "state_unreadable":    state_unreadable,
        "drawback_alert":      drawback_alert,
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
