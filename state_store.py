"""
state_store.py — Single source of truth for shared JSON state on disk.

Before this module existed, risk_manager.py, executor.py, profit_protection.py,
sizing.py, and dashboard.py each kept their own copy of "load positions.json /
today's stats / risk state," and every copy caught a JSON parse error the same
way: silently return a fresh, permissive default (trading_halted=False,
pnl=0.0, positions=[]). A corrupted file — e.g. from a crash mid-write — would
quietly erase a halt or a day's loss tracking instead of stopping trading
until a human checked.

This module centralizes that state in one place with two changes:
  1. Writes are atomic (temp file + os.replace) so a crash mid-write can't
     leave a half-written file behind in the first place.
  2. Reads FAIL CLOSED: a file that exists but fails to parse is reported to
     the caller as unreadable/corrupted, not silently swapped for a safe-
     looking default. Risk-relevant callers (risk_manager.check_all_risk)
     treat "unreadable" as an automatic halt.
"""

import os
import json
import logging
from datetime import date, datetime
from pathlib import Path
from dataclasses import dataclass, asdict

log = logging.getLogger("state_store")

LOG_DIR               = Path("logs")
POSITIONS_FILE        = LOG_DIR / "positions.json"
JOURNAL_FILE          = LOG_DIR / "trade_journal.json"
RISK_STATE_FILE       = LOG_DIR / "risk_state.json"
WEEKLY_STATS_FILE     = LOG_DIR / "weekly_stats.json"
PROTECTION_STATE_FILE = LOG_DIR / "profit_protection_state.json"


def daily_stats_file(for_date: date = None) -> Path:
    d = for_date or date.today()
    return LOG_DIR / f"stats_{d.isoformat()}.json"


def monthly_stats_file(for_date: date = None) -> Path:
    d = for_date or date.today()
    return LOG_DIR / f"monthly_{d.strftime('%Y_%m')}.json"


def atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically: write to a sibling temp file, then os.replace()
    it over the target. os.replace is atomic on both POSIX and Windows when
    source and destination are on the same volume, so a crash mid-write
    leaves either the old file or the new one, never a truncated hybrid."""
    LOG_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _load_or(path: Path):
    """Returns (data, error). `data is None and error is None` means the file
    doesn't exist (a normal state, e.g. first run of the day). `error` set
    means the file exists but couldn't be parsed — the caller must decide
    how to fail closed for its own use case; this helper never papers over
    corruption with a silent default."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        log.error(f"state_store: {path} exists but failed to parse: {e}")
        return None, e


# ── Positions ────────────────────────────────────────────────────────────────

def load_positions() -> list:
    """Best-effort read for display/non-critical use. Returns [] if the file
    is missing OR corrupted — for a hard guarantee that corruption was
    actually detected (not silently treated as "no positions"), use
    load_positions_checked() instead."""
    data, err = _load_or(POSITIONS_FILE)
    if err is not None or data is None:
        return []
    return data


def load_positions_checked() -> tuple:
    """Returns (positions, ok). ok=False means the file exists but is
    unreadable — callers that gate trading or exit-monitoring MUST treat
    this as "we don't actually know what's open" and fail closed, rather
    than proceeding as if there were zero positions (which would let new
    trades stack on top of untracked ones, or skip exit checks on positions
    that are actually still open)."""
    data, err = _load_or(POSITIONS_FILE)
    if err is not None:
        return [], False
    return (data or []), True


def save_positions(positions: list) -> None:
    atomic_write_json(POSITIONS_FILE, positions)


# ── Trade journal ────────────────────────────────────────────────────────────

def load_journal() -> list:
    data, err = _load_or(JOURNAL_FILE)
    if err is not None or data is None:
        return []
    return data


def append_journal(entry: dict, keep: int = 500) -> None:
    journal = load_journal()
    journal.insert(0, entry)
    atomic_write_json(JOURNAL_FILE, journal[:keep])


# ── Daily stats ──────────────────────────────────────────────────────────────

def _empty_daily_stats() -> dict:
    return {"date": date.today().isoformat(), "pnl": 0.0, "trade_count": 0,
            "winners": 0, "losers": 0}


def load_daily_stats() -> dict:
    """Fails CLOSED. If today's stats file exists but can't be parsed, this
    returns an empty-looking stats dict with `_corrupted=True` set, instead
    of a plain $0.00 day that looks identical to a genuinely fresh morning.
    risk_manager.check_all_risk() checks that flag first and halts new
    entries until the file is restored, rather than trading on top of an
    unknown day's P&L."""
    data, err = _load_or(daily_stats_file())
    if err is not None:
        stats = _empty_daily_stats()
        stats["_corrupted"] = True
        return stats
    if data is None:
        return _empty_daily_stats()
    return data


def save_daily_stats(stats: dict) -> None:
    atomic_write_json(daily_stats_file(), stats)


# ── Risk state ───────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    consecutive_losses: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    peak_daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    last_updated: str = ""


def load_risk_state() -> RiskState:
    """Fails CLOSED. A risk_state.json that exists but fails to parse (or
    whose shape no longer matches RiskState) comes back HALTED with a clear
    reason, instead of a fresh RiskState() that silently clears whatever
    halt — consecutive losses, manual halt, anything — was actually in
    effect."""
    data, err = _load_or(RISK_STATE_FILE)
    if err is not None:
        return RiskState(trading_halted=True, halt_reason="STATE_FILE_CORRUPTED")
    if data is None:
        return RiskState()
    try:
        return RiskState(**data)
    except TypeError as e:
        log.error(f"state_store: risk_state.json has an unexpected shape: {e}")
        return RiskState(trading_halted=True, halt_reason="STATE_FILE_CORRUPTED")


def save_risk_state(rs: RiskState) -> None:
    rs.last_updated = datetime.now().isoformat()
    atomic_write_json(RISK_STATE_FILE, asdict(rs))


# ── Monthly stats ────────────────────────────────────────────────────────────

def _empty_monthly_stats() -> dict:
    return {"month": date.today().strftime("%Y-%m"), "pnl": 0.0, "trade_count": 0,
            "winners": 0, "losers": 0, "strategy_pnl": {}}


def load_monthly_stats() -> dict:
    data, err = _load_or(monthly_stats_file())
    if err is not None:
        stats = _empty_monthly_stats()
        stats["_corrupted"] = True
        return stats
    if data is None:
        return _empty_monthly_stats()
    return data


def save_monthly_stats(stats: dict) -> None:
    atomic_write_json(monthly_stats_file(), stats)


# ── Weekly stats ─────────────────────────────────────────────────────────────

def load_weekly_stats_raw():
    """Returns (data, err) with no default applied — profit_protection.py
    owns the week-start-date validation logic, so it decides what an empty
    week looks like."""
    return _load_or(WEEKLY_FILE := WEEKLY_STATS_FILE)


def save_weekly_stats(stats: dict) -> None:
    atomic_write_json(WEEKLY_STATS_FILE, stats)


# ── Profit-protection state ──────────────────────────────────────────────────

def load_protection_state_raw():
    return _load_or(PROTECTION_STATE_FILE)


def save_protection_state(state: dict) -> None:
    atomic_write_json(PROTECTION_STATE_FILE, state)
