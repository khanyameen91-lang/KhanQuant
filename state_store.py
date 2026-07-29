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
import time
import logging
import contextlib
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

# Cross-process advisory locking, stdlib only (no filelock dependency —
# the server doesn't have it installed and this doesn't warrant one).
# fcntl on POSIX (the Ubuntu server, which is what actually matters for
# production), msvcrt on Windows (dev machine).
try:
    import fcntl
    _LOCK_IMPL = "fcntl"
except ImportError:
    fcntl = None
    try:
        import msvcrt
        _LOCK_IMPL = "msvcrt"
    except ImportError:
        msvcrt = None
        _LOCK_IMPL = None


def daily_stats_file(for_date: date = None) -> Path:
    d = for_date or date.today()
    return LOG_DIR / f"stats_{d.isoformat()}.json"


def monthly_stats_file(for_date: date = None) -> Path:
    d = for_date or date.today()
    return LOG_DIR / f"monthly_{d.strftime('%Y_%m')}.json"


@contextlib.contextmanager
def file_lock(name: str, timeout: float = 10.0):
    """
    Cross-process advisory lock, held for the duration of the `with` block.

    bot.py (scan loop) and dashboard.py (manual close-position button) are
    separate processes that both read-modify-write the same positions.json
    and stats files. Without a lock, a manual close racing an automated
    exit can lose one of the two updates entirely — classic read-modify-
    write race, and the losing update is silently gone.

    Best-effort by design: if the platform has no lock primitive, or the
    lock can't be acquired within `timeout`, this logs and proceeds rather
    than blocking trading indefinitely. A missed lock is a small
    correctness risk; a hung trading bot is a bigger one.
    """
    LOG_DIR.mkdir(exist_ok=True)
    lock_path = LOG_DIR / f".{name}.lock"

    if _LOCK_IMPL is None:
        yield
        return

    fh = open(lock_path, "a+b")
    acquired = False
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                if _LOCK_IMPL == "fcntl":
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except (OSError, IOError):
                time.sleep(0.05)

        if not acquired:
            log.warning(f"state_store: couldn't acquire {name} lock within {timeout}s — proceeding unlocked")
        yield
    finally:
        if acquired:
            try:
                if _LOCK_IMPL == "fcntl":
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                else:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        fh.close()


def atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically: write to a sibling temp file, then os.replace()
    it over the target. os.replace is atomic on both POSIX and Windows when
    source and destination are on the same volume, so a crash mid-write
    leaves either the old file or the new one, never a truncated hybrid.

    Note: atomicity protects against a *torn* file, not against a lost
    update between two processes that each read-then-wrote. Use
    file_lock() around a read-modify-write sequence for that.
    """
    LOG_DIR.mkdir(exist_ok=True)
    # Unique temp name per process so two concurrent writers can't clobber
    # each other's temp file mid-write and produce a corrupt result.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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


def update_positions(mutator):
    """
    Read-modify-write positions.json under a cross-process lock.

    `mutator` takes the current positions list and returns the new one
    (or None to abort the write). Use this instead of a bare
    load_positions() ... save_positions() pair anywhere the new value
    depends on the old one — otherwise bot.py's scan loop and
    dashboard.py's manual close can interleave and silently drop one of
    the two updates.

    Returns whatever the mutator returned.
    """
    with file_lock("positions"):
        positions, ok = load_positions_checked()
        if not ok:
            log.error("state_store: refusing to modify unreadable positions.json")
            return None
        updated = mutator(positions)
        if updated is not None:
            atomic_write_json(POSITIONS_FILE, updated)
        return updated


# ── Trade journal ────────────────────────────────────────────────────────────

def load_journal() -> list:
    data, err = _load_or(JOURNAL_FILE)
    if err is not None or data is None:
        return []
    return data


def append_journal(entry: dict, keep: int = 500) -> None:
    with file_lock("journal"):
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


def update_daily_stats(mutator):
    """
    Read-modify-write today's stats file under a cross-process lock. Same
    rationale as update_positions() — accumulating P&L and incrementing
    trade/win/loss counters is a read-modify-write, so two processes
    closing positions at the same moment could otherwise lose one trade's
    P&L from the daily total entirely.

    Refuses to write if the existing file is corrupted (fails closed
    rather than overwriting an unreadable day with a fresh-looking one).
    """
    with file_lock("daily_stats"):
        stats = load_daily_stats()
        if stats.get("_corrupted"):
            log.error("state_store: refusing to modify unreadable daily stats file")
            return None
        updated = mutator(stats)
        if updated is not None:
            atomic_write_json(daily_stats_file(), updated)
        return updated


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
