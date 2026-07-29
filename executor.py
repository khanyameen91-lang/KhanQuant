"""
executor.py — Order Execution & Position Management

Places, monitors, and closes options orders via Tastytrade API.
Enforces all risk rules before any order is submitted.
"""

import os
import json
import time
from datetime import datetime, date
from pathlib import Path
from auth import session
import alerts
import state_store
import profit_protection

# ── Risk Configuration (loaded from env) ──────────────────────────────────────
# Kept here for backward compatibility (bot.py's v1-fallback stub imports
# these directly) — the actual risk *decision* now always goes through
# risk_manager.check_all_risk(); see check_risk() below.
DAILY_LOSS_LIMIT = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
MAX_POSITION_SIZE = float(os.environ.get("MAX_POSITION_SIZE", 150))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", 3))
# How old a position's last successful price refresh can be before
# check_exits() stops trusting it for P&L-based stop-loss/profit-target
# decisions. Default 6 min = 3 missed 2-minute scan cycles in a row.
STALE_PRICE_THRESHOLD_SEC = int(os.environ.get("STALE_PRICE_THRESHOLD_SEC", 360))

# ── Trade Log ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
POSITIONS_FILE = LOG_DIR / "positions.json"
def _daily_stats_file() -> Path:
    return LOG_DIR / f"stats_{date.today().isoformat()}.json"
JOURNAL_FILE = LOG_DIR / "trade_journal.json"


# Journal/positions/daily-stats load+save now live in state_store.py (single
# source of truth, atomic writes, fails closed on corrupted state). Kept as
# aliases here since the rest of this file — and bot.py's v1-fallback stub —
# calls them by these names.
_load_journal = state_store.load_journal
_load_positions = state_store.load_positions
_save_positions = state_store.save_positions
_load_daily_stats = state_store.load_daily_stats
_save_daily_stats = state_store.save_daily_stats


def _append_journal(entry: dict):
    state_store.append_journal(entry)


def _apply_close_to_daily_stats(pnl: float) -> None:
    """Fold a closed trade's P&L into today's stats under a lock."""
    def _mutate(stats):
        stats["pnl"] = round(stats.get("pnl", 0.0) + pnl, 2)
        stats["trade_count"] = stats.get("trade_count", 0) + 1
        if pnl >= 0:
            stats["winners"] = stats.get("winners", 0) + 1
        else:
            stats["losers"] = stats.get("losers", 0) + 1
        return stats
    state_store.update_daily_stats(_mutate)


def _remove_position_from_state(position: dict) -> tuple:
    """Drop a closed position from positions.json under a lock.
    Returns (count_before, count_after)."""
    counts = {"before": 0, "after": 0}

    def _mutate(positions):
        counts["before"] = len(positions)
        target_id = position.get("order_id", position.get("id", ""))
        remaining = [p for p in positions
                     if p.get("order_id", p.get("id", str(id(p)))) != target_id]
        # fallback: remove by symbol if id didn't match
        if len(remaining) == counts["before"]:
            remaining = [p for p in positions if p.get("symbol") != position.get("symbol")]
        counts["after"] = len(remaining)
        return remaining

    state_store.update_positions(_mutate)
    return counts["before"], counts["after"]


# ── Risk Checks ────────────────────────────────────────────────────────────────
def check_risk(setup: dict) -> tuple[bool, str]:
    """
    Run all risk checks before allowing a trade.
    Returns (approved: bool, reason: str)

    This used to be its own, narrower 5-check reimplementation of
    risk_manager.check_all_risk() — no halt-state check, no delta/drawdown/
    monthly/sector/strategy-concentration checks. Two independently
    maintained risk gates is exactly how one of them quietly drifts out of
    sync with the other (e.g. a future caller that reaches place_order()
    without going through bot.py's upstream check_all_risk() call would
    have bypassed the trading halt entirely). There is now exactly one risk
    gate: risk_manager.check_all_risk(). This function delegates to it so
    every order-placement path — today's and any future one — is covered.
    """
    import risk_manager
    return risk_manager.check_all_risk(setup)


# ── Order Building ─────────────────────────────────────────────────────────────
def _build_spread_order(setup: dict, account_number: str) -> dict:
    """Build a debit vertical spread order."""
    long_leg = setup["long_leg"]
    short_leg = setup["short_leg"]
    return {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(setup["net_debit"], 2)),
        "price-effect": "Debit",
        "legs": [
            {"instrument-type": "Equity Option", "symbol": long_leg["symbol"],
             "quantity": 1, "action": "Buy to Open"},
            {"instrument-type": "Equity Option", "symbol": short_leg["symbol"],
             "quantity": 1, "action": "Sell to Open"},
        ]
    }


def _build_credit_spread_order(setup: dict) -> dict:
    """Build a credit spread order (bull put or bear call)."""
    short_leg = setup["short_leg"]
    long_leg = setup["long_leg"]
    return {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(setup["net_credit"], 2)),
        "price-effect": "Credit",
        "legs": [
            {"instrument-type": "Equity Option", "symbol": short_leg["symbol"],
             "quantity": 1, "action": "Sell to Open"},
            {"instrument-type": "Equity Option", "symbol": long_leg["symbol"],
             "quantity": 1, "action": "Buy to Open"},
        ]
    }


def _build_iron_condor_order(setup: dict) -> dict:
    """Build a 4-leg iron condor order."""
    return {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(setup["net_credit"], 2)),
        "price-effect": "Credit",
        "legs": [
            {"instrument-type": "Equity Option", "symbol": setup["short_put"]["symbol"],
             "quantity": 1, "action": "Sell to Open"},
            {"instrument-type": "Equity Option", "symbol": setup["long_put"]["symbol"],
             "quantity": 1, "action": "Buy to Open"},
            {"instrument-type": "Equity Option", "symbol": setup["short_call"]["symbol"],
             "quantity": 1, "action": "Sell to Open"},
            {"instrument-type": "Equity Option", "symbol": setup["long_call"]["symbol"],
             "quantity": 1, "action": "Buy to Open"},
        ]
    }


def _build_strangle_order(setup: dict) -> dict:
    """Build a long strangle order (buy OTM call + OTM put)."""
    lc = setup["long_call"]
    lp = setup["long_put"]
    return {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(lc["ask"] + lp["ask"], 2)),
        "price-effect": "Debit",
        "legs": [
            {"instrument-type": "Equity Option", "symbol": lc["symbol"],
             "quantity": 1, "action": "Buy to Open"},
            {"instrument-type": "Equity Option", "symbol": lp["symbol"],
             "quantity": 1, "action": "Buy to Open"},
        ]
    }


def _build_long_option_order(setup: dict) -> dict:
    """Build a single-leg long option order (long call, long put, or 0DTE scalp)."""
    contract = setup["contract"]
    return {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(contract["ask"], 2)),
        "price-effect": "Debit",
        "legs": [
            {"instrument-type": "Equity Option", "symbol": contract["symbol"],
             "quantity": 1, "action": "Buy to Open"},
        ]
    }


# ── Order Placement ────────────────────────────────────────────────────────────
def place_order(setup: dict, decision: dict) -> dict | None:
    """
    Place an options order after passing risk checks.
    Returns position dict if successful, None if failed.
    """
    account_number = session.account_number

    # Run risk checks first
    approved, reason = check_risk(setup)
    if not approved:
        print(f"🚫 Trade blocked: {reason}")
        return None

    # Build order payload — route by strategy type
    strategy_label = setup.get("strategy_name", setup["type"])
    t = setup["type"]
    if t == "spread":
        order = _build_spread_order(setup, account_number)
    elif t == "credit_spread":
        order = _build_credit_spread_order(setup)
    elif t == "iron_condor":
        order = _build_iron_condor_order(setup)
    elif t == "strangle":
        order = _build_strangle_order(setup)
    else:  # long_option, 0dte_scalp
        order = _build_long_option_order(setup)

    print(f"📤 Placing order: {setup['symbol']} {strategy_label}...")

    # SANDBOX MODE: log the order but don't send it to the broker
    if os.environ.get("TRADING_MODE", "sandbox").lower() == "sandbox":
        print(f"🟡 SANDBOX MODE — order simulated (not sent to Tastytrade):")
        print(f"   {order}")
        order_id = f"SIM-{int(time.time())}"
        now = datetime.now()
        position = {
            "order_id": order_id,
            "symbol": setup["symbol"],
            "strategy": strategy_label,
            "type": setup["type"],
            "direction": setup["direction"],
            "expiration": setup["expiration"],
            "max_loss": setup["max_loss"],
            "max_profit": setup.get("max_profit"),
            "entry_date": date.today().isoformat(),
            "entry_time": now.isoformat(),
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning"),
            "setup": setup,
            # ── ML training metadata ──────────────────────────────────────
            "technical_score":     decision.get("technical_score", 50),
            "flow_score":          decision.get("flow_score", 50),
            "sentiment_score":     decision.get("sentiment_score", 50),
            "volatility_score":    decision.get("volatility_score", 50),
            "regime_score":        decision.get("regime_score", 50),
            "institutional_score": decision.get("institutional_score", 50),
            "ev_score":            decision.get("ev_score", 50),
            "liquidity_score":     decision.get("liquidity_score", 50),
            "rs_score":            decision.get("rs_score", 50),
            "regime_type":         decision.get("regime_type", "unknown"),
            "iv_rank":             decision.get("iv_rank", 50),
            "dte":                 decision.get("dte", 14),
            "hour_of_day":         now.hour + now.minute / 60,
            "day_of_week":         now.weekday(),
            "ml_win_prob":         decision.get("ml_win_prob", 0),
        }
        state_store.update_positions(lambda ps: ps + [position])
        alerts.trade_opened({
            "symbol": setup["symbol"],
            "strategy": f"[SANDBOX] {strategy_label}",
            "direction": setup["direction"],
            "expiration": setup["expiration"],
            "max_profit": setup.get("max_profit", 0) or 0,
            "max_loss": setup["max_loss"],
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning", ""),
        })
        return position

    try:
        resp = session.post(
            f"/accounts/{account_number}/orders",
            body=order
        )

        order_data = resp.get("data", {})
        order_id = order_data.get("order", {}).get("id")

        if not order_id:
            print(f"❌ Order placement failed: {resp}")
            alerts.error("place_order", Exception(f"No order ID returned: {resp}"))
            return None

        print(f"✅ Order placed! ID: {order_id}")

        # An accepted order ID is NOT a fill. This used to record the
        # position as open the instant the broker returned an ID, so a
        # limit order that was accepted but never filled (price moved
        # away, or it sat and expired) would be tracked locally as a real
        # position that doesn't exist at the broker — with exit checks
        # and stop-losses running against it forever.
        fill_status = _await_fill(account_number, order_id)
        if fill_status != "Filled":
            print(f"⚠️  Order {order_id} did not fill (status: {fill_status}) — not recording a position")
            alerts.degraded(
                "order_unfilled",
                f"{setup['symbol']} {strategy_label} order {order_id} ended as "
                f"'{fill_status}' rather than Filled — no position recorded",
                cooldown_minutes=5,
            )
            return None

        # Build position record
        now = datetime.now()
        position = {
            "order_id": order_id,
            "symbol": setup["symbol"],
            "strategy": strategy_label,
            "type": setup["type"],
            "direction": setup["direction"],
            "expiration": setup["expiration"],
            "max_loss": setup["max_loss"],
            "max_profit": setup.get("max_profit"),
            "entry_date": date.today().isoformat(),
            "entry_time": now.isoformat(),
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning"),
            "setup": setup,
            # ── ML training metadata ──────────────────────────────────────
            "technical_score":     decision.get("technical_score", 50),
            "flow_score":          decision.get("flow_score", 50),
            "sentiment_score":     decision.get("sentiment_score", 50),
            "volatility_score":    decision.get("volatility_score", 50),
            "regime_score":        decision.get("regime_score", 50),
            "institutional_score": decision.get("institutional_score", 50),
            "ev_score":            decision.get("ev_score", 50),
            "liquidity_score":     decision.get("liquidity_score", 50),
            "rs_score":            decision.get("rs_score", 50),
            "regime_type":         decision.get("regime_type", "unknown"),
            "iv_rank":             decision.get("iv_rank", 50),
            "dte":                 decision.get("dte", 14),
            "hour_of_day":         now.hour + now.minute / 60,
            "day_of_week":         now.weekday(),
            "ml_win_prob":         decision.get("ml_win_prob", 0),
        }

        # Save position
        state_store.update_positions(lambda ps: ps + [position])

        # Send alert
        alerts.trade_opened({
            "symbol": setup["symbol"],
            "strategy": strategy_label,
            "direction": setup["direction"],
            "expiration": setup["expiration"],
            "max_profit": setup.get("max_profit", 0) or 0,
            "max_loss": setup["max_loss"],
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning", ""),
        })

        return position

    except Exception as e:
        print(f"❌ Order error: {e}")
        alerts.error("place_order", e)
        return None


def get_order_status(account_number: str, order_id: str):
    """Current broker-side status for an order, or None if unreadable."""
    try:
        resp = session.get(f"/accounts/{account_number}/orders/{order_id}")
        return resp.get("data", {}).get("status")
    except Exception as e:
        print(f"⚠️  Order status fetch error for {order_id}: {e}")
        return None


# Terminal states — no point polling further once an order reaches one.
_ORDER_DONE_STATES = {"Filled", "Cancelled", "Rejected", "Expired", "Removed"}


def _await_fill(account_number: str, order_id: str,
                timeout_sec: float = 45.0, poll_sec: float = 3.0) -> str:
    """
    Poll an order until it fills, reaches another terminal state, or the
    timeout expires. Returns the final status string ("Filled",
    "Rejected", "Live" if still working when we give up, etc.).

    A limit order that's still 'Live' at timeout isn't an error — it just
    hasn't filled yet, and the caller should not record it as an open
    position. It stays working at the broker and will be picked up by
    reconcile_with_broker() on a later cycle if it does fill.
    """
    deadline = time.time() + timeout_sec
    status = None
    while time.time() < deadline:
        status = get_order_status(account_number, order_id)
        if status in _ORDER_DONE_STATES:
            return status
        time.sleep(poll_sec)
    return status or "Unknown"


def reconcile_with_broker() -> dict:
    """
    Compare locally-tracked positions against what the broker actually
    reports, and alert on any mismatch.

    Nothing anywhere used to do this. A local file that's lost or
    corrupted, a manual trade placed outside the bot, an order that was
    accepted but never filled, or a position closed at the broker by
    something other than the bot would all drift silently — the bot would
    keep managing a position that isn't there, or ignore one that is.

    Read-only and advisory: it reports, it doesn't auto-mutate local
    state. Auto-"fixing" a mismatch is exactly the kind of destructive
    guess that should stay a human decision.

    Skipped entirely in sandbox mode, where local state IS the source of
    truth by design (no real broker positions exist to compare against).
    """
    if os.environ.get("TRADING_MODE", "sandbox").lower() == "sandbox":
        return {"skipped": "sandbox mode"}

    try:
        account_number = session.account_number
        resp = session.get(f"/accounts/{account_number}/positions")
        broker_items = resp.get("data", {}).get("items", [])
    except Exception as e:
        alerts.degraded("reconcile", f"Couldn't fetch broker positions: {e}", cooldown_minutes=60)
        return {"error": str(e)}

    broker_symbols = {p.get("symbol") for p in broker_items if p.get("symbol")}
    local_positions = _load_positions()
    local_symbols = {p.get("symbol") for p in local_positions if p.get("symbol")}

    only_local = local_symbols - broker_symbols
    only_broker = broker_symbols - local_symbols

    if only_local:
        alerts.risk_warning(
            f"⚠️ Reconciliation: {len(only_local)} position(s) tracked locally but "
            f"NOT open at the broker: {', '.join(sorted(only_local))}. "
            f"Likely an unfilled order or a fill the bot missed — needs review."
        )
    if only_broker:
        alerts.risk_warning(
            f"⚠️ Reconciliation: {len(only_broker)} position(s) open at the broker but "
            f"NOT tracked by the bot: {', '.join(sorted(only_broker))}. "
            f"These are unmanaged — no stop-loss or exit logic is running on them."
        )

    return {
        "local_count": len(local_symbols),
        "broker_count": len(broker_symbols),
        "only_local": sorted(only_local),
        "only_broker": sorted(only_broker),
        "in_sync": not only_local and not only_broker,
    }


# ── Position Monitoring ────────────────────────────────────────────────────────
def get_position_pnl(position: dict):
    """
    Calculate current P&L for an open position. Returns None (not a silent
    0.0) if it genuinely can't be determined — a real loss on a position
    whose fetch is failing shouldn't read as "flat," which used to look
    identical to a real $0 P&L both in the stop-loss check and on the
    dashboard.

    In sandbox mode, use the yfinance-refreshed unrealized_pnl field.
    In live mode, fetch from Tastytrade API.
    """
    # Sandbox: use locally computed P&L from refresh_position_prices()
    if os.environ.get("TRADING_MODE", "sandbox").lower() == "sandbox":
        return float(position.get("unrealized_pnl", 0.0))

    try:
        account_number = session.account_number
        resp = session.get(f"/accounts/{account_number}/positions")
        positions_data = resp.get("data", {}).get("items", [])

        for pos in positions_data:
            if pos.get("symbol") == position["symbol"]:
                return float(pos.get("today-pnl", pos.get("pnl", 0)))
        # Position not found in the broker's own list — that's not "$0
        # P&L," that's "we don't know," and worth flagging distinctly
        # since it could mean the position was closed outside the bot.
        alerts.degraded(
            f"pnl_missing_{position.get('symbol', '?')}",
            "Position not found in broker's live position list",
            cooldown_minutes=30,
        )
        return None
    except Exception as e:
        print(f"⚠️  PnL fetch error: {e}")
        alerts.degraded(f"pnl_fetch_{position.get('symbol', '?')}", f"PnL fetch error: {e}", cooldown_minutes=30)
        return None


def close_position(position: dict, reason: str = "Bot decision") -> bool:
    """
    Close an open position by placing a closing order.

    Thin wrapper around _close_position_inner() that enforces the
    duplicate-close guard. A close request that succeeded at the broker
    but whose HTTP response was lost (timeout) would raise inside, leaving
    the position in local state — and the next 2-minute scan cycle would
    submit a SECOND closing order against a position that's already flat,
    which on a short leg means opening an unintended naked position. Also
    covers the bot's exit check racing the dashboard's manual close button
    on the same position from a different process.
    """
    pos_key = _position_key(position)
    if not _claim_close(pos_key):
        print(f"close_position: {position.get('symbol')} already has a close in flight — skipping duplicate")
        return False

    closed_ok = False
    try:
        closed_ok = _close_position_inner(position, reason)
        return closed_ok
    finally:
        # Only release the claim if the close did NOT succeed. A
        # successful close already removed the position from local state,
        # so there's nothing left to re-close; holding the claim until it
        # ages out is the safer side to err on if anything is still in
        # flight at the broker.
        if not closed_ok:
            _release_close(pos_key)


def _close_position_inner(position: dict, reason: str = "Bot decision") -> bool:
    import os, sys
    # Write debug to file so we can see it regardless of stdout buffering
    _dbg = LOG_DIR / "close_debug.log"
    def _log(msg):
        try:
            with open(_dbg, "a") as f:
                f.write(f"{datetime.now().isoformat()} {msg}\n")
        except Exception:
            pass
        print(msg, flush=True)

    _log(f"close_position called: symbol={position.get('symbol')} reason={reason}")

    # Load .env so this works when called from the dashboard process (separate from bot)
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
        _log("dotenv loaded OK")
    except Exception as de:
        _log(f"dotenv load failed: {de}")

    # ── SANDBOX FAST PATH: no auth or API needed ───────────────────────────────
    is_sandbox = (
        os.environ.get("TRADING_MODE", "sandbox").lower() == "sandbox"
        or "[SANDBOX]" in position.get("strategy", "")
        or position.get("sandbox", False)
    )
    _log(f"is_sandbox={is_sandbox} TRADING_MODE={os.environ.get('TRADING_MODE','NOT_SET')}")

    if is_sandbox:
        try:
            pnl = float(position.get("unrealized_pnl", 0.0))
            # Update daily stats + remove from open positions. Both are
            # read-modify-writes on files the dashboard process also
            # writes (its manual close-position button calls right into
            # this same function), so they go through the locked helpers
            # rather than a bare load/save pair that could interleave and
            # silently drop one process's update.
            _apply_close_to_daily_stats(pnl)
            before, after = _remove_position_from_state(position)
            _log(f"SANDBOX close OK: pnl=${pnl:.2f} positions_before={before} positions_after={after}")
            try:
                alerts.trade_closed(position, pnl)
            except Exception:
                pass
            return True
        except Exception as e:
            _log(f"SANDBOX close ERROR: {e}")
            import traceback
            _log(traceback.format_exc())
            return False

    # ── LIVE MODE: authenticate and send order ─────────────────────────────────
    username = os.environ.get("TASTY_USERNAME", "")
    password = os.environ.get("TASTY_PASSWORD", "")
    _log(f"LIVE MODE: user={'SET' if username else 'MISSING'}")
    if not username or not password:
        _log("❌ TASTY_USERNAME/PASSWORD not set")
        return False

    session.ensure_valid(username, password)
    account_number = session.account_number
    _log(f"account_number={account_number}")
    if not account_number:
        _log(f"❌ account_number is None after auth")
        return False
    setup = position.get("setup", {})

    print(f"📤 Closing {position['symbol']} {position.get('strategy','')} LIVE — {reason}")

    try:
        raw_type = position.get("type", "long_option").lower().replace("-", "_").replace(" ", "_")

        # Normalize type variants to canonical names
        if "iron_condor" in raw_type:
            t = "iron_condor"
        elif "strangle" in raw_type or "straddle" in raw_type:
            t = "strangle"
        elif "credit" in raw_type or "bull_put" in raw_type or "bear_call" in raw_type:
            t = "credit_spread"
        elif "debit" in raw_type or "bull_call" in raw_type or "bear_put" in raw_type or "spread" in raw_type:
            t = "spread"
        else:
            t = "long_option"

        print(f"  Position type: '{raw_type}' → closing as '{t}'")

        if t == "iron_condor":
            order = {
                "time-in-force": "Day", "order-type": "Market", "price-effect": "Debit",
                "legs": [
                    {"instrument-type": "Equity Option", "symbol": setup.get("short_put", {}).get("symbol", ""),
                     "quantity": 1, "action": "Buy to Close"},
                    {"instrument-type": "Equity Option", "symbol": setup.get("long_put", {}).get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                    {"instrument-type": "Equity Option", "symbol": setup.get("short_call", {}).get("symbol", ""),
                     "quantity": 1, "action": "Buy to Close"},
                    {"instrument-type": "Equity Option", "symbol": setup.get("long_call", {}).get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                ]
            }
        elif t == "strangle":
            order = {
                "time-in-force": "Day", "order-type": "Market", "price-effect": "Credit",
                "legs": [
                    {"instrument-type": "Equity Option", "symbol": setup.get("long_call", {}).get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                    {"instrument-type": "Equity Option", "symbol": setup.get("long_put", {}).get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                ]
            }
        elif t == "spread":
            long_leg  = setup.get("long_leg", {})
            short_leg = setup.get("short_leg", {})
            order = {
                "time-in-force": "Day", "order-type": "Market", "price-effect": "Credit",
                "legs": [
                    {"instrument-type": "Equity Option", "symbol": long_leg.get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                    {"instrument-type": "Equity Option", "symbol": short_leg.get("symbol", ""),
                     "quantity": 1, "action": "Buy to Close"},
                ]
            }
        elif t == "credit_spread":
            short_leg = setup.get("short_leg", {})
            long_leg  = setup.get("long_leg", {})
            order = {
                "time-in-force": "Day", "order-type": "Market", "price-effect": "Debit",
                "legs": [
                    {"instrument-type": "Equity Option", "symbol": short_leg.get("symbol", ""),
                     "quantity": 1, "action": "Buy to Close"},
                    {"instrument-type": "Equity Option", "symbol": long_leg.get("symbol", ""),
                     "quantity": 1, "action": "Sell to Close"},
                ]
            }
        else:  # long_option, 0dte_scalp
            contract = setup.get("contract", {})
            sym = contract.get("symbol", "") if contract else ""
            order = {
                "time-in-force": "Day", "order-type": "Market", "price-effect": "Credit",
                "legs": [
                    {"instrument-type": "Equity Option", "symbol": sym,
                     "quantity": 1, "action": "Sell to Close"},
                ]
            }

        # Send live order
        session.post(f"/accounts/{account_number}/orders", body=order)

        # Calculate final PnL, update stats, remove position, log journal
        pnl = get_position_pnl(position)
        if pnl is None:
            # Couldn't confirm the real fill P&L from the broker (already
            # alerted inside get_position_pnl). Recording it as $0 would
            # silently corrupt today's stats and the win/loss counters —
            # log it clearly instead and record a neutral placeholder that
            # a human needs to reconcile manually, rather than pretending
            # it was a scratch trade.
            _log(f"WARNING: could not determine final P&L for {position.get('symbol')} — recording as $0, needs manual reconciliation")
            pnl = 0.0

        # Both are read-modify-writes on files the dashboard process also
        # writes — see the sandbox close path above for the full rationale.
        _apply_close_to_daily_stats(pnl)
        _remove_position_from_state(position)

        try:
            alerts.trade_closed(position, pnl)
        except Exception:
            pass

        closed_at = datetime.now()
        entry_time = position.get("entry_time", "")
        try:
            held_minutes = (closed_at - datetime.fromisoformat(entry_time)).seconds // 60 if entry_time else 0
        except Exception:
            held_minutes = 0

        _append_journal({
            "symbol": position["symbol"], "strategy": position.get("strategy", ""),
            "type": position.get("type", ""), "direction": position.get("direction", ""),
            "order_id": position.get("order_id", ""), "entry_date": position.get("entry_date", ""),
            "entry_time": entry_time, "closed_at": closed_at.isoformat(),
            "expiration": position.get("expiration", ""), "held_minutes": held_minutes,
            "hour_of_day": position.get("hour_of_day", closed_at.hour),
            "day_of_week": position.get("day_of_week", closed_at.weekday()),
            "pnl": round(pnl, 2), "max_loss": position.get("max_loss", 0),
            "max_profit": position.get("max_profit", 0), "close_reason": reason,
            "reasoning": position.get("reasoning", ""),
            "confidence": position.get("confidence", 0),
            # Sub-scores are recorded as null when genuinely unavailable
            # rather than defaulted to 50 — the ML feature extractor
            # encodes that distinction (see ml_engine._SCORE_KEYS), so a
            # subsystem that was down doesn't train the model as though
            # it had returned a real neutral reading.
            "technical_score": position.get("technical_score"),
            "flow_score": position.get("flow_score"),
            "sentiment_score": position.get("sentiment_score"),
            "volatility_score": position.get("volatility_score"),
            "regime_score": position.get("regime_score"),
            "institutional_score": position.get("institutional_score"),
            "ev_score": position.get("ev_score"),
            "liquidity_score": position.get("liquidity_score"),
            "rs_score": position.get("rs_score"),
            "regime_type": position.get("regime_type", "unknown"),
            "iv_rank": position.get("iv_rank", 50), "dte": position.get("dte", 14),
            "ml_win_prob": position.get("ml_win_prob", 0),
        })

        print(f"✅ LIVE position closed. P&L: ${pnl:+.2f}")
        return True

    except Exception as e:
        print(f"❌ LIVE close error: {e}")
        try:
            alerts.error("close_position", e)
        except Exception:
            pass
        return False


def check_exits(brain) -> int:
    """
    Check all open positions for exit conditions.
    Returns number of positions closed.
    """
    positions = _load_positions()
    closed = 0

    # Read the CURRENT protection state once per cycle — this used to be
    # computed (tightened stop under RESTRICT/REDUCE_HALF) but never
    # actually reach this function, which hardcoded a flat 75% stop
    # regardless of protection level. Also carries the "was up 5%, now
    # below 2%" drawback flag, which used to only log a message.
    protection = profit_protection.get_protection_status()
    stop_frac = protection.get("stop_level", 0.75)

    if protection.get("drawback_alert"):
        for position in positions:
            close_position(position, "Profit drawback protection — closing all positions")
            closed += 1
        if closed:
            alerts.risk_warning(
                f"🚨 Drawback protection closed {closed} position(s): "
                f"was up ${protection['peak_daily_pnl']:.2f}, now ${protection['daily_pnl']:.2f}"
            )
        return closed

    for position in positions:
        pnl = get_position_pnl(position)
        max_loss = position.get("max_loss", 150)
        max_profit = position.get("max_profit") or max_loss * 2

        # pnl can be None now (live-mode fetch failure — see
        # get_position_pnl) instead of a silent 0.0 that looked
        # indistinguishable from a genuinely flat position.
        pnl_pct = (pnl / max_loss) * 100 if (max_loss and pnl is not None) else 0

        # Calculate DTE remaining
        try:
            exp_date = date.fromisoformat(position["expiration"])
            dte_remaining = (exp_date - date.today()).days
        except Exception:
            dte_remaining = 999

        # Hard exit rules
        if dte_remaining <= 1:
            close_position(position, "1 DTE — forced close")
            closed += 1
            continue

        if pnl is None:
            alerts.degraded(
                f"pnl_fetch_{position.get('symbol', '?')}",
                "Couldn't fetch this position's live P&L from the broker — "
                "skipping stop/target checks this cycle (DTE-based exit still applies)",
                cooldown_minutes=30,
            )
            continue

        # Price staleness check. refresh_position_prices() used to fail
        # silently per-position (a chain fetch error, a missing strike) and
        # just leave unrealized_pnl at whatever value it last had — with no
        # way to tell "this is current" from "this hasn't updated in
        # hours." A real loss could sit unnoticed indefinitely because it
        # never crossed the stop-loss threshold computed from a frozen
        # number, while the dashboard kept showing a plausible P&L the
        # whole time. Skip the P&L-based checks below (not the DTE check
        # above, which doesn't need a live price) when the price is too
        # old to trust, and alert — a position with a stuck price feed is
        # exactly the kind of thing that needs a human to look at it.
        updated_at = position.get("price_updated_at")
        price_age_sec = None
        if updated_at:
            try:
                price_age_sec = (datetime.now() - datetime.fromisoformat(updated_at)).total_seconds()
            except Exception:
                price_age_sec = None

        if updated_at and (price_age_sec is None or price_age_sec > STALE_PRICE_THRESHOLD_SEC):
            age_desc = "an unreadable timestamp" if price_age_sec is None else f"{price_age_sec/60:.0f} min"
            alerts.degraded(
                f"stale_price_{position.get('symbol', '?')}",
                f"No successful price refresh in {age_desc} — skipping P&L-based "
                f"stop/target checks this cycle (DTE-based exit still applies)",
                cooldown_minutes=30,
            )
            continue
        # updated_at is None (no timestamp at all yet) just means this
        # position hasn't been through its first refresh cycle — not a
        # failure, nothing to alert about, just nothing to check yet either.
        elif updated_at is None:
            continue

        if pnl <= -(max_loss * stop_frac):
            close_position(position, f"Stop loss hit ({pnl_pct:.0f}%)")
            closed += 1
            continue

        if max_profit and pnl >= (max_profit * 0.50):
            close_position(position, f"Profit target hit ({pnl_pct:.0f}%)")
            closed += 1
            continue

        # Ask Claude if we should close
        position_with_stats = {
            **position,
            "days_held": (date.today() - date.fromisoformat(position["entry_date"])).days,
            "dte_remaining": dte_remaining,
        }
        claude_decision = brain.should_close_position(position_with_stats, pnl_pct)
        if claude_decision.get("decision") == "CLOSE":
            close_position(position, f"Claude: {claude_decision.get('reasoning', '')[:50]}")
            closed += 1

    return closed


def _get_option_mid(chain, opt_type: str, strike: float) -> float:
    """Fetch mid price for a single option leg from a yfinance chain."""
    df = chain.calls if opt_type == "C" else chain.puts
    row = df[df["strike"] == strike]
    if row.empty:
        all_strikes = df["strike"]
        nearest = all_strikes.iloc[(all_strikes - strike).abs().argsort()[:1]].iloc[0]
        row = df[df["strike"] == nearest]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    bid  = float(r.get("bid", 0) or 0)
    ask  = float(r.get("ask", 0) or 0)
    last = float(r.get("lastPrice", 0) or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return last


def _position_key(pos: dict) -> str:
    """Stable identity for matching a position across a reload."""
    return str(pos.get("order_id") or pos.get("id") or pos.get("symbol", ""))


# How long a close-in-flight claim stays valid. Long enough to cover a
# hung HTTP request, short enough that a genuinely failed close can be
# retried on a later cycle rather than being blocked forever.
_CLOSE_CLAIM_TTL_SEC = 300


def _claim_close(pos_key: str) -> bool:
    """
    Atomically claim the right to close this position. Returns False if
    another close for the same position is already in flight (possibly in
    the other process). Claims older than the TTL are treated as stale and
    reclaimable, so a crashed close attempt doesn't wedge the position
    permanently.
    """
    claims_file = LOG_DIR / "close_claims.json"
    now = time.time()
    with state_store.file_lock("close_claims"):
        try:
            claims = json.loads(claims_file.read_text()) if claims_file.exists() else {}
        except Exception:
            claims = {}

        claims = {k: v for k, v in claims.items() if now - v < _CLOSE_CLAIM_TTL_SEC}
        if pos_key in claims:
            return False

        claims[pos_key] = now
        state_store.atomic_write_json(claims_file, claims)
        return True


def _release_close(pos_key: str) -> None:
    """Drop a close claim once the close is fully resolved."""
    claims_file = LOG_DIR / "close_claims.json"
    with state_store.file_lock("close_claims"):
        try:
            claims = json.loads(claims_file.read_text()) if claims_file.exists() else {}
        except Exception:
            return
        if claims.pop(pos_key, None) is not None:
            state_store.atomic_write_json(claims_file, claims)


def refresh_position_prices():
    """
    Fetch current market prices for all open positions via yfinance
    and update positions.json with live P&L data.
    Handles single-leg options and two-leg spreads.

    Prices are fetched into a side dict first and only merged back into
    positions.json at the end under a lock, re-reading the file fresh.
    Fetching takes seconds (network-bound, one chain per position), and
    writing back a list captured before all that would resurrect any
    position closed in the meantime — by the bot's own exit check or by
    the dashboard's manual close button running in a separate process.
    """
    positions = _load_positions()
    if not positions:
        return

    try:
        import yfinance as yf
    except ImportError:
        return

    price_updates = {}   # position_key -> dict of fields to merge back
    for pos in positions:
        try:
            setup      = pos.get("setup", {})
            symbol     = pos.get("symbol", "")
            expiration = pos.get("expiration", "")
            pos_type   = pos.get("type", "long_option")

            if not symbol or not expiration:
                continue

            ticker = yf.Ticker(symbol)
            try:
                chain = ticker.option_chain(expiration)
            except Exception:
                continue

            # ── Spread positions (credit or debit, two legs) ──────────────────
            if pos_type in ("credit_spread", "debit_spread", "bull_put_spread",
                            "bear_call_spread", "bull_call_spread", "bear_put_spread"):
                long_leg  = setup.get("long_leg", {})
                short_leg = setup.get("short_leg", {})
                opt_type  = setup.get("opt_type", "C")

                long_strike  = float(long_leg.get("strike", 0))
                short_strike = float(short_leg.get("strike", 0))
                if long_strike <= 0 or short_strike <= 0:
                    continue

                long_price  = _get_option_mid(chain, opt_type, long_strike)
                short_price = _get_option_mid(chain, opt_type, short_strike)
                if long_price == 0 and short_price == 0:
                    continue

                current_spread_value = short_price - long_price  # positive = credit spread value

                if setup.get("net_credit"):
                    # Credit spread: P&L = entry_credit - current_spread_value
                    entry_credit = float(setup["net_credit"])
                    unrealized_pnl = round((entry_credit - current_spread_value) * 100, 2)
                    max_profit = float(setup.get("max_profit") or pos.get("max_profit") or entry_credit * 100 or 1)
                    pnl_pct = round(unrealized_pnl / max_profit * 100, 1)
                else:
                    # Debit spread: P&L = current_spread_value - entry_debit
                    entry_debit = float(setup.get("net_debit", 0))
                    unrealized_pnl = round((current_spread_value - entry_debit) * 100, 2)
                    max_profit = float(setup.get("max_profit") or pos.get("max_profit") or 1)
                    pnl_pct = round(unrealized_pnl / max_profit * 100, 1)

                price_updates[_position_key(pos)] = {
                    "unrealized_pnl":   unrealized_pnl,
                    "pnl_pct":          pnl_pct,
                    "price_updated_at": datetime.now().isoformat(),
                    "price_stale":      False,
                }

            # ── Single-leg options ────────────────────────────────────────────
            else:
                contract    = setup.get("contract", {})
                entry_price = float(contract.get("ask", 0) or 0)
                opt_type    = setup.get("opt_type", "C")
                strike      = float(contract.get("strike", 0))

                if entry_price <= 0 or strike <= 0:
                    continue

                current_price = _get_option_mid(chain, opt_type, strike)
                if current_price == 0:
                    continue

                unrealized_pnl = round((current_price - entry_price) * 100, 2)
                pnl_pct = round((current_price - entry_price) / entry_price * 100, 1)

                price_updates[_position_key(pos)] = {
                    "current_price":    round(current_price, 2),
                    "entry_price":      round(entry_price, 2),
                    "unrealized_pnl":   unrealized_pnl,
                    "pnl_pct":          pnl_pct,
                    "price_updated_at": datetime.now().isoformat(),
                    "price_stale":      False,
                }

        except Exception as e:
            print(f"  Price refresh error {pos.get('symbol', '?')}: {e}")

    if not price_updates:
        return

    def _merge(current_positions):
        for p in current_positions:
            fields = price_updates.get(_position_key(p))
            if fields:
                p.update(fields)
        return current_positions

    state_store.update_positions(_merge)


def get_open_positions() -> list:
    """Return current open positions."""
    return _load_positions()


def get_daily_stats() -> dict:
    """Return today's trading stats."""
    return _load_daily_stats()
