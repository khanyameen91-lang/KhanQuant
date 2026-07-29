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

# ── Risk Configuration (loaded from env) ──────────────────────────────────────
DAILY_LOSS_LIMIT = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
MAX_POSITION_SIZE = float(os.environ.get("MAX_POSITION_SIZE", 150))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", 3))

# ── Trade Log ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
POSITIONS_FILE = LOG_DIR / "positions.json"
def _daily_stats_file() -> Path:
    return LOG_DIR / f"stats_{date.today().isoformat()}.json"
JOURNAL_FILE = LOG_DIR / "trade_journal.json"


def _load_journal() -> list:
    if JOURNAL_FILE.exists():
        try:
            return json.loads(JOURNAL_FILE.read_text())
        except Exception:
            return []
    return []


def _append_journal(entry: dict):
    journal = _load_journal()
    journal.insert(0, entry)
    journal = journal[:500]  # keep last 500 trades
    JOURNAL_FILE.write_text(json.dumps(journal, indent=2))


def _load_positions() -> list:
    """Load open positions from disk."""
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text())
        except Exception:
            return []
    return []


def _save_positions(positions: list):
    """Persist open positions to disk."""
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


def _load_daily_stats() -> dict:
    """Load today's trading stats."""
    f = _daily_stats_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {"date": date.today().isoformat(), "pnl": 0.0, "trade_count": 0,
            "winners": 0, "losers": 0}


def _save_daily_stats(stats: dict):
    """Persist daily stats to disk."""
    _daily_stats_file().write_text(json.dumps(stats, indent=2))


# ── Risk Checks ────────────────────────────────────────────────────────────────
def check_risk(setup: dict) -> tuple[bool, str]:
    """
    Run all risk checks before allowing a trade.
    Returns (approved: bool, reason: str)
    """
    positions = _load_positions()
    stats = _load_daily_stats()

    # Check 1: Max open positions
    if len(positions) >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions reached ({MAX_OPEN_POSITIONS})"

    # Check 2: Daily loss limit
    if stats["pnl"] <= -DAILY_LOSS_LIMIT:
        return False, f"Daily loss limit hit (${abs(stats['pnl']):.0f} lost today)"

    # Check 3: Max loss per trade
    if setup.get("max_loss", 0) > MAX_POSITION_SIZE:
        return False, f"Setup max loss ${setup['max_loss']:.0f} exceeds limit ${MAX_POSITION_SIZE:.0f}"

    # Check 4: Don't trade same symbol twice
    open_symbols = [p["symbol"] for p in positions]
    if setup["symbol"] in open_symbols:
        return False, f"Already have an open position in {setup['symbol']}"

    # Check 5: Remaining daily budget
    remaining_budget = DAILY_LOSS_LIMIT - abs(min(stats["pnl"], 0))
    if setup.get("max_loss", 0) > remaining_budget:
        return False, f"Trade max loss ${setup['max_loss']:.0f} exceeds remaining daily budget ${remaining_budget:.0f}"

    return True, "All risk checks passed"


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
        positions = _load_positions()
        positions.append(position)
        _save_positions(positions)
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
        positions = _load_positions()
        positions.append(position)
        _save_positions(positions)

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


# ── Position Monitoring ────────────────────────────────────────────────────────
def get_position_pnl(position: dict) -> float:
    """
    Calculate current P&L for an open position.
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
        return 0.0
    except Exception as e:
        print(f"⚠️  PnL fetch error: {e}")
        return 0.0


def close_position(position: dict, reason: str = "Bot decision") -> bool:
    """
    Close an open position by placing a closing order.
    """
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
            # Update daily stats
            stats = _load_daily_stats()
            stats["pnl"] = round(stats.get("pnl", 0.0) + pnl, 2)
            stats["trade_count"] = stats.get("trade_count", 0) + 1
            if pnl >= 0:
                stats["winners"] = stats.get("winners", 0) + 1
            else:
                stats["losers"] = stats.get("losers", 0) + 1
            _save_daily_stats(stats)
            # Remove from open positions
            positions = _load_positions()
            target_id = position.get("order_id", position.get("id", ""))
            before = len(positions)
            positions = [p for p in positions if p.get("order_id", p.get("id", str(id(p)))) != target_id]
            # fallback: remove by symbol if id didn't match
            if len(positions) == before:
                positions = [p for p in positions if p.get("symbol") != position.get("symbol")]
            _save_positions(positions)
            _log(f"SANDBOX close OK: pnl=${pnl:.2f} positions_before={before} positions_after={len(positions)}")
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
        stats = _load_daily_stats()
        stats["pnl"] = round(stats.get("pnl", 0.0) + pnl, 2)
        stats["trade_count"] = stats.get("trade_count", 0) + 1
        if pnl >= 0:
            stats["winners"] = stats.get("winners", 0) + 1
        else:
            stats["losers"] = stats.get("losers", 0) + 1
        _save_daily_stats(stats)

        positions = _load_positions()
        target_id = position.get("order_id", position.get("id", ""))
        positions = [p for p in positions if p.get("order_id", p.get("id", str(id(p)))) != target_id]
        _save_positions(positions)

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
            "technical_score": position.get("technical_score", 50),
            "flow_score": position.get("flow_score", 50),
            "sentiment_score": position.get("sentiment_score", 50),
            "volatility_score": position.get("volatility_score", 50),
            "regime_score": position.get("regime_score", 50),
            "institutional_score": position.get("institutional_score", 50),
            "ev_score": position.get("ev_score", 50),
            "liquidity_score": position.get("liquidity_score", 50),
            "rs_score": position.get("rs_score", 50),
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

    for position in positions:
        pnl = get_position_pnl(position)
        max_loss = position.get("max_loss", 150)
        max_profit = position.get("max_profit") or max_loss * 2

        pnl_pct = (pnl / max_loss) * 100 if max_loss else 0

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

        if pnl <= -(max_loss * 0.75):
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


def refresh_position_prices():
    """
    Fetch current market prices for all open positions via yfinance
    and update positions.json with live P&L data.
    Handles single-leg options and two-leg spreads.
    """
    positions = _load_positions()
    if not positions:
        return

    try:
        import yfinance as yf
    except ImportError:
        return

    updated = False
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

                pos["unrealized_pnl"]   = unrealized_pnl
                pos["pnl_pct"]          = pnl_pct
                pos["price_updated_at"] = datetime.now().strftime("%H:%M:%S")
                updated = True

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

                pos["current_price"]    = round(current_price, 2)
                pos["entry_price"]      = round(entry_price, 2)
                pos["unrealized_pnl"]   = unrealized_pnl
                pos["pnl_pct"]          = pnl_pct
                pos["price_updated_at"] = datetime.now().strftime("%H:%M:%S")
                updated = True

        except Exception as e:
            print(f"  Price refresh error {pos.get('symbol', '?')}: {e}")

    if updated:
        _save_positions(positions)


def get_open_positions() -> list:
    """Return current open positions."""
    return _load_positions()


def get_daily_stats() -> dict:
    """Return today's trading stats."""
    return _load_daily_stats()
