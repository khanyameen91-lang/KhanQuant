"""
bot.py — Main Trading Bot Orchestrator (v2)

Full pipeline:
  1. Regime detection (every 30 min)
  2. Flow analysis (per symbol, per scan)
  3. Market data snapshot (option chains + quotes)
  4. Composite scoring (6 factors per setup)
  5. Claude AI final judgment
  6. Dynamic position sizing
  7. Risk checks
  8. Order placement
  9. Position monitoring with dynamic exits
  10. ML nightly retrain

Scans every 10 minutes during market hours (9:35–3:45 ET).
"""

import sys
import os
import time
import logging
import schedule
import concurrent.futures
from pathlib import Path
from datetime import datetime, date, time as dtime
from dotenv import load_dotenv

load_dotenv()

# ── Logging to file (safe for systemd — no broken pipe) ───────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Replace print with log.info so all output goes to file
import builtins
_orig_print = builtins.print
def _safe_print(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
        log.info(msg)
    except Exception:
        pass
builtins.print = _safe_print

# Suppress broken pipe errors on stdout/stderr
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import auth
import alerts
import market_data
import claude_brain
import executor

# Suppress yfinance 404 noise (ETFs like SPY/QQQ have no fundamentals data)
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Intelligence engine ────────────────────────────────────────────────────────
try:
    from intelligence import get_intelligence_scores, check_trade_allowed, get_pre_trade_summary
    INTEL_AVAILABLE = bool(os.environ.get("FINNHUB_API_KEY", ""))
    if INTEL_AVAILABLE:
        print("Intelligence engine loaded (Finnhub)")
    else:
        print("Intelligence engine loaded (passthrough mode — no API key)")
except ImportError as _ie:
    print(f"intelligence.py not found — skipping intel: {_ie}")
    INTEL_AVAILABLE = False
    def get_intelligence_scores(symbols): return {"symbols": {}, "macro": {}}
    def check_trade_allowed(symbol, intel): return True, "No intel engine"
    def get_pre_trade_summary(*a, **kw): return ""

# ── Profit protection + weekly limits ─────────────────────────────────────────
try:
    from profit_protection import (
        get_protection_status, update_weekly_pnl,
        is_strategy_allowed, get_adjusted_max_loss, get_stop_level,
        get_weekly_summary,
    )
    PROTECTION_AVAILABLE = True
    print("Profit protection loaded")
except ImportError as _ie:
    print(f"profit_protection.py not found: {_ie}")
    PROTECTION_AVAILABLE = False
    def get_protection_status(): return {"level":"NONE","size_multiplier":1.0,"stop_level":0.75,
                                          "strategies_allowed":["all"],"new_entries_halted":False,
                                          "daily_pnl":0,"weekly_pnl":0,"weekly_halted":False,
                                          "weekly_halt_reason":"","messages":[]}
    def update_weekly_pnl(delta): pass
    def is_strategy_allowed(stype, prot): return True
    def get_adjusted_max_loss(base, prot): return base
    def get_stop_level(prot): return 0.75
    def get_weekly_summary(): return {}

# ── Regime size multiplier ────────────────────────────────────────────────────
try:
    from regime import get_regime_size_multiplier
except ImportError:
    def get_regime_size_multiplier(regime): return 1.0

# ── v2 modules (graceful fallback if packages not yet installed) ───────────────
V2_AVAILABLE = False
try:
    from regime        import detect_regime, MarketRegime
    from flow_analyzer import analyze_flow
    from scoring       import score_trade, TradeScore
    from sizing        import calculate_position_size
    from risk_manager  import check_all_risk, update_consecutive_losses, get_risk_summary
    from ml_engine     import nightly_retrain, is_strategy_disabled, predict_win_probability
    V2_AVAILABLE = True
    print("v2 engine loaded (indicators, regime, scoring, ML)")
except ImportError as _e:
    print(f"v2 modules not available — running in v1 mode: {_e}")
    print("Install: pip install yfinance pandas pandas-ta numpy scikit-learn --break-system-packages")

    # Stubs so the rest of bot.py doesn't crash
    def detect_regime():
        class _R:
            condition = "UNKNOWN"; trend = "UNKNOWN"; vix = 20.0
            summary = "Regime detection unavailable (packages not installed)"
        return _R()
    def analyze_flow(symbol, price): return {}
    def score_trade(*a, **kw):
        class _S:
            confidence = 0; action = "SKIP"
            technical_score = flow_score = sentiment_score = 50
            volatility_score = regime_score = institutional_score = 50
            technical_signals = flow_signals = sentiment_signals = key_risks = []
            regime_summary = ""
        return _S()
    def calculate_position_size(confidence, strategy_type, vix=20, use_kelly=True):
        return {"max_loss": float(os.environ.get("MAX_POSITION_SIZE", 150)),
                "approved": True, "vix_mult": 1.0}
    def check_all_risk(setup):
        from executor import _load_positions, _load_daily_stats, MAX_OPEN_POSITIONS, DAILY_LOSS_LIMIT, MAX_POSITION_SIZE
        positions = _load_positions(); stats = _load_daily_stats()
        if len(positions) >= MAX_OPEN_POSITIONS: return False, "Max positions"
        if stats["pnl"] <= -DAILY_LOSS_LIMIT: return False, "Daily limit"
        if setup.get("max_loss", 0) > MAX_POSITION_SIZE: return False, "Size limit"
        return True, "OK"
    def get_risk_summary():
        from executor import _load_daily_stats, _load_positions, DAILY_LOSS_LIMIT, MAX_OPEN_POSITIONS
        stats = _load_daily_stats(); pos = _load_positions()
        wt = stats.get("winners", 0); tc = max(1, stats.get("trade_count", 1))
        return {"daily_pnl": stats["pnl"], "trading_halted": False, "halt_reason": "",
                "open_positions": len(pos), "max_positions": MAX_OPEN_POSITIONS,
                "win_rate": round(wt/tc*100, 1), "portfolio_delta": 0,
                "daily_loss_limit": DAILY_LOSS_LIMIT,
                "remaining_risk_budget": DAILY_LOSS_LIMIT - abs(min(stats["pnl"], 0))}
    def nightly_retrain(): return {}
    def is_strategy_disabled(s): return False
    def predict_win_probability(d): return 0.5
    class MarketRegime:
        condition = "UNKNOWN"; trend = "UNKNOWN"; vix = 20.0
        summary = "v1 mode"

# ── Market Hours ───────────────────────────────────────────────────────────────
MARKET_OPEN            = dtime(9, 35)
MARKET_CLOSE           = dtime(15, 45)
SCAN_INTERVAL_MINUTES  = 2
REGIME_REFRESH_MINUTES = 30

# Global regime cache (refreshed every 30 min)
_current_regime: MarketRegime = None
_regime_updated_at: float     = 0.0


def _now_et():
    """Return current datetime in US/Eastern time."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # fallback: UTC-4 (EDT) offset
        from datetime import timezone, timedelta
        return datetime.now(timezone(timedelta(hours=-4)))


def is_market_open() -> bool:
    now_et = _now_et()
    if now_et.weekday() >= 5:
        return False
    t = now_et.time().replace(tzinfo=None)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def is_eod() -> bool:
    t = _now_et().time().replace(tzinfo=None)
    return dtime(15, 55) <= t <= dtime(16, 5)


def is_paused() -> bool:
    from pathlib import Path
    return Path("logs/bot_paused.txt").exists()


# ── Regime Management ──────────────────────────────────────────────────────────

def get_regime() -> MarketRegime:
    """Return current regime, refreshing if stale."""
    global _current_regime, _regime_updated_at
    if _current_regime is None or (time.time() - _regime_updated_at) > REGIME_REFRESH_MINUTES * 60:
        print("Detecting market regime...")
        _current_regime = detect_regime()
        _regime_updated_at = time.time()
    return _current_regime


def refresh_regime():
    """Force regime refresh (called on schedule)."""
    global _current_regime, _regime_updated_at
    _current_regime = detect_regime()
    _regime_updated_at = time.time()


# ── Protection level alert tracker ────────────────────────────────────────────
_last_protection_level: str = "NONE"

def _alert_protection_change(new_level: str, daily_pnl: float):
    """Fire Telegram alert only when protection level actually changes."""
    global _last_protection_level
    if new_level != _last_protection_level:
        _last_protection_level = new_level
        emoji = {"NONE": "✅", "REDUCE_25": "🟡", "REDUCE_HALF": "🟠",
                 "RESTRICT": "🔴", "HALT": "🚫"}.get(new_level, "⚠️")
        try:
            alerts.risk_warning(
                f"{emoji} Protection level → {new_level}\n"
                f"Daily P&L: ${daily_pnl:+.2f}\n"
                f"New entries halted — protecting gains"
            )
        except Exception as e:
            print(f"Telegram alert error: {e}")


# ── Main Scan Cycle ────────────────────────────────────────────────────────────

def scan_and_trade():
    """
    Full 10-minute scan cycle:
    1. Check exits on all open positions
    2. Detect market regime
    3. Get market snapshots (option chains)
    4. Score each setup (6 factors)
    5. Claude final judgment
    6. Dynamic sizing
    7. Risk check + order placement
    """
    try:
        _scan_and_trade_inner()
        # Refresh regardless of whether this cycle actually did anything
        # (market closed / paused are legitimate no-ops) — the point of
        # this timestamp is purely "the scan loop itself is still alive
        # and completing," which health_check() below alerts on if it
        # goes stale. Distinct from logs/heartbeat.txt, which only proves
        # the process is running, not that scanning is actually working.
        Path("logs/last_successful_scan.txt").write_text(str(time.time()))
    except BrokenPipeError:
        pass  # systemd pipe closed — safe to ignore
    except Exception as e:
        log.error(f"Scan error: {e}")
        try:
            alerts.send(f"⚠️ Bot error in scan: {e}")
        except Exception:
            pass


def _scan_and_trade_inner():
    """Actual scan logic — called by scan_and_trade() with error wrapping."""
    if not is_market_open():
        return
    if is_paused():
        print("Bot paused — skipping scan")
        return

    now_et  = _now_et()
    now_str = now_et.strftime('%H:%M:%S ET')
    print(f"\n{'='*55}")
    print(f"SCAN: {now_str}")
    print(f"{'='*55}")

    # ── Time-of-day rules ──────────────────────────────────────────────────────
    hour_min = now_et.hour + now_et.minute / 60

    # No new entries in first 15 minutes (market open volatility)
    if hour_min < 9 + 50/60:   # before 9:50 ET
        print(f"Time gate: First 15 min of open — skipping new entries (now {now_str})")
        # Still check exits on open positions
        open_positions = executor.get_open_positions()
        if open_positions:
            executor.refresh_position_prices()
            executor.check_exits(claude_brain)
        return

    # Force-close all 0DTE positions by 15:30 ET
    if hour_min >= 15.5:       # 15:30 ET
        open_positions = executor.get_open_positions()
        for pos in open_positions:
            if pos.get("type") == "0dte_scalp":
                executor.close_position(pos, "0DTE forced close — 15:30 ET")
                print(f"  Force-closed 0DTE {pos['symbol']} at {now_str}")

        # This branch used to return here, which meant non-0DTE positions
        # got NO exit check at all between 15:30 and 15:45 — and after
        # 15:45 is_market_open() is False, so the scan returns early and
        # nothing is monitored again until 09:35 the next morning. A
        # stop-loss breach in that final 15 minutes went unhandled and
        # then sat through the overnight gap. Keep checking exits on
        # everything still open; only new ENTRIES are off the table
        # this late.
        remaining = executor.get_open_positions()
        if remaining:
            executor.refresh_position_prices()
            closed = executor.check_exits(claude_brain)
            if closed:
                print(f"  Late-session exit check closed {closed} position(s)")
        return

    # Refresh Tastytrade session
    try:
        auth.session.ensure_valid(
            os.environ["TASTY_USERNAME"],
            os.environ["TASTY_PASSWORD"]
        )
    except Exception as e:
        print(f"Session refresh error: {e}")

    # ── Step 0: Reconcile local state against the broker ───────────────────────
    # No-op in sandbox mode. In live mode this catches local state drifting
    # from reality — an unfilled order tracked as open, a manual trade the
    # bot doesn't know about, a position closed outside the bot. Advisory
    # only: it alerts, it doesn't auto-mutate state.
    try:
        recon = executor.reconcile_with_broker()
        if recon.get("in_sync") is False:
            print(f"⚠️  Broker reconciliation mismatch: {recon}")
    except Exception as e:
        print(f"Reconciliation error: {e}")

    # ── Step 1: Exit management ────────────────────────────────────────────────
    open_positions = executor.get_open_positions()
    print(f"Open positions: {len(open_positions)}/{executor.MAX_OPEN_POSITIONS}")

    # Refresh live P&L for all open positions
    if open_positions:
        executor.refresh_position_prices()

    if open_positions:
        closed = executor.check_exits(claude_brain)
        if closed:
            print(f"Closed {closed} position(s)")
            # Update consecutive loss counter
            stats = executor.get_daily_stats()
            # Simple heuristic: if today's PnL dropped after close, it was a loss
            # (The journal has exact data; this is a fast check)

    # ── Step 2: Risk check before scanning ────────────────────────────────────
    risk_summary = get_risk_summary()
    if risk_summary["trading_halted"]:
        print(f"HALT: {risk_summary['halt_reason']}")
        return

    if risk_summary["daily_pnl"] <= -executor.DAILY_LOSS_LIMIT:
        print(f"Daily loss limit hit. No new trades.")
        return

    open_positions = executor.get_open_positions()
    if len(open_positions) >= executor.MAX_OPEN_POSITIONS:
        print(f"At max positions ({executor.MAX_OPEN_POSITIONS}). Skipping scan.")
        return

    # ── Step 2b: Profit protection + weekly limits ────────────────────────────
    protection = get_protection_status()
    for msg in protection.get("messages", []):
        print(msg)

    if protection["new_entries_halted"]:
        msg = f"PROFIT PROTECTION HALT: {protection['level']} — protecting daily gains"
        print(msg)
        # Alert once per level change (check against last known level)
        _alert_protection_change(protection["level"], protection.get("daily_pnl", 0))
        return

    if protection.get("weekly_halted"):
        msg = f"WEEKLY LOSS HALT: {protection['weekly_halt_reason']}"
        print(msg)
        alerts.risk_warning(f"🔴 Weekly loss limit hit\n{protection['weekly_halt_reason']}\n"
                            f"Weekly P&L: ${protection.get('weekly_pnl', 0):+.2f}")
        return

    if protection["level"] != "NONE":
        print(f"Protection level: {protection['level']} | "
              f"Size: {protection['size_multiplier']:.0%} | "
              f"Daily P&L: ${protection['daily_pnl']:+.2f}")

    # ── Step 3: Regime detection ───────────────────────────────────────────────
    regime = get_regime()
    print(f"Regime: {regime.summary}")

    # Block all trades on FOMC day
    if hasattr(regime, 'is_fomc_day') and regime.is_fomc_day:
        print("FOMC DAY — No new positions. Capital preservation mode.")
        return

    regime_size_mult = get_regime_size_multiplier(regime)
    if regime_size_mult < 1.0:
        print(f"Regime size constraint: {regime_size_mult:.0%} "
              f"({regime.calendar_event if hasattr(regime, 'calendar_event') else 'special event'})")

    # ── Step 4: Market snapshots ───────────────────────────────────────────────
    watchlist = market_data.get_watchlist()
    print(f"Scanning {watchlist}...")

    # Keep the stream subscribed to near-the-money contracts on the
    # watchlist's nearest expiration. Bounded by MAX_SUBSCRIPTIONS inside
    # the client — an options chain has thousands of contracts and
    # subscribing to all of them is how you exhaust a small box.
    try:
        import dxlink_stream
        if dxlink_stream.is_enabled() and dxlink_stream.get_client().status()["connected"]:
            _subscribe_atm_contracts(watchlist)
    except Exception as e:
        print(f"DXLink subscribe skipped: {e}")

    try:
        snapshots = market_data.get_market_snapshot()
        print(f"Found {len(snapshots)} option setups")
    except Exception as e:
        print(f"Market scan error: {e}")
        if not isinstance(e, BrokenPipeError):
            alerts.error("market_scan", e)
        return

    if not snapshots:
        print("No valid setups this cycle")
        return

    # ── Step 5: Flow analysis (parallel for speed) ────────────────────────────
    symbols_needed = list(set(s["quote"]["symbol"] for s in snapshots))
    prices         = {s["quote"]["symbol"]: s["quote"]["price"] for s in snapshots}
    flow_cache     = {}

    print(f"Analyzing options flow for {symbols_needed}...")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(analyze_flow, sym, prices[sym]): sym
                       for sym in symbols_needed}
            for future in concurrent.futures.as_completed(futures):
                sym = futures[future]
                try:
                    flow_cache[sym] = future.result()
                except Exception as e:
                    print(f"  Flow error {sym}: {e}")
                    flow_cache[sym] = {}
    except Exception as e:
        print(f"Flow analysis error: {e}")
        flow_cache = {sym: {} for sym in symbols_needed}

    # ── Step 5b: Global intelligence check ───────────────────────────────────
    print("Fetching market intelligence...")
    try:
        intel = get_intelligence_scores(symbols_needed)
        macro = intel.get("macro", {})
        if macro:
            print(f"  Macro risk: {macro.get('macro_risk_score', 0)}/100 | "
                  f"Market sentiment: {macro.get('market_sentiment', 50)}/100")
            if macro.get("upcoming_events"):
                print(f"  Upcoming events: {', '.join(macro['upcoming_events'][:3])}")
    except Exception as e:
        print(f"Intelligence fetch error: {e}")
        intel = {"symbols": {}, "macro": {}}

    # ── Step 6: Composite scoring (v2) or direct Claude (v1 fallback) ────────
    if V2_AVAILABLE:
        print("Scoring all setups (v2 engine)...")
        scored_snapshots = []
        intel_contexts   = {}   # symbol -> pre-trade summary string for Claude

        for snap in snapshots:
            sym       = snap["quote"]["symbol"]
            setup     = snap["setup"]
            direction = setup.get("direction", "neutral")
            stype     = setup.get("type", "long_option")
            iv_rank   = snap.get("iv_rank", 50)
            flow      = flow_cache.get(sym, {})

            # No new 0DTE entries after 14:30 ET
            if stype == "0dte_scalp" and hour_min >= 14.5:
                print(f"  [TIME GATE] {sym} 0DTE — no new entries after 14:30 ET")
                continue

            if is_strategy_disabled(stype):
                print(f"  Skipping {sym} {stype} — disabled by ML")
                continue

            # ── Intelligence gate ───────────────────────────────────────────
            intel_ok, intel_reason = check_trade_allowed(sym, intel)
            if not intel_ok:
                print(f"  [INTEL BLOCK] {sym}: {intel_reason}")
                continue

            # ── Profit protection strategy gate ─────────────────────────────
            if not is_strategy_allowed(stype, protection):
                print(f"  [PROTECTION BLOCK] {sym} {stype} — not allowed at "
                      f"protection level {protection['level']}")
                continue

            try:
                ts = score_trade(
                    symbol=sym, direction=direction, strategy_type=stype,
                    iv_rank=iv_rank, regime=regime, flow=flow,
                    snapshot=snap,       # Pass snapshot for EV + liquidity scoring
                    ml_win_prob=None,    # Will be set after ML check below
                    skip_sentiment=False,
                )
            except Exception as e:
                import traceback
                print(f"  Scoring error {sym}: {e}")
                print(f"  Traceback: {traceback.format_exc().splitlines()[-3]}")
                continue

            if ts.action != "SKIP":
                scored_snapshots.append((snap, ts))
                print(f"  {sym} {setup.get('strategy_name',stype)}: {ts.confidence:.0f}/100 → {ts.action}")
                # Build intel context string for Claude
                try:
                    intel_contexts[sym] = get_pre_trade_summary(
                        symbol=sym,
                        direction=direction,
                        strategy=stype,
                        intel=intel,
                        technical_signals=list(ts.technical_signals),
                        flow_signals=list(ts.flow_signals),
                        score=ts.confidence,
                    )
                except Exception as e:
                    print(f"  Intel context error {sym}: {e}")
                    intel_contexts[sym] = ""
            else:
                print(f"  {sym} {setup.get('strategy_name',stype)}: {ts.confidence:.0f}/100 → SKIP")

        if not scored_snapshots:
            print("No setups met confidence threshold this cycle")
            return

        print(f"\nClaude reviewing {len(scored_snapshots)} qualified setups...")
        approved = claude_brain.rank_scored_setups(scored_snapshots, intel_contexts)

    else:
        # ── v1 fallback: send raw snapshots directly to Claude ────────────────
        print(f"Sending {len(snapshots)} setups to Claude (v1 mode)...")
        try:
            approved_raw = claude_brain.rank_setups(snapshots)
        except Exception as e:
            print(f"Claude analysis error: {e}")
            return
        # Wrap in v2-compatible format
        approved = [
            {**d, "scale": "FULL", "snapshot": d.get("snapshot", {}),
             "pre_score": d.get("confidence", 0)}
            for d in approved_raw
        ]

    if not approved:
        print("Claude approved no setups this cycle")
        return

    # ── Step 8: Execute best setup ────────────────────────────────────────────
    best = approved[0]
    confidence  = best.get("confidence", 0)
    scale       = best.get("scale", "FULL")
    setup       = best["snapshot"]["setup"]
    symbol      = setup["symbol"]

    print(f"\nBest setup: {symbol} {setup.get('strategy_name','')} ({confidence}% confidence)")

    # Dynamic sizing
    size_scale = {"FULL": 1.0, "HALF": 0.5, "QUARTER": 0.25}.get(scale, 1.0)
    sizing_result = calculate_position_size(
        confidence    = confidence,
        strategy_type = setup["type"],
        vix           = regime.vix,
        use_kelly     = True,
    )

    if not sizing_result.get("approved"):
        print(f"Sizing engine: {sizing_result.get('reason')}")
        return

    recommended_max_loss = round(sizing_result["max_loss"] * size_scale)

    # Apply profit protection size multiplier
    protection_mult = protection.get("size_multiplier", 1.0)
    recommended_max_loss = round(recommended_max_loss * protection_mult)

    # Apply regime size multiplier
    recommended_max_loss = round(recommended_max_loss * regime_size_mult)

    print(f"Position size: ${recommended_max_loss} "
          f"(scale={scale} | protection={protection_mult:.0%} | "
          f"regime={regime_size_mult:.0%} | VIX adj={sizing_result['vix_mult']})")

    # Override setup's max_loss with dynamically sized value
    if recommended_max_loss < setup.get("max_loss", 9999):
        setup["max_loss"] = recommended_max_loss

    # Full risk check
    sector_ok, sector_msg = True, "OK"
    try:
        from risk_manager import check_sector_concentration
        sector_ok, sector_msg = check_sector_concentration(symbol)
    except Exception:
        pass

    if not sector_ok:
        print(f"Risk: {sector_msg}")
        return

    approved_risk, risk_reason = check_all_risk(setup)
    if not approved_risk:
        print(f"Risk blocked: {risk_reason}")
        return

    # ML win probability boost/check
    ml_prob = predict_win_probability({
        "confidence": confidence,
        "technical_score": scored_snapshots[0][1].technical_score if scored_snapshots else 50,
        "flow_score":      scored_snapshots[0][1].flow_score if scored_snapshots else 50,
        "direction":       setup.get("direction", "neutral"),
        "max_loss":        setup.get("max_loss", 100),
        "dte":             best["snapshot"].get("dte", 14),
        "iv_rank":         best["snapshot"].get("iv_rank", 50),
    })
    print(f"ML win probability: {ml_prob*100:.0f}%")

    if ml_prob < 0.35 and confidence < 85:
        print(f"ML model low-confidence prediction — skipping")
        return

    # Grab TradeScore for best setup
    best_ts = scored_snapshots[0][1] if scored_snapshots else None

    # Place order — pass full metadata for journal + ML
    position = executor.place_order(setup, {
        "confidence":          confidence,
        "reasoning":           best.get("reasoning", ""),
        "key_risk":            best.get("key_risk", ""),
        # 10-factor sub-scores
        "technical_score":     best_ts.technical_score     if best_ts else 50,
        "flow_score":          best_ts.flow_score          if best_ts else 50,
        "sentiment_score":     best_ts.sentiment_score     if best_ts else 50,
        "volatility_score":    best_ts.volatility_score    if best_ts else 50,
        "regime_score":        best_ts.regime_score        if best_ts else 50,
        "institutional_score": best_ts.institutional_score if best_ts else 50,
        "ev_score":            getattr(best_ts, "ev_score", 50)        if best_ts else 50,
        "liquidity_score":     getattr(best_ts, "liquidity_score", 50) if best_ts else 50,
        "rs_score":            getattr(best_ts, "rs_score", 50)        if best_ts else 50,
        # Market context
        "regime_type":  regime.condition,
        "iv_rank":      best["snapshot"].get("iv_rank", 50),
        "dte":          best["snapshot"].get("dte", 14),
        "ml_win_prob":  ml_prob,
    })

    if position:
        print(f"Trade executed: {position['symbol']} {position['strategy']}")
    else:
        print("Order failed or blocked by risk rules")


# ── End of Day ─────────────────────────────────────────────────────────────────

def end_of_day():
    """Send daily summary and run ML retrain."""
    stats = executor.get_daily_stats()
    risk  = get_risk_summary()

    alerts.daily_summary({
        "trade_count": stats["trade_count"],
        "total_pnl":   stats["pnl"],
        "winners":     stats["winners"],
        "losers":      stats["losers"],
        "win_rate":    risk["win_rate"],
        "portfolio_delta": risk["portfolio_delta"],
    })

    print(f"\nEOD: {stats['trade_count']} trades | P&L: ${stats['pnl']:+.2f} | "
          f"Win rate: {risk['win_rate']}%")

    # Flag anything being carried overnight. Once the session closes there
    # is genuinely no way to exit — the options market is shut — so the
    # honest handling is to make the overnight exposure explicit rather
    # than implying it's still being watched. Exit checks resume at 09:35
    # ET; gap risk between now and then is unmanaged by design.
    overnight = executor.get_open_positions()
    if overnight:
        lines = [f"• {p.get('symbol')} {p.get('strategy', '')} "
                 f"(max loss ${p.get('max_loss', 0):.0f}, exp {p.get('expiration', '?')})"
                 for p in overnight]
        alerts.risk_warning(
            f"🌙 Holding {len(overnight)} position(s) overnight — unmonitored until "
            f"09:35 ET tomorrow (market closed, no exits possible):\n" + "\n".join(lines)
        )
        print(f"EOD: {len(overnight)} position(s) held overnight — alerted")

    # Nightly ML retrain
    print("Starting nightly ML retrain...")
    try:
        nightly_retrain()
    except Exception as e:
        print(f"ML retrain error: {e}")


def _subscribe_atm_contracts(watchlist, strikes_per_side: int = 6):
    """
    Subscribe the DXLink stream to near-the-money contracts for each
    watchlist symbol on its nearest expiration.

    Only ATM-ish strikes matter: those are what the strategy builders
    actually select (delta targets 0.20-0.45), and they're where accurate
    Greeks change decisions. Subscribing whole chains would blow the
    client's subscription cap on strikes the bot will never trade.
    """
    import dxlink_stream
    client = dxlink_stream.get_client()
    syms = []
    for symbol in watchlist:
        try:
            resp = auth.session.get(f"/option-chains/{symbol}/nested")
            items = resp.get("data", {}).get("items", [])
            if not items:
                continue
            exps = items[0].get("expirations", [])
            if not exps:
                continue
            quote = market_data.get_quote(symbol)
            spot = quote["price"] if quote else 0
            if not spot:
                continue
            strikes = sorted(exps[0].get("strikes", []),
                             key=lambda s: abs(float(s.get("strike-price", 0)) - spot))
            for s in strikes[:strikes_per_side]:
                for key in ("call-streamer-symbol", "put-streamer-symbol"):
                    if s.get(key):
                        syms.append(s[key])
        except Exception:
            continue
    if syms:
        added = client.subscribe(syms)
        if added:
            print(f"DXLink: subscribed {added} new contracts "
                  f"({client.status()['subscribed']}/{client.status()['max_subscriptions']})")


def write_health_status():
    """
    Publish data-source health to logs/health_status.json for the
    dashboard to read.

    The dashboard used to determine this itself by importing sentiment /
    claude_brain / market_data inside its request handler. Those pull in
    yfinance, pandas and numpy, which ballooned the dashboard process
    from ~46MB to ~400MB — and on this 956MB instance with no swap, the
    OOM killer took it out seven times. The bot already has all of these
    modules resident, so it is both cheaper and more authoritative for it
    to publish status here and let the dashboard just read a small file.
    """
    sources = []
    try:
        provider = claude_brain._llm_provider
        cooling = False
        try:
            import sentiment as _s
            cooling = _s._in_llm_cooldown()
        except Exception:
            pass
        if provider == "scoring":
            sources.append({"name": "LLM decision layer",
                            "detail": "no provider configured — pure scoring fallback",
                            "status": "down"})
        elif cooling:
            sources.append({"name": "LLM decision layer",
                            "detail": f"{provider} — rate limited, sentiment paused",
                            "status": "degraded"})
        else:
            sources.append({"name": "LLM decision layer",
                            "detail": provider, "status": "live"})
    except Exception as e:
        sources.append({"name": "LLM decision layer", "detail": str(e)[:80], "status": "down"})

    try:
        import intelligence
        has_key = bool(getattr(intelligence, "FINNHUB_KEY", ""))
        sources.append({"name": "News & macro intel",
                        "detail": "Finnhub" if has_key else "no API key — neutral defaults",
                        "status": "live" if has_key else "down"})
    except Exception as e:
        sources.append({"name": "News & macro intel", "detail": str(e)[:80], "status": "down"})

    try:
        import dxlink_stream
        if not dxlink_stream.is_enabled():
            sources.append({"name": "Option chain & Greeks",
                            "detail": "yfinance chain + Black-Scholes Greeks "
                                      "(DXLink streaming available but disabled)",
                            "status": "modeled"})
        else:
            st = dxlink_stream.get_client().status()
            if st["connected"]:
                sources.append({"name": "Option chain & Greeks",
                                "detail": f"DXLink real-time — {st['greeks_cached']} contracts "
                                          f"with broker Greeks, rest Black-Scholes",
                                "status": "live"})
            else:
                sources.append({"name": "Option chain & Greeks",
                                "detail": f"DXLink enabled but disconnected "
                                          f"({st['last_error'] or 'connecting'}) — "
                                          f"falling back to Black-Scholes",
                                "status": "degraded"})
    except Exception as e:
        sources.append({"name": "Option chain & Greeks",
                        "detail": f"Black-Scholes ({str(e)[:40]})", "status": "modeled"})
    sources.append({"name": "Underlying quotes",
                    "detail": "Tastytrade NBBO (yfinance fallback if unavailable)",
                    "status": "live"})

    try:
        import state_store
        state_store.atomic_write_json(
            Path("logs/health_status.json"),
            {"updated_at": datetime.now().isoformat(), "sources": sources},
        )
    except Exception as e:
        print(f"health status write error: {e}")


def health_check():
    """Write heartbeat every 5 min so dashboard stays Online."""
    Path("logs/heartbeat.txt").write_text(str(time.time()))
    write_health_status()

    # heartbeat.txt only proves the process is alive — a bot that's thrown
    # an exception on every scan for six hours still writes this file and
    # still shows "Online, green" on the dashboard. Cross-check against
    # last_successful_scan.txt (written by scan_and_trade() above) so a
    # scan loop that's stopped actually completing gets flagged instead of
    # silently looking identical to "market's just quiet today."
    scan_marker = Path("logs/last_successful_scan.txt")
    stale_after_sec = SCAN_INTERVAL_MINUTES * 60 * 3
    if scan_marker.exists():
        try:
            age_sec = time.time() - float(scan_marker.read_text())
            if age_sec > stale_after_sec:
                alerts.degraded("scan_loop",
                                 f"No successful scan cycle in {age_sec/60:.0f} min "
                                 f"(expected every {SCAN_INTERVAL_MINUTES} min) — "
                                 f"process is alive but scanning may be stuck/erroring",
                                 cooldown_minutes=30)
        except Exception:
            pass

    open_pos   = executor.get_open_positions()
    stats      = executor.get_daily_stats()
    risk       = get_risk_summary()
    protection = get_protection_status()
    weekly     = get_weekly_summary()

    prot_str = (f" | PROT:{protection['level']}" if protection["level"] != "NONE" else "")
    weekly_str = f" | Wk P&L:${weekly.get('pnl', 0):+.0f}" if weekly else ""

    print(f"Health: {_now_et().strftime('%H:%M')} ET | "
          f"Pos: {len(open_pos)} | "
          f"P&L: ${stats['pnl']:+.2f} | "
          f"Regime: {_current_regime.condition if _current_regime else 'Unknown'}"
          f"{prot_str}{weekly_str}")


# ── Scheduler Setup ────────────────────────────────────────────────────────────

_daily_job_last_run = {}


def _run_daily_at_et(job_name: str, et_hour: int, et_minute: int, fn):
    """
    Run `fn` once per calendar day, at or after the given ET wall-clock
    time.

    schedule.every().day.at("16:00") fires on the SERVER's local clock,
    and this bot runs on a UTC host — so "16:00" was actually firing at
    12:00 ET, i.e. the "end of day" summary and ML retrain ran in the
    middle of every trading session, reporting a half-finished day.
    Rather than hardcoding a UTC offset (which breaks twice a year at DST
    boundaries), this is called from a frequent tick and self-guards on
    the current ET time plus a once-per-day marker.
    """
    now_et = _now_et()
    today = now_et.date()
    if _daily_job_last_run.get(job_name) == today:
        return
    if (now_et.hour, now_et.minute) < (et_hour, et_minute):
        return
    _daily_job_last_run[job_name] = today
    print(f"Running daily job '{job_name}' at {now_et.strftime('%H:%M')} ET")
    try:
        fn()
    except Exception as e:
        print(f"Daily job '{job_name}' error: {e}")
        alerts.error(f"daily_job_{job_name}", e)


def _daily_job_tick():
    # 16:00 ET — after the 15:45 close, so the summary covers a finished day.
    _run_daily_at_et("end_of_day", 16, 0, end_of_day)
    # 22:00 ET — second retrain well after the session.
    _run_daily_at_et("nightly_retrain", 22, 0, nightly_retrain)


def setup_schedule():
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scan_and_trade)
    schedule.every(REGIME_REFRESH_MINUTES).minutes.do(refresh_regime)
    schedule.every(2).minutes.do(health_check)
    schedule.every(1).minutes.do(_daily_job_tick)


# ── Main Entry ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   CLAUDE OPTIONS TRADER v2.0")
    print("   Broker: Tastytrade | Brain: Claude AI + ML")
    print("=" * 55)

    required = ["TASTY_USERNAME", "TASTY_PASSWORD",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return

    # Authenticate
    print("Authenticating with Tastytrade...")
    while True:
        try:
            auth.initialize()
            if not auth.is_authenticated():
                print("Auth failed: no valid token. Update TASTY_REFRESH_TOKEN in .env and restart.")
                # This used to keep the process alive so the dashboard's
                # heartbeat still reported "online" — true, but silently:
                # no alert was ever sent, so a broken refresh token could
                # sit unnoticed for days while zero trades were placed.
                # Now it alerts once immediately, then again on a cooldown
                # for as long as auth stays broken.
                alerts.error("startup_auth", Exception(
                    "No valid Tastytrade token — bot process is alive but cannot "
                    "authenticate, trade, or see real positions until "
                    "TASTY_REFRESH_TOKEN is fixed in .env."
                ))
                while not auth.is_authenticated():
                    print("Retrying auth in 5 minutes...")
                    time.sleep(300)
                    auth.initialize()
                    if not auth.is_authenticated():
                        alerts.degraded("tastytrade_auth", "Still no valid token after retry", cooldown_minutes=30)
            print(f"Authenticated. Account: {auth.session.account_number}")
            break
        except Exception as e:
            print(f"Auth failed: {e}")
            alerts.degraded("tastytrade_auth", f"Auth raised an exception: {e}", cooldown_minutes=30)
            print("Retrying in 5 minutes...")
            time.sleep(300)
            # Loop back and retry — this used to recurse into main() again,
            # growing the call stack by one frame per retry (and re-sending
            # bot_started()) until a long enough outage caused a
            # RecursionError and killed the process outright.

    mode = os.environ.get("TRADING_MODE", "sandbox").upper()
    print(f"Mode: {mode}")
    print(f"Daily loss limit: ${executor.DAILY_LOSS_LIMIT:.0f}")
    print(f"Max positions: {executor.MAX_OPEN_POSITIONS}")
    print(f"Scan interval: every {SCAN_INTERVAL_MINUTES} min")
    print(f"Market hours: {MARKET_OPEN.strftime('%H:%M')} — {MARKET_CLOSE.strftime('%H:%M')} ET")

    # Real-time streaming (opt-in via DXLINK_ENABLED). Failing to connect
    # is deliberately NOT fatal — market_data falls back to Black-Scholes
    # Greeks, which is exactly how the bot ran before streaming existed.
    try:
        import dxlink_stream
        if dxlink_stream.is_enabled():
            print("DXLink streaming: connecting…")
            if dxlink_stream.get_client().start():
                print("DXLink streaming: connected (real broker Greeks active)")
            else:
                print(f"DXLink streaming: unavailable "
                      f"({dxlink_stream.get_client().last_error}) — using Black-Scholes")
        else:
            print("DXLink streaming: disabled (set DXLINK_ENABLED=true to enable)")
    except Exception as e:
        print(f"DXLink streaming: init error {e} — using Black-Scholes")

    alerts.bot_started()
    setup_schedule()
    health_check()  # immediate heartbeat

    # Detect regime immediately
    try:
        get_regime()
    except Exception as e:
        print(f"Initial regime detection error: {e}")

    # First scan if market is open
    if is_market_open():
        print("\nMarket is open — running first scan...")
        scan_and_trade()
    else:
        print("\nMarket closed — waiting...")

    print(f"\nRunning... (Ctrl+C to stop)\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        alerts.bot_stopped("Manual stop (Ctrl+C)")


if __name__ == "__main__":
    main()
