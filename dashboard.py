"""
dashboard.py — Web Dashboard for Claude Options Trading Bot

Accessible from any device via Cloudflare Tunnel.
Reads from the same log files the bot writes.
Run with: python dashboard.py
"""

import os
import json
import glob
import queue
import threading
from datetime import date, datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response, stream_with_context, redirect
from dotenv import load_dotenv
import auth
import state_store

load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

LOG_DIR = state_store.LOG_DIR
POSITIONS_FILE = state_store.POSITIONS_FILE
BOT_LOG_FILE = LOG_DIR / "bot.log"

# ── SSE log streaming ──────────────────────────────────────────────────────────
log_subscribers = []

def tail_log_file():
    """Background thread that tails bot.log and pushes to SSE subscribers."""
    BOT_LOG_FILE.parent.mkdir(exist_ok=True)
    BOT_LOG_FILE.touch(exist_ok=True)
    with open(BOT_LOG_FILE, "r") as f:
        f.seek(0, 2)  # seek to end
        while True:
            line = f.readline()
            if line:
                for q in list(log_subscribers):
                    try:
                        q.put(line.strip())
                    except Exception:
                        pass
            else:
                threading.Event().wait(0.5)

threading.Thread(target=tail_log_file, daemon=True).start()


# ── Data helpers ───────────────────────────────────────────────────────────────
# Positions/daily-stats loading lives in state_store.py (single source of
# truth shared with the bot process). load_daily_stats() surfaces a
# `_corrupted` flag instead of silently showing a fake $0.00 day when the
# stats file is unreadable — see state_store.load_daily_stats().
load_positions = state_store.load_positions
load_daily_stats = state_store.load_daily_stats


def load_trade_history(limit=50):
    """Load all historical stats files, newest first."""
    pattern = str(LOG_DIR / "stats_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    history = []
    for f in files[:limit]:
        try:
            data = json.loads(Path(f).read_text())
            history.append(data)
        except Exception:
            pass
    return history


def load_recent_log(lines=100):
    if BOT_LOG_FILE.exists():
        try:
            all_lines = BOT_LOG_FILE.read_text().splitlines()
            return all_lines[-lines:]
        except Exception:
            pass
    return []


def is_bot_running():
    """Check if bot process is running by looking for a recent heartbeat."""
    heartbeat = LOG_DIR / "heartbeat.txt"
    if not heartbeat.exists():
        return False
    try:
        ts = float(heartbeat.read_text().strip())
        return (datetime.now().timestamp() - ts) < 300  # alive if heartbeat < 5 min ago
    except Exception:
        return False


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    positions = load_positions()
    stats = load_daily_stats()
    history = load_trade_history(30)

    # Build P&L chart data (last 30 days)
    chart_labels = [h.get("date", "") for h in reversed(history)]
    chart_pnl = [round(h.get("pnl", 0), 2) for h in reversed(history)]
    cumulative = []
    running = 0
    for p in chart_pnl:
        running += p
        cumulative.append(round(running, 2))

    return jsonify({
        "bot_running": is_bot_running(),
        "trading_mode": os.environ.get("TRADING_MODE", "sandbox").upper(),
        "positions": positions,
        "stats": stats,
        "history": history[:10],
        "chart": {
            "labels": chart_labels,
            "daily_pnl": chart_pnl,
            "cumulative": cumulative,
        },
        "risk": {
            "daily_loss_limit": float(os.environ.get("DAILY_LOSS_LIMIT", 200)),
            "max_position_size": float(os.environ.get("MAX_POSITION_SIZE", 150)),
            "max_open_positions": int(os.environ.get("MAX_OPEN_POSITIONS", 3)),
        }
    })


@app.route("/api/log")
def api_log():
    return jsonify({"lines": load_recent_log(100)})


@app.route("/api/log/stream")
@app.route("/stream")
def api_log_stream():
    """Server-Sent Events endpoint for live log streaming."""
    q = queue.Queue()
    log_subscribers.append(q)

    def generate():
        # Send recent history first
        for line in load_recent_log(30):
            yield f"data: {line}\n\n"
        # Then stream new lines
        try:
            while True:
                try:
                    line = q.get(timeout=30)
                    yield f"data: {line}\n\n"
                except queue.Empty:
                    yield "data: \n\n"  # keepalive
        finally:
            if q in log_subscribers:
                log_subscribers.remove(q)

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/oauth/callback")
def oauth_callback():
    code = request.args.get("code")
    if not code:
        return "<h2>❌ No code received from Tastytrade.</h2>", 400
    success = auth.exchange_code(code)
    if success:
        return """<html><body style='font-family:sans-serif;text-align:center;padding:60px;background:#f0f4f8'>
            <h1 style='color:#059669'>✅ Tastytrade Connected!</h1>
            <p style='color:#475569'>Your bot is now authorized. You can close this tab.</p>
            <script>setTimeout(()=>window.location='/',2000)</script>
        </body></html>"""
    return "<h2>❌ Authorization failed. Try again.</h2>", 400


@app.route("/api/auth-status")
def api_auth_status():
    return jsonify({
        "authenticated": auth.is_authenticated(),
        "account_number": auth.get_account_number(),
        "auth_url": auth.get_auth_url(),
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update risk settings in .env file."""
    data = request.json
    env_path = Path(".env")
    if not env_path.exists():
        return jsonify({"error": ".env file not found"}), 404

    content = env_path.read_text()
    updates = {
        "DAILY_LOSS_LIMIT": data.get("daily_loss_limit"),
        "MAX_POSITION_SIZE": data.get("max_position_size"),
        "MAX_OPEN_POSITIONS": data.get("max_open_positions"),
    }
    for key, val in updates.items():
        if val is not None:
            import re
            content = re.sub(rf"^{key}=.*$", f"{key}={val}", content, flags=re.MULTILINE)

    env_path.write_text(content)
    return jsonify({"ok": True, "message": "Settings saved. Restart bot to apply."})


PAUSE_FILE = LOG_DIR / "bot_paused.txt"
WATCHLIST_FILE = LOG_DIR / "watchlist.json"
JOURNAL_FILE = LOG_DIR / "trade_journal.json"


@app.route("/api/bot/pause", methods=["POST"])
def api_bot_pause():
    """Toggle bot pause state."""
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()
        return jsonify({"paused": False})
    else:
        PAUSE_FILE.write_text("paused")
        return jsonify({"paused": True})


@app.route("/api/bot/paused")
def api_bot_paused():
    return jsonify({"paused": PAUSE_FILE.exists()})


@app.route("/api/positions/close", methods=["POST"])
def api_close_position():
    """Manually close a position by order_id."""
    data = request.json
    order_id = data.get("order_id")
    positions = load_positions()
    pos = next((p for p in positions if p.get("order_id") == order_id), None)
    if not pos:
        return jsonify({"error": "Position not found"}), 404
    try:
        import executor
        success = executor.close_position(pos, "Manual close via dashboard")
        return jsonify({"ok": success})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist", methods=["GET", "POST"])
def api_watchlist():
    """Get or update the watchlist."""
    default = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    if request.method == "GET":
        if WATCHLIST_FILE.exists():
            try:
                return jsonify(json.loads(WATCHLIST_FILE.read_text()))
            except Exception:
                pass
        return jsonify(default)
    else:
        data = request.json or {}
        # Accept both "symbols" and "watchlist" keys
        symbols = data.get("watchlist", data.get("symbols", default))
        symbols = [s.upper().strip() for s in symbols if s.strip()]
        WATCHLIST_FILE.write_text(json.dumps(symbols))
        return jsonify({"ok": True, "symbols": symbols})


@app.route("/api/journal")
def api_journal():
    """Get trade journal entries."""
    if JOURNAL_FILE.exists():
        try:
            return jsonify(json.loads(JOURNAL_FILE.read_text()))
        except Exception:
            pass
    return jsonify([])


# ── v2 API endpoints ───────────────────────────────────────────────────────────

@app.route("/api/regime")
def api_regime():
    """Return current market regime from cache file."""
    regime_file = LOG_DIR / "regime_cache.json"
    if regime_file.exists():
        try:
            data = json.loads(regime_file.read_text())
            regime = data.get("regime", {})
            return jsonify({
                "available": True,
                "condition":    regime.get("condition", "UNKNOWN"),
                "trend":        regime.get("trend", "UNKNOWN"),
                "vol":          regime.get("vol", "UNKNOWN"),
                "vol_direction": regime.get("vol_direction", "STABLE"),
                "vix":          regime.get("vix", 20.0),
                "vix_pct":      regime.get("vix_pct", 50.0),
                "spy_vs_ema50": regime.get("spy_vs_ema50", 0.0),
                "summary":      regime.get("summary", ""),
                "preferred_strategies": regime.get("preferred_strategies", []),
                "avoid_strategies":     regime.get("avoid_strategies", []),
                "cached_at": data.get("ts", 0),
            })
        except Exception:
            pass
    return jsonify({"available": False, "summary": "Regime detection not yet run"})


@app.route("/api/risk")
def api_risk():
    """Return full risk summary from risk_manager."""
    try:
        from risk_manager import get_risk_summary
        return jsonify({"available": True, **get_risk_summary()})
    except ImportError:
        # Fallback: build from files
        stats = load_daily_stats()
        positions = load_positions()
        daily_loss_limit = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
        wt = stats.get("winners", 0)
        tc = max(1, stats.get("trade_count", 1))
        return jsonify({
            "available": False,
            "daily_pnl": stats.get("pnl", 0),
            "daily_loss_limit": daily_loss_limit,
            "remaining_risk_budget": daily_loss_limit - abs(min(stats.get("pnl", 0), 0)),
            "open_positions": len(positions),
            "max_positions": int(os.environ.get("MAX_OPEN_POSITIONS", 3)),
            "win_rate": round(wt / tc * 100, 1),
            "trading_halted": False,
            "halt_reason": "",
            "portfolio_delta": 0,
            "portfolio_theta": 0,
            "portfolio_vega":  0,
        })
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ml")
def api_ml():
    """Return ML engine stats."""
    ml_stats_file = LOG_DIR / "ml_stats.json"
    if ml_stats_file.exists():
        try:
            data = json.loads(ml_stats_file.read_text())
            overall = data.get("strategy_stats", {}).get("overall", {})
            return jsonify({
                "available": True,
                "last_retrain":        data.get("last_retrain"),
                "trades_analyzed":     data.get("trades_analyzed", 0),
                "win_rate":            overall.get("win_rate", 0),
                "profit_factor":       overall.get("profit_factor", 0),
                "sharpe":              overall.get("sharpe", 0),
                "sortino":             overall.get("sortino", 0),
                "max_drawdown":        overall.get("max_drawdown", 0),
                "disabled_strategies": data.get("disabled_strategies", []),
                "model_status":        data.get("model", {}).get("status", "not_trained"),
                "model_accuracy":      data.get("model", {}).get("cv_accuracy", 0),
                "strategy_stats":      data.get("strategy_stats", {}),
                "weights":             data.get("weights", {}),
            })
        except Exception:
            pass
    return jsonify({"available": False, "message": "ML engine not yet trained (needs trade data)"})


@app.route("/api/flow")
def api_flow():
    """Return latest options flow analysis from cache."""
    oi_file   = LOG_DIR / "oi_snapshot.json"
    flow_log  = LOG_DIR / "flow_log.json"
    if flow_log.exists():
        try:
            return jsonify({"available": True, **json.loads(flow_log.read_text())})
        except Exception:
            pass
    return jsonify({"available": False, "message": "Flow data collected on next scan"})


@app.route("/api/scores")
def api_scores():
    """Return latest per-symbol composite scores from last scan."""
    scores_file = LOG_DIR / "last_scores.json"
    if scores_file.exists():
        try:
            return jsonify({"available": True, "scores": json.loads(scores_file.read_text())})
        except Exception:
            pass
    return jsonify({"available": False, "scores": []})


@app.route("/api/overview")
def api_overview():
    """Single endpoint returning everything the dashboard needs."""
    positions = load_positions()
    stats     = load_daily_stats()
    history   = load_trade_history(30)

    chart_labels = [h.get("date", "") for h in reversed(history)]
    chart_pnl    = [round(h.get("pnl", 0), 2) for h in reversed(history)]
    cumulative   = []
    running = 0
    for p in chart_pnl:
        running += p
        cumulative.append(round(running, 2))

    # Watchlist
    watchlist = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    if WATCHLIST_FILE.exists():
        try:
            watchlist = json.loads(WATCHLIST_FILE.read_text())
        except Exception:
            pass

    # Risk
    try:
        from risk_manager import get_risk_summary
        risk = get_risk_summary()
        risk["available"] = True
        # Normalize field names for dashboard
        risk.setdefault("net_delta", risk.pop("portfolio_delta", 0))
        risk.setdefault("net_theta", risk.pop("portfolio_theta", 0))
        risk.setdefault("net_gamma", 0)
        risk.setdefault("net_vega", risk.pop("portfolio_vega", 0))
        risk.setdefault("drawdown_pct", 0)
        risk.setdefault("consecutive_losses", 0)
        risk.setdefault("trading_halted", False)
    except Exception:
        daily_loss_limit = float(os.environ.get("DAILY_LOSS_LIMIT", 200))
        risk = {
            "available": False,
            "daily_pnl": stats.get("pnl", 0),
            "daily_loss_limit": daily_loss_limit,
            "net_delta": 0, "net_theta": 0, "net_gamma": 0, "net_vega": 0,
            "drawdown_pct": 0, "consecutive_losses": 0, "trading_halted": False,
        }

    # Regime
    regime = {"available": False}
    regime_file = LOG_DIR / "regime_cache.json"
    if regime_file.exists():
        try:
            d = json.loads(regime_file.read_text()).get("regime", {})
            regime = {
                "available":  True,
                "condition":  d.get("condition", "UNKNOWN"),
                "trend":      d.get("trend", "UNKNOWN"),
                "vol_regime": d.get("vol", d.get("vol_regime", "NORMAL_VOL")),
                "vix":        d.get("vix", 20.0),
                "spy_change": d.get("spy_vs_ema50", d.get("spy_change", 0.0)),
                "qqq_change": d.get("qqq_change", 0.0),
                "summary":    d.get("summary", ""),
                "preferred_strategies": d.get("preferred_strategies", []),
                "avoid_strategies":     d.get("avoid_strategies", []),
            }
        except Exception:
            pass

    # ML
    ml = {"available": False, "model_trained": False}
    ml_file = LOG_DIR / "ml_stats.json"
    if ml_file.exists():
        try:
            md = json.loads(ml_file.read_text())
            overall = md.get("strategy_stats", {}).get("overall", {})
            ml = {
                "available":          True,
                "model_trained":      md.get("model", {}).get("status") == "trained",
                "model_accuracy":     md.get("model", {}).get("cv_accuracy", 0),
                "overall_win_rate":   overall.get("win_rate", 0),
                "sharpe":             overall.get("sharpe", 0),
                "profit_factor":      overall.get("profit_factor", 0),
                "total_trades":       md.get("trades_analyzed", 0),
                "last_retrain":       md.get("last_retrain", "—"),
                "disabled_strategies": md.get("disabled_strategies", []),
            }
        except Exception:
            pass

    # Scores (last scan)
    scores = []
    scores_file = LOG_DIR / "last_scores.json"
    if scores_file.exists():
        try:
            scores = json.loads(scores_file.read_text())
        except Exception:
            pass

    # Intelligence
    intel = {"available": False}
    intel_file = LOG_DIR / "intelligence_cache.json"
    if intel_file.exists():
        try:
            id_ = json.loads(intel_file.read_text())
            macro = id_.get("macro", {})
            intel = {
                "available":       True,
                "macro_risk":      macro.get("macro_risk_score", 0),
                "market_sentiment": macro.get("market_sentiment", 50),
                "risk_on":         macro.get("risk_on", True),
                "yield_10yr":      macro.get("yield_10yr", 0),
                "macro_signals":   macro.get("macro_signals", [])[:3],
                "upcoming_events": macro.get("upcoming_events", []),
                "market_blocked":  macro.get("market_blocked", False),
                "blocked_symbols": [
                    sym for sym, v in id_.get("symbols", {}).items()
                    if v.get("trade_blocked")
                ],
            }
        except Exception:
            pass

    # Profit protection
    protection = {"available": False, "level": "NONE", "size_multiplier": 1.0}
    try:
        from profit_protection import get_protection_status
        prot = get_protection_status()
        protection = {
            "available":       True,
            "level":           prot.get("level", "NONE"),
            "size_multiplier": prot.get("size_multiplier", 1.0),
            "daily_pnl":       prot.get("daily_pnl", 0),
            "weekly_pnl":      prot.get("weekly_pnl", 0),
            "weekly_halted":   prot.get("weekly_halted", False),
            "messages":        prot.get("messages", []),
        }
    except Exception:
        pass

    # Stats shape for dashboard
    wt = stats.get("winners", 0)
    tc = max(1, stats.get("trade_count", 1))
    overview_stats = {
        "today_pnl":    stats.get("pnl", 0),
        "all_time_pnl": sum(h.get("pnl", 0) for h in history),
        "win_rate":     wt / tc if tc > 0 else 0,
        "trade_count":  stats.get("trade_count", 0),
        "recent_trades": [],
    }

    return jsonify({
        "bot_running":  is_bot_running(),
        "mode":         os.environ.get("TRADING_MODE", "sandbox").upper(),
        "last_scan":    None,
        "positions":    positions,
        "stats":        overview_stats,
        "chart":        {"labels": chart_labels, "values": cumulative},
        "risk":         risk,
        "regime":       regime,
        "ml":           ml,
        "scores":       scores,
        "watchlist":    watchlist,
        "intel":        intel,
        "protection":   protection,
    })


# ── Phase 2 API endpoints ─────────────────────────────────────────────────────

@app.route("/api/intel")
def api_intel():
    """Return latest intelligence engine data."""
    intel_file = LOG_DIR / "intelligence_cache.json"
    if intel_file.exists():
        try:
            data = json.loads(intel_file.read_text())
            macro = data.get("macro", {})
            symbols = data.get("symbols", {})
            return jsonify({
                "available":       True,
                "macro_risk":      macro.get("macro_risk_score", 0),
                "market_sentiment": macro.get("market_sentiment", 50),
                "market_urgency":  macro.get("market_urgency", 0),
                "risk_on":         macro.get("risk_on", True),
                "yield_10yr":      macro.get("yield_10yr", 0),
                "gold_chg":        macro.get("gold_chg", 0),
                "dxy_chg":         macro.get("dxy_chg", 0),
                "oil_chg":         macro.get("oil_chg", 0),
                "btc_chg":         macro.get("btc_chg", 0),
                "macro_signals":   macro.get("macro_signals", []),
                "upcoming_events": macro.get("upcoming_events", []),
                "next_event_hours": macro.get("next_event_hours", 999),
                "market_headlines": macro.get("market_headlines", []),
                "market_blocked":  macro.get("market_blocked", False),
                "symbol_intel":    {
                    sym: {
                        "news_sentiment":   v.get("news_sentiment_score", 50),
                        "event_risk":       v.get("event_risk_score", 0),
                        "social_sentiment": v.get("social_sentiment_score", 50),
                        "analyst_score":    v.get("analyst_score", 50),
                        "insider_score":    v.get("insider_score", 50),
                        "trade_blocked":    v.get("trade_blocked", False),
                        "block_reason":     v.get("block_reason", ""),
                        "earnings_soon":    v.get("earnings_soon", False),
                        "earnings_days":    v.get("earnings_days_away", 999),
                        "headlines":        v.get("headlines", [])[:2],
                    }
                    for sym, v in symbols.items()
                },
                "cached_at": data.get("ts", 0),
            })
        except Exception as e:
            return jsonify({"available": False, "error": str(e)})
    return jsonify({"available": False, "message": "Intelligence data not yet fetched"})


@app.route("/api/protection")
def api_protection():
    """Return profit protection status."""
    try:
        from profit_protection import get_protection_status, get_weekly_summary
        prot  = get_protection_status()
        weekly = get_weekly_summary()
        return jsonify({
            "available":       True,
            "level":           prot.get("level", "NONE"),
            "size_multiplier": prot.get("size_multiplier", 1.0),
            "daily_pnl":       prot.get("daily_pnl", 0),
            "weekly_pnl":      prot.get("weekly_pnl", 0),
            "weekly_halted":   prot.get("weekly_halted", False),
            "weekly_halt_reason": prot.get("weekly_halt_reason", ""),
            "messages":        prot.get("messages", []),
            "strategies_allowed": prot.get("strategies_allowed", []),
            "new_entries_halted": prot.get("new_entries_halted", False),
            "weekly_loss_limit": weekly.get("weekly_loss_limit", 600),
            "weekly_trade_count": weekly.get("trade_count", 0),
        })
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/monthly")
def api_monthly():
    """Return monthly performance breakdown."""
    try:
        from risk_manager import get_monthly_summary
        return jsonify({"available": True, **get_monthly_summary()})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/equity-curve")
def api_equity_curve():
    """Return equity curve from trade journal. Optional ?since=YYYY-MM-DD to exclude old data."""
    if JOURNAL_FILE.exists():
        try:
            from datetime import date as _date, timedelta
            journal = json.loads(JOURNAL_FILE.read_text())
            # Optional since= filter (defaults to 30 days ago to exclude stale historical data)
            since_param = request.args.get("since", "")
            if since_param:
                since_cutoff = since_param[:10]
            else:
                since_cutoff = (_date.today() - timedelta(days=30)).isoformat()
            # Build daily P&L by date
            points = {}
            for t in journal:
                closed = t.get("closed_at", t.get("opened_at", ""))[:10]
                if closed and closed >= since_cutoff:
                    points[closed] = points.get(closed, 0) + t.get("pnl", 0)
            # Sort and cumulate
            sorted_dates = sorted(points.keys())
            cum = 0
            labels, values, daily = [], [], []
            for d in sorted_dates:
                cum += points[d]
                labels.append(d)
                values.append(round(cum, 2))
                daily.append(round(points[d], 2))
            return jsonify({
                "available": True,
                "labels": labels, "cumulative": values, "daily": daily,
                "total_pnl": round(cum, 2),
                "total_trades": len(journal),
                "since": since_cutoff,
            })
        except Exception as e:
            return jsonify({"available": False, "error": str(e)})
    return jsonify({"available": False, "labels": [], "cumulative": [], "daily": []})


@app.route("/api/strategy-performance")
def api_strategy_performance():
    """Return per-strategy performance breakdown."""
    try:
        from ml_engine import compute_all_strategy_stats, _load_journal
        trades = _load_journal()
        if not trades:
            return jsonify({"available": False, "message": "No trades yet"})
        stats = compute_all_strategy_stats(trades)
        return jsonify({"available": True, "strategies": stats})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/data-integrity")
def api_data_integrity():
    """
    Report the real status of every data source feeding trade decisions.

    Every score in this system has had exactly one visual state: a number.
    A dead API key, a rate-limited LLM, or a Black-Scholes-modeled Greek
    all rendered as a perfectly plausible value with nothing to
    distinguish them from a real measurement. This endpoint adds the
    second dimension — is that number live, degraded, or modeled.
    """
    sources = []

    # NOTE: this endpoint must stay non-blocking. It polls every 30s from
    # the browser, and an earlier version made live broker/quote API calls
    # inline — which hung the request for 30s+ and made the health panel
    # itself a source of load and instability. Everything below reads
    # local state (token file, cooldown ledger, config) only. Never add a
    # network call here; if live data is needed, have the bot write it to
    # a file on its own cycle and read that.

    # Broker session — read the shared token file directly. The dashboard
    # is a separate process from the bot and never authenticates itself,
    # so checking this process's in-memory auth state would always report
    # "down" even with a perfectly healthy bot session.
    try:
        import auth
        tok = auth.TOKEN_FILE
        if tok.exists():
            data = json.loads(tok.read_text())
            expires = datetime.fromisoformat(data["expires_at"])
            mins = (expires - datetime.utcnow()).total_seconds() / 60
            if mins > 0:
                sources.append({"name": "Tastytrade — order execution",
                                "detail": f"account {data.get('account_number','?')} · token valid {mins:.0f}m",
                                "status": "live"})
            elif data.get("refresh_token"):
                sources.append({"name": "Tastytrade — order execution",
                                "detail": "access token expired — refresh token present",
                                "status": "degraded"})
            else:
                sources.append({"name": "Tastytrade — order execution",
                                "detail": "token expired, no refresh token", "status": "down"})
        else:
            sources.append({"name": "Tastytrade — order execution",
                            "detail": "no saved token", "status": "down"})
    except Exception as e:
        sources.append({"name": "Tastytrade — order execution",
                        "detail": str(e)[:80], "status": "down"})

    # Underlying quotes — report the configured source, not a live probe.
    sources.append({
        "name": "Underlying quotes",
        "detail": "Tastytrade NBBO (yfinance fallback if unavailable)",
        "status": "live",
    })

    # Option chain + Greeks. Honest label: the chain is real market data
    # but the Greeks are computed, not broker-supplied.
    sources.append({
        "name": "Option chain & Greeks",
        "detail": "yfinance chain + Black-Scholes Greeks (not broker-supplied)",
        "status": "modeled",
    })

    # LLM provider — decisions and sentiment share this quota
    try:
        import claude_brain, sentiment
        provider = claude_brain._llm_provider
        if provider == "scoring":
            sources.append({"name": "LLM decision layer",
                            "detail": "no provider configured — pure scoring fallback",
                            "status": "down"})
        elif sentiment._in_llm_cooldown():
            sources.append({"name": "LLM decision layer",
                            "detail": f"{provider} — rate limited, sentiment paused",
                            "status": "degraded"})
        else:
            sources.append({"name": "LLM decision layer",
                            "detail": provider, "status": "live"})
    except Exception as e:
        sources.append({"name": "LLM decision layer", "detail": str(e)[:80], "status": "down"})

    # News / macro intelligence
    try:
        import intelligence
        has_key = bool(getattr(intelligence, "FINNHUB_KEY", ""))
        sources.append({
            "name": "News & macro intel",
            "detail": "Finnhub" if has_key else "no API key — neutral defaults",
            "status": "live" if has_key else "down",
        })
    except Exception as e:
        sources.append({"name": "News & macro intel", "detail": str(e)[:80], "status": "down"})

    # Recently-degraded subsystems, from the alert cooldown ledger
    recent = []
    try:
        cd_file = LOG_DIR / "alert_cooldowns.json"
        if cd_file.exists():
            for name, ts in json.loads(cd_file.read_text()).items():
                age_min = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 60
                if age_min < 180:
                    recent.append({"source": name, "minutes_ago": round(age_min)})
    except Exception:
        pass

    # Scan-loop liveness: heartbeat alone only proves the process exists.
    scan_age_min = None
    try:
        marker = LOG_DIR / "last_successful_scan.txt"
        if marker.exists():
            scan_age_min = round((datetime.now().timestamp() - float(marker.read_text())) / 60, 1)
    except Exception:
        pass

    return jsonify({
        "available": True,
        "sources": sources,
        "recent_degradations": sorted(recent, key=lambda r: r["minutes_ago"]),
        "last_successful_scan_min_ago": scan_age_min,
    })


@app.route("/api/exposure")
def api_exposure():
    """
    Correlated exposure, grouped — not a flat per-symbol list.

    Three open positions in three different symbols reads as diversified
    on a naive count, but SPY + QQQ + IWM is one concentrated bet on the
    broad market. Groups by the same sector buckets risk_manager uses to
    gate trades, and reports beta-weighted delta alongside raw delta.
    """
    try:
        import risk_manager
        positions = load_positions()
        greeks = risk_manager.get_portfolio_greeks()

        groups = {}
        for p in positions:
            sym = p.get("symbol", "?")
            sector = risk_manager._SECTOR_MAP.get(sym, "OTHER")
            g = groups.setdefault(sector, {"sector": sector, "symbols": [],
                                            "count": 0, "max_loss": 0.0})
            g["symbols"].append(sym)
            g["count"] += 1
            g["max_loss"] += float(p.get("max_loss", 0) or 0)

        # 2 positions per sector is risk_manager's concentration limit
        for g in groups.values():
            g["limit"] = 2
            g["used_pct"] = round(min(g["count"] / 2 * 100, 100), 1)
            g["betas"] = {s: risk_manager.get_beta(s) for s in g["symbols"]}

        max_delta = risk_manager.MAX_PORTFOLIO_DELTA
        bwd = greeks.get("beta_weighted_delta", 0.0)
        return jsonify({
            "available": True,
            "groups": sorted(groups.values(), key=lambda g: -g["count"]),
            "net_delta": greeks.get("net_delta", 0.0),
            "beta_weighted_delta": bwd,
            "max_portfolio_delta": max_delta,
            "beta_weighted_used_pct": round(min(abs(bwd) / max_delta * 100, 100), 1) if max_delta else 0,
            "open_positions": len(positions),
        })
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


# ── Action endpoints ───────────────────────────────────────────────────────────

@app.route("/api/force-scan", methods=["POST"])
def api_force_scan():
    """Write a flag file that tells the bot to scan immediately."""
    try:
        (LOG_DIR / "force_scan.flag").write_text("1")
        return jsonify({"ok": True, "message": "Scan triggered — check the log in a moment."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/close-all", methods=["POST"])
def api_close_all():
    """Write a flag file telling the bot to close all positions."""
    try:
        (LOG_DIR / "close_all.flag").write_text("1")
        return jsonify({"ok": True, "message": "Close-all signal sent to bot."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/close-position", methods=["POST"])
def api_close_position_new():
    """Manually close a position by id or symbol."""
    data = request.json or {}
    pos_id = data.get("id") or data.get("order_id")
    positions = load_positions()
    pos = next((p for p in positions if str(p.get("order_id", p.get("id", ""))) == str(pos_id) or p.get("symbol") == pos_id), None)
    if not pos:
        return jsonify({"ok": False, "message": "Position not found"}), 404
    try:
        import executor
        success = executor.close_position(pos, "Manual close via dashboard")
        return jsonify({"ok": success, "message": "Position closed." if success else "Close failed — check server logs."})
    except Exception as e:
        import traceback
        print(f"❌ /api/close-position exception: {traceback.format_exc()}")
        return jsonify({"ok": False, "message": f"Error: {e}"}), 500


@app.route("/api/manual-trade", methods=["POST"])
def api_manual_trade():
    """Queue a manual trade scan for a specific symbol."""
    data = request.json or {}
    symbol   = data.get("symbol", "").upper()
    strategy = data.get("strategy", "long_option")
    direction = data.get("direction", "bullish")
    max_loss  = float(data.get("max_loss", 100))
    if not symbol:
        return jsonify({"ok": False, "message": "Symbol required"}), 400
    try:
        flag = {"symbol": symbol, "strategy": strategy, "direction": direction, "max_loss": max_loss}
        (LOG_DIR / "manual_trade.json").write_text(json.dumps(flag))
        return jsonify({"ok": True, "message": f"Manual scan queued for {symbol}. Bot will process it next cycle."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


if __name__ == "__main__":
    LOG_DIR.mkdir(exist_ok=True)
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
