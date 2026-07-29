"""
alerts.py — Telegram Notifications

Sends formatted trade alerts to your phone via Telegram.
All bot activity (trades, errors, daily summaries) flows through here.
"""

import os
import requests
from datetime import datetime


# ── Core send function ─────────────────────────────────────────────────────────
def send(message: str, silent: bool = False) -> bool:
    """
    Send a message to your Telegram chat.
    silent=True sends without a notification sound (for low-priority updates).
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[ALERT — Telegram not configured]: {message}")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_notification": silent
            },
            timeout=10
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"⚠️  Telegram alert failed: {e}")
        return False


# ── Formatted alert types ──────────────────────────────────────────────────────
def trade_opened(trade: dict):
    """Alert when a new position is opened."""
    direction = "📈" if trade["direction"] == "bullish" else "📉"
    msg = (
        f"{direction} <b>TRADE OPENED</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Symbol:</b> {trade['symbol']}\n"
        f"🎯 <b>Strategy:</b> {trade['strategy']}\n"
        f"📅 <b>Expiration:</b> {trade['expiration']}\n"
        f"💰 <b>Max Profit:</b> ${trade['max_profit']:.0f}\n"
        f"🛑 <b>Max Loss:</b> ${trade['max_loss']:.0f}\n"
        f"🤖 <b>Claude confidence:</b> {trade['confidence']}%\n"
        f"💭 <b>Reasoning:</b> {trade['reasoning']}\n"
        f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)


def trade_closed(trade: dict, pnl: float):
    """Alert when a position is closed."""
    emoji = "✅" if pnl >= 0 else "❌"
    msg = (
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Symbol:</b> {trade['symbol']}\n"
        f"🎯 <b>Strategy:</b> {trade['strategy']}\n"
        f"💵 <b>P&L:</b> ${pnl:+.2f}\n"
        f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)


def daily_summary(stats: dict):
    """End-of-day performance summary."""
    net = stats['total_pnl']
    emoji = "🟢" if net >= 0 else "🔴"
    msg = (
        f"{emoji} <b>DAILY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Trades today:</b> {stats['trade_count']}\n"
        f"✅ <b>Winners:</b> {stats['winners']}\n"
        f"❌ <b>Losers:</b> {stats['losers']}\n"
        f"💵 <b>Net P&L:</b> ${net:+.2f}\n"
        f"📅 <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}"
    )
    send(msg)


def risk_warning(message: str):
    """High-priority alert for risk limit breaches."""
    msg = (
        f"⚠️ <b>RISK WARNING</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{message}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)  # always audible


def bot_started():
    send(
        f"🤖 <b>Trading bot started</b>\n"
        f"📅 {datetime.now().strftime('%B %d, %Y %H:%M')}\n"
        f"✅ Monitoring markets..."
    )


def bot_stopped(reason: str = "Manual stop"):
    send(
        f"🛑 <b>Trading bot stopped</b>\n"
        f"Reason: {reason}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )


def error(context: str, err: Exception):
    """Alert on unexpected errors so you know the bot is in trouble."""
    msg = (
        f"🚨 <b>BOT ERROR</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 <b>Where:</b> {context}\n"
        f"💬 <b>Error:</b> {str(err)[:200]}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)


# ── Phase 3: Protection / Halt / Regime / Weekly alerts ──────────────────────

def protection_level_change(new_level: str, daily_pnl: float, size_mult: float):
    """Alert when profit protection level changes."""
    emoji = {"NONE": "✅", "REDUCE_25": "🟡", "REDUCE_HALF": "🟠",
             "RESTRICT": "🔴", "HALT": "🚫"}.get(new_level, "⚠️")
    msg = (
        f"{emoji} <b>PROTECTION LEVEL: {new_level}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Daily P&L:</b> ${daily_pnl:+.2f}\n"
        f"📉 <b>Size multiplier:</b> {size_mult:.0%}\n"
        f"🔒 New entries: {'HALTED' if new_level == 'HALT' else 'restricted'}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)


def halt_triggered(reason: str, daily_pnl: float = 0, weekly_pnl: float = 0):
    """Alert when trading is halted (daily/weekly limit)."""
    msg = (
        f"🚫 <b>TRADING HALTED</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 <b>Reason:</b> {reason}\n"
        f"💵 <b>Daily P&L:</b> ${daily_pnl:+.2f}\n"
        f"📅 <b>Weekly P&L:</b> ${weekly_pnl:+.2f}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg)


def weekly_summary(stats: dict):
    """Weekly performance summary (send every Friday EOD)."""
    net = stats.get("weekly_pnl", 0)
    emoji = "🟢" if net >= 0 else "🔴"
    trades = stats.get("trade_count", 0)
    winners = stats.get("winners", 0)
    win_rate = round(winners / max(1, trades) * 100, 1)
    msg = (
        f"{emoji} <b>WEEKLY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Net P&L:</b> ${net:+.2f}\n"
        f"📊 <b>Trades:</b> {trades} ({winners}W / {trades - winners}L)\n"
        f"🎯 <b>Win rate:</b> {win_rate}%\n"
        f"📅 <b>Week ending:</b> {datetime.now().strftime('%B %d, %Y')}"
    )
    send(msg)


def regime_change(old_regime: str, new_regime: str, vix: float = 0):
    """Alert when market regime shifts significantly."""
    risk_regimes = {"VOLATILE", "BEAR", "STRONG_BEAR", "CRASH", "PANIC", "FOMC"}
    is_risk_on = new_regime in risk_regimes
    emoji = "⚠️" if is_risk_on else "📊"
    msg = (
        f"{emoji} <b>REGIME CHANGE</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📤 <b>From:</b> {old_regime}\n"
        f"📥 <b>To:</b> {new_regime}\n"
        f"📊 <b>VIX:</b> {vix:.1f}\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}"
    )
    send(msg, silent=not is_risk_on)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("Sending test alert...")
    result = send("🤖 <b>Test alert!</b>\nYour trading bot alerts are working correctly.")
    print("✅ Sent!" if result else "❌ Failed — check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
