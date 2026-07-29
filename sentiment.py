"""
sentiment.py — News, Events & Sentiment Engine

Aggregates news and event data, scores sentiment using
Claude AI, and flags high-impact upcoming events.

Sources:
  - yfinance news (free, no API key)
  - Earnings calendar (yfinance)
  - Economic calendar (hardcoded major events + date detection)
  - Finviz news (scraped)

Output: Sentiment Score (0-100) + event flags for Claude context.
"""

import os
import time
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import anthropic
    _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    CLAUDE_AVAILABLE = True
except Exception:
    CLAUDE_AVAILABLE = False

SENTIMENT_CACHE_FILE = Path("logs/sentiment_cache.json")
CACHE_TTL = 1800  # 30 minutes — news doesn't change that fast


# ── News Fetching ──────────────────────────────────────────────────────────────

def _fetch_yfinance_news(symbol: str, max_items: int = 10) -> list:
    """Fetch recent news via yfinance (free, no key needed)."""
    if not YF_AVAILABLE:
        return []
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        items = []
        for n in news[:max_items]:
            items.append({
                "title":     n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link":      n.get("link", ""),
                "timestamp": n.get("providerPublishTime", 0),
                "source":    "yfinance",
            })
        return items
    except Exception as e:
        print(f"⚠️  News fetch error {symbol}: {e}")
        return []


def _fetch_market_news(max_items: int = 15) -> list:
    """Fetch broad market news from SPY, QQQ, VIX."""
    all_news = []
    seen_titles = set()
    for sym in ["SPY", "QQQ", "^VIX"]:
        for item in _fetch_yfinance_news(sym, max_items=5):
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                all_news.append(item)
    return sorted(all_news, key=lambda x: x["timestamp"], reverse=True)[:max_items]


# ── Earnings Calendar ─────────────────────────────────────────────────────────

def _get_earnings_info(symbol: str) -> dict:
    """
    Check if symbol has earnings in the next 5 days or had earnings in last 2 days.
    """
    if not YF_AVAILABLE:
        return {"has_earnings_soon": False, "days_to_earnings": None}
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or cal.empty:
            return {"has_earnings_soon": False, "days_to_earnings": None}

        # yfinance calendar returns a DataFrame with columns as dates
        if hasattr(cal, "columns"):
            earnings_dates = list(cal.columns)
        else:
            return {"has_earnings_soon": False, "days_to_earnings": None}

        today = date.today()
        for ed in earnings_dates:
            try:
                ed_date = pd.Timestamp(ed).date() if hasattr(ed, "date") else date.fromisoformat(str(ed)[:10])
                delta = (ed_date - today).days
                if -2 <= delta <= 5:
                    return {
                        "has_earnings_soon": True,
                        "days_to_earnings": delta,
                        "earnings_date": str(ed_date),
                        "earnings_timing": "pre_earnings" if delta > 0 else "post_earnings",
                    }
            except Exception:
                continue

        return {"has_earnings_soon": False, "days_to_earnings": None}
    except Exception:
        return {"has_earnings_soon": False, "days_to_earnings": None}


# ── Economic Calendar ─────────────────────────────────────────────────────────

# High-impact events we track by date pattern
# These are approximate — update monthly for precision
_HIGH_IMPACT_EVENTS = [
    # Format: (month, day, event_name)
    # Add known dates here; the detector also checks FOMC meeting patterns
]

def _get_economic_events_today() -> list:
    """
    Check for known high-impact economic events today/tomorrow.
    Uses a simple heuristic: FOMC meetings are typically 2nd/3rd week
    of Jan, Mar, May, Jun, Jul, Sep, Nov, Dec.
    """
    today = date.today()
    events = []

    # FOMC detection heuristic (8 meetings/year, roughly every 6-7 weeks)
    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
    if today.month in fomc_months and 25 <= today.day <= 31:
        # End of month in FOMC months often means upcoming meeting
        events.append({
            "event": "FOMC Meeting (possible)",
            "impact": "HIGH",
            "note": "Fed may announce rate decision",
        })

    # CPI releases: typically 2nd week of month
    if 10 <= today.day <= 14:
        events.append({
            "event": "CPI Release Window",
            "impact": "HIGH",
            "note": "CPI data typically released 2nd week of month",
        })

    # Jobs report: first Friday of month
    if today.weekday() == 4 and today.day <= 7:  # First Friday
        events.append({
            "event": "NFP / Jobs Report",
            "impact": "EXTREME",
            "note": "Non-farm payrolls — avoid directional options",
        })

    return events


# ── AI Sentiment Scoring ──────────────────────────────────────────────────────

def _score_sentiment_with_claude(headlines: list, symbol: str) -> dict:
    """
    Use Claude to score market sentiment from news headlines.
    Returns score 0-100 and brief analysis.
    """
    if not CLAUDE_AVAILABLE or not headlines:
        return {"score": 50, "sentiment": "NEUTRAL", "summary": "No news data"}

    # Check cache
    cache = _load_cache()
    cache_key = f"sentiment_{symbol}"
    if cache_key in cache:
        if time.time() - cache[cache_key]["ts"] < CACHE_TTL:
            return cache[cache_key]["data"]

    headlines_text = "\n".join(
        f"- {h['title']} ({h.get('publisher', 'unknown')})"
        for h in headlines[:8]
    )

    prompt = f"""Analyze these recent news headlines for {symbol} and rate the market sentiment.

Headlines:
{headlines_text}

Rate the current sentiment for options trading:
- 0-30 = Strongly bearish (sell rallies, buy puts)
- 31-45 = Mildly bearish
- 46-55 = Neutral
- 56-70 = Mildly bullish
- 71-100 = Strongly bullish (buy dips, buy calls)

Respond with ONLY valid JSON:
{{"score": 0-100, "sentiment": "BULLISH/BEARISH/NEUTRAL", "key_theme": "one sentence summary", "risk_flags": ["any specific risks"]}}"""

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        # Extract first JSON object only (guard against extra text after closing brace)
        brace = raw.find("{")
        if brace != -1:
            depth, end = 0, -1
            for i, ch in enumerate(raw[brace:], brace):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            raw = raw[brace:end] if end != -1 else raw[brace:]
        result = json.loads(raw)

        # Cache it
        cache[cache_key] = {"data": result, "ts": time.time()}
        _save_cache(cache)
        return result
    except Exception as e:
        pass  # Sentiment API returned empty — silently use neutral fallback
        return {"score": 50, "sentiment": "NEUTRAL", "key_theme": "Unable to score", "risk_flags": []}


def _load_cache() -> dict:
    if SENTIMENT_CACHE_FILE.exists():
        try:
            return json.loads(SENTIMENT_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    SENTIMENT_CACHE_FILE.parent.mkdir(exist_ok=True)
    SENTIMENT_CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── Full Sentiment Analysis ────────────────────────────────────────────────────

def analyze_sentiment(symbol: str) -> dict:
    """
    Full sentiment analysis for a symbol.
    Combines stock-specific news + market news + earnings + economic events.
    """
    # Fetch news
    stock_news  = _fetch_yfinance_news(symbol, max_items=8)
    market_news = _fetch_market_news(max_items=5)
    all_headlines = stock_news + market_news

    # Earnings info
    earnings = _get_earnings_info(symbol)

    # Economic events
    eco_events = _get_economic_events_today()

    # Score with Claude (or fallback)
    sentiment_score = _score_sentiment_with_claude(all_headlines, symbol)

    # Boost/penalize based on earnings proximity
    base_score = sentiment_score.get("score") or 50
    earnings_adj = 0
    if earnings.get("has_earnings_soon"):
        dte = earnings.get("days_to_earnings", 99)
        if 1 <= dte <= 3:
            # Earnings imminent: IV usually elevated, risky for directional
            earnings_adj = -10
        elif dte == 0:
            earnings_adj = -15  # Day of earnings — very risky

    # Economic event penalty
    eco_adj = 0
    for ev in eco_events:
        if ev.get("impact") == "EXTREME":
            eco_adj -= 15
        elif ev.get("impact") == "HIGH":
            eco_adj -= 5

    adjusted_score = max(0, min(100, base_score + earnings_adj + eco_adj))

    return {
        "symbol":           symbol,
        "score":            adjusted_score,
        "base_score":       base_score,
        "sentiment":        sentiment_score.get("sentiment", "NEUTRAL"),
        "key_theme":        sentiment_score.get("key_theme", ""),
        "risk_flags":       sentiment_score.get("risk_flags", []),
        "news_count":       len(stock_news),
        "recent_headlines": [h["title"] for h in stock_news[:3]],
        "earnings":         earnings,
        "eco_events":       eco_events,
        "earnings_adj":     earnings_adj,
        "eco_adj":          eco_adj,
    }


# ── Sentiment Score (0-100) ────────────────────────────────────────────────────

def compute_sentiment_score(symbol: str, direction: str) -> dict:
    """
    Return a 0-100 sentiment score aligned with trade direction.
    Bullish trade gets higher score when sentiment is bullish, and vice versa.
    """
    data = analyze_sentiment(symbol)
    raw_score = data["score"]
    is_bull = direction == "bullish"

    # For bearish trades, invert the sentiment score
    if not is_bull:
        directional_score = 100 - raw_score
    else:
        directional_score = raw_score

    signals = []

    sentiment = data["sentiment"]
    if (is_bull and sentiment == "BULLISH") or (not is_bull and sentiment == "BEARISH"):
        signals.append(f"✓ Sentiment aligned ({sentiment})")
    elif (is_bull and sentiment == "BEARISH") or (not is_bull and sentiment == "BULLISH"):
        signals.append(f"✗ Sentiment opposes ({sentiment})")
    else:
        signals.append(f"~ Sentiment neutral")

    if data["earnings"]["has_earnings_soon"]:
        dte = data["earnings"].get("days_to_earnings", 0)
        signals.append(f"⚠️  Earnings in {dte} day(s) — elevated IV risk")

    for ev in data["eco_events"]:
        signals.append(f"⚠️  {ev['event']} — {ev['note']}")

    for flag in data["risk_flags"]:
        signals.append(f"⚠️  {flag}")

    return {
        "score":          round(directional_score, 1),
        "direction":      direction,
        "sentiment":      sentiment,
        "key_theme":      data["key_theme"],
        "signals":        signals,
        "earnings":       data["earnings"],
        "eco_events":     data["eco_events"],
        "headlines":      data["recent_headlines"],
        "raw_data":       data,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Analyzing SPY sentiment...")
    result = compute_sentiment_score("SPY", "bullish")
    print(f"\nSentiment Score (bullish): {result['score']}/100")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Theme: {result['key_theme']}")
    print(f"Headlines: {result['headlines']}")
    print(f"Signals:")
    for s in result["signals"]:
        print(f"  {s}")
