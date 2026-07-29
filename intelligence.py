"""
intelligence.py — Global Market Intelligence Engine (v2)

Multi-source real-time awareness engine. Monitors:
  - Finnhub: company news, economic calendar, earnings, social sentiment,
             analyst recommendations, insider transactions
  - yfinance: Treasury yields (10yr/30yr), Gold, Oil, Dollar Index (DXY),
              Bitcoin (risk-on/off proxy)

Scores (all 0-100):
  news_sentiment_score      - bullish vs bearish news tone (50=neutral)
  news_urgency_score        - breaking/time-sensitive news
  event_risk_score          - risk of adverse surprise from this symbol
  macro_risk_score          - market-wide event risk
  sector_risk_score         - sector-level risk
  social_sentiment_score    - Reddit/Twitter/social media tone
  analyst_score             - analyst recommendation consensus
  insider_score             - insider buying vs selling
  headline_confidence_score - how many sources agree

Trade gate: symbol blocked on high event risk + bearish sentiment.
"""

import os
import json
import time
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

import alerts

FINNHUB_KEY  = os.environ.get("FINNHUB_API_KEY", "")
BASE_URL     = "https://finnhub.io/api/v1"
LOG_DIR      = Path("logs")
CACHE_FILE   = LOG_DIR / "intelligence_cache.json"
CACHE_TTL    = 110   # seconds

# yfinance for macro data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ── Sector mapping ─────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "SPY":  "market",      "QQQ":  "technology",
    "AAPL": "technology",  "MSFT": "technology",  "NVDA": "technology",
    "AMD":  "technology",  "META": "technology",  "AMZN": "consumer",
    "TSLA": "auto",        "PLTR": "technology",
}

# ── Keyword classifiers ────────────────────────────────────────────────────────
BEARISH_WORDS = [
    "crash", "collapse", "plunge", "slump", "recession", "default", "bankruptcy",
    "fraud", "investigation", "lawsuit", "fine", "penalty", "downgrade", "miss",
    "disappoint", "cut guidance", "layoff", "recall", "breach", "hack", "tariff",
    "sanction", "war", "conflict", "attack", "crisis", "inflation", "rate hike",
    "hawkish", "sell-off", "correction", "bear market", "downside", "loss",
    "decline", "drop", "fall", "weak", "slowdown", "contraction", "deficit",
]

BULLISH_WORDS = [
    "surge", "soar", "rally", "boom", "growth", "beat", "exceed", "record high",
    "upgrade", "outperform", "partnership", "acquisition", "deal", "contract",
    "approval", "breakthrough", "dividend", "buyback", "revenue beat",
    "earnings beat", "raise guidance", "bullish", "dovish", "rate cut",
    "stimulus", "strong", "recovery", "expansion", "profit", "gain", "rise",
]

URGENCY_WORDS = [
    "breaking", "urgent", "alert", "just in", "developing", "flash", "emergency",
    "halt", "suspend", "circuit breaker", "fed speaks", "fomc", "powell",
    "cpi report", "jobs report", "gdp", "nonfarm", "rate decision",
    "war", "nuclear", "cyber attack", "market close", "trading halt",
]

BLOCK_WORDS = [
    "trading halt", "circuit breaker", "sec investigation", "doj investigation",
    "accounting fraud", "restatement", "going concern", "chapter 11",
    "fbi raid", "ceo arrested", "massive recall", "nuclear", "bioterror",
]

# ── High-impact economic events ────────────────────────────────────────────────
HIGH_IMPACT_EVENTS = [
    "nonfarm", "cpi", "pce", "fomc", "fed rate", "gdp", "retail sales",
    "unemployment", "ppi", "ism manufacturing", "ism services", "jolts",
    "consumer confidence", "housing starts", "durable goods",
]


# ── Finnhub helpers ────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> any:
    if not FINNHUB_KEY:
        return None
    params["token"] = FINNHUB_KEY
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # 403 means the configured Finnhub key doesn't have access to this
        # endpoint (a premium tier). This used to be suppressed entirely —
        # not even a print — so the caller's scorer silently returns its
        # neutral-50 default and nothing anywhere records that the feature
        # is actually just unavailable, not "reads as neutral." Still don't
        # spam the console every 2-minute scan, but do surface it via the
        # rate-limited degraded() alert so it's visible at least once.
        msg = str(e)
        if "403" in msg or "404" in msg:
            alerts.degraded(f"finnhub_{endpoint.split('/')[0]}",
                             f"{endpoint} returned {msg[:80]} — likely not available on this API tier")
        else:
            print(f"  [intel] {endpoint}: {e}")
        return None


def _fetch_market_news() -> list:
    """General financial news — top 20 items."""
    data = _get("news", {"category": "general", "minId": 0})
    return data[:20] if isinstance(data, list) else []


def _fetch_symbol_news(symbol: str) -> list:
    """Company-specific news for last 2 days."""
    today = date.today().isoformat()
    two_days_ago = (date.today() - timedelta(days=2)).isoformat()
    data = _get("company-news", {"symbol": symbol, "from": two_days_ago, "to": today})
    return data[:10] if isinstance(data, list) else []


def _fetch_economic_calendar() -> list:
    """Upcoming economic events for next 7 days."""
    today = date.today().isoformat()
    week_out = (date.today() + timedelta(days=7)).isoformat()
    data = _get("calendar/economic", {"from": today, "to": week_out})
    if isinstance(data, dict):
        return data.get("economicCalendar", [])
    return []


def _fetch_earnings_calendar(symbols: list) -> list:
    """Earnings announcements for next 7 days."""
    today = date.today().isoformat()
    week_out = (date.today() + timedelta(days=7)).isoformat()
    data = _get("calendar/earnings", {"from": today, "to": week_out})
    if isinstance(data, dict):
        items = data.get("earningsCalendar", [])
        return [e for e in items if e.get("symbol") in symbols]
    return []


def _fetch_symbol_sentiment(symbol: str) -> dict:
    """Finnhub news sentiment score for a symbol."""
    data = _get("news-sentiment", {"symbol": symbol})
    return data if isinstance(data, dict) else {}


def _fetch_social_sentiment(symbol: str) -> dict:
    """
    Finnhub social media sentiment — Reddit + Twitter/X mentions and score.
    Returns bullish/bearish ratio and mention volume.
    """
    today     = date.today().isoformat()
    week_ago  = (date.today() - timedelta(days=7)).isoformat()
    data = _get("stock/social-sentiment", {"symbol": symbol, "from": week_ago, "to": today})
    if not isinstance(data, dict):
        return {}
    reddit  = data.get("reddit", [])
    twitter = data.get("twitter", [])
    return {
        "reddit":  reddit[-1]  if reddit  else {},
        "twitter": twitter[-1] if twitter else {},
        "reddit_mentions":  sum(r.get("mention", 0) for r in reddit[-3:]),
        "twitter_mentions": sum(t.get("mention", 0) for t in twitter[-3:]),
        "reddit_score":  reddit[-1].get("score", 0)  if reddit  else 0,
        "twitter_score": twitter[-1].get("score", 0) if twitter else 0,
    }


def _fetch_analyst_recommendations(symbol: str) -> list:
    """
    Finnhub analyst recommendation trends (Buy/Hold/Sell consensus).
    Returns last 2 months of data.
    """
    data = _get("stock/recommendation", {"symbol": symbol})
    return data[:2] if isinstance(data, list) else []


def _fetch_insider_transactions(symbol: str) -> dict:
    """
    Finnhub insider transactions (buying/selling by executives and directors).
    Significant insider buying = bullish signal.
    """
    today    = date.today().isoformat()
    month_ago = (date.today() - timedelta(days=30)).isoformat()
    data = _get("stock/insider-transactions", {"symbol": symbol, "from": month_ago, "to": today})
    if not isinstance(data, dict):
        return {}
    transactions = data.get("data", [])
    buys  = [t for t in transactions if t.get("transactionType") in ("Buy", "P - Purchase")]
    sells = [t for t in transactions if t.get("transactionType") in ("Sale", "S - Sale")]
    return {
        "buy_count":    len(buys),
        "sell_count":   len(sells),
        "buy_shares":   sum(t.get("share", 0) for t in buys),
        "sell_shares":  sum(t.get("share", 0) for t in sells),
        "net_sentiment": "bullish" if len(buys) > len(sells) else "bearish" if len(sells) > len(buys) else "neutral",
    }


# ── Macro market data from yfinance ───────────────────────────────────────────

def _fetch_macro_market_data() -> dict:
    """
    Fetch key macro indicators from yfinance:
    - 10yr Treasury yield (^TNX) — rising = bearish for equities
    - 30yr Treasury yield (^TYX)
    - Gold (GC=F) — rising = risk-off / uncertainty
    - Crude Oil (CL=F) — context for energy and inflation
    - Dollar Index (DX-Y.NYB) — rising dollar = risk-off
    - Bitcoin (BTC-USD) — risk-on/off proxy
    """
    if not YF_AVAILABLE:
        return {}

    tickers = {
        "yield_10yr": "^TNX",
        "yield_30yr": "^TYX",
        "gold":       "GC=F",
        "oil":        "CL=F",
        "dxy":        "DX-Y.NYB",
        "btc":        "BTC-USD",
    }

    result = {}
    for name, ticker_sym in tickers.items():
        try:
            t    = yf.Ticker(ticker_sym)
            hist = t.history(period="5d", interval="1d")
            if hist is not None and len(hist) >= 2:
                curr  = float(hist["Close"].iloc[-1])
                prev  = float(hist["Close"].iloc[-2])
                chg   = (curr - prev) / prev * 100
                result[name] = {"value": round(curr, 2), "change_pct": round(chg, 2)}
        except Exception:
            pass

    return result


def _score_macro_market(macro_data: dict) -> dict:
    """
    Score macro market conditions for equity risk.
    Returns a risk score (0-100) and list of signals.
    """
    if not macro_data:
        return {"macro_market_risk": 30, "signals": [], "risk_on": True}

    risk_score = 30   # Baseline low risk
    signals    = []
    risk_on_signals = 0
    risk_off_signals = 0

    # Treasury yields (rising yield = bearish for growth stocks)
    yield_10 = macro_data.get("yield_10yr", {})
    if yield_10:
        chg = yield_10.get("change_pct", 0)
        val = yield_10.get("value", 4.0)
        if chg > 2:
            risk_score += 15
            risk_off_signals += 1
            signals.append(f"⚠️ 10yr yield surging +{chg:.1f}% ({val:.2f}%) — equity headwind")
        elif chg < -2:
            risk_score -= 10
            risk_on_signals += 1
            signals.append(f"✓ 10yr yield falling {chg:.1f}% — equity tailwind")
        if val > 5.0:
            risk_score += 10
            signals.append(f"⚠️ 10yr yield high ({val:.2f}%) — valuation pressure")

    # Gold (rising = fear/uncertainty)
    gold = macro_data.get("gold", {})
    if gold:
        chg = gold.get("change_pct", 0)
        if chg > 1.5:
            risk_score += 10
            risk_off_signals += 1
            signals.append(f"⚠️ Gold surging +{chg:.1f}% — risk-off signal")
        elif chg < -1:
            risk_score -= 5
            risk_on_signals += 1
            signals.append(f"✓ Gold falling — risk-on environment")

    # Dollar Index (strong dollar = risk-off)
    dxy = macro_data.get("dxy", {})
    if dxy:
        chg = dxy.get("change_pct", 0)
        if chg > 0.5:
            risk_score += 8
            risk_off_signals += 1
            signals.append(f"⚠️ Dollar strengthening +{chg:.1f}% — risk-off pressure")
        elif chg < -0.5:
            risk_score -= 5
            risk_on_signals += 1
            signals.append(f"✓ Dollar weakening — emerging market/equity tailwind")

    # Oil (large moves = macro uncertainty)
    oil = macro_data.get("oil", {})
    if oil:
        chg = oil.get("change_pct", 0)
        if abs(chg) > 3:
            risk_score += 8
            signals.append(f"⚠️ Oil {'+' if chg > 0 else ''}{chg:.1f}% — energy market volatility")

    # Bitcoin (risk proxy — falling BTC = risk-off)
    btc = macro_data.get("btc", {})
    if btc:
        chg = btc.get("change_pct", 0)
        if chg < -5:
            risk_score += 8
            risk_off_signals += 1
            signals.append(f"⚠️ BTC -{abs(chg):.1f}% — risk-off across asset classes")
        elif chg > 5:
            risk_on_signals += 1
            signals.append(f"✓ BTC +{chg:.1f}% — broad risk appetite")

    risk_score = max(0, min(100, risk_score))
    risk_on    = risk_on_signals > risk_off_signals

    return {
        "macro_market_risk": risk_score,
        "signals":           signals[:5],
        "risk_on":           risk_on,
        "risk_on_signals":   risk_on_signals,
        "risk_off_signals":  risk_off_signals,
        "yield_10yr":        macro_data.get("yield_10yr", {}).get("value", 0),
        "gold_chg":          macro_data.get("gold", {}).get("change_pct", 0),
        "dxy_chg":           macro_data.get("dxy", {}).get("change_pct", 0),
        "oil_chg":           macro_data.get("oil", {}).get("change_pct", 0),
        "btc_chg":           macro_data.get("btc", {}).get("change_pct", 0),
    }


# ── Keyword scoring ────────────────────────────────────────────────────────────

def _score_text(text: str) -> dict:
    """Quick rule-based scoring of a news headline/summary."""
    text_lower = text.lower()

    bearish_hits = sum(1 for w in BEARISH_WORDS if w in text_lower)
    bullish_hits = sum(1 for w in BULLISH_WORDS if w in text_lower)
    urgency_hits = sum(1 for w in URGENCY_WORDS if w in text_lower)
    block_hits   = sum(1 for w in BLOCK_WORDS   if w in text_lower)

    # Sentiment: 50=neutral, higher=bullish, lower=bearish
    total = bearish_hits + bullish_hits
    if total == 0:
        sentiment = 50
    else:
        sentiment = round(50 + (bullish_hits - bearish_hits) / total * 50)

    urgency  = min(100, urgency_hits * 25 + (20 if block_hits > 0 else 0))
    blocking = block_hits > 0

    return {
        "sentiment": max(0, min(100, sentiment)),
        "urgency":   urgency,
        "blocking":  blocking,
        "bearish":   bearish_hits,
        "bullish":   bullish_hits,
    }


def _score_articles(articles: list) -> dict:
    """Aggregate scores across a list of news articles."""
    if not articles:
        return {"sentiment": 50, "urgency": 0, "blocking": False, "count": 0, "headlines": []}

    sentiments, urgencies, blocking = [], [], False
    headlines = []

    for a in articles[:8]:
        headline = a.get("headline", a.get("summary", ""))
        if not headline:
            continue
        s = _score_text(headline)
        sentiments.append(s["sentiment"])
        urgencies.append(s["urgency"])
        if s["blocking"]:
            blocking = True
        headlines.append(headline[:120])

    avg_sentiment = round(sum(sentiments) / len(sentiments)) if sentiments else 50
    max_urgency   = max(urgencies) if urgencies else 0

    return {
        "sentiment": avg_sentiment,
        "urgency":   max_urgency,
        "blocking":  blocking,
        "count":     len(headlines),
        "headlines": headlines[:5],
    }


# ── Economic calendar scoring ─────────────────────────────────────────────────

def _score_economic_calendar(events: list) -> dict:
    """Score upcoming economic events for macro risk."""
    if not events:
        return {"macro_risk": 10, "upcoming_events": [], "next_event_hours": 999}

    high_impact = []
    now = datetime.now()

    # US-centric keywords — skip foreign events (CPIF=Swedish, BOE=UK, etc.)
    US_ONLY_EVENTS = {"nonfarm", "cpi", "pce", "fomc", "fed rate", "gdp",
                      "retail sales", "unemployment", "ppi", "ism", "jolts",
                      "consumer confidence", "housing starts", "durable goods",
                      "us ", "u.s.", "federal reserve", "treasury"}
    FOREIGN_SKIP = {"cpif", "boe", "ecb", "rba", "boj", "rbnz", "snb",
                    "swedish", "german", "eurozone", "uk ", "china ", "japan"}

    for e in events:
        name = (e.get("event") or "").lower()
        impact = (e.get("impact") or "").lower()
        country = (e.get("country") or "").upper()
        event_time_str = e.get("time") or e.get("date") or ""

        # Skip non-US events
        if country and country not in ("US", "USD", ""):
            continue
        if any(k in name for k in FOREIGN_SKIP):
            continue

        is_high = any(k in name for k in HIGH_IMPACT_EVENTS) or impact in ("high", "3")
        if not is_high:
            continue

        # Parse event time
        hours_away = 999
        try:
            if "T" in event_time_str:
                et = datetime.fromisoformat(event_time_str.replace("Z", ""))
            else:
                et = datetime.fromisoformat(event_time_str)
            hours_away = (et - now).total_seconds() / 3600
        except Exception:
            pass

        # Skip events that already happened (negative hours = past)
        if hours_away < -0.5:
            continue

        high_impact.append({
            "name":       e.get("event", "Unknown"),
            "hours_away": round(hours_away, 1),
            "actual":     e.get("actual"),
            "estimate":   e.get("estimate"),
        })

    if not high_impact:
        return {"macro_risk": 10, "upcoming_events": [], "next_event_hours": 999}

    # Sort by proximity
    high_impact.sort(key=lambda x: x["hours_away"])
    next_hours = high_impact[0]["hours_away"]

    # Risk is highest when event is imminent
    if next_hours < 0.5:       macro_risk = 85   # happening now
    elif next_hours < 2:       macro_risk = 70   # within 2 hours
    elif next_hours < 6:       macro_risk = 50   # same day
    elif next_hours < 24:      macro_risk = 30   # tomorrow
    else:                      macro_risk = 15   # later this week

    return {
        "macro_risk":       macro_risk,
        "upcoming_events":  [e["name"] for e in high_impact[:3]],
        "next_event_hours": next_hours,
    }


def _score_earnings(earnings: list, symbol: str) -> dict:
    """Check if symbol has earnings coming up."""
    for e in earnings:
        if e.get("symbol") != symbol:
            continue
        date_str = e.get("date") or ""
        try:
            earnings_date = date.fromisoformat(date_str)
            days_away = (earnings_date - date.today()).days
            if days_away <= 1:
                return {"earnings_soon": True, "days_away": days_away, "event_risk_boost": 40}
            elif days_away <= 3:
                return {"earnings_soon": True, "days_away": days_away, "event_risk_boost": 20}
            elif days_away <= 7:
                return {"earnings_soon": True, "days_away": days_away, "event_risk_boost": 10}
        except Exception:
            pass
    return {"earnings_soon": False, "days_away": 999, "event_risk_boost": 0}


# ── New v2 scoring functions ───────────────────────────────────────────────────

def _score_social_sentiment(social: dict, direction: str = "bullish") -> float:
    """
    Score social media sentiment 0-100.
    Reddit score > 0 = bullish buzz. Twitter score > 0 = bullish.
    """
    if not social:
        return 50.0

    r_score = float(social.get("reddit_score", 0))
    t_score = float(social.get("twitter_score", 0))
    r_mentions = int(social.get("reddit_mentions", 0))
    t_mentions = int(social.get("twitter_mentions", 0))

    # Composite: blend reddit and twitter scores
    total_mentions = r_mentions + t_mentions
    if total_mentions > 0:
        # Weight by volume
        blended = (r_score * r_mentions + t_score * t_mentions) / total_mentions
    else:
        blended = (r_score + t_score) / 2

    # Convert to 0-100 scale (scores typically range -1 to +1)
    score = 50 + blended * 50
    score = max(0, min(100, score))

    # For bearish trades, invert (bearish sentiment = high score)
    if direction == "bearish":
        score = 100 - score

    return round(score, 1)


def _score_analyst_recommendations(recs: list) -> float:
    """
    Score analyst consensus 0-100.
    More buy ratings = higher score.
    Uses the most recent month's data.
    """
    if not recs:
        return 50.0

    latest = recs[0]  # most recent
    strong_buy = int(latest.get("strongBuy", 0))
    buy        = int(latest.get("buy", 0))
    hold       = int(latest.get("hold", 0))
    sell       = int(latest.get("sell", 0))
    strong_sell = int(latest.get("strongSell", 0))

    total = strong_buy + buy + hold + sell + strong_sell
    if total == 0:
        return 50.0

    # Weighted score: strongBuy=5, buy=4, hold=3, sell=2, strongSell=1
    weighted = (strong_buy*5 + buy*4 + hold*3 + sell*2 + strong_sell*1) / total
    # Convert 1-5 to 0-100
    score = (weighted - 1) / 4 * 100
    return round(max(0, min(100, score)), 1)


def _score_insider_activity(insider: dict) -> float:
    """
    Score insider buying vs selling 0-100.
    Net buyers = bullish signal.
    """
    if not insider:
        return 50.0

    buys  = insider.get("buy_count", 0)
    sells = insider.get("sell_count", 0)
    total = buys + sells

    if total == 0:
        return 50.0

    buy_ratio = buys / total
    score     = 20 + buy_ratio * 60   # 0 buys=20, all buys=80, neither extreme

    # Large volume insider buys = stronger signal
    buy_shares  = insider.get("buy_shares", 0)
    sell_shares = insider.get("sell_shares", 0)
    if buy_shares > sell_shares * 2:
        score = min(100, score + 15)
    elif sell_shares > buy_shares * 2:
        score = max(0, score - 15)

    return round(score, 1)


# ── Main intelligence function ─────────────────────────────────────────────────

def get_intelligence_scores(symbols: list) -> dict:
    """
    Main entry point. Returns per-symbol intelligence scores + global macro score.

    Returns:
    {
      "macro": { macro_risk, upcoming_events, next_event_hours },
      "symbols": {
        "AAPL": {
          news_sentiment_score,
          news_urgency_score,
          event_risk_score,
          sector_risk_score,
          trade_blocked,
          block_reason,
          headlines,
          earnings_soon,
        },
        ...
      },
      "ts": timestamp,
    }
    """
    # Check cache first
    cached = _load_cache()
    if cached:
        return cached

    print("  [intel] Fetching market intelligence (v2)...")
    result = {"symbols": {}, "ts": time.time()}

    # ── 1. Market-wide news ────────────────────────────────────────────────────
    market_news    = _fetch_market_news()
    market_scores  = _score_articles(market_news)

    # ── 2. Economic calendar ───────────────────────────────────────────────────
    econ_events    = _fetch_economic_calendar()
    macro_scores   = _score_economic_calendar(econ_events)

    # ── 3. Macro market data (yields, gold, oil, DXY, BTC) ────────────────────
    macro_market   = _fetch_macro_market_data()
    macro_mkt_score = _score_macro_market(macro_market)

    # Blend macro risk from all sources
    macro_risk = max(
        macro_scores["macro_risk"],
        min(60, market_scores["urgency"]),
        int(macro_mkt_score["macro_market_risk"] * 0.5),  # weight at 50%
    )
    if market_scores["sentiment"] < 30:
        macro_risk = min(100, macro_risk + 20)

    result["macro"] = {
        "macro_risk_score":    macro_risk,
        "market_sentiment":    market_scores["sentiment"],
        "market_urgency":      market_scores["urgency"],
        "upcoming_events":     macro_scores["upcoming_events"],
        "next_event_hours":    macro_scores["next_event_hours"],
        "market_headlines":    market_scores["headlines"][:3],
        "market_blocked":      market_scores["blocking"],
        # New macro fields
        "macro_market_risk":   macro_mkt_score["macro_market_risk"],
        "risk_on":             macro_mkt_score["risk_on"],
        "macro_signals":       macro_mkt_score["signals"],
        "yield_10yr":          macro_mkt_score.get("yield_10yr", 0),
        "gold_chg":            macro_mkt_score.get("gold_chg", 0),
        "dxy_chg":             macro_mkt_score.get("dxy_chg", 0),
        "oil_chg":             macro_mkt_score.get("oil_chg", 0),
        "btc_chg":             macro_mkt_score.get("btc_chg", 0),
    }

    # ── 4. Earnings calendar ───────────────────────────────────────────────────
    earnings = _fetch_earnings_calendar(symbols)

    # ── 5. Per-symbol scores ───────────────────────────────────────────────────
    for symbol in symbols:
        sym_news    = _fetch_symbol_news(symbol)
        sym_scores  = _score_articles(sym_news)
        sym_earn    = _score_earnings(earnings, symbol)
        sector      = SECTOR_MAP.get(symbol, "general")

        # Finnhub news sentiment
        finnhub_sentiment = _fetch_symbol_sentiment(symbol)
        finnhub_sent = finnhub_sentiment.get("sentiment", {}).get("bullishPercent", 0.5)

        # v2: Social sentiment (Reddit + Twitter)
        social      = _fetch_social_sentiment(symbol)
        social_sent = _score_social_sentiment(social)

        # v2: Analyst recommendations
        analyst_recs  = _fetch_analyst_recommendations(symbol)
        analyst_score = _score_analyst_recommendations(analyst_recs)

        # v2: Insider transactions
        insider       = _fetch_insider_transactions(symbol)
        insider_score = _score_insider_activity(insider)

        # Blend all sentiment sources
        sentiment_components = [sym_scores["sentiment"]]
        if finnhub_sent > 0:
            sentiment_components.append(finnhub_sent * 100)
        if social.get("reddit_mentions", 0) + social.get("twitter_mentions", 0) > 0:
            sentiment_components.append(social_sent)

        blended_sentiment = round(sum(sentiment_components) / len(sentiment_components))

        # Headline confidence: how many sources agree on direction
        bullish_sources = sum([
            1 if blended_sentiment > 55 else 0,
            1 if analyst_score > 60 else 0,
            1 if social_sent > 55 else 0,
            1 if insider_score > 60 else 0,
        ])
        bearish_sources = sum([
            1 if blended_sentiment < 45 else 0,
            1 if analyst_score < 40 else 0,
            1 if social_sent < 45 else 0,
            1 if insider_score < 40 else 0,
        ])
        total_sources = 4
        dominant     = max(bullish_sources, bearish_sources)
        headline_confidence = round(dominant / total_sources * 100)

        # Event risk: news urgency + earnings boost + macro spillover
        event_risk = min(100,
            sym_scores["urgency"]
            + sym_earn["event_risk_boost"]
            + (macro_risk // 4)
        )

        # Sector risk
        sector_risk = min(100, round(
            (100 - market_scores["sentiment"]) * 0.4
            + sym_scores["urgency"] * 0.3
        ))

        # Trade blocked?
        trade_blocked = False
        block_reason  = ""
        if sym_scores["blocking"]:
            trade_blocked = True
            block_reason  = "Breaking negative news detected"
        elif result["macro"]["market_blocked"]:
            trade_blocked = True
            block_reason  = "High-impact market event in progress"
        elif sym_earn["earnings_soon"] and sym_earn["days_away"] <= 1:
            trade_blocked = True
            block_reason  = f"Earnings in {sym_earn['days_away']} day(s)"
        elif event_risk > 75 and blended_sentiment < 35:
            trade_blocked = True
            block_reason  = "High event risk with bearish news"

        result["symbols"][symbol] = {
            "news_sentiment_score":      blended_sentiment,
            "news_urgency_score":        sym_scores["urgency"],
            "event_risk_score":          event_risk,
            "sector_risk_score":         sector_risk,
            "macro_risk_score":          macro_risk,
            "social_sentiment_score":    social_sent,
            "analyst_score":             analyst_score,
            "insider_score":             insider_score,
            "headline_confidence_score": headline_confidence,
            "trade_blocked":             trade_blocked,
            "block_reason":              block_reason,
            "earnings_soon":             sym_earn["earnings_soon"],
            "earnings_days_away":        sym_earn["days_away"],
            "headlines":                 sym_scores["headlines"][:3],
            "sector":                    sector,
            # Raw data for dashboard
            "analyst_recs":   analyst_recs[0] if analyst_recs else {},
            "insider_summary": insider,
            "social_mentions": {
                "reddit":  social.get("reddit_mentions", 0),
                "twitter": social.get("twitter_mentions", 0),
            },
        }

    _save_cache(result)
    print(f"  [intel] Done. Macro risk: {macro_risk}/100 | "
          f"Market sentiment: {market_scores['sentiment']}/100")
    return result


def check_trade_allowed(symbol: str, intel: dict) -> tuple:
    """
    Returns (allowed: bool, reason: str).
    Call before placing any trade.
    """
    sym_intel = intel.get("symbols", {}).get(symbol, {})
    macro     = intel.get("macro", {})

    if sym_intel.get("trade_blocked"):
        return False, sym_intel.get("block_reason", "Blocked by intelligence engine")

    if macro.get("market_blocked"):
        return False, "Market-wide blocking event detected"

    if macro.get("macro_risk_score", 0) > 80:
        next_hrs = macro.get("next_event_hours", 999)
        events   = macro.get("upcoming_events", [])
        if next_hrs < 1:   # only block within 1 hour (was 2h — too aggressive)
            return False, f"High-impact US macro event imminent (<1h): {', '.join(events[:2])}"

    return True, "Intelligence check passed"


def get_trade_context(symbol: str, direction: str, intel: dict) -> str:
    """
    Returns a natural language summary of the intelligence context for this trade.
    Passed to Claude as additional context.
    """
    sym_intel  = intel.get("symbols", {}).get(symbol, {})
    macro      = intel.get("macro", {})

    sentiment  = sym_intel.get("news_sentiment_score", 50)
    urgency    = sym_intel.get("news_urgency_score", 0)
    event_risk = sym_intel.get("event_risk_score", 0)
    macro_risk = macro.get("macro_risk_score", 0)
    headlines  = sym_intel.get("headlines", [])
    events     = macro.get("upcoming_events", [])
    earnings   = sym_intel.get("earnings_soon", False)
    earn_days  = sym_intel.get("earnings_days_away", 999)
    next_hrs   = macro.get("next_event_hours", 999)

    sentiment_label = "bullish" if sentiment > 60 else "bearish" if sentiment < 40 else "neutral"
    direction_aligned = (
        (direction == "bullish" and sentiment > 55) or
        (direction == "bearish" and sentiment < 45) or
        (sentiment_label == "neutral")
    )

    lines = [f"INTELLIGENCE CONTEXT for {symbol} ({direction.upper()}):"]

    # News sentiment
    lines.append(f"- News sentiment: {sentiment}/100 ({sentiment_label}) — "
                 f"{'aligned with' if direction_aligned else 'AGAINST'} trade direction")

    # Headlines
    if headlines:
        lines.append(f"- Recent headlines: {' | '.join(headlines[:2])}")
    else:
        lines.append("- No significant recent news found")

    # Earnings
    if earnings and earn_days <= 7:
        lines.append(f"- EARNINGS WARNING: Earnings in {earn_days} day(s) — elevated volatility risk")

    # Event risk
    if event_risk > 50:
        lines.append(f"- Event risk score: {event_risk}/100 (elevated)")
    else:
        lines.append(f"- Event risk score: {event_risk}/100 (low)")

    # Macro events
    if events and next_hrs < 24:
        lines.append(f"- Upcoming macro event within {next_hrs:.1f}h: {', '.join(events[:2])}")
        lines.append(f"- Macro risk score: {macro_risk}/100")
    else:
        lines.append(f"- No high-impact macro events within 24h. Macro risk: {macro_risk}/100")

    return "\n".join(lines)


def get_pre_trade_summary(symbol: str, direction: str, strategy: str, intel: dict,
                          technical_signals: list, flow_signals: list,
                          score: float) -> str:
    """
    Produce a structured trade approval context string for Claude.
    Claude uses this to write the final 'Trade approved because...' explanation.
    """
    allowed, reason = check_trade_allowed(symbol, intel)
    context = get_trade_context(symbol, direction, intel)

    tech_summary  = " | ".join(technical_signals[:3]) if technical_signals else "No signals"
    flow_summary  = " | ".join(flow_signals[:2]) if flow_signals else "No flow data"

    return f"""
PRE-TRADE INTELLIGENCE GATE: {"PASS" if allowed else "BLOCK — " + reason}

{context}

TECHNICAL SIGNALS: {tech_summary}
OPTIONS FLOW: {flow_summary}
COMPOSITE SCORE: {score:.0f}/100

INSTRUCTION: If approving this trade, begin your reasoning with:
"Trade approved because {symbol} has [summarize key bullish/bearish factors], [flow context],
[news/macro context], and [volatility context]."
If blocking, begin with: "Trade blocked because [specific reason]."
"""


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < CACHE_TTL:
                return data
        except Exception:
            pass
    return {}


def _save_cache(data: dict):
    LOG_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    symbols = ["AAPL", "TSLA", "NVDA", "SPY", "META"]
    intel = get_intelligence_scores(symbols)

    print("\nMACRO:")
    print(f"  Risk: {intel['macro']['macro_risk_score']}/100")
    print(f"  Market sentiment: {intel['macro']['market_sentiment']}/100")
    print(f"  Upcoming: {intel['macro']['upcoming_events']}")

    print("\nSYMBOLS:")
    for sym, s in intel["symbols"].items():
        blocked = " [BLOCKED]" if s["trade_blocked"] else ""
        print(f"  {sym}: sentiment={s['news_sentiment_score']} "
              f"urgency={s['news_urgency_score']} "
              f"event_risk={s['event_risk_score']}{blocked}")
        if s.get("block_reason"):
            print(f"    Reason: {s['block_reason']}")

    print("\nTRADE CONTEXT (AAPL bullish):")
    print(get_trade_context("AAPL", "bullish", intel))
