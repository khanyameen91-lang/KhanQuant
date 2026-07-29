"""
claude_brain.py — LLM Trade Decision Engine

Uses Groq (free) as the primary LLM for trade decisions.
Falls back to pure scoring engine if no LLM is available.

Priority order:
  1. Groq API (free, Llama 3.1 70B) — set GROQ_API_KEY in .env
  2. Anthropic API (paid) — set ANTHROPIC_API_KEY in .env
  3. Pure scoring engine — no API key needed, auto-approves ≥ 75 score

To use Groq (free):
  1. Sign up at https://console.groq.com (free, no credit card)
  2. Create an API key
  3. Add to .env: GROQ_API_KEY=gsk_xxxxxxxxxxxx
  4. That's it — the bot will use Groq automatically
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ── LLM client setup (lazy — only imported if key exists) ─────────────────────
GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def _get_llm_client():
    """Return (client, provider) or (None, 'scoring') if no keys configured."""
    if GROQ_KEY:
        try:
            from groq import Groq
            return Groq(api_key=GROQ_KEY), "groq"
        except ImportError:
            pass  # groq not installed — fall through
    if ANTHROPIC_KEY:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=ANTHROPIC_KEY), "anthropic"
        except ImportError:
            pass
    return None, "scoring"

_llm_client, _llm_provider = _get_llm_client()

print(f"claude_brain: using provider={_llm_provider}")


# ── Shared system prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the final judgment layer of an institutional options trading system running in PAPER TRADING (sandbox) mode.

You receive pre-computed scores from 6 specialized analysis engines. Your job is to:
1. Review the composite confidence score and all sub-scores
2. Check if there are any red flags the scoring missed
3. Confirm or override the automated recommendation
4. Set the final confidence level and execution decision

PAPER TRADING CONTEXT: This is a sandbox — no real money. Be willing to execute on high-confidence setups to validate the strategy. Do not be overly conservative.

ACCOUNT CONSTRAINTS:
- Account: ~$5,000 paper
- Max open positions: 3
- Daily loss limit: $500 (managed by sizing engine)
- DO NOT second-guess position sizing — the sizing engine already calculated the actual dollar risk. The "Max Loss" in the setup is the option premium, not the actual position size.
- Only defined-risk trades

SCORING INTERPRETATION:
- Technical (0-100): Multi-timeframe momentum, trend, RSI, MACD
- Flow (0-100): Options flow direction, unusual activity, PC ratio
- Sentiment (0-100): News, earnings risk, economic events
- Volatility (0-100): IV rank vs strategy fit, VIX environment
- Regime (0-100): Market regime fit for this strategy
- Institutional (0-100): Block trades, GEX, OI accumulation

EXECUTION THRESHOLDS:
- ≥80: Strong conviction — EXECUTE (confirm unless clear red flag)
- 70-79: Good setup — EXECUTE if no red flags
- <70: SKIP unless compelling qualitative data not captured by scoring

OVERRIDE CONDITIONS (when you should override and SKIP):
- Earnings announcement within 2 days (IV crush risk)
- Major macro event (FOMC, CPI) same or next day
- Stock-specific crisis (FDA rejection, fraud, major lawsuit)
- IV rank > 80 for long options (too expensive)

Respond ONLY with valid JSON — no markdown, no preamble:
{
  "decision": "EXECUTE" or "SKIP",
  "confidence": 0-100,
  "override": true/false,
  "reasoning": "brief 1-sentence reason",
  "key_risk": "main risk factor",
  "scale": "FULL" or "HALF" or "SKIP"
}"""


def _call_llm(prompt: str, system: str, max_tokens: int = 512) -> str:
    """Call the available LLM provider and return raw text response."""
    if _llm_provider == "groq":
        response = _llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    elif _llm_provider == "anthropic":
        response = _llm_client.messages.create(
            model="claude-haiku-4-5-20251001",   # cheapest Anthropic model
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    raise RuntimeError("No LLM provider configured")


def _scoring_fallback(score, snapshot: dict, reason: str) -> dict:
    """
    Pure scoring engine decision — no LLM required.
    Auto-approves setups scoring ≥ 75 with a higher bar.
    """
    threshold = 75
    if score.confidence >= threshold:
        scale = "FULL" if score.confidence >= 85 else "HALF"
        return {
            "decision": "EXECUTE",
            "confidence": score.confidence,
            "override": False,
            "reasoning": f"Scoring engine approval ({score.confidence:.0f}/100) — {reason}",
            "key_risk": "No LLM review",
            "scale": scale,
            "snapshot": snapshot,
            "pre_score": score.confidence,
        }
    else:
        return {
            "decision": "SKIP",
            "confidence": score.confidence,
            "override": False,
            "reasoning": f"Score {score.confidence:.0f} below threshold ({threshold}) — {reason}",
            "key_risk": "Low confidence",
            "scale": "SKIP",
            "snapshot": snapshot,
            "pre_score": score.confidence,
        }


def analyze_scored_setup(snapshot: dict, score, intel_context: str = "") -> dict:
    """
    Send a fully scored trade setup to the LLM for final judgment.
    Falls back to scoring engine if no LLM is available.
    """
    # Pure scoring mode — no LLM
    if _llm_provider == "scoring":
        result = _scoring_fallback(score, snapshot, "no LLM configured")
        action = result["decision"]
        print(f"  [SCORING MODE] {action} — {result['reasoning'][:60]}")
        return result

    quote    = snapshot["quote"]
    setup    = snapshot["setup"]
    iv_rank  = snapshot["iv_rank"]
    dte      = snapshot["dte"]
    strategy = setup.get("strategy_name", setup.get("type", ""))

    intel_section = f"\n{intel_context}\n" if intel_context else ""

    prompt = f"""
Review this pre-scored options trade and make the final execution decision.

SYMBOL & MARKET:
- Symbol: {quote['symbol']}
- Price: ${quote['price']:.2f} (Day: {quote['change_pct']:+.2f}%)
- Volume: {quote['volume']:,}
- IV Rank: {iv_rank:.0f}/100

PROPOSED TRADE:
- Strategy: {strategy}
- Direction: {setup['direction']}
- Expiration: {setup['expiration']} ({dte} DTE)
- Max Loss: ${setup['max_loss']:.0f}
- Max Profit: ${setup.get('max_profit') or 'unlimited'}
{f"- Net Credit: ${setup.get('net_credit', 0):.2f}" if setup.get('net_credit') else ""}
{f"- Net Debit: ${setup.get('net_debit', 0):.2f}" if setup.get('net_debit') else ""}

PRE-COMPUTED SCORES (automated engines):
- COMPOSITE CONFIDENCE: {score.confidence:.0f}/100 → {score.action}
- Technical Score:     {score.technical_score:.0f}/100
- Flow Score:          {score.flow_score:.0f}/100
- Sentiment Score:     {score.sentiment_score:.0f}/100
- Volatility Score:    {score.volatility_score:.0f}/100
- Regime Score:        {score.regime_score:.0f}/100
- Institutional Score: {score.institutional_score:.0f}/100

MARKET REGIME: {score.regime_summary}

TECHNICAL SIGNALS:
{chr(10).join(score.technical_signals[:6]) if score.technical_signals else "None computed"}

FLOW SIGNALS:
{chr(10).join(score.flow_signals[:4]) if score.flow_signals else "None computed"}

SENTIMENT SIGNALS:
{chr(10).join(score.sentiment_signals[:3]) if score.sentiment_signals else "None computed"}

KEY RISKS FLAGGED: {', '.join(score.key_risks) if score.key_risks else 'None'}
{intel_section}
Current time: {datetime.now().strftime('%A %B %d, %Y %H:%M ET')}

Confirm the automated recommendation or override it. Respond with JSON only.
"""

    try:
        raw = _call_llm(prompt, SYSTEM_PROMPT, max_tokens=512)
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extract first JSON object
        start = raw.find("{")
        if start >= 0:
            depth, end = 0, -1
            for i, ch in enumerate(raw[start:], start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: end = i + 1; break
            if end > start: raw = raw[start:end]
        decision = json.loads(raw)
        decision["snapshot"] = snapshot
        decision["pre_score"] = score.confidence
        return decision

    except json.JSONDecodeError as e:
        print(f"Warning: LLM returned invalid JSON: {e}")
        return {
            "decision": "EXECUTE" if score.action in ("AUTO_EXECUTE", "ALERT") else "SKIP",
            "confidence": score.confidence,
            "override": False,
            "reasoning": "Scoring engine recommendation (LLM parse error)",
            "key_risk": "Unknown",
            "scale": "HALF",
            "snapshot": snapshot,
            "pre_score": score.confidence,
        }
    except Exception as e:
        print(f"Warning: LLM API error ({_llm_provider}): {e}")
        result = _scoring_fallback(score, snapshot, f"{_llm_provider} API error")
        print(f"  FALLBACK: {result['reasoning'][:80]}")
        return result


def rank_scored_setups(scored_snapshots: list, intel_contexts: dict = None) -> list:
    """
    Review all pre-scored setups and return only approved ones.
    scored_snapshots: list of (snapshot, TradeScore) tuples.
    intel_contexts: optional dict of symbol -> intel context string
    Returns decisions sorted by confidence.
    """
    decisions = []
    intel_contexts = intel_contexts or {}

    for snapshot, score in scored_snapshots:
        if score.action == "SKIP":
            print(f"  Skipping {snapshot['quote']['symbol']} "
                  f"{snapshot['setup'].get('strategy_name','')} "
                  f"— score {score.confidence:.0f} below threshold")
            continue

        setup  = snapshot["setup"]
        symbol = setup["symbol"]
        label  = f"{symbol} {setup.get('strategy_name', setup['type'])}"
        print(f"  [{_llm_provider.upper()}] reviewing {label} (score: {score.confidence:.0f})...")

        intel_ctx = intel_contexts.get(symbol, "")
        decision  = analyze_scored_setup(snapshot, score, intel_ctx)

        if decision.get("decision") == "EXECUTE":
            decisions.append(decision)
            override_note = " [OVERRIDE]" if decision.get("override") else ""
            print(f"     EXECUTE{override_note} — confidence: {decision.get('confidence')}%")
        else:
            print(f"     SKIP — {decision.get('reasoning', '')[:60]}")

    decisions.sort(key=lambda d: d.get("confidence", 0), reverse=True)
    return decisions


def should_close_position(position: dict, current_pnl_pct: float) -> dict:
    """
    Decide whether to close an existing position.
    Falls back to rule-based logic if no LLM available.
    """
    # Rule-based fallback (works without any LLM)
    if _llm_provider == "scoring":
        # Hard rules only
        dte = position.get("dte_remaining", 99)
        if dte <= 1:
            return {"decision": "CLOSE", "reasoning": "DTE ≤ 1 — expiration risk", "urgency": "IMMEDIATE"}
        if current_pnl_pct <= -75:
            return {"decision": "CLOSE", "reasoning": "Stop loss -75%", "urgency": "IMMEDIATE"}
        if current_pnl_pct >= 50:
            return {"decision": "CLOSE", "reasoning": "Take profit +50%", "urgency": "NORMAL"}
        return {"decision": "HOLD", "reasoning": "No exit rule triggered", "urgency": "NORMAL"}

    strategy  = position.get("strategy", "options position")
    direction = position.get("direction", "neutral")
    symbol    = position.get("symbol", "")
    dte       = position.get("dte_remaining", 0)
    days_held = position.get("days_held", 0)

    prompt = f"""
Should I close this options position?

POSITION:
- Symbol: {symbol}
- Strategy: {strategy}
- Direction: {direction}
- Entry date: {position['entry_date']}
- Days held: {days_held}
- DTE remaining: {dte}
- Current P&L: {current_pnl_pct:+.1f}%
- Max loss: ${position['max_loss']:.0f}
- Max profit: ${position.get('max_profit') or 'N/A'}

MANDATORY EXIT RULES (do not override):
- CLOSE if DTE ≤ 1 (expiration risk)
- CLOSE if P&L ≤ -75% of max loss (stop loss)
- CLOSE if P&L ≥ +50% of max profit (take profit)
- CLOSE 0DTE positions by 3:30pm ET

DISCRETIONARY RULES:
- Consider closing at -50% if thesis clearly failed
- Consider closing if adverse news emerged after entry
- For iron condors: close if price approaches short strike

Current time: {datetime.now().strftime('%A %B %d, %Y %H:%M ET')}

Respond with JSON only:
{{"decision": "CLOSE" or "HOLD", "reasoning": "brief reason", "urgency": "IMMEDIATE" or "NORMAL"}}
"""

    try:
        raw = _call_llm(prompt, "You are a disciplined options risk manager. Apply exit rules strictly. Respond with valid JSON only.", max_tokens=256)
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        if start >= 0:
            depth, end = 0, -1
            for i, ch in enumerate(raw[start:], start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: end = i + 1; break
            if end > start: raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"Warning: Close decision error: {e}")
        # Fallback: apply hard rules only
        dte = position.get("dte_remaining", 99)
        if dte <= 1 or current_pnl_pct <= -75:
            return {"decision": "CLOSE", "reasoning": "Hard rule triggered", "urgency": "IMMEDIATE"}
        if current_pnl_pct >= 50:
            return {"decision": "CLOSE", "reasoning": "Take profit", "urgency": "NORMAL"}
        return {"decision": "HOLD", "reasoning": "Error — defaulting to hold", "urgency": "NORMAL"}


# ── v1 fallback: analyze raw (unscored) snapshots ─────────────────────────────
def rank_setups(snapshots: list) -> list:
    """v1 compatibility: analyze raw snapshots without pre-scoring."""
    decisions = []
    for snapshot in snapshots:
        setup = snapshot["setup"]
        label = f"{setup['symbol']} {setup.get('strategy_name', setup.get('type',''))}"
        print(f"  Analyzing {label}...")
        from scoring import TradeScore
        ts = TradeScore(
            symbol=setup["symbol"],
            direction=setup.get("direction", "neutral"),
            strategy_type=setup.get("type", "long_option"),
            confidence=0, action="WATCHLIST",
        )
        decision = analyze_scored_setup(snapshot, ts)
        if decision.get("decision") == "EXECUTE":
            decisions.append(decision)
            print(f"     EXECUTE — confidence: {decision.get('confidence')}%")
        else:
            print(f"     SKIP — {decision.get('reasoning','')[:60]}")
    decisions.sort(key=lambda d: d.get("confidence", 0), reverse=True)
    return decisions


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Provider: {_llm_provider}")
    from scoring import TradeScore
    mock_score = TradeScore(
        symbol="SPY", direction="bullish", strategy_type="long_option",
        technical_score=78, flow_score=72, sentiment_score=65,
        volatility_score=80, regime_score=70, institutional_score=60,
        confidence=73.5, action="WATCHLIST",
        technical_signals=["✓ EMA aligned", "✓ RSI healthy (55)"],
        flow_signals=["✓ Flow bullish", "✓ PC ratio bullish (0.72)"],
        sentiment_signals=["~ Sentiment neutral"],
        regime_summary="NEUTRAL market | RANGING | VIX 18.5",
    )
    mock_snapshot = {
        "quote": {"symbol": "SPY", "price": 542.30, "change_pct": 0.85, "volume": 45_000_000},
        "setup": {"type": "long_option", "strategy_name": "Long Call (Bullish)",
                  "direction": "bullish", "symbol": "SPY", "expiration": "2026-08-15",
                  "max_loss": 250, "max_profit": None},
        "iv_rank": 28, "dte": 17,
    }
    result = analyze_scored_setup(mock_snapshot, mock_score)
    print(json.dumps({k: v for k, v in result.items() if k != "snapshot"}, indent=2))
