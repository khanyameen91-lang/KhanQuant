"""
scoring.py — Composite 6-Factor Confidence Scoring Engine

Combines all signal sources into a single trade confidence score (0-100).

Factors (each 0-100):
  1. Technical Score    — Multi-timeframe indicators, trend, momentum
  2. Flow Score         — Options flow, unusual activity, PC ratio
  3. Sentiment Score    — News, earnings, economic events
  4. Volatility Score   — IV rank vs strategy, VIX positioning
  5. Regime Score       — Market regime fit for strategy type
  6. Institutional Score — Large trade detection, OI accumulation

Confidence thresholds:
  ≥ 90 → AUTO_EXECUTE
  ≥ 80 → ALERT (execute if in budget)
  ≥ 70 → WATCHLIST
  < 70 → SKIP

Weights are dynamically adjusted by:
  - Market regime (volatile = flow matters more)
  - ML engine feedback (winning strategies get higher weight)
  - Strategy type (0DTE requires higher bar)
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from indicators import compute_technical_score
from flow_analyzer import compute_flow_score, analyze_flow
from sentiment import compute_sentiment_score
from regime import MarketRegime, regime_strategy_score

# Graceful imports for new v3 modules
try:
    from expected_value import compute_ev, get_ev_summary
    EV_AVAILABLE = True
except ImportError:
    EV_AVAILABLE = False
    def compute_ev(setup, ml_win_prob=None): return {"ev_score": 50, "ev_dollars": 0, "approved": True, "reasoning": "EV engine not available"}

try:
    from liquidity import score_setup_liquidity, get_liquidity_summary
    LIQ_AVAILABLE = True
except ImportError:
    LIQ_AVAILABLE = False
    def score_setup_liquidity(snapshot): return {"liquidity_score": 50, "approved": True, "signals": []}

try:
    from relative_strength import compute_rs_score
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    def compute_rs_score(symbol, direction="bullish"): return {"rs_score": 50, "signals": []}


# ── Score Weights (10-Factor Engine) ─────────────────────────────────────────
# Base weights — must sum to 1.0
BASE_WEIGHTS = {
    "technical":     0.25,   # Multi-TF momentum, trend, RSI, MACD
    "flow":          0.20,   # Options flow, sweeps, PC ratio
    "sentiment":     0.12,   # News, intel, economic events
    "volatility":    0.12,   # IV rank vs strategy fit
    "regime":        0.10,   # Market regime alignment
    "institutional": 0.06,   # Block trades, GEX, OI accumulation
    "ev":            0.08,   # Expected value (mathematical edge)
    "liquidity":     0.04,   # Bid/ask spread, OI, volume
    "rs":            0.03,   # Relative strength vs SPY/sector
}

# Minimum confidence to even consider a trade (RAISED for institutional quality)
MIN_CONFIDENCE = {
    "long_option":   65,
    "spread":        65,
    "debit_spread":  65,
    "credit_spread": 65,
    "iron_condor":   65,
    "strangle":      65,
    "0dte_scalp":    75,     # 0DTE still requires higher bar
}

# ML weight override file
ML_WEIGHT_FILE = Path("logs/ml_weights.json")


@dataclass
class TradeScore:
    symbol: str
    direction: str
    strategy_type: str

    # Sub-scores (each 0-100)
    technical_score:     float = 50.0
    flow_score:          float = 50.0
    sentiment_score:     float = 50.0
    volatility_score:    float = 50.0
    regime_score:        float = 50.0
    institutional_score: float = 50.0
    ev_score:            float = 50.0   # Expected Value score
    liquidity_score:     float = 50.0   # Options liquidity score
    rs_score:            float = 50.0   # Relative strength score

    # Weighted composite
    confidence:     float = 50.0
    action:         str   = "SKIP"   # AUTO_EXECUTE | ALERT | WATCHLIST | SKIP

    # Supporting details
    technical_signals:  list = None
    flow_signals:       list = None
    sentiment_signals:  list = None
    regime_summary:     str  = ""
    key_risks:          list = None
    regime_multiplier:  float = 1.0
    weights_used:       dict = None

    def __post_init__(self):
        if self.technical_signals is None:
            self.technical_signals = []
        if self.flow_signals is None:
            self.flow_signals = []
        if self.sentiment_signals is None:
            self.sentiment_signals = []
        if self.key_risks is None:
            self.key_risks = []
        if self.weights_used is None:
            self.weights_used = {}

    def to_dict(self) -> dict:
        return asdict(self)


# ── Volatility Score ───────────────────────────────────────────────────────────

def compute_volatility_score(iv_rank: float, strategy_type: str, vix: float = 20.0) -> dict:
    """
    Score how well current IV conditions match the strategy type.

    Credit strategies (condors, credit spreads) thrive in HIGH IV.
    Debit strategies (long options, debit spreads) thrive in LOW IV.
    """
    score = 50.0
    signals = []

    # IV rank fit for strategy
    is_premium_seller = strategy_type in ("credit_spread", "iron_condor")
    is_buyer          = strategy_type in ("long_option", "0dte_scalp", "strangle", "spread")

    if is_premium_seller:
        if iv_rank >= 70:
            score = 90
            signals.append(f"✓ IV rank {iv_rank:.0f} — excellent for premium selling")
        elif iv_rank >= 50:
            score = 75
            signals.append(f"✓ IV rank {iv_rank:.0f} — good for premium selling")
        elif iv_rank < 30:
            score = 30
            signals.append(f"✗ IV rank {iv_rank:.0f} — too low for premium selling")
        else:
            score = 55
            signals.append(f"~ IV rank {iv_rank:.0f} — acceptable for premium selling")

    elif is_buyer:
        if iv_rank <= 25:
            score = 90
            signals.append(f"✓ IV rank {iv_rank:.0f} — cheap options, excellent for buying")
        elif iv_rank <= 45:
            score = 75
            signals.append(f"✓ IV rank {iv_rank:.0f} — reasonable premium for buying")
        elif iv_rank >= 70:
            score = 25
            signals.append(f"✗ IV rank {iv_rank:.0f} — expensive options (IV crush risk)")
        else:
            score = 55
            signals.append(f"~ IV rank {iv_rank:.0f} — moderate option cost")

    # VIX adjustment
    if vix >= 35:
        score = max(0, score - 15)
        signals.append(f"✗ VIX {vix:.0f} — extreme fear (position sizing reduced)")
    elif vix >= 25:
        score = max(0, score - 5)
        signals.append(f"~ VIX {vix:.0f} — elevated volatility")
    elif vix <= 15:
        if is_buyer:
            score = min(100, score + 5)
            signals.append(f"✓ VIX {vix:.0f} — low vol, cheap options")
        else:
            signals.append(f"~ VIX {vix:.0f} — low vol (spreads may be tight)")

    return {"score": round(score, 1), "signals": signals}


# ── Regime Score ───────────────────────────────────────────────────────────────

def compute_regime_score(setup_type: str, direction: str, regime: MarketRegime) -> dict:
    """Score how well the setup fits the current market regime."""
    score = 50.0
    signals = []

    # Base fit from regime preferences
    multiplier = regime_strategy_score(setup_type, regime)
    if multiplier > 1.0:
        score = 75
        signals.append(f"✓ {setup_type} preferred in {regime.trend} regime")
    elif multiplier < 1.0:
        score = 25
        signals.append(f"✗ {setup_type} not ideal in {regime.trend} regime")
    else:
        score = 50
        signals.append(f"~ {setup_type} neutral for {regime.trend} regime")

    # Directional alignment with market
    cond = regime.condition
    is_bull = direction == "bullish"
    is_neutral = direction == "neutral"

    if is_neutral:
        if cond in ("NEUTRAL", "RANGING"):
            score = min(100, score + 20)
            signals.append(f"✓ Neutral strategy matches {cond} market")
        elif cond in ("STRONG_BULL", "STRONG_BEAR"):
            score = max(0, score - 20)
            signals.append(f"✗ Neutral strategy in strong trending market")
    elif is_bull:
        if cond in ("STRONG_BULL", "BULL"):
            score = min(100, score + 20)
            signals.append(f"✓ Bullish direction aligns with {cond} market")
        elif cond in ("STRONG_BEAR", "BEAR"):
            score = max(0, score - 20)
            signals.append(f"✗ Bullish direction against {cond} market")
    else:  # bearish
        if cond in ("STRONG_BEAR", "BEAR"):
            score = min(100, score + 20)
            signals.append(f"✓ Bearish direction aligns with {cond} market")
        elif cond in ("STRONG_BULL", "BULL"):
            score = max(0, score - 20)
            signals.append(f"✗ Bearish direction against {cond} market")

    # Trend strength bonus
    if regime.trend_strength > 40 and setup_type in ("long_option", "spread"):
        score = min(100, score + 5)
        signals.append(f"✓ Strong ADX {regime.trend_strength:.0f} — trend trade")

    return {
        "score": round(max(0, min(100, score)), 1),
        "signals": signals,
        "regime_summary": regime.summary,
        "multiplier": multiplier,
    }


# ── Institutional Score ────────────────────────────────────────────────────────

def compute_institutional_score(flow: dict, direction: str) -> dict:
    """
    Score institutional activity signals (blocks, OI accumulation, gamma positioning).
    Derived from flow_analyzer output.
    """
    score = 50.0
    signals = []
    is_bull = direction == "bullish"

    # Block trades (large premium = institutional conviction)
    call_prem = flow.get("call_premium_moved", 0)
    put_prem  = flow.get("put_premium_moved", 0)

    if call_prem > 100000 and is_bull:
        score = min(100, score + 25)
        signals.append(f"✓ Large call block trades (${call_prem/1000:.0f}K premium)")
    elif put_prem > 100000 and not is_bull:
        score = min(100, score + 25)
        signals.append(f"✓ Large put block trades (${put_prem/1000:.0f}K premium)")
    elif call_prem > 25000 and is_bull:
        score = min(100, score + 10)
        signals.append(f"✓ Call blocks detected (${call_prem/1000:.0f}K)")
    elif put_prem > 25000 and not is_bull:
        score = min(100, score + 10)
        signals.append(f"✓ Put blocks detected (${put_prem/1000:.0f}K)")

    # GEX regime
    gex = flow.get("gamma_exposure", {})
    gex_positive = gex.get("gex_positive", True)
    if is_bull and gex_positive:
        score = min(100, score + 10)
        signals.append("✓ Positive GEX — dealers provide stability")
    elif not is_bull and not gex_positive:
        score = min(100, score + 10)
        signals.append("✓ Negative GEX — dealers amplify moves")
    elif (is_bull and not gex_positive) or (not is_bull and gex_positive):
        score = max(0, score - 5)
        signals.append("~ GEX opposes trade direction slightly")

    # OI accumulation
    call_oi = flow.get("call_oi_rising", 0)
    put_oi  = flow.get("put_oi_rising", 0)
    if is_bull and call_oi > 2:
        score = min(100, score + 10)
        signals.append(f"✓ Call OI accumulating on {call_oi} strikes")
    elif not is_bull and put_oi > 2:
        score = min(100, score + 10)
        signals.append(f"✓ Put OI accumulating on {put_oi} strikes")

    # Unusual activity
    if flow.get("has_unusual_flow"):
        unusual_bull = flow.get("unusual_call_count", 0) > flow.get("unusual_put_count", 0)
        if (is_bull and unusual_bull) or (not is_bull and not unusual_bull):
            score = min(100, score + 5)
            signals.append("✓ Unusual options activity in trade direction")

    return {"score": round(max(0, min(100, score)), 1), "signals": signals}


# ── Weight Loading (ML-adjusted) ──────────────────────────────────────────────

def _load_weights(regime: MarketRegime) -> dict:
    """Load base weights, apply regime adjustments, and apply ML overrides."""
    weights = dict(BASE_WEIGHTS)

    # Apply regime adjustments
    adj = regime.score_adjustments
    for k, factor in adj.items():
        component = k.replace("_weight", "")
        if component in weights:
            weights[component] *= factor

    # Apply ML overrides
    if ML_WEIGHT_FILE.exists():
        try:
            ml_weights = json.loads(ML_WEIGHT_FILE.read_text())
            for k, v in ml_weights.items():
                if k in weights:
                    weights[k] = v
        except Exception:
            pass

    # Normalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ── Main Scoring Function ─────────────────────────────────────────────────────

def score_trade(
    symbol: str,
    direction: str,
    strategy_type: str,
    iv_rank: float,
    regime: MarketRegime,
    flow: dict,                    # from analyze_flow()
    snapshot: dict = None,         # full snapshot for liquidity + EV scoring
    ml_win_prob: float = None,     # ML win probability for EV calculation
    skip_sentiment: bool = False,
) -> "TradeScore":
    """
    Compute the full 10-factor composite confidence score for a trade setup.

    New in v3: EV score, Liquidity score, Relative Strength score.
    Thresholds raised: 95+ = EXECUTE, 90-94 = ALERT, 80-89 = WATCHLIST.
    """
    ts = TradeScore(symbol=symbol, direction=direction, strategy_type=strategy_type)

    # ── 1. Technical ──────────────────────────────────────────────────────────
    tech = compute_technical_score(symbol, direction)
    ts.technical_score   = float(tech.get("score") or 50)
    ts.technical_signals = tech.get("signals", [])

    # ── 2. Flow ───────────────────────────────────────────────────────────────
    flow_score_result = compute_flow_score(flow, direction)
    ts.flow_score   = float(flow_score_result.get("score") or 50)
    ts.flow_signals = flow_score_result.get("signals", [])

    # ── 3. Sentiment ──────────────────────────────────────────────────────────
    sent_data = {}
    if not skip_sentiment:
        sent = compute_sentiment_score(symbol, direction)
        ts.sentiment_score   = float(sent.get("score") or 50)
        ts.sentiment_signals = sent.get("signals", [])
        sent_data            = sent.get("raw_data", {})
        for flag in sent.get("key_risks", []):
            ts.key_risks.append(flag)
    else:
        ts.sentiment_score   = 50.0
        ts.sentiment_signals = ["~ Sentiment skipped for speed"]

    # ── 4. Volatility ─────────────────────────────────────────────────────────
    vol = compute_volatility_score(iv_rank, strategy_type, vix=regime.vix)
    ts.volatility_score = float(vol.get("score") or 50)

    # ── 5. Regime ─────────────────────────────────────────────────────────────
    reg = compute_regime_score(strategy_type, direction, regime)
    ts.regime_score      = float(reg.get("score") or 50)
    ts.regime_summary    = reg.get("regime_summary", "")
    ts.regime_multiplier = float(reg.get("multiplier") or 1.0)

    # ── 6. Institutional ──────────────────────────────────────────────────────
    inst = compute_institutional_score(flow, direction)
    ts.institutional_score = float(inst.get("score") or 50)

    # ── 7. Expected Value ─────────────────────────────────────────────────────
    if EV_AVAILABLE and snapshot:
        ev_result      = compute_ev(snapshot.get("setup", {}), ml_win_prob)
        ts.ev_score    = float(ev_result.get("ev_score", 50))
        if not ev_result.get("approved"):
            ts.key_risks.append(f"Negative EV: {ev_result.get('reasoning', '')[:80]}")
    else:
        ts.ev_score = 50.0   # Neutral if not computable

    # ── 8. Liquidity ──────────────────────────────────────────────────────────
    # liquidity_approved gates the final action below (see "Action
    # classification") — liquidity.py's own stated purpose is "if you can't
    # get a fair fill, don't trade," but until now its pass/fail result only
    # moved the composite score by 4% of its weight, so a contract that
    # failed its own OI/volume/spread minimums could still score high
    # enough on the other 9 factors to trade anyway.
    liquidity_approved = True
    if LIQ_AVAILABLE and snapshot:
        liq_result       = score_setup_liquidity(snapshot)
        ts.liquidity_score = float(liq_result.get("liquidity_score", 50))
        if not liq_result.get("approved"):
            liquidity_approved = False
            ts.key_risks.append(f"Liquidity: {liq_result.get('reject_reason', 'Poor liquidity')}")
    else:
        ts.liquidity_score = 50.0

    # ── 9. Relative Strength ──────────────────────────────────────────────────
    if RS_AVAILABLE:
        rs_result   = compute_rs_score(symbol, direction)
        ts.rs_score = float(rs_result.get("rs_score", 50))
    else:
        ts.rs_score = 50.0

    # ── Weighted composite (10 factors) ───────────────────────────────────────
    weights = _load_weights(regime)
    ts.weights_used = weights

    raw_confidence = (
        ts.technical_score     * weights.get("technical",     0.25) +
        ts.flow_score          * weights.get("flow",          0.20) +
        ts.sentiment_score     * weights.get("sentiment",     0.12) +
        ts.volatility_score    * weights.get("volatility",    0.12) +
        ts.regime_score        * weights.get("regime",        0.10) +
        ts.institutional_score * weights.get("institutional", 0.06) +
        ts.ev_score            * weights.get("ev",            0.08) +
        ts.liquidity_score     * weights.get("liquidity",     0.04) +
        ts.rs_score            * weights.get("rs",            0.03)
    )

    # Apply regime strategy multiplier
    raw_confidence *= ts.regime_multiplier

    # Hard override: FOMC day = 0 regardless of score
    if regime.is_fomc_day:
        raw_confidence = 0.0
        ts.key_risks.append("FOMC Day — no new positions")

    # Apply earnings penalty
    if not skip_sentiment and sent_data.get("earnings", {}).get("has_earnings_soon"):
        raw_confidence *= 0.85

    # EV hard check: if EV is deeply negative, cap score at 60.
    # (Reuses ev_result computed in step 7 above instead of calling
    # compute_ev() a second time for the same setup.)
    if EV_AVAILABLE and snapshot and ev_result.get("ev_dollars", 0) < -20:
        raw_confidence = min(raw_confidence, 60.0)

    ts.confidence = round(max(0, min(100, raw_confidence)), 1)

    # ── Action classification (RAISED THRESHOLDS) ─────────────────────────────
    min_conf = MIN_CONFIDENCE.get(strategy_type, 80)

    if not liquidity_approved:
        ts.action = "SKIP"           # Liquidity veto — no confidence score overrides this
    elif ts.confidence >= 80:
        ts.action = "AUTO_EXECUTE"   # Strong conviction — execute
    elif ts.confidence >= 70:
        ts.action = "ALERT"          # Good setup — execute if budget allows
    elif ts.confidence >= min_conf:
        ts.action = "WATCHLIST"
    else:
        ts.action = "SKIP"

    return ts


def print_score_report(ts: "TradeScore"):
    """Pretty-print a 10-factor score report."""
    print(f"\n{'='*60}")
    print(f"  Score Report: {ts.symbol} {ts.direction.upper()} {ts.strategy_type}")
    print(f"{'='*60}")
    print(f"  CONFIDENCE:  {ts.confidence:.0f}/100  →  {ts.action}")
    print(f"  ─────────────────────────────────────")
    print(f"  Technical:   {ts.technical_score:.0f}/100   (25%)")
    print(f"  Flow:        {ts.flow_score:.0f}/100   (20%)")
    print(f"  EV:          {ts.ev_score:.0f}/100   (8%)")
    print(f"  Sentiment:   {ts.sentiment_score:.0f}/100   (12%)")
    print(f"  Volatility:  {ts.volatility_score:.0f}/100   (12%)")
    print(f"  Regime:      {ts.regime_score:.0f}/100   (10%)")
    print(f"  Institution: {ts.institutional_score:.0f}/100   (6%)")
    print(f"  Liquidity:   {ts.liquidity_score:.0f}/100   (4%)")
    print(f"  Rel Strength:{ts.rs_score:.0f}/100   (3%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Regime:      {ts.regime_summary}")
    if ts.key_risks:
        print(f"  Risks: {' | '.join(ts.key_risks[:3])}")
    print(f"{'='*60}")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from auth import initialize
    initialize()
    from regime import detect_regime
    from flow_analyzer import analyze_flow

    symbol = "SPY"
    price  = 540.0

    print(f"Scoring {symbol} bullish long_option...")
    regime = detect_regime()
    flow   = analyze_flow(symbol, price)
    ts     = score_trade(symbol, "bullish", "long_option", 30, regime, flow)
    print_score_report(ts)
