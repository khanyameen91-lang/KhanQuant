"""
market_data.py — Market Data & Option Chain Analysis

Pulls live quotes, option chains, and calculates IV rank
from Tastytrade's API. Generates setups for all major
day-trading options strategies:
  - Long calls / puts (low IV, directional)
  - Debit spreads (moderate IV, directional)
  - Credit spreads (high IV, directional bias)
  - Iron condors (high IV, neutral)
  - Straddles / strangles (low IV, big move expected)
  - 0DTE scalps (SPY/QQQ same-day weeklies)
"""

import json
import statistics
from datetime import datetime, date, timedelta
from pathlib import Path
from auth import session
import options_pricing

# ── yfinance for quotes/chains — this account's Tastytrade API access is
# real (api.tastytrade.com, not a cert/sandbox environment; DXLink
# real-time streaming is available), but the chain-fetch path here still
# goes through yfinance for now. Greeks are no longer hardcoded, though —
# see options_pricing.py — they're computed via Black-Scholes from each
# contract's own strike/spot/DTE/IV, which yfinance does provide.
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

_DEFAULT_WATCHLIST = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "META", "AMD"]
_WATCHLIST_FILE = Path("logs/watchlist.json")

# Symbols that support 0DTE (daily expirations)
ZERO_DTE_SYMBOLS = {"SPY", "QQQ", "IWM"}


def get_watchlist() -> list:
    """Load watchlist from file, falling back to defaults."""
    if _WATCHLIST_FILE.exists():
        try:
            return json.loads(_WATCHLIST_FILE.read_text())
        except Exception:
            pass
    return _DEFAULT_WATCHLIST


WATCHLIST = get_watchlist()


# ── Quotes ─────────────────────────────────────────────────────────────────────
def get_quote(symbol: str) -> dict:
    """
    Live price/bid/ask/volume — Tastytrade's own real-time quote feed
    first (real NBBO bid/ask from the broker), yfinance as fallback only
    if Tastytrade is unavailable. This used to be the other way around,
    and the yfinance path fabricated bid/ask as price ± $0.01 — a
    synthetic 2-cent spread with no relationship to the real market. Real
    Tastytrade SPY spreads observed during verification were ~2 cents too,
    but genuinely quoted, not invented.

    IV/IV-rank always come from the yfinance-based calculation below
    regardless of which source won the price — Tastytrade's REST snapshot
    endpoint (/market-data/by-type) doesn't expose those fields; getting
    real per-contract IV/Greeks from Tastytrade requires the DXLink
    streaming feed, which is a separate, larger integration (see
    options_pricing.py for the Black-Scholes Greeks used in the meantime).
    """
    q = _get_quote_tastytrade(symbol)
    if not q and _YF_AVAILABLE:
        q = _get_quote_yfinance_price(symbol)
    if not q:
        return None
    iv, iv_rank = _compute_iv_and_rank(symbol, q["price"])
    q["iv"] = iv
    q["iv_rank"] = iv_rank
    return q


def _compute_iv_and_rank(symbol: str, price: float) -> tuple:
    """
    Best-effort IV + IV-rank via yfinance's option chain and a 1yr
    realized-vol distribution. Split out from quote fetching so it's
    reused no matter which source (Tastytrade or yfinance) provided the
    actual price — Tastytrade's REST quote endpoint doesn't return IV.
    """
    iv_rank = 50.0
    iv = 0.25
    if not _YF_AVAILABLE or price <= 0:
        return iv, iv_rank
    try:
        ticker = yf.Ticker(symbol)
        opts = ticker.options
        if opts:
            exp = opts[0]
            chain = ticker.option_chain(exp)
            atm_calls = chain.calls
            atm_calls = atm_calls[abs(atm_calls["strike"] - price) < price * 0.05]
            if not atm_calls.empty:
                iv = float(atm_calls["impliedVolatility"].mean())

        # Compute 52-week IV rank via rolling HV distribution
        hist_1y = ticker.history(period="1y", interval="1d", auto_adjust=True)
        if hist_1y is not None and len(hist_1y) >= 60:
            log_ret = (hist_1y["Close"] / hist_1y["Close"].shift(1)).apply(
                lambda x: x if x > 0 else float("nan")
            ).apply(float).pipe(lambda s: s.map(lambda x: __import__("math").log(x) if x > 0 else float("nan")))
            # 21-day rolling realized vol → annualized
            rv_series = log_ret.rolling(21).std() * (252 ** 0.5)
            rv_series = rv_series.dropna()
            if len(rv_series) >= 20:
                rv_min  = float(rv_series.quantile(0.05))
                rv_max  = float(rv_series.quantile(0.95))
                if rv_max > rv_min:
                    iv_rank = round((iv - rv_min) / (rv_max - rv_min) * 100, 1)
                    iv_rank = max(0.0, min(100.0, iv_rank))
                else:
                    iv_rank = 50.0
    except Exception:
        pass
    return iv, iv_rank


def _get_quote_yfinance_price(symbol: str) -> dict:
    """Fallback price/bid/ask/volume from yfinance, used only if
    Tastytrade's own quote feed is unavailable. Still fabricates a
    synthetic 2-cent spread (yfinance doesn't provide real bid/ask for
    equities) — this is now the fallback case, not the default."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        hist = ticker.history(period="2d", interval="1d")
        price = float(info.last_price or 0)
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        volume = int(info.three_month_average_volume or info.last_volume or 0)
        return {
            "symbol": symbol,
            "price": price,
            "bid": price - 0.01,
            "ask": price + 0.01,
            "mid": price,
            "volume": volume,
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        print(f"⚠️  yfinance quote error for {symbol}: {e}")
        return None


def _get_quote_tastytrade(symbol: str) -> dict:
    """
    Real-time equity quote from Tastytrade's own market-data snapshot
    endpoint. This used to call /market-data/quotes, which returns a 404
    on this account/API version — verified live against the real account
    before this fix (see /market-data/by-type, which does work and returns
    real bid/ask/volume).
    """
    try:
        resp = session.get("/market-data/by-type", params={"equity[]": symbol})
        items = resp.get("data", {}).get("items", [])
        if not items:
            return None
        q = items[0]
        bid = float(q.get("bid", 0) or 0)
        ask = float(q.get("ask", 0) or 0)
        last = float(q.get("last", 0) or 0)
        mark = float(q.get("mark", 0) or 0)
        mid_ba = (bid + ask) / 2 if (bid or ask) else 0
        price = last or mark or mid_ba
        prev_close = float(q.get("prev-close", 0) or 0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        return {
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "mid": round((bid + ask) / 2, 4) if (bid or ask) else price,
            "volume": int(float(q.get("volume", 0) or 0)),
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        print(f"⚠️  Tastytrade quote error for {symbol}: {e}")
        return None


def get_quotes_batch(symbols: list) -> dict:
    """
    Batch-fetch quotes in a single Tastytrade API call instead of one
    round-trip per symbol, falling back to per-symbol get_quote() (which
    itself falls back to yfinance) for any symbol Tastytrade didn't
    return.
    """
    results = {}
    try:
        resp = session.get("/market-data/by-type", params={"equity[]": symbols})
        items = resp.get("data", {}).get("items", [])
        for q in items:
            sym = q.get("symbol")
            if not sym:
                continue
            bid = float(q.get("bid", 0) or 0)
            ask = float(q.get("ask", 0) or 0)
            last = float(q.get("last", 0) or 0)
            mark = float(q.get("mark", 0) or 0)
            mid_ba = (bid + ask) / 2 if (bid or ask) else 0
            price = last or mark or mid_ba
            prev_close = float(q.get("prev-close", 0) or 0)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            iv, iv_rank = _compute_iv_and_rank(sym, price)
            results[sym] = {
                "symbol": sym, "price": price, "bid": bid, "ask": ask,
                "mid": round((bid + ask) / 2, 4) if (bid or ask) else price,
                "volume": int(float(q.get("volume", 0) or 0)),
                "change_pct": round(change_pct, 2),
                "iv": iv, "iv_rank": iv_rank,
            }
    except Exception as e:
        print(f"⚠️  Batch quote error: {e}")

    for symbol in symbols:
        if symbol not in results:
            q = get_quote(symbol)
            if q:
                results[symbol] = q
    return results


# ── Option Chains (yfinance primary, Tastytrade fallback for live) ─────────────
def get_expirations(symbol: str) -> list:
    if _YF_AVAILABLE:
        return _get_expirations_yf(symbol)
    return _get_expirations_tastytrade(symbol)


def _get_expirations_yf(symbol: str) -> list:
    try:
        ticker = yf.Ticker(symbol)
        exps = ticker.options  # tuple of expiration date strings YYYY-MM-DD
        result = sorted(list(exps))
        print(f"[exp] {symbol}: {result[:5]}")
        return result
    except Exception as e:
        print(f"⚠️  Expiration error for {symbol}: {e}")
        return []


def _get_expirations_tastytrade(symbol: str) -> list:
    try:
        resp = session.get(f"/option-chains/{symbol}/expirations")
        expirations = resp.get("data", {}).get("items", [])
        dates = [exp.get("expiration-date") for exp in expirations if exp.get("expiration-date")]
        return sorted(dates)
    except Exception as e:
        print(f"⚠️  Expiration error for {symbol}: {e}")
        return []


def get_option_chain(symbol: str, expiration: str, option_type: str = None) -> list:
    if _YF_AVAILABLE:
        return _get_chain_yf(symbol, expiration, option_type)
    return _get_chain_tastytrade(symbol, expiration, option_type)


def _get_chain_yf(symbol: str, expiration: str, option_type: str = None) -> list:
    """Fetch option chain from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        chain = ticker.option_chain(expiration)
        price = ticker.fast_info.last_price or 0
        contracts = []

        def _safe_int(val):
            try:
                import math
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0
                return int(val)
            except Exception:
                return 0

        def _safe_float(val, default=0.0):
            try:
                import math
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return default
                return float(val)
            except Exception:
                return default

        def _parse(df, opt_type):
            if option_type and opt_type != option_type:
                return
            for _, row in df.iterrows():
                strike     = _safe_float(row.get("strike", 0))
                bid        = _safe_float(row.get("bid", 0))
                ask        = _safe_float(row.get("ask", 0))
                last_price = _safe_float(row.get("lastPrice", 0))
                # yfinance often returns NaN for bid/ask → use lastPrice as fallback
                if ask == 0 and last_price > 0:
                    ask = last_price
                    bid = round(last_price * 0.95, 2)
                mid    = (bid + ask) / 2 if (bid > 0 or ask > 0) else 0
                if mid == 0:
                    continue  # skip contracts with no pricing data
                iv     = _safe_float(row.get("impliedVolatility", 0.25)) * 100
                # Real Black-Scholes Greeks — this used to be a linear
                # moneyness formula standing in for delta (no IV or
                # time-to-expiry term at all) plus flat hardcoded constants
                # for gamma/theta/vega (0.02 / -mid*0.01 / mid*0.1 for
                # EVERY contract regardless of strike or expiration). Every
                # delta-targeted strike selection and every portfolio Greek
                # exposure check downstream was working off those fake
                # numbers. See options_pricing.py.
                dte_days = options_pricing.days_to_expiration(expiration)
                greeks = options_pricing.black_scholes_greeks(price, strike, dte_days, iv, opt_type)
                contracts.append({
                    "symbol":        f"{symbol}{expiration.replace('-','')}{opt_type}{int(strike*1000):08d}",
                    "underlying":    symbol,
                    "expiration":    expiration,
                    "strike":        strike,
                    "type":          opt_type,
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "volume":        _safe_int(row.get("volume", 0)),
                    "open_interest": _safe_int(row.get("openInterest", 0)),
                    "delta":         greeks["delta"],
                    "gamma":         greeks["gamma"],
                    "theta":         greeks["theta"],
                    "vega":          greeks["vega"],
                    "iv":            iv,
                })

        _parse(chain.calls, "C")
        _parse(chain.puts, "P")
        return contracts
    except Exception as e:
        print(f"⚠️  Option chain error for {symbol}: {e}")
        return []


def _get_chain_tastytrade(symbol: str, expiration: str, option_type: str = None) -> list:
    try:
        params = {"expiration-date": expiration}
        resp = session.get(f"/option-chains/{symbol}/nested", params=params)
        expirations = resp.get("data", {}).get("items", [])
        contracts = []
        for exp in expirations:
            if exp.get("expiration-date") != expiration:
                continue
            for strike_data in exp.get("strikes", []):
                strike = float(strike_data.get("strike-price", 0))
                for opt_type, key in [("C", "call"), ("P", "put")]:
                    if option_type and opt_type != option_type:
                        continue
                    contract = strike_data.get(key, {})
                    if not contract:
                        continue
                    contracts.append({
                        "symbol":        contract.get("symbol", ""),
                        "underlying":    symbol,
                        "expiration":    expiration,
                        "strike":        strike,
                        "type":          opt_type,
                        "bid":           float(contract.get("bid", 0)),
                        "ask":           float(contract.get("ask", 0)),
                        "mid":           (float(contract.get("bid", 0)) + float(contract.get("ask", 0))) / 2,
                        "volume":        int(contract.get("volume", 0)),
                        "open_interest": int(contract.get("open-interest", 0)),
                        "delta":         float(contract.get("delta", 0)),
                        "gamma":         float(contract.get("gamma", 0)),
                        "theta":         float(contract.get("theta", 0)),
                        "vega":          float(contract.get("vega", 0)),
                        "iv":            float(contract.get("implied-volatility", 0)) * 100,
                    })
        return contracts
    except Exception as e:
        print(f"⚠️  Option chain error for {symbol}: {e}")
        return []


# ── Expiration Pickers ─────────────────────────────────────────────────────────
def pick_expiration(symbol: str, min_dte: int = 7, max_dte: int = 30) -> str:
    expirations = get_expirations(symbol)
    today = date.today()
    candidates = []
    for exp_str in expirations:
        try:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                candidates.append((dte, exp_str))
        except ValueError:
            continue
    if not candidates:
        print(f"[exp] {symbol}: no expiration in {min_dte}-{max_dte} DTE (today={today}, checked {len(expirations)} dates)")
        return None
    target_dte = (min_dte + max_dte) / 2
    candidates.sort(key=lambda x: abs(x[0] - target_dte))
    return candidates[0][1]


def pick_0dte_expiration(symbol: str) -> str:
    """Pick today's or nearest expiration (0-1 DTE) for 0DTE scalping."""
    expirations = get_expirations(symbol)
    today = date.today()
    candidates = []
    for exp_str in expirations:
        try:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if 0 <= dte <= 1:
                candidates.append((dte, exp_str))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ── Strike Finders ─────────────────────────────────────────────────────────────
def find_strikes(contracts: list, target_delta: float, option_type: str) -> dict:
    typed = [c for c in contracts if c["type"] == option_type]
    if not typed:
        return None
    if option_type == "P":
        target_delta = -abs(target_delta)
    return min(typed, key=lambda c: abs(c["delta"] - target_delta))


def find_atm_strike(contracts: list, price: float, option_type: str) -> dict:
    """Find the at-the-money contract."""
    typed = [c for c in contracts if c["type"] == option_type]
    if not typed:
        return None
    return min(typed, key=lambda c: abs(c["strike"] - price))


# ── Strategy Builders ──────────────────────────────────────────────────────────

def build_long_option(symbol: str, direction: str, expiration: str) -> dict:
    """Long call or put. Best when IV is low and big directional move expected."""
    opt_type = "C" if direction == "bullish" else "P"
    target_delta = 0.45 if direction == "bullish" else -0.45
    contracts = get_option_chain(symbol, expiration, opt_type)
    print(f"[build_long] {symbol} {direction} {expiration}: {len(contracts)} contracts")
    if not contracts:
        return None
    contract = find_strikes(contracts, abs(target_delta), opt_type)
    if not contract:
        return None
    cost = contract["ask"] * 100
    print(f"[build_long] {symbol} {direction}: strike={contract['strike']} ask={contract['ask']} cost=${cost:.0f}")
    if cost > 2000 or cost <= 0:
        print(f"[build_long] {symbol} {direction}: FILTERED cost ${cost:.0f} > 2000 or <= 0")
        return None
    return {
        "type": "long_option",
        "strategy_name": f"Long {opt_type} ({direction.title()})",
        "direction": direction,
        "symbol": symbol,
        "expiration": expiration,
        "opt_type": opt_type,
        "contract": contract,
        "max_loss": round(cost, 2),
        "max_profit": None,
        "delta": contract["delta"],
        "iv": contract["iv"],
    }


def build_debit_spread(symbol: str, direction: str, expiration: str) -> dict:
    """Vertical debit spread. Defined risk, directional. Best for moderate IV."""
    opt_type = "C" if direction == "bullish" else "P"
    contracts = get_option_chain(symbol, expiration, opt_type)
    if not contracts:
        return None
    quote = get_quote(symbol)
    if not quote:
        return None

    if direction == "bullish":
        short_leg = find_strikes(contracts, 0.35, "C")
        if not short_leg:
            return None
        long_candidates = [c for c in contracts if c["strike"] < short_leg["strike"]]
        if not long_candidates:
            return None
        long_leg = max(long_candidates, key=lambda c: c["strike"])
    else:
        short_leg = find_strikes(contracts, -0.35, "P")
        if not short_leg:
            return None
        long_candidates = [c for c in contracts if c["strike"] > short_leg["strike"]]
        if not long_candidates:
            return None
        long_leg = min(long_candidates, key=lambda c: c["strike"])

    width = abs(long_leg["strike"] - short_leg["strike"])
    net_debit = long_leg["ask"] - short_leg["bid"]
    max_profit = (width - net_debit) * 100
    max_loss = net_debit * 100

    if max_loss > 1000 or max_loss <= 0:
        return None
    if max_profit / max_loss < 0.5:
        return None

    return {
        "type": "spread",
        "strategy_name": f"{'Bull Call' if direction == 'bullish' else 'Bear Put'} Spread",
        "direction": direction,
        "symbol": symbol,
        "expiration": expiration,
        "opt_type": opt_type,
        "long_leg": long_leg,
        "short_leg": short_leg,
        "width": width,
        "net_debit": round(net_debit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "risk_reward": round(max_profit / max_loss, 2),
    }


def build_credit_spread(symbol: str, direction: str, expiration: str) -> dict:
    """
    Vertical credit spread (sell premium). Best when IV is high.
    Bullish = sell put spread (bull put spread).
    Bearish = sell call spread (bear call spread).
    """
    if direction == "bullish":
        opt_type = "P"
        # Sell OTM put, buy further OTM put below it
        short_leg = find_strikes(get_option_chain(symbol, expiration, "P"), -0.30, "P")
        if not short_leg:
            return None
        contracts = get_option_chain(symbol, expiration, "P")
        long_candidates = [c for c in contracts if c["strike"] < short_leg["strike"]]
        if not long_candidates:
            return None
        long_leg = max(long_candidates, key=lambda c: c["strike"])
    else:
        opt_type = "C"
        # Sell OTM call, buy further OTM call above it
        short_leg = find_strikes(get_option_chain(symbol, expiration, "C"), 0.30, "C")
        if not short_leg:
            return None
        contracts = get_option_chain(symbol, expiration, "C")
        long_candidates = [c for c in contracts if c["strike"] > short_leg["strike"]]
        if not long_candidates:
            return None
        long_leg = min(long_candidates, key=lambda c: c["strike"])

    width = abs(long_leg["strike"] - short_leg["strike"])
    net_credit = short_leg["bid"] - long_leg["ask"]
    max_profit = net_credit * 100
    max_loss = (width - net_credit) * 100

    if max_loss > 1000 or max_loss <= 0 or net_credit <= 0:
        return None
    if max_profit / max_loss < 0.25:  # need at least 25% return on risk
        return None

    return {
        "type": "credit_spread",
        "strategy_name": f"{'Bull Put' if direction == 'bullish' else 'Bear Call'} Credit Spread",
        "direction": direction,
        "symbol": symbol,
        "expiration": expiration,
        "opt_type": opt_type,
        "short_leg": short_leg,
        "long_leg": long_leg,
        "width": width,
        "net_credit": round(net_credit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "risk_reward": round(max_profit / max_loss, 2),
    }


def build_iron_condor(symbol: str, expiration: str) -> dict:
    """
    Iron condor: sell OTM put spread + sell OTM call spread.
    Best when IV is high and market is expected to stay range-bound.
    """
    put_contracts = get_option_chain(symbol, expiration, "P")
    call_contracts = get_option_chain(symbol, expiration, "C")
    quote = get_quote(symbol)
    if not put_contracts or not call_contracts or not quote:
        return None

    # Sell ~0.20 delta put and call (OTM)
    short_put = find_strikes(put_contracts, -0.20, "P")
    short_call = find_strikes(call_contracts, 0.20, "C")
    if not short_put or not short_call:
        return None

    # Buy further OTM legs for defined risk
    long_put_candidates = [c for c in put_contracts if c["strike"] < short_put["strike"]]
    long_call_candidates = [c for c in call_contracts if c["strike"] > short_call["strike"]]
    if not long_put_candidates or not long_call_candidates:
        return None

    long_put = max(long_put_candidates, key=lambda c: c["strike"])
    long_call = min(long_call_candidates, key=lambda c: c["strike"])

    put_width = abs(short_put["strike"] - long_put["strike"])
    call_width = abs(short_call["strike"] - long_call["strike"])

    put_credit = short_put["bid"] - long_put["ask"]
    call_credit = short_call["bid"] - long_call["ask"]
    total_credit = put_credit + call_credit

    max_profit = total_credit * 100
    max_loss = (max(put_width, call_width) - total_credit) * 100

    if max_loss > 1000 or max_loss <= 0 or total_credit <= 0:
        return None
    if max_profit / max_loss < 0.20:
        return None

    return {
        "type": "iron_condor",
        "strategy_name": "Iron Condor",
        "direction": "neutral",
        "symbol": symbol,
        "expiration": expiration,
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
        "put_width": put_width,
        "call_width": call_width,
        "net_credit": round(total_credit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "risk_reward": round(max_profit / max_loss, 2),
        "breakevens": {
            "lower": round(short_put["strike"] - total_credit, 2),
            "upper": round(short_call["strike"] + total_credit, 2),
        },
    }


def build_strangle(symbol: str, direction: str, expiration: str) -> dict:
    """
    Long strangle: buy OTM call + OTM put.
    Best when IV is low and a big move is expected (earnings, catalyst).
    direction is 'neutral' but we build both sides.
    """
    put_contracts = get_option_chain(symbol, expiration, "P")
    call_contracts = get_option_chain(symbol, expiration, "C")
    if not put_contracts or not call_contracts:
        return None

    # Buy ~0.25 delta options (OTM)
    long_call = find_strikes(call_contracts, 0.25, "C")
    long_put = find_strikes(put_contracts, -0.25, "P")
    if not long_call or not long_put:
        return None

    total_cost = (long_call["ask"] + long_put["ask"]) * 100
    if total_cost > 2000 or total_cost <= 0:
        return None

    return {
        "type": "strangle",
        "strategy_name": "Long Strangle",
        "direction": "neutral",
        "symbol": symbol,
        "expiration": expiration,
        "long_call": long_call,
        "long_put": long_put,
        "max_loss": round(total_cost, 2),
        "max_profit": None,
        "breakeven_upper": round(long_call["strike"] + (total_cost / 100), 2),
        "breakeven_lower": round(long_put["strike"] - (total_cost / 100), 2),
    }


def build_0dte_scalp(symbol: str, direction: str, expiration: str) -> dict:
    """
    0DTE scalp: buy ATM or near-ATM call/put with same-day expiration.
    High gamma plays — small moves can produce large % gains.
    Only for SPY, QQQ, IWM which have daily expirations.
    """
    opt_type = "C" if direction == "bullish" else "P"
    target_delta = 0.50 if direction == "bullish" else -0.50
    contracts = get_option_chain(symbol, expiration, opt_type)
    if not contracts:
        return None

    contract = find_strikes(contracts, abs(target_delta), opt_type)
    if not contract:
        return None

    cost = contract["ask"] * 100
    # 0DTE max risk is tighter — we want to risk less per trade
    if cost > 1000 or cost <= 0:
        return None

    return {
        "type": "0dte_scalp",
        "strategy_name": f"0DTE {opt_type} Scalp ({direction.title()})",
        "direction": direction,
        "symbol": symbol,
        "expiration": expiration,
        "opt_type": opt_type,
        "contract": contract,
        "max_loss": round(cost, 2),
        "max_profit": None,
        "delta": contract["delta"],
        "gamma": contract["gamma"],
        "theta": contract["theta"],
        "iv": contract["iv"],
    }


# ── Market Snapshot ────────────────────────────────────────────────────────────
def get_market_snapshot() -> list:
    """
    Build a full snapshot of all watchlist symbols with all applicable strategies.
    Returns a list of setups ready to send to Claude for ranking.
    """
    snapshots = []
    quotes = get_quotes_batch(WATCHLIST)
    today = date.today()
    now = datetime.now().time()

    for symbol, quote in quotes.items():
        iv_rank = quote.get("iv_rank", 0)
        change_pct = quote.get("change_pct", 0)
        price = quote.get("price", 0)
        print(f"[snapshot] {symbol}: price=${price:.2f} iv_rank={iv_rank:.0f} chg={change_pct:+.2f}%")
        setups = []

        # ── 0DTE scalps (SPY, QQQ, IWM only, during market hours) ────────────
        if symbol in ZERO_DTE_SYMBOLS:
            zero_exp = pick_0dte_expiration(symbol)
            if zero_exp:
                dte_0 = (date.fromisoformat(zero_exp) - today).days
                # Bullish if up >0.3%, bearish if down >0.3%, both if flat
                if change_pct > 0.3:
                    s = build_0dte_scalp(symbol, "bullish", zero_exp)
                    if s:
                        snapshots.append({"quote": quote, "setup": s, "dte": dte_0,
                                          "expiration": zero_exp, "iv_rank": iv_rank})
                elif change_pct < -0.3:
                    s = build_0dte_scalp(symbol, "bearish", zero_exp)
                    if s:
                        snapshots.append({"quote": quote, "setup": s, "dte": dte_0,
                                          "expiration": zero_exp, "iv_rank": iv_rank})
                else:
                    for d in ["bullish", "bearish"]:
                        s = build_0dte_scalp(symbol, d, zero_exp)
                        if s:
                            snapshots.append({"quote": quote, "setup": s, "dte": dte_0,
                                              "expiration": zero_exp, "iv_rank": iv_rank})

        # ── Standard expirations (7-21 DTE) ───────────────────────────────────
        expiration = pick_expiration(symbol, min_dte=7, max_dte=21)
        if not expiration:
            print(f"[snapshot] {symbol}: no expiration found, skipping")
            continue
        dte = (date.fromisoformat(expiration) - today).days

        if iv_rank >= 60:
            # High IV: favor selling premium
            # Iron condor (neutral / range-bound)
            ic = build_iron_condor(symbol, expiration)
            if ic:
                snapshots.append({"quote": quote, "setup": ic, "dte": dte,
                                  "expiration": expiration, "iv_rank": iv_rank})

            # Credit spreads (directional bias from price action)
            for direction in (["bullish"] if change_pct > 0 else ["bearish"] if change_pct < 0 else ["bullish", "bearish"]):
                cs = build_credit_spread(symbol, direction, expiration)
                if cs:
                    snapshots.append({"quote": quote, "setup": cs, "dte": dte,
                                      "expiration": expiration, "iv_rank": iv_rank})

        elif iv_rank >= 35:
            # Moderate IV: debit spreads
            for direction in ["bullish", "bearish"]:
                ds = build_debit_spread(symbol, direction, expiration)
                if ds:
                    snapshots.append({"quote": quote, "setup": ds, "dte": dte,
                                      "expiration": expiration, "iv_rank": iv_rank})

        else:
            # Low IV: buy options or strangles (expecting a move)
            for direction in ["bullish", "bearish"]:
                lo = build_long_option(symbol, direction, expiration)
                if lo:
                    snapshots.append({"quote": quote, "setup": lo, "dte": dte,
                                      "expiration": expiration, "iv_rank": iv_rank})

            # Strangle if IV is very low (expecting catalyst)
            if iv_rank < 20:
                st = build_strangle(symbol, "neutral", expiration)
                if st:
                    snapshots.append({"quote": quote, "setup": st, "dte": dte,
                                      "expiration": expiration, "iv_rank": iv_rank})

    return snapshots


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from auth import initialize
    initialize()

    print("Fetching market snapshot...")
    snapshots = get_market_snapshot()
    print(f"Found {len(snapshots)} potential setups:")
    for s in snapshots:
        setup = s["setup"]
        print(f"  {setup['symbol']} | {setup.get('strategy_name','?')} | "
              f"IV Rank: {s['iv_rank']:.0f} | DTE: {s['dte']} | "
              f"Max Loss: ${setup['max_loss']:.0f} | "
              f"Max Profit: ${setup.get('max_profit') or 'unlimited'}")
