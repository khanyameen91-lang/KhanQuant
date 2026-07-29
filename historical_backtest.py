"""
historical_backtest.py — Historical strategy replay.

WHAT THIS IS, PRECISELY
-----------------------
backtester.py (the existing module) is a performance REPORT over trades the
bot has already taken live. It cannot evaluate a strategy or a parameter
change before risking money on it, because it only knows about trades that
already happened.

This module replays a strategy over historical market data instead, so a
change can be checked before it goes live.

WHAT IS REAL vs MODELED — read this before trusting a number
------------------------------------------------------------
REAL:    underlying price paths (actual historical daily OHLCV), realized
         volatility computed from those prices, trading calendar, the
         strategy's own entry/exit rules, commissions.
MODELED: option prices. Historical option chains are not available from
         the data sources this bot uses (yfinance serves only CURRENT
         chains), so entry and exit premiums are computed with
         Black-Scholes using trailing realized volatility as an implied-
         volatility proxy.

That last point is a material limitation and the results carry it:
  * Real IV almost always exceeds realized vol (the variance risk
    premium), so premium-SELLING strategies (credit spreads, iron
    condors) are priced CONSERVATIVELY here — they would typically
    collect more premium in reality than this model credits them.
  * Premium-BUYING strategies (long options, debit spreads) are
    correspondingly priced optimistically — they'd usually pay more.
  * IV crush around earnings and event-driven vol spikes are not
    modeled at all.

So: use this to compare strategies and parameters against each other, and
to catch outright broken logic. Do not read the absolute P&L as a
prediction of live results.
"""

import os
import math
import statistics
from datetime import date, datetime, timedelta

import options_pricing

try:
    import yfinance as yf
    import pandas as pd
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── Cost model ───────────────────────────────────────────────────────────────
# Tastytrade: $1.00/contract to open, $0 to close on equity options.
COMMISSION_PER_CONTRACT_OPEN = float(os.environ.get("COMMISSION_PER_CONTRACT", 1.00))
COMMISSION_PER_CONTRACT_CLOSE = 0.0
# Slippage as a fraction of the option's theoretical price, applied
# against you on both entry and exit. Real fills land inside a bid/ask
# spread; assuming mid-price fills is the single most common way a
# backtest flatters itself.
SLIPPAGE_PCT = float(os.environ.get("BACKTEST_SLIPPAGE_PCT", 0.02))

CONTRACT_MULTIPLIER = 100


def _realized_vol(closes, lookback: int = 21) -> float:
    """Annualized realized volatility (%) over the trailing window — the
    stand-in for implied volatility. Returns a percentage to match the
    units options_pricing.black_scholes_greeks expects."""
    if len(closes) < lookback + 1:
        return 20.0
    window = closes[-(lookback + 1):]
    rets = [math.log(window[i] / window[i - 1])
            for i in range(1, len(window)) if window[i - 1] > 0]
    if len(rets) < 2:
        return 20.0
    return max(5.0, min(150.0, statistics.stdev(rets) * math.sqrt(252) * 100))


def _price_option(spot, strike, dte, iv_pct, opt_type) -> float:
    return options_pricing.black_scholes_greeks(spot, strike, dte, iv_pct, opt_type)["price"]


def _find_strike_by_delta(spot, dte, iv_pct, opt_type, target_delta, step=1.0):
    """Pick the strike whose Black-Scholes delta is closest to target.
    Mirrors how market_data.find_strikes() selects live strikes, so the
    backtest exercises the same strike-selection logic the bot uses."""
    best, best_err = None, 1e9
    lo, hi = spot * 0.75, spot * 1.25
    k = round(lo / step) * step
    while k <= hi:
        d = options_pricing.black_scholes_greeks(spot, k, dte, iv_pct, opt_type)["delta"]
        err = abs(abs(d) - abs(target_delta))
        if err < best_err:
            best, best_err = k, err
        k += step
    return best


# ── Strategy definitions ─────────────────────────────────────────────────────
# Delta targets mirror market_data.py's live builders so the replay tests
# the same strategy shapes the bot actually trades.
STRATEGY_SPECS = {
    "long_option":   {"kind": "single", "delta": 0.45, "debit": True},
    "debit_spread":  {"kind": "vertical", "long_delta": 0.45, "short_delta": 0.35, "debit": True},
    "credit_spread": {"kind": "vertical", "short_delta": 0.30, "long_delta": 0.20, "debit": False},
    "iron_condor":   {"kind": "condor", "short_delta": 0.20, "long_delta": 0.10},
}


def _position_value(spot, legs, dte, iv_pct) -> float:
    """Net value of a position's legs, per share (multiply by 100 for $)."""
    total = 0.0
    for leg in legs:
        px = _price_option(spot, leg["strike"], dte, iv_pct, leg["type"])
        total += px * leg["qty"]   # qty is +1 long, -1 short
    return total


def _build_legs(spec, spot, dte, iv_pct, direction):
    """Construct the position's legs for a given strategy + direction."""
    kind = spec["kind"]

    if kind == "single":
        t = "C" if direction == "bullish" else "P"
        k = _find_strike_by_delta(spot, dte, iv_pct, t, spec["delta"])
        return [{"strike": k, "type": t, "qty": 1}]

    if kind == "vertical":
        if spec["debit"]:
            t = "C" if direction == "bullish" else "P"
            kl = _find_strike_by_delta(spot, dte, iv_pct, t, spec["long_delta"])
            ks = _find_strike_by_delta(spot, dte, iv_pct, t, spec["short_delta"])
            return [{"strike": kl, "type": t, "qty": 1},
                    {"strike": ks, "type": t, "qty": -1}]
        # credit spread: sell closer to the money, buy further out for protection
        t = "P" if direction == "bullish" else "C"
        ks = _find_strike_by_delta(spot, dte, iv_pct, t, spec["short_delta"])
        kl = _find_strike_by_delta(spot, dte, iv_pct, t, spec["long_delta"])
        return [{"strike": ks, "type": t, "qty": -1},
                {"strike": kl, "type": t, "qty": 1}]

    if kind == "condor":
        sp = _find_strike_by_delta(spot, dte, iv_pct, "P", spec["short_delta"])
        lp = _find_strike_by_delta(spot, dte, iv_pct, "P", spec["long_delta"])
        sc = _find_strike_by_delta(spot, dte, iv_pct, "C", spec["short_delta"])
        lc = _find_strike_by_delta(spot, dte, iv_pct, "C", spec["long_delta"])
        return [{"strike": sp, "type": "P", "qty": -1},
                {"strike": lp, "type": "P", "qty": 1},
                {"strike": sc, "type": "C", "qty": -1},
                {"strike": lc, "type": "C", "qty": 1}]

    raise ValueError(f"unknown strategy kind: {kind}")


def backtest_strategy(
    symbol: str = "SPY",
    strategy: str = "credit_spread",
    period: str = "2y",
    dte: int = 14,
    hold_days: int = 7,
    profit_target_pct: float = 0.50,
    stop_loss_pct: float = 0.75,
    entry_every_n_days: int = 5,
    direction_rule: str = "trend",
) -> dict:
    """
    Replay `strategy` on `symbol` over `period` of real historical prices.

    direction_rule:
      "trend"   — bullish when spot is above its 20-day average, else bearish
      "bullish" / "bearish" — fixed
      "neutral" — for condors (direction is irrelevant)

    Exits on whichever comes first: profit target, stop loss, or hold_days.
    Both are measured against max profit / max loss the same way
    executor.check_exits() measures them live.
    """
    if not _DEPS_OK:
        return {"status": "deps_unavailable", "detail": "yfinance/pandas required"}
    if strategy not in STRATEGY_SPECS:
        return {"status": "unknown_strategy", "detail": strategy}

    spec = STRATEGY_SPECS[strategy]
    hist = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if hist is None or len(hist) < 60:
        return {"status": "insufficient_history", "bars": 0 if hist is None else len(hist)}

    closes = [float(c) for c in hist["Close"].tolist()]
    dates = [d.date() if hasattr(d, "date") else d for d in hist.index.tolist()]

    trades = []
    i = 40  # need lookback for realized vol + trend
    while i < len(closes) - hold_days - 1:
        spot = closes[i]
        iv = _realized_vol(closes[: i + 1])

        if direction_rule == "trend":
            sma20 = sum(closes[i - 20:i]) / 20
            direction = "bullish" if spot >= sma20 else "bearish"
        elif direction_rule in ("bullish", "bearish"):
            direction = direction_rule
        else:
            direction = "neutral"

        legs = _build_legs(spec, spot, dte, iv, direction)

        entry_val = _position_value(spot, legs, dte, iv)
        is_debit = entry_val > 0

        # Slippage works against you in both directions: you pay more for
        # a debit, you collect less for a credit.
        slip = abs(entry_val) * SLIPPAGE_PCT
        entry_fill = entry_val + slip if is_debit else entry_val + slip
        commission = COMMISSION_PER_CONTRACT_OPEN * len(legs)

        # Max profit / max loss for defined-risk structures
        if is_debit:
            widths = [abs(legs[0]["strike"] - legs[1]["strike"])] if len(legs) > 1 else [None]
            width = widths[0]
            max_loss = abs(entry_fill) * CONTRACT_MULTIPLIER + commission
            max_profit = ((width - abs(entry_fill)) * CONTRACT_MULTIPLIER
                          if width else abs(entry_fill) * CONTRACT_MULTIPLIER * 2)
        else:
            credit = abs(entry_fill)
            if spec["kind"] == "condor":
                width = abs(legs[0]["strike"] - legs[1]["strike"])
            else:
                width = abs(legs[0]["strike"] - legs[1]["strike"])
            max_profit = credit * CONTRACT_MULTIPLIER
            max_loss = max((width - credit) * CONTRACT_MULTIPLIER, 1.0) + commission

        # Walk forward day by day until an exit rule fires
        exit_reason, exit_idx, pnl = "time", i + hold_days, 0.0
        for j in range(1, hold_days + 1):
            k = i + j
            if k >= len(closes):
                break
            spot_j = closes[k]
            dte_j = max(dte - j, 0)
            iv_j = _realized_vol(closes[: k + 1])
            val_j = _position_value(spot_j, legs, dte_j, iv_j)

            # Mark-to-market P&L before exit costs
            if is_debit:
                gross = (val_j - entry_fill) * CONTRACT_MULTIPLIER
            else:
                gross = (abs(entry_fill) - abs(val_j)) * CONTRACT_MULTIPLIER
                if val_j > 0:   # short position moved against us
                    gross = (abs(entry_fill) - val_j) * CONTRACT_MULTIPLIER

            if gross >= max_profit * profit_target_pct:
                exit_reason, exit_idx, pnl = "profit_target", k, gross
                break
            if gross <= -(max_loss * stop_loss_pct):
                exit_reason, exit_idx, pnl = "stop_loss", k, gross
                break
            exit_idx, pnl = k, gross

        exit_slip = abs(pnl) * SLIPPAGE_PCT * 0.5
        net = pnl - exit_slip - commission - (COMMISSION_PER_CONTRACT_CLOSE * len(legs))

        trades.append({
            "entry_date": str(dates[i]),
            "exit_date": str(dates[min(exit_idx, len(dates) - 1)]),
            "direction": direction,
            "iv_used": round(iv, 1),
            "entry_value": round(entry_fill, 4),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "pnl": round(net, 2),
            "exit_reason": exit_reason,
        })
        i += entry_every_n_days

    return _summarize(symbol, strategy, trades, period)


def _summarize(symbol, strategy, trades, period) -> dict:
    if not trades:
        return {"status": "no_trades", "symbol": symbol, "strategy": strategy}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    total = sum(pnls)

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    mean = total / len(pnls)
    sd = statistics.stdev(pnls) if len(pnls) > 1 else 0.0
    # Per-trade Sharpe — deliberately NOT annualized. ml_engine.py's
    # existing metric multiplies by sqrt(252) as though each trade were a
    # trading day, which massively inflates it at ~1 trade/day or fewer.
    sharpe = (mean / sd) if sd > 0 else 0.0

    reasons = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    return {
        "status": "ok",
        "symbol": symbol,
        "strategy": strategy,
        "period": period,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "total_pnl": round(total, 2),
        "avg_pnl": round(mean, 2),
        "profit_factor": round(sum(wins) / sum(losses), 2) if losses else float("inf"),
        "sharpe_per_trade": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "exit_breakdown": reasons,
        "costs_modeled": {
            "commission_per_contract_open": COMMISSION_PER_CONTRACT_OPEN,
            "slippage_pct": SLIPPAGE_PCT,
        },
        "caveat": ("Option prices are Black-Scholes MODELED from realized vol, not "
                   "historical option quotes. Premium-selling results are conservative; "
                   "premium-buying results are optimistic. Compare strategies to each "
                   "other, do not read absolute P&L as a live-results prediction."),
        "sample_trades": trades[:5],
    }


def compare_strategies(symbol: str = "SPY", period: str = "2y", **kw) -> dict:
    """Run every strategy over the same window so they can be ranked
    against each other — the comparison this module is actually good for,
    given the modeled-pricing caveat above."""
    out = {}
    for strat in STRATEGY_SPECS:
        rule = "neutral" if strat == "iron_condor" else "trend"
        out[strat] = backtest_strategy(symbol=symbol, strategy=strat,
                                        period=period, direction_rule=rule, **kw)
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Historical strategy replay")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--strategy", default=None, help="omit to compare all")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--dte", type=int, default=14)
    ap.add_argument("--hold-days", type=int, default=7)
    args = ap.parse_args()

    if args.strategy:
        res = backtest_strategy(symbol=args.symbol, strategy=args.strategy,
                                 period=args.period, dte=args.dte,
                                 hold_days=args.hold_days)
        print(json.dumps(res, indent=2, default=str))
    else:
        results = compare_strategies(symbol=args.symbol, period=args.period,
                                      dte=args.dte, hold_days=args.hold_days)
        print(f"\n{'Strategy':<16}{'Trades':>7}{'Win%':>8}{'Total P&L':>12}"
              f"{'PF':>7}{'Sharpe':>9}{'MaxDD':>10}")
        print("-" * 69)
        for name, r in results.items():
            if r.get("status") != "ok":
                print(f"{name:<16}{r.get('status')}")
                continue
            print(f"{name:<16}{r['trades']:>7}{r['win_rate']:>8}"
                  f"{r['total_pnl']:>12}{r['profit_factor']:>7}"
                  f"{r['sharpe_per_trade']:>9}{r['max_drawdown']:>10}")
        print(f"\n{results[list(results)[0]]['caveat']}\n")
