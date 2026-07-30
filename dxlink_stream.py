"""
dxlink_stream.py — Real-time quotes and Greeks via Tastytrade's DXLink feed.

WHY THIS EXISTS
---------------
Everything else in this bot gets option data from yfinance: delayed,
no real bid/ask for many strikes, and no Greeks at all (options_pricing.py
computes Black-Scholes Greeks as a stand-in). Tastytrade provides a real
DXLink/dxFeed streaming feed carrying genuine NBBO quotes AND
broker-calculated Greeks per contract. This account has valid
entitlements for it (verified against /api-quote-tokens).

MEMORY DISCIPLINE — read before changing anything here
------------------------------------------------------
This runs on a 956MB instance with no swap, alongside bot.py (~150MB).
Yesterday the dashboard was OOM-killed seven times for importing pandas
into a web handler. This module is therefore built to a strict budget:

  * stdlib + `websockets` only. No pandas, no numpy, no yfinance.
  * MAX_SUBSCRIPTIONS caps tracked symbols. An options chain has
    thousands of contracts; subscribing to all of them would grow the
    snapshot dict without bound.
  * Snapshots are last-value-only (one small dict per symbol). No tick
    history, no accumulating buffers — that is what would actually eat
    the box.

Run standalone to smoke-test:  python3 dxlink_stream.py --symbols SPY,QQQ
"""

import os
import ssl
import json
import time
import asyncio
import logging
import threading
from datetime import datetime

log = logging.getLogger("dxlink")

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

# Hard ceiling on tracked symbols. Protects both memory and the feed's
# own subscription limits.
MAX_SUBSCRIPTIONS = int(os.environ.get("DXLINK_MAX_SUBSCRIPTIONS", 200))
KEEPALIVE_SEC = 30
# Consider a snapshot stale past this age; callers should fall back to
# their existing data source rather than trade on a frozen quote.
SNAPSHOT_MAX_AGE_SEC = float(os.environ.get("DXLINK_MAX_AGE_SEC", 30))

_QUOTE_FIELDS = ["eventType", "eventSymbol", "bidPrice", "askPrice", "bidSize", "askSize"]
_GREEKS_FIELDS = ["eventType", "eventSymbol", "price", "volatility",
                  "delta", "gamma", "theta", "vega", "rho"]


def get_streamer_token() -> dict:
    """Fetch a DXLink token + websocket URL from Tastytrade."""
    import auth
    resp = auth.session.get("/api-quote-tokens")
    data = resp.get("data", {})
    return {"token": data.get("token"), "url": data.get("dxlink-url")}


class DXLinkClient:
    """
    Streaming client maintaining a last-value snapshot per symbol.

    Thread-safe for readers: get_quote()/get_greeks() take a lock and
    return copies, so the bot's synchronous scan loop can read while the
    asyncio loop writes from its own thread.
    """

    def __init__(self, token: str = None, url: str = None):
        self._token = token
        self._url = url
        self._quotes = {}     # symbol -> {bid, ask, bid_size, ask_size, ts}
        self._greeks = {}     # symbol -> {delta, gamma, theta, vega, iv, ts}
        # event type -> ordered field names, supplied by FEED_CONFIG.
        # COMPACT payloads carry no field names, so without this the
        # data is undecodable.
        self._event_fields = {}
        self._lock = threading.Lock()
        self._subscribed = set()
        self._pending = set()
        self._ws = None
        self._loop = None
        self._thread = None
        self._running = False
        self._connected = threading.Event()
        self._channel = 1
        self.last_error = None
        self.connect_count = 0

    # ── Public, synchronous API (safe to call from the bot's scan loop) ──

    def start(self, timeout: float = 20.0) -> bool:
        """Start the background stream. Returns True once authorized."""
        if not _WS_AVAILABLE:
            self.last_error = "websockets package not installed"
            return False
        if self._running:
            return self._connected.is_set()

        if not self._token or not self._url:
            try:
                creds = get_streamer_token()
                self._token, self._url = creds["token"], creds["url"]
            except Exception as e:
                self.last_error = f"token fetch failed: {e}"
                return False

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name="dxlink")
        self._thread.start()
        return self._connected.wait(timeout)

    def stop(self, timeout: float = 5.0):
        """Stop cleanly. Closes the socket from inside the event loop and
        lets the loop unwind on its own, rather than calling loop.stop()
        out from under pending tasks (which produced a burst of
        'Event loop is closed' / 'Task was destroyed but it is pending'
        noise on shutdown)."""
        self._running = False
        loop, ws = self._loop, self._ws
        if loop and ws:
            try:
                asyncio.run_coroutine_threadsafe(ws.close(), loop).result(timeout=2)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)
        self._connected.clear()

    def subscribe(self, streamer_symbols) -> int:
        """
        Add symbols to the feed. Takes dxFeed STREAMER symbols (e.g.
        ".SPY260729C500"), not OCC symbols — the option-chain endpoint
        returns these as call-streamer-symbol / put-streamer-symbol.
        Returns the number actually queued after the cap is applied.
        """
        syms = [s for s in streamer_symbols if s]
        with self._lock:
            room = MAX_SUBSCRIPTIONS - len(self._subscribed)
            if room <= 0:
                return 0
            new = [s for s in syms if s not in self._subscribed][:room]
            self._pending.update(new)
            self._subscribed.update(new)
        if new and self._loop and self._connected.is_set():
            asyncio.run_coroutine_threadsafe(self._flush_subscriptions(), self._loop)
        return len(new)

    def get_quote(self, streamer_symbol: str):
        """Latest quote, or None if absent/stale."""
        with self._lock:
            q = self._quotes.get(streamer_symbol)
            if not q:
                return None
            if time.time() - q["ts"] > SNAPSHOT_MAX_AGE_SEC:
                return None
            return dict(q)

    def get_greeks(self, streamer_symbol: str):
        """Latest broker-calculated Greeks, or None if absent/stale."""
        with self._lock:
            g = self._greeks.get(streamer_symbol)
            if not g:
                return None
            if time.time() - g["ts"] > SNAPSHOT_MAX_AGE_SEC:
                return None
            return dict(g)

    def status(self) -> dict:
        with self._lock:
            return {
                "available": _WS_AVAILABLE,
                "connected": self._connected.is_set(),
                "subscribed": len(self._subscribed),
                "max_subscriptions": MAX_SUBSCRIPTIONS,
                "quotes_cached": len(self._quotes),
                "greeks_cached": len(self._greeks),
                "connect_count": self.connect_count,
                "last_error": self.last_error,
            }

    # ── Internals ───────────────────────────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_forever())
        except Exception as e:
            self.last_error = str(e)
            log.error(f"dxlink loop exited: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _connect_forever(self):
        backoff = 2
        while self._running:
            try:
                await self._session()
                backoff = 2          # reset only after a clean session
            except Exception as e:
                self.last_error = str(e)
                log.warning(f"dxlink session ended: {e}; retry in {backoff}s")
                self._connected.clear()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _session(self):
        ssl_ctx = ssl.create_default_context()
        async with websockets.connect(self._url, ssl=ssl_ctx,
                                       max_size=2**20, ping_interval=None) as ws:
            self._ws = ws
            self.connect_count += 1

            await self._send(ws, {"type": "SETUP", "channel": 0,
                                   "version": "0.1-khanquant/1.0",
                                   "keepaliveTimeout": 60,
                                   "acceptKeepaliveTimeout": 60})
            await self._send(ws, {"type": "AUTH", "channel": 0, "token": self._token})

            keepalive = asyncio.create_task(self._keepalive(ws))
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    await self._handle(ws, json.loads(raw))
            finally:
                keepalive.cancel()
                self._connected.clear()

    async def _send(self, ws, msg):
        await ws.send(json.dumps(msg))

    async def _keepalive(self, ws):
        while True:
            await asyncio.sleep(KEEPALIVE_SEC)
            try:
                await self._send(ws, {"type": "KEEPALIVE", "channel": 0})
            except Exception:
                return

    async def _handle(self, ws, msg):
        mtype = msg.get("type")

        if mtype == "AUTH_STATE":
            if msg.get("state") == "AUTHORIZED":
                await self._send(ws, {"type": "CHANNEL_REQUEST",
                                       "channel": self._channel,
                                       "service": "FEED",
                                       "parameters": {"contract": "AUTO"}})
            elif msg.get("state") == "UNAUTHORIZED":
                self.last_error = "DXLink rejected the token"

        elif mtype == "CHANNEL_OPENED":
            await self._send(ws, {
                "type": "FEED_SETUP", "channel": self._channel,
                "acceptAggregationPeriod": 1,
                "acceptDataFormat": "COMPACT",
                "acceptEventFields": {"Quote": _QUOTE_FIELDS, "Greeks": _GREEKS_FIELDS},
            })

        elif mtype == "FEED_CONFIG":
            # The server declares the ACTUAL field order per event type
            # here, and it may differ from what we asked for. COMPACT
            # payloads carry no field names, so this mapping is the only
            # way to decode them — parse it before subscribing, and
            # subscribe only once the first config arrives.
            for event_type, fields in (msg.get("eventFields") or {}).items():
                self._event_fields[event_type] = fields
            if not self._connected.is_set():
                self._connected.set()
            await self._flush_subscriptions()

        elif mtype == "FEED_DATA":
            self._ingest(msg.get("data", []))

        elif mtype == "ERROR":
            self.last_error = f"{msg.get('error')}: {msg.get('message')}"
            log.error(f"dxlink error: {self.last_error}")

    async def _flush_subscriptions(self):
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        if not pending or not self._ws:
            return
        # Chunk so a large first subscribe doesn't build one huge frame.
        for i in range(0, len(pending), 100):
            chunk = pending[i:i + 100]
            add = []
            for s in chunk:
                add.append({"symbol": s, "type": "Quote"})
                add.append({"symbol": s, "type": "Greeks"})
            try:
                await self._send(self._ws, {"type": "FEED_SUBSCRIPTION",
                                             "channel": self._channel, "add": add})
            except Exception as e:
                log.warning(f"subscription flush failed: {e}")
                with self._lock:
                    self._pending.update(chunk)
                return

    def _ingest(self, data):
        """
        Parse COMPACT-format FEED_DATA.

        Verified against the live feed, the shape is:

            data = ["Quote", ["Quote", ".SPY260730P739", 2.86, 2.88,
                              "Quote", ".SPY260730C739", 1.47, 1.48, ...]]

        i.e. data[0] is the event type as a STRING, and data[1] is one
        flat list of records back-to-back with no delimiters. Field names
        are NOT in the payload — they come from the FEED_CONFIG message
        (self._event_fields), and record width is simply the number of
        declared fields.

        (An earlier version of this assumed data[0] was a list of field
        names. Because it's a string, every message failed an isinstance
        check and was skipped — the stream connected and subscribed
        cleanly while silently decoding nothing.)
        """
        now = time.time()
        try:
            if len(data) < 2 or not isinstance(data[0], str):
                return
            event_type = data[0]
            values = data[1]
            fields = self._event_fields.get(event_type)
            if not fields or not isinstance(values, list):
                return

            width = len(fields)
            idx = {name: pos for pos, name in enumerate(fields)}
            sym_pos = idx.get("eventSymbol", 1)

            for off in range(0, len(values) - width + 1, width):
                row = values[off:off + width]
                sym = row[sym_pos]
                if not isinstance(sym, str) or not sym:
                    continue

                def num(field):
                    p = idx.get(field)
                    if p is None:
                        return None
                    try:
                        f = float(row[p])
                        return None if f != f else f   # drop NaN
                    except (TypeError, ValueError):
                        return None

                if event_type == "Quote":
                    bid, ask = num("bidPrice"), num("askPrice")
                    if bid is None and ask is None:
                        continue
                    with self._lock:
                        self._quotes[sym] = {
                            "bid": bid, "ask": ask,
                            "bid_size": num("bidSize"), "ask_size": num("askSize"),
                            "mid": round((bid + ask) / 2, 4) if (bid and ask) else None,
                            "ts": now,
                        }
                elif event_type == "Greeks":
                    with self._lock:
                        self._greeks[sym] = {
                            "delta": num("delta"), "gamma": num("gamma"),
                            "theta": num("theta"), "vega": num("vega"),
                            "rho": num("rho"),
                            "iv": num("volatility"), "price": num("price"),
                            "ts": now,
                        }
        except Exception as e:
            log.warning(f"dxlink parse error: {e}")


def to_streamer_symbol(underlying: str, expiration: str, opt_type: str,
                        strike: float) -> str:
    """
    Build a dxFeed streamer symbol, e.g. ('SPY','2026-07-30','C',738.0)
    -> '.SPY260730C738'.

    Note the strike has no zero padding and drops a trailing '.0' —
    matching the format observed on the live feed. This exists so callers
    holding an OCC-style contract can look up streaming data without
    re-fetching the chain just to read call-streamer-symbol.
    """
    y, m, d = expiration.split("-")
    strike_str = f"{strike:g}"
    return f".{underlying.upper()}{y[2:]}{m}{d}{opt_type.upper()}{strike_str}"


# Module-level singleton so the bot keeps one connection, not one per call.
_client = None


def get_client() -> DXLinkClient:
    global _client
    if _client is None:
        _client = DXLinkClient()
    return _client


def is_enabled() -> bool:
    """
    Streaming is OPT-IN and defaults OFF.

    The client is tested and its overhead measured at ~10MB for 200 live
    contracts, which this box can absorb. But switching the Greeks source
    changes what every strike-selection and portfolio-risk calculation
    sees, so it should be turned on deliberately rather than silently
    taking effect on deploy. Enable with DXLINK_ENABLED=true in .env.
    """
    return os.environ.get("DXLINK_ENABLED", "").lower() in ("1", "true", "yes")


def enrich_contract(contract: dict) -> dict:
    """
    Overlay real streaming bid/ask and broker-calculated Greeks onto a
    contract dict built from the yfinance chain.

    Returns the contract unchanged if streaming is disabled, not
    connected, or has no fresh data for this symbol — so the caller keeps
    its Black-Scholes values (see options_pricing.py) rather than losing
    data. Adds "greeks_source": "dxlink" when real Greeks were applied,
    so the dashboard can report honestly which one is in play.
    """
    if not is_enabled():
        return contract
    try:
        client = get_client()
        if not client.status()["connected"]:
            return contract
        sym = to_streamer_symbol(contract["underlying"], contract["expiration"],
                                  contract["type"], contract["strike"])
        q = client.get_quote(sym)
        g = client.get_greeks(sym)
        if q:
            if q.get("bid") is not None:
                contract["bid"] = q["bid"]
            if q.get("ask") is not None:
                contract["ask"] = q["ask"]
            if q.get("mid") is not None:
                contract["mid"] = q["mid"]
            contract["quote_source"] = "dxlink"
        if g and g.get("delta") is not None:
            contract["delta"] = g["delta"]
            contract["gamma"] = g.get("gamma", contract.get("gamma"))
            contract["theta"] = g.get("theta", contract.get("theta"))
            contract["vega"] = g.get("vega", contract.get("vega"))
            if g.get("iv") is not None:
                contract["iv"] = g["iv"] * 100
            contract["greeks_source"] = "dxlink"
        return contract
    except Exception as e:
        log.debug(f"dxlink enrich skipped for {contract.get('symbol')}: {e}")
        return contract


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="DXLink streaming smoke test")
    ap.add_argument("--symbols", default="SPY", help="comma-separated underlyings")
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--strikes", type=int, default=6, help="ATM strikes per side")
    args = ap.parse_args()

    import auth, market_data
    auth.initialize()

    # Collect streamer symbols for near-the-money contracts on the
    # nearest expiration — a realistic subscription, not the whole chain.
    streamer_syms = []
    for underlying in [s.strip().upper() for s in args.symbols.split(",")]:
        try:
            resp = auth.session.get(f"/option-chains/{underlying}/nested")
            items = resp.get("data", {}).get("items", [])
            if not items:
                print(f"  {underlying}: no chain returned")
                continue
            exps = items[0].get("expirations", [])
            if not exps:
                continue
            strikes = exps[0].get("strikes", [])
            q = market_data.get_quote(underlying)
            spot = q["price"] if q else 0
            strikes.sort(key=lambda s: abs(float(s.get("strike-price", 0)) - spot))
            for s in strikes[:args.strikes]:
                for key in ("call-streamer-symbol", "put-streamer-symbol"):
                    if s.get(key):
                        streamer_syms.append(s[key])
            print(f"  {underlying}: spot {spot}, exp {exps[0].get('expiration-date')}, "
                  f"{len(strikes[:args.strikes])*2} contracts")
        except Exception as e:
            print(f"  {underlying}: chain error {e}")

    if not streamer_syms:
        raise SystemExit("no streamer symbols resolved")

    c = get_client()
    print(f"\nConnecting… ({len(streamer_syms)} contracts)")
    if not c.start():
        raise SystemExit(f"connect failed: {c.last_error}")
    print("Connected and authorized.")
    c.subscribe(streamer_syms)

    deadline = time.time() + args.seconds
    while time.time() < deadline:
        time.sleep(5)
        st = c.status()
        print(f"  quotes={st['quotes_cached']} greeks={st['greeks_cached']} "
              f"subs={st['subscribed']} reconnects={st['connect_count']-1}")

    print("\nSample live data:")
    shown = 0
    for sym in streamer_syms:
        q, g = c.get_quote(sym), c.get_greeks(sym)
        if q or g:
            print(f"  {sym}")
            if q:
                print(f"     quote  bid={q['bid']} ask={q['ask']} mid={q['mid']}")
            if g:
                print(f"     greeks delta={g['delta']} gamma={g['gamma']} "
                      f"theta={g['theta']} vega={g['vega']} iv={g['iv']}")
            shown += 1
            if shown >= 4:
                break
    if not shown:
        print("  (no data received — market may be closed)")
    c.stop()
