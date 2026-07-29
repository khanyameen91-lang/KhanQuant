"""
auth.py — Tastytrade OAuth2 Authentication

Uses OAuth2 with multiple grant type fallbacks:
1. Saved token (fastest)
2. Refresh token (if access token expired)
3. Resource Owner Password Credentials (username/password → token directly)
4. Authorization Code via browser (fallback)
"""

import os
import json
import requests
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_URL      = "https://api.tastytrade.com"
TOKEN_URL     = "https://api.tastytrade.com/oauth/token"
AUTH_URL      = "https://my.tastytrade.com/oauth/authorize"
REDIRECT      = "http://localhost:5000/oauth/callback"

CLIENT_ID     = os.environ.get("TASTY_CLIENT_ID",     "c1960e4c-3cb4-420d-a06d-071245e22111")
CLIENT_SECRET = os.environ.get("TASTY_CLIENT_SECRET", "cb1b9d22fd62e60cc23af7480195ad7be077d82d")
USERNAME      = os.environ.get("TASTY_USERNAME",      "")
PASSWORD      = os.environ.get("TASTY_PASSWORD",      "")
GRANT_REFRESH_TOKEN = os.environ.get("TASTY_REFRESH_TOKEN", "")
TOKEN_FILE    = Path(".oauth_token.json")

_access_token   = None
_refresh_token  = None
_account_number = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_headers():
    return {"Authorization": f"Bearer {_access_token}",
            "Content-Type": "application/json"}


def _fetch_account_number():
    resp = requests.get(f"{BASE_URL}/customers/me/accounts",
                        headers=_get_headers(), timeout=10)
    resp.raise_for_status()
    items = resp.json()["data"]["items"]
    if not items:
        raise RuntimeError("No accounts found.")
    return items[0]["account"]["account-number"]


def _save_tokens(access, refresh, expires_at):
    global _account_number
    TOKEN_FILE.write_text(json.dumps({
        "access_token":   access,
        "refresh_token":  refresh,
        "expires_at":     expires_at.isoformat(),
        "account_number": _account_number,
    }))
    try: TOKEN_FILE.chmod(0o600)
    except: pass


def _store(d):
    """Parse a token response dict and store tokens."""
    global _access_token, _refresh_token, _account_number
    _access_token  = d["access_token"]
    _refresh_token = d.get("refresh_token", _refresh_token)
    expires = datetime.utcnow() + timedelta(seconds=d.get("expires_in", 3600))
    if not _account_number:
        _account_number = _fetch_account_number()
    _save_tokens(_access_token, _refresh_token, expires)
    print(f"✅ Authenticated! Account: {_account_number}")


# ── Grant type 1: Load saved token ─────────────────────────────────────────────
def _load_saved():
    global _access_token, _refresh_token, _account_number
    if not TOKEN_FILE.exists():
        return False
    try:
        data      = json.loads(TOKEN_FILE.read_text())
        expires   = datetime.fromisoformat(data["expires_at"])
        _refresh_token  = data.get("refresh_token")
        _account_number = data.get("account_number")
        if expires > datetime.utcnow() + timedelta(minutes=5):
            _access_token = data["access_token"]
            print(f"✅ Loaded saved token (expires {expires.strftime('%H:%M UTC')})")
            return True
        if _refresh_token:
            return _do_refresh()
    except Exception as e:
        print(f"⚠️  Saved token error: {e}")
    return False


# ── Grant type 2: Refresh token ────────────────────────────────────────────────
def _do_refresh():
    global _access_token
    try:
        resp = requests.post(TOKEN_URL, data={
            "grant_type":    "refresh_token",
            "refresh_token": _refresh_token,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }, timeout=15)
        if resp.status_code == 200:
            _store(resp.json())
            print("✅ Token refreshed")
            return True
    except Exception as e:
        print(f"⚠️  Refresh error: {e}")
    return False


# ── Grant type 3: Resource Owner Password Credentials ─────────────────────────
def _do_password_grant():
    print("🔑 Trying OAuth2 password grant...")
    try:
        resp = requests.post(TOKEN_URL, data={
            "grant_type":    "password",
            "username":      USERNAME,
            "password":      PASSWORD,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope":         "read trade openid",
        }, timeout=15)
        print(f"   Response: {resp.status_code}")
        if resp.status_code == 200:
            _store(resp.json())
            return True
        else:
            print(f"   ❌ Password grant failed: {resp.text[:200]}")
    except Exception as e:
        print(f"   ❌ Password grant error: {e}")
    return False


# ── Grant type 4: Authorization code via browser ───────────────────────────────
def _do_browser_auth():
    from urllib.parse import urlencode
    params = urlencode({
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT,
        "response_type": "code",
        "scope":         "read trade openid",
    })
    url = f"{AUTH_URL}?{params}"
    print(f"\n🌐 Tastytrade auth required. Visit this URL to re-authorize:")
    print(f"   {url}\n")
    print("💡 After approving, the dashboard will catch the callback.")
    # DO NOT call webbrowser.open() — hangs on headless server
    # Just print the URL so the user can authorize manually


# ── Token exchange (called by dashboard /oauth/callback route) ─────────────────
def exchange_code(code: str) -> bool:
    global _account_number
    try:
        resp = requests.post(TOKEN_URL, data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }, timeout=15)
        resp.raise_for_status()
        _store(resp.json())
        return True
    except Exception as e:
        print(f"❌ Code exchange failed: {e} — {getattr(e, 'response', None) and e.response.text}")
        return False


# ── Grant type 0: Use grant refresh token from .env ───────────────────────────
def _do_grant_token():
    """Use the long-lived grant refresh token stored in .env."""
    global _refresh_token
    if not GRANT_REFRESH_TOKEN:
        return False
    print("🔑 Using grant refresh token from .env...")
    _refresh_token = GRANT_REFRESH_TOKEN
    return _do_refresh()


# ── Main entry point ───────────────────────────────────────────────────────────
def initialize():
    if _load_saved():
        return

    # Try grant refresh token (from Tastytrade OAuth app → Create Grant)
    if _do_grant_token():
        return

    # Fall back to browser-based OAuth
    _do_browser_auth()


def is_authenticated():
    return _access_token is not None


def get_auth_url():
    from urllib.parse import urlencode
    params = urlencode({
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT,
        "response_type": "code",
        "scope":         "read trade openid",
    })
    return f"{AUTH_URL}?{params}"


def get_account_number():
    return _account_number


# ── Session proxy (used by market_data, executor, bot) ────────────────────────
class _SessionProxy:
    @property
    def token(self):          return _access_token
    @property
    def account_number(self): return _account_number
    @property
    def headers(self):        return _get_headers()

    def ensure_valid(self, username=None, password=None):
        """Refresh token if missing or expired (within 10-minute buffer)."""
        global _access_token, _account_number
        need_refresh = not _access_token
        if not need_refresh and TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text())
                expires = datetime.fromisoformat(data["expires_at"])
                if expires <= datetime.utcnow() + timedelta(minutes=10):
                    need_refresh = True
            except Exception:
                need_refresh = True
        if need_refresh:
            ok = _do_refresh() or _do_password_grant()
            if ok and not _account_number:
                try:
                    _account_number = _fetch_account_number()
                except Exception as e:
                    print(f"⚠️  Could not fetch account number: {e}")

    def _refresh_and_retry(self, method, url, **kwargs):
        """On 401: force re-auth then retry once."""
        global _access_token
        _access_token = None  # force full refresh
        _do_refresh() or _do_password_grant()
        resp = method(url, headers=_get_headers(), timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get(self, endpoint, params=None):
        resp = requests.get(f"{BASE_URL}{endpoint}",
                            headers=_get_headers(), params=params, timeout=15)
        if resp.status_code == 401:
            return self._refresh_and_retry(requests.get, f"{BASE_URL}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, endpoint, body):
        resp = requests.post(f"{BASE_URL}{endpoint}",
                             headers=_get_headers(), json=body, timeout=15)
        if resp.status_code == 401:
            return self._refresh_and_retry(requests.post, f"{BASE_URL}{endpoint}", json=body)
        resp.raise_for_status()
        return resp.json()

    def delete(self, endpoint):
        resp = requests.delete(f"{BASE_URL}{endpoint}",
                               headers=_get_headers(), timeout=15)
        if resp.status_code == 401:
            return self._refresh_and_retry(requests.delete, f"{BASE_URL}{endpoint}")
        resp.raise_for_status()
        return resp.json()


session = _SessionProxy()


if __name__ == "__main__":
    initialize()
    print(f"Authenticated: {is_authenticated()}")
    print(f"Account: {get_account_number()}")
