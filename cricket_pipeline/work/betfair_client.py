"""Betfair Exchange API adapter — used when BET_MODE=live.

This is a *scaffold*. Live-trading requires:
  1. A funded Betfair account
  2. Application Key (https://developer.betfair.com)
  3. Username + password
  4. SSL client certificate (.pem) for non-interactive login
  5. Strict review of every code path before enabling

By default this module raises NotImplementedError on every action — the user
must explicitly fill in their credentials AND set `BETFAIR_LIVE_CONFIRMED=yes`
to actually transact.

Why a scaffold rather than a working client out of the box?
  - Real money. Don't auto-place trades from autonomous code without
    deliberate human opt-in.
  - The Betfair flow (login → keepalive → list events → list markets →
    list runners → list prices → placeOrders → settle) is non-trivial to
    test without an account.

Reading list:
  - https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Getting+Started
  - https://github.com/betcode-org/betfair  (community Python client)
"""
from __future__ import annotations

import os
from typing import Any


def _live_confirmed() -> bool:
    return os.environ.get("BETFAIR_LIVE_CONFIRMED", "").lower() in ("1", "yes", "true")


def _load_credentials() -> dict[str, str]:
    return {
        "app_key":  os.environ.get("BETFAIR_APP_KEY", ""),
        "username": os.environ.get("BETFAIR_USERNAME", ""),
        "password": os.environ.get("BETFAIR_PASSWORD", ""),
        "cert":     os.environ.get("BETFAIR_CERT_PATH", ""),
        "key":      os.environ.get("BETFAIR_KEY_PATH", ""),
    }


def login() -> str:
    """Returns a session token. Raises if any prereq is missing."""
    if not _live_confirmed():
        raise NotImplementedError(
            "Live trading requires BETFAIR_LIVE_CONFIRMED=yes in env, plus "
            "valid Betfair credentials. See cricket_pipeline/work/betfair_client.py."
        )
    creds = _load_credentials()
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Betfair credentials: {missing}")
    # Real implementation would POST to https://identitysso-cert.betfair.com/api/certlogin
    # using the .pem cert + key, parse response, return sessionToken.
    raise NotImplementedError("Implement Betfair certlogin here.")


def list_cricket_markets() -> list[dict]:
    """Returns market catalogues for upcoming cricket events."""
    raise NotImplementedError("Implement listMarketCatalogue.")


def get_runner_prices(market_id: str) -> list[dict]:
    """Returns ladder/back-lay prices for runners in a market."""
    raise NotImplementedError("Implement listMarketBook.")


def place(decision: Any) -> dict:
    """Place a bet for the given decision. Returns Betfair's response.
    Stub — raises until live mode is implemented."""
    raise NotImplementedError(
        "Live placeOrders not implemented. The bet engine will fall back to "
        "paper mode. Implement in betfair_client.place() and set "
        "BETFAIR_LIVE_CONFIRMED=yes to enable."
    )
