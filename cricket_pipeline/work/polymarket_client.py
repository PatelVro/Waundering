"""Polymarket adapter (scaffold).

Polymarket is one of the few venues that:
  - is reachable from Canada (via a crypto wallet — USDC on Polygon)
  - has a public REST + WebSocket API
  - lists cricket markets occasionally (IPL final, T20 World Cup, etc.)

This module is a SCAFFOLD. To enable live placement you need:
  1. A Polygon-network wallet funded with USDC
  2. The CLOB API credentials (api_key + secret + passphrase) generated via
     https://docs.polymarket.com/#authentication
  3. POLYMARKET_LIVE_CONFIRMED=yes in env

Reading list:
  - https://docs.polymarket.com/
  - https://github.com/Polymarket/py-clob-client
"""
from __future__ import annotations

import os
from typing import Any


def _live_confirmed() -> bool:
    return os.environ.get("POLYMARKET_LIVE_CONFIRMED", "").lower() in ("1", "yes", "true")


def _creds() -> dict:
    return {
        "api_key":      os.environ.get("POLYMARKET_API_KEY", ""),
        "api_secret":   os.environ.get("POLYMARKET_API_SECRET", ""),
        "passphrase":   os.environ.get("POLYMARKET_PASSPHRASE", ""),
        "wallet":       os.environ.get("POLYMARKET_WALLET_ADDR", ""),
        "private_key":  os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
    }


def list_cricket_markets() -> list[dict]:
    """List active cricket markets via Polymarket's gamma API (no auth)."""
    import requests
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/events",
            params={"active": "true", "tag_slug": "cricket"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []


def place(decision: Any) -> dict:
    """Place an order. Stub — raises until you opt in + install py_clob_client."""
    if not _live_confirmed():
        raise NotImplementedError(
            "Polymarket live trading requires POLYMARKET_LIVE_CONFIRMED=yes plus "
            "API + wallet credentials. See cricket_pipeline/work/polymarket_client.py."
        )
    creds = _creds()
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise RuntimeError(f"Missing Polymarket credentials: {missing}")
    # Real implementation:
    #   from py_clob_client.client import ClobClient
    #   client = ClobClient(host=..., chain_id=137, key=creds["private_key"])
    #   client.create_or_derive_api_creds()
    #   order_args = OrderArgs(price=..., size=..., side='BUY', token_id=...)
    #   signed = client.create_order(order_args)
    #   return client.post_order(signed)
    raise NotImplementedError("Implement Polymarket order placement in polymarket_client.place()")
