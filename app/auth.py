"""API keys and rate limiting.

Only SHA-256 hashes of keys are stored. Two sources, merged:
- data/api_keys.json, written by scripts/create_api_key.py and re-read when it changes,
  so new or revoked keys take effect without a restart;
- the VS_API_KEYS setting ("label:scope:sha256;..."), for hosts like Hugging Face
  Spaces where keys are configured as secrets.

Scopes:
- "full": search and add images (the mentor's curl key);
- "search": search only, per-key rate limit (a server-side proxy such as the Lovable
  edge function, where every visitor arrives from the same IP);
- "web": search only, stricter limit per key *and* client IP, for a key that ends up
  in a public frontend and is therefore visible to anyone who opens it.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KEY_PREFIX = "vs_"
SCOPES = ("full", "search", "web")
PERMISSIONS = {"full": {"search", "write"}, "search": {"search"}, "web": {"search"}}


@dataclass(frozen=True)
class Principal:
    label: str
    scope: str

    def can(self, permission: str) -> bool:
        return permission in PERMISSIONS.get(self.scope, set())


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def parse_env_keys(spec: str) -> dict[str, Principal]:
    keys = {}
    for item in spec.replace(",", ";").split(";"):
        if not item.strip():
            continue
        label, scope, digest = (part.strip() for part in item.split(":"))
        if scope not in SCOPES:
            raise ValueError(f"VS_API_KEYS: unknown scope '{scope}' for '{label}'")
        keys[digest.lower()] = Principal(label, scope)
    return keys


class KeyStore:
    def __init__(self, path: Path, env_spec: str = ""):
        self.path = path
        self._env_keys = parse_env_keys(env_spec)
        self._lock = threading.Lock()
        self._mtime: int | None = -1
        self._file_keys: dict[str, Principal] = {}

    def verify(self, key: str) -> Principal | None:
        """Return who the key belongs to, or None if it is unknown or revoked."""
        self._refresh()
        digest = hash_key(key)
        return self._env_keys.get(digest) or self._file_keys.get(digest)

    def create(self, label: str, scope: str = "full") -> str:
        if scope not in SCOPES:
            raise ValueError(f"Scope must be one of {SCOPES}.")
        entries = self._read()
        if any(e["label"] == label and not e["revoked"] for e in entries):
            raise ValueError(f"An active key labelled '{label}' already exists.")
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        entries.append({
            "label": label,
            "scope": scope,
            "hash": hash_key(key),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "revoked": False,
        })
        self._write(entries)
        return key

    def revoke(self, label: str) -> bool:
        entries = self._read()
        found = False
        for e in entries:
            if e["label"] == label and not e["revoked"]:
                e["revoked"] = True
                found = True
        self._write(entries)
        return found

    def list(self) -> list[dict]:
        return [{k: v for k, v in e.items() if k != "hash"} for e in self._read()]

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if mtime == self._mtime:
            return
        with self._lock:
            self._file_keys = {
                e["hash"]: Principal(e["label"], e.get("scope", "full"))
                for e in self._read() if not e["revoked"]
            }
            self._mtime = mtime

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text("utf-8"))["keys"]

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"keys": entries}, indent=2), "utf-8")
        os.replace(tmp, self.path)


class RateLimiter:
    """Sliding one-minute window per client, in memory (one server process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, client: str) -> float | None:
        """Record a request. Returns seconds to wait if the client is over the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return 60 - (now - hits[0])
            hits.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for stale in [c for c, h in self._hits.items() if not h or now - h[-1] >= 60]:
                    del self._hits[stale]
            return None
