"""Download an image from a user-supplied URL without letting it reach internal hosts (SSRF)."""
from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlsplit

import httpx

from app.errors import BadRequestError, PayloadTooLargeError


async def _ensure_public_host(host: str, port: int) -> None:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port)
    except OSError as exc:
        raise BadRequestError(f"image_url host '{host}' could not be resolved.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise BadRequestError("image_url must point to a public internet address.")


async def fetch_image(url: str, max_bytes: int, timeout: float = 10.0) -> bytes:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BadRequestError("image_url must be an http(s) URL.")
    await _ensure_public_host(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))

    # Redirects are not followed: the target could be an internal address.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            async with client.stream("GET", url, headers={"User-Agent": "visual-search/1.0"}) as resp:
                if resp.is_redirect:
                    raise BadRequestError("image_url redirects; pass the final image URL instead.")
                if resp.status_code != 200:
                    raise BadRequestError(f"image_url returned HTTP {resp.status_code}.")
                chunks, total = [], 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise PayloadTooLargeError(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise BadRequestError(f"Could not download image_url: {exc.__class__.__name__}.") from exc
    return b"".join(chunks)
