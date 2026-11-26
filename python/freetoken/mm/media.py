"""Client-supplied image handling: ref collection, the accept gate, and byte fetching.

Raises plain ValueError on bad input; the server layer maps it to its wire error type.
The message names the class of failure only: no local path, allowlist root, errno or
library text reaches the client (that detail goes to the server log).
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
import stat
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.server.args import ServerArgs

logger = init_logger(__name__)

_MAX_IMAGE_BYTES = 32 << 20
_MAX_B64_CHARS = ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 65_536
_MAX_REDIRECTS = 5


def collect_image_refs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pop the image refs out of rendered messages in prompt order; the bare {"type": "image"} parts stay for the chat template."""
    refs: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                ref = part.pop("freetoken_ref", None)
                if ref:
                    refs.append(ref)
    return refs


def image_reject_reason(config: ServerArgs) -> str | None:
    """None when this server can accept image inputs, else the client-facing reason."""
    if "image" in config.served_modalities:
        return None
    if config.mm.text_model_only:
        return "image input is disabled on this server (--text-model-only)"
    if "vision" in config.mm.disabled_encoders:
        return "image input is disabled on this server (--mm-disable)"
    return "the served model has no vision support"


def _check_media_domain(url: str, config: ServerArgs) -> None:
    """Reject a URL whose hostname is not in --allowed-media-domains; empty allowlist admits any domain."""
    from urllib.parse import urlparse

    raw = config.allowed_media_domains
    allowed = {d.strip().lower().rstrip(".") for d in raw.split(",") if d.strip()}
    if not allowed:
        return
    host = (urlparse(url).hostname or "").rstrip(".")
    if host not in allowed:
        raise ValueError(
            f"the URL must be from one of the allowed domains: {sorted(allowed)}; "
            f"input URL domain: {host or '<none>'}"
        )


def _assert_public_ip(raw_ip: str, *, err_msg: str) -> None:
    ip = ipaddress.ip_address(raw_ip)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:
        raise ValueError(err_msg)


async def _check_url_target(
    url: str, config: ServerArgs, *, is_redirect: bool, resolve_dns: bool = True
) -> None:
    """Validate URL scheme, domain allowlist, and non-private/non-loopback target IP(s)."""
    err_private = (
        "the URL redirects to a private or loopback address"
        if is_redirect
        else "the URL resolves to a private or loopback address"
    )
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "the URL redirects to a non-http(s) location"
            if is_redirect
            else "unsupported image URL scheme"
        )
    try:
        _check_media_domain(url, config)
    except ValueError:
        if is_redirect:
            raise ValueError("the URL redirects outside the allowed domains") from None
        raise
    parsed = urlparse(url)
    if "%" in (parsed.netloc or ""):
        raise ValueError(err_private)
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise ValueError("the image URL has no hostname")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(err_private)
    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False
    if is_ip_literal:
        _assert_public_ip(host, err_msg=err_private)
        return
    if not resolve_dns:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo, host, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        logger.warning("image URL DNS lookup failed for %r: %r", host, exc)
        raise ValueError("the image URL is unreachable") from exc
    if not infos:
        raise ValueError("the image URL is unreachable")
    for info in infos:
        addr = info[4][0].split("%", 1)[0]
        _assert_public_ip(addr, err_msg=err_private)


def _check_redirect_target(url: str, config: ServerArgs) -> None:
    """Synchronous IP/allowlist check helper retained for direct unit tests; async fetch runs _check_url_target."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("the URL redirects to a non-http(s) location")
    try:
        _check_media_domain(url, config)
    except ValueError:
        raise ValueError("the URL redirects outside the allowed domains") from None
    parsed = urlparse(url)
    if "%" in (parsed.netloc or ""):
        raise ValueError("the URL redirects to a private or loopback address")
    host = (parsed.hostname or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("the URL redirects to a private or loopback address")
    try:
        _assert_public_ip(
            host, err_msg="the URL redirects to a private or loopback address"
        )
    except ValueError as exc:
        if "private or loopback" in str(exc):
            raise
        return


async def _fetch_remote_media(url: str, config: ServerArgs) -> bytes:
    """GET an http(s) ref, following redirects one hop at a time so every target is re-checked."""
    import httpx

    resolve_dns = getattr(httpx.AsyncClient, "__module__", "").startswith("httpx")
    await _check_url_target(url, config, is_redirect=False, resolve_dns=resolve_dns)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            async with client.stream("GET", url) as resp:
                net_stream = (
                    resp.extensions.get("network_stream")
                    if hasattr(resp, "extensions")
                    else None
                )
                if net_stream is not None and hasattr(net_stream, "get_extra_info"):
                    peer = net_stream.get_extra_info("peername")
                    if isinstance(peer, tuple) and peer and isinstance(peer[0], str):
                        peer_ip = peer[0].split("%", 1)[0]
                        try:
                            ipaddress.ip_address(peer_ip)
                        except ValueError:
                            pass
                        else:
                            _assert_public_ip(
                                peer_ip,
                                err_msg="the URL resolves to a private or loopback address",
                            )
                if (
                    resp.next_request is not None
                ):  # a 3xx with a Location: httpx built the hop
                    url = str(resp.next_request.url)
                    await _check_url_target(
                        url, config, is_redirect=True, resolve_dns=resolve_dns
                    )
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > _MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > _MAX_IMAGE_BYTES:
                        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                return bytes(buf)
    raise ValueError(f"the URL redirects more than {_MAX_REDIRECTS} times")


def _load_local_media(url: str, config: ServerArgs) -> bytes:
    """Read a file:// ref; the resolved path must be a strict regular-file subpath of --allowed-local-media-path."""
    from pathlib import Path
    from urllib.request import url2pathname

    root = config.allowed_local_media_path
    if not root:
        raise ValueError("cannot load local files without --allowed-local-media-path")
    spec = urlparse(url)
    # RFC 8089: empty authority and localhost identify the local machine.
    if spec.netloc and spec.netloc.lower().rstrip(".") != "localhost":
        raise ValueError("file:// URLs must have no host or use localhost")
    filepath = Path(url2pathname(spec.path or ""))
    # resolve() follows symlinks, so a link inside the root escaping it is rejected too
    resolved = filepath.resolve()
    if Path(root).resolve() not in resolved.parents:
        raise ValueError(
            "the file path must be a subpath of --allowed-local-media-path"
        )
    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("the local file is missing or unreadable")
    if st.st_size > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
    raw = resolved.read_bytes()
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
    return raw


def _decode_b64_bounded(payload: str) -> bytes:
    if len(payload) > _MAX_B64_CHARS:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
    cleaned = "".join(payload.split())
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise ValueError("image data is not valid base64") from exc
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
    return raw


async def _fetch_one(ref: dict[str, Any], config: ServerArgs) -> bytes:
    """One ref to bytes. Our own ValueErrors are already client-safe and pass through; anything
    a library raises is reduced to its failure class and logged with the detail."""
    data = ref.get("data") or ""
    if ref.get("kind") == "b64":
        return _decode_b64_bounded(data)
    if data.startswith("data:"):
        if "," not in data:
            raise ValueError("image data is not valid base64")
        header, payload = data.split(",", 1)
        if ";base64" not in header.lower():
            raise ValueError("data URL must be base64-encoded (;base64)")
        return _decode_b64_bounded(payload)
    if data.startswith(("http://", "https://")):
        import httpx

        try:
            return await _fetch_remote_media(data, config)
        except ValueError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"the image URL returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ValueError("timed out fetching the image URL") from exc
        except Exception as exc:
            logger.warning("image URL fetch failed: %r", exc)
            raise ValueError("the image URL is unreachable") from exc
    if data.startswith("file://"):
        try:
            return await asyncio.to_thread(_load_local_media, data, config)
        except ValueError:
            raise
        except OSError as exc:
            logger.warning("local image read failed: %r", exc)
            raise ValueError("the local file is missing or unreadable") from exc
    raise ValueError("unsupported image source (expect http(s)/file url or base64)")


async def fetch_image_bytes(
    refs: list[dict[str, Any]], config: ServerArgs
) -> list[bytes]:
    """Resolve collected image refs (URLs / base64) to raw bytes, in order."""
    out: list[bytes] = []
    for ref in refs:
        try:
            out.append(await _fetch_one(ref, config))
        except ValueError as exc:
            raise ValueError(f"could not load image: {exc}") from exc
    return out


__all__ = ["collect_image_refs", "fetch_image_bytes", "image_reject_reason"]
