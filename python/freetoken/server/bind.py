"""Bind the HTTP server to one address or to a comma-separated list of them.

``--host 127.0.0.1,172.17.0.1`` serves loopback clients and containers that reach the host
through the Docker bridge (``host.docker.internal`` resolves to the ``docker0`` address) without
listening on every interface the way ``0.0.0.0`` does. A single address keeps plain
``uvicorn.run``; a list binds one socket per address and hands them all to one uvicorn server.
"""

from __future__ import annotations

import socket
from typing import Any, List

import uvicorn


def split_hosts(host: str) -> List[str]:
    """``" a , b ,a"`` -> ``["a", "b"]``: trimmed, empty entries dropped, first occurrence kept."""
    hosts: List[str] = []
    for part in host.split(","):
        part = part.strip()
        if part and part not in hosts:
            hosts.append(part)
    if not hosts:
        raise ValueError(f"--host {host!r} names no address")
    return hosts


def bind_sockets(hosts: List[str], port: int) -> List[socket.socket]:
    """One listening-ready TCP socket per address, all on ``port``. Closes what it opened if a
    later bind fails, so a bad address is reported without leaking the earlier sockets."""
    socks: List[socket.socket] = []
    try:
        for host in hosts:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            socks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                raise OSError(
                    exc.errno, f"cannot bind {host}:{port}: {exc.strerror}"
                ) from exc
            sock.set_inheritable(True)
    except BaseException:
        for sock in socks:
            sock.close()
        raise
    return socks


def serve(app: Any, host: str, port: int) -> None:
    """``uvicorn.run(app, host=host, port=port)``, generalised to a comma-separated host list."""
    hosts = split_hosts(host)
    if len(hosts) == 1:
        uvicorn.run(app, host=hosts[0], port=port)
        return
    server = uvicorn.Server(uvicorn.Config(app, host=hosts[0], port=port))
    server.run(sockets=bind_sockets(hosts, port))
