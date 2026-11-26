"""Bind the HTTP server to one address or to a comma-separated list of them.

``--host 127.0.0.1,172.17.0.1`` serves loopback clients and containers that reach the host
through the Docker bridge (``host.docker.internal`` resolves to the ``docker0`` address) without
listening on every interface the way ``0.0.0.0`` does. A single address keeps plain
``uvicorn.run``; a list binds one socket per address and hands them all to one uvicorn server.
"""

from __future__ import annotations

import socket
from typing import Any

import uvicorn


def split_hosts(host: str) -> list[str]:
    """``" a , b ,a"`` -> ``["a", "b"]``: trimmed, bracket-unwrapped, empty entries dropped, first occurrence kept."""
    hosts: list[str] = []
    for part in host.split(","):
        part = part.strip()
        if part.startswith("[") and part.endswith("]"):
            part = part[1:-1].strip()
        if part and part not in hosts:
            hosts.append(part)
    if not hosts:
        raise ValueError(f"--host {host!r} names no address")
    return hosts


def bind_sockets(hosts: list[str], port: int) -> list[socket.socket]:
    """One listening-ready TCP socket per address, all on ``port``. Closes what it opened if a
    later bind fails, so a bad address is reported without leaking the earlier sockets."""
    socks: list[socket.socket] = []
    try:
        for host in hosts:
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise OSError(
                    exc.errno, f"cannot resolve {host}:{port}: {exc.strerror}"
                ) from exc
            if not infos:
                raise OSError(f"cannot resolve {host}:{port}: no addresses")
            family, socktype, proto, _, sockaddr = infos[0]
            sock = socket.socket(family, socktype, proto)
            socks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind(sockaddr)
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


def serve(app: Any, host: str, port: int, **uvicorn_kwargs: Any) -> None:
    """``uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)``, generalised to a comma-separated host list."""
    hosts = split_hosts(host)
    if len(hosts) == 1:
        uvicorn.run(app, host=hosts[0], port=port, **uvicorn_kwargs)
        return
    server = uvicorn.Server(
        uvicorn.Config(app, host=hosts[0], port=port, **uvicorn_kwargs)
    )
    socks = bind_sockets(hosts, port)
    try:
        server.run(sockets=socks)
    finally:
        for sock in socks:
            sock.close()
