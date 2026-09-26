"""--host with a comma-separated address list (server/bind.py)."""

import socket
import threading
import time
import urllib.request

import pytest
import uvicorn

from freetoken.server.bind import bind_sockets, split_hosts


def test_split_hosts_trims_drops_empty_and_dedupes():
    assert split_hosts("127.0.0.1") == ["127.0.0.1"]
    assert split_hosts(" 127.0.0.1 , 172.17.0.1 ,,127.0.0.1") == [
        "127.0.0.1",
        "172.17.0.1",
    ]
    with pytest.raises(ValueError):
        split_hosts(" , ")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _second_loopback_available() -> bool:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.2", 0))
        return True
    except OSError:
        return False


def test_bind_sockets_closes_what_it_opened_when_a_later_bind_fails():
    port = _free_port()
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen()
        with pytest.raises(OSError, match="cannot bind 127.0.0.1"):
            bind_sockets(["127.0.0.1"], port)
    # the port is free again: nothing leaked from the failed call
    socks = bind_sockets(["127.0.0.1"], port)
    for s in socks:
        s.close()


@pytest.mark.skipif(
    not _second_loopback_available(), reason="needs 127.0.0.2 (Linux loopback /8)"
)
def test_one_server_answers_on_every_listed_address():
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    port = _free_port()
    hosts = split_hosts("127.0.0.1,127.0.0.2")
    server = uvicorn.Server(
        uvicorn.Config(app, host=hosts[0], port=port, log_level="warning")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": bind_sockets(hosts, port)}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        for host in hosts:
            with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as r:
                assert r.status == 200 and r.read() == b"ok"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
