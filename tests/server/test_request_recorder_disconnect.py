"""The request recorder must not hide a client disconnect from a non-streaming handler.

Real sockets on purpose: the failure only shows through uvicorn's receive channel (a
BaseHTTPMiddleware wrapper never yields ``http.disconnect`` while the handler computes), which
TestClient does not model.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

from fastapi import FastAPI, Request

from freetoken.server.api_server import _RecordRequestMiddleware


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app: FastAPI, port: int):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    return server


def _post_and_hang_up(port: int) -> None:
    body = b'{"prompt": "x"}'
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.sendall(
            b"POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        time.sleep(0.2)  # the handler is now polling


def test_non_streaming_handler_sees_the_client_leave():
    app = FastAPI()
    app.add_middleware(_RecordRequestMiddleware)
    seen: dict[str, float | None] = {}

    @app.post("/v1/completions")
    async def slow(request: Request):
        t0 = time.monotonic()
        for _ in range(60):
            await asyncio.sleep(0.05)
            if await request.is_disconnected():
                seen["after"] = time.monotonic() - t0
                return {"detected": True}
        seen["after"] = None
        return {"detected": False}

    port = _free_port()
    server = _serve(app, port)
    try:
        _post_and_hang_up(port)
        deadline = time.monotonic() + 5
        while "after" not in seen and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        server.should_exit = True
    assert seen.get("after") is not None, "handler never saw the disconnect"
    assert seen["after"] < 2.0
