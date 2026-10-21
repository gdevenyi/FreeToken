import pytest

import freetoken.engine.graph as graph_mod
from freetoken.env import ENV


class _Graph:
    def raw_cuda_graph_exec(self):
        return 1234


class _Stream:
    cuda_stream = 5678
    synced = False

    def synchronize(self):
        self.synced = True


@pytest.mark.parametrize("enabled", [True, False])
def test_graph_upload_follows_env(monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(graph_mod, "_cu_graph_upload", lambda: lambda exec_, stream: calls.append((exec_, stream)) or 0)
    monkeypatch.setattr(ENV.GRAPH_UPLOAD, "value", enabled)
    stream = _Stream()
    graph_mod._upload_graph(_Graph(), stream)
    assert calls == ([(1234, 5678)] if enabled else [])
    assert stream.synced is enabled
