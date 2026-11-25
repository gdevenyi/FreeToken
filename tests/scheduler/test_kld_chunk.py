"""A /v1/internal/kld chunk through the real (unbound) Scheduler methods on CPU hybrid managers:
refused unless enabled, prefilled cold in one pass, scored by the lm_head hook, freed without a
prefix-cache entry, and its rows returned on the terminal reply."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from freetoken.benchmark.kld import KldCapture
from freetoken.core import SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.message import AbortBackendMsg, DetokenizeMsg, ErrorReplyMsg, UserMsg
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

N_CTX, VOCAB, HIDDEN, PAGES = 16, 50, 8, 64


def _base_file(path, n_chunk=2):
    """A base file of the llama.cpp layout; the row values only need to decode."""
    rng = np.random.default_rng(0)
    n_rows, nv = N_CTX - 1 - N_CTX // 2, 2 * ((VOCAB + 1) // 2) + 4
    tokens = rng.integers(0, VOCAB, size=(n_chunk, N_CTX)).astype("<i4")
    rows = rng.integers(0, 65535, size=(n_chunk * n_rows, nv)).astype("<u2")
    rows[:, :4] = np.array([16 / 65535, -17.0], dtype="<f4").view("<u2")
    with open(path, "wb") as f:
        f.write(
            b"_logits_"
            + struct.pack("<Iii", N_CTX, VOCAB, n_chunk)
            + tokens.tobytes()
            + rows.tobytes()
        )
    return tokens


def _setup(enabled=True):
    g = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate="silu",
    )
    pool = LinearStatePool(
        group=g,
        num_slots=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
    )
    pt = torch.zeros(4, PAGES, dtype=torch.int32)
    cm = CacheManager(PAGES, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm, dm = TableManager(max_running_reqs=4, page_table=pt), DecodeManager(page_size=1)
    sent = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        decode_manager=dm,
        prefill_manager=PrefillManager(cm, tm, dm),
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        spec=None,
        config=SimpleNamespace(
            page_size=1,
            kld_endpoint=enabled,
            model_config=SimpleNamespace(vocab_size=VOCAB),
            tp_info=SimpleNamespace(is_primary=lambda: True),
        ),
        engine=SimpleNamespace(max_seq_len=1024, encoder_cache=None),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=sent.extend,
        _kv_usage_pages=cm.page_usage,
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _pending_abort_acks=set(),
        _last_data=None,
    )
    for name in (
        "_free_req_resources",
        "_commit_tokens",
        "_admit_kld",
        "_attach_kld",
        "_finish_kld",
    ):
        setattr(stub, name, getattr(Scheduler, name).__get__(stub))
    return cm, pool, sent, stub


def _chunk_msg(path, tokens, chunk=1, uid=7):
    spec = {"path": str(path), "chunk": chunk, "slice_rows": 3, "bos_token_id": None}
    return UserMsg(
        uid=uid,
        input_ids=torch.from_numpy(tokens[chunk].astype(np.int32)),
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True, kld=spec),
    )


def test_kld_chunk_prefills_cold_alone_and_returns_its_rows(tmp_path):
    tokens = _base_file(tmp_path / "b.kld")
    cm, pool, sent, stub = _setup()
    free_pages, free_slots = len(cm.free_slots), pool.num_free_slots
    Scheduler._process_one_msg(stub, _chunk_msg(tmp_path / "b.kld", tokens))
    assert sent == [] and stub.prefill_manager.runnable

    batch = stub.prefill_manager.schedule_next_batch(4096)
    (req,) = batch.reqs
    assert req.cached_len == 0 and req.extend_len == N_CTX
    cm.allocate_paged(batch.reqs)
    Scheduler._attach_kld(stub, batch)
    assert isinstance(batch.kld, KldCapture)

    w = torch.randn(VOCAB, HIDDEN)
    calls = []
    batch.kld.consume(
        torch.randn(N_CTX, HIDDEN),
        lambda rows: calls.append(rows.shape[0]) or rows @ w.T,
    )
    assert calls == [3, 3, 1]  # rows 8..14 in slices of 3
    req.complete_one()
    Scheduler._process_last_data(
        stub,
        (
            SimpleNamespace(batch=batch),
            (
                None,
                torch.tensor([3], dtype=torch.int32),
                SimpleNamespace(synchronize=lambda: None),
            ),
        ),
    )
    (msg,) = [m for m in sent if isinstance(m, DetokenizeMsg)]
    assert msg.finished and msg.kld["error"] is None and msg.kld["chunk"] == 1
    assert len(msg.kld["kld"]) == len(msg.kld["same_top"]) == N_CTX - 1 - N_CTX // 2
    assert stub._kld_jobs == {}
    # freed, and nothing entered the prefix cache
    assert len(cm.free_slots) == free_pages and pool.num_free_slots == free_slots
    assert (
        cm.prefix_cache.full_evictable == 0
        and cm.prefix_cache.mamba_evictable_size == 0
    )
    cm.check_integrity()


@pytest.mark.parametrize(
    ("enabled", "chunk_tokens", "match"),
    [
        (False, 1, "--enable-kld-endpoint"),
        (True, 0, "differ"),  # chunk 1's spec with chunk 0's tokens
    ],
)
def test_kld_chunk_is_refused(tmp_path, enabled, chunk_tokens, match):
    tokens = _base_file(tmp_path / "b.kld")
    _cm, _pool, sent, stub = _setup(enabled=enabled)
    msg = _chunk_msg(tmp_path / "b.kld", tokens)
    msg.input_ids = torch.from_numpy(tokens[chunk_tokens].astype(np.int32))
    Scheduler._process_one_msg(stub, msg)
    (err,) = sent
    assert isinstance(err, ErrorReplyMsg) and err.uid == 7 and match in err.error
    assert not stub.prefill_manager.runnable and not getattr(stub, "_kld_jobs", None)


def test_kld_chunk_split_over_passes_reports_an_error(tmp_path):
    tokens = _base_file(tmp_path / "b.kld")
    _cm, _pool, _sent, stub = _setup()
    Scheduler._process_one_msg(stub, _chunk_msg(tmp_path / "b.kld", tokens))
    batch = stub.prefill_manager.schedule_next_batch(
        N_CTX // 2
    )  # a budget below the chunk: two passes
    Scheduler._attach_kld(stub, batch)
    assert batch.kld is None
    assert "one pass" in Scheduler._finish_kld(stub, 7)["error"]


def test_abort_drops_the_kld_job(tmp_path):
    tokens = _base_file(tmp_path / "b.kld")
    _cm, _pool, _sent, stub = _setup()
    Scheduler._process_one_msg(stub, _chunk_msg(tmp_path / "b.kld", tokens))
    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=7))
    assert stub._kld_jobs == {} and not stub.prefill_manager.runnable
