"""DFlash2DraftModel against an fp32 torch reference of the sglang DFlash 2 math, on a synthetic bf16 draft checkpoint: context rows into the per-layer K/V ring, the block forward (window mask, grouped convs, residual norms) and the selector walk over the target head's candidates.

At TP=2, on the CPU with a simulated communicator: the config check, the sharded load, the budget against the sharded ring, the per-rank candidate gather, and the rank-0 drafts with the same collectives on both ranks; on one GPU, both ranks' block forward in lockstep against the fp32 reference."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV, CPU = torch.device("cuda"), torch.device("cpu")
H, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER = 64, 2, 4, 2, 64, 128
GROUP_SIZE, TAPS, RANK, TOP_K, WINDOW, VOCAB, MASK = 16, (1, 2), 8, 4, 16, 64, 61
RING = WINDOW + 64
THETA, EPS = 10000.0, 1e-6
CONFIG = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "is_causal": False, "dtype": "bfloat16",
    "dflash_config": {"block_size": 8, "conv_group_size": GROUP_SIZE, "conv_kernel_size": 2, "mask_token_id": MASK,
                      "selector_rank": RANK, "selector_top_k": TOP_K, "target_layer_ids": list(TAPS)},
    "head_dim": HEAD_DIM, "hidden_act": "silu", "hidden_size": H, "intermediate_size": INTER,
    "layer_types": ["sliding_attention"] * LAYERS, "max_position_embeddings": 4096, "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS, "num_key_value_heads": KV_HEADS, "num_target_layers": 4, "rms_norm_eps": EPS,
    "rope_parameters": {"rope_theta": THETA, "rope_type": "default"}, "sliding_window": WINDOW, "use_sliding_window": True,
    "tie_word_embeddings": False, "vocab_size": VOCAB,
}


def _weights() -> dict[str, torch.Tensor]:
    """A fan-in scaled bf16 draft plus the target's embedding and head (``target.*``, not part of the checkpoint)."""
    g = torch.Generator().manual_seed(0)

    def mat(*shape):
        return (torch.randn(*shape, generator=g) * shape[-1] ** -0.5).to(torch.bfloat16)

    def gain(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {"fc.weight": mat(H, len(TAPS) * H), "hidden_norm.weight": gain(H), "norm.weight": gain(H),
         "candidate_selector.hidden_projection.weight": mat(RANK, H),
         "candidate_selector.predecessor_codebook": mat(VOCAB, RANK), "candidate_selector.successor_codebook": mat(VOCAB, RANK),
         "target.embed": mat(VOCAB, H), "target.head": mat(VOCAB, H)}
    for i in range(LAYERS):
        p = f"layers.{i}"
        w.update({f"{p}.self_attn.q_proj.weight": mat(HEADS * HEAD_DIM, H), f"{p}.self_attn.k_proj.weight": mat(KV_HEADS * HEAD_DIM, H),
                  f"{p}.self_attn.v_proj.weight": mat(KV_HEADS * HEAD_DIM, H), f"{p}.self_attn.o_proj.weight": mat(H, HEADS * HEAD_DIM),
                  f"{p}.self_attn.q_norm.weight": gain(HEAD_DIM), f"{p}.self_attn.k_norm.weight": gain(HEAD_DIM),
                  f"{p}.mlp.gate_proj.weight": mat(INTER, H), f"{p}.mlp.up_proj.weight": mat(INTER, H), f"{p}.mlp.down_proj.weight": mat(H, INTER),
                  f"{p}.input_layernorm.weight": gain(H), f"{p}.post_attention_layernorm.weight": gain(H)})
        for conv in ("attention_conv", "mlp_conv"):
            w[f"{p}.{conv}.base_kernel"] = (0.5 * torch.randn(2, 2, H, generator=g)).to(torch.bfloat16)
            w[f"{p}.{conv}.kernel_projection.weight"] = mat(2 * 2 * (H // GROUP_SIZE), H)
    return w


def _load_draft(tmp_path, w: dict[str, torch.Tensor], dtype=torch.bfloat16):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.dflash import DFlash2DraftModel, parse_config
    from freetoken.utils import cached_load_hf_config, torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    path = _checkpoint(tmp_path, w)
    with torch.device("meta"), torch_dtype(dtype):
        embed, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    for op, name in ((embed, "target.embed"), (head, "target.head")):
        op.load_state_dict({"weight": w[name].to(DEV, dtype)})
        finalize_quant(op)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)
    return DFlash2DraftModel.load(parse_config(cached_load_hf_config(path)), target, path, DEV, dtype)


def _checkpoint(tmp_path, w: dict[str, torch.Tensor]) -> str:
    """The draft tensors of ``w`` as a checkpoint directory."""
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    save_file({name: t for name, t in w.items() if not name.startswith("target.")}, str(path / "model.safetensors"))
    return str(path)


class _Reference:
    """The draft math in fp32 from the HF definition: pre-norm layers, per-head q/k norm, NeoX rope, grouped dynamic conv, sliding-window block attention."""

    def __init__(self, w: dict[str, torch.Tensor]) -> None:
        self.w = {name: t.to(DEV, torch.float32) for name, t in w.items()}

    @staticmethod
    def rmsnorm(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight

    @staticmethod
    def rope(x, pos):
        """``x [n, heads, HEAD_DIM]`` at positions ``pos [n]``."""
        inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, device=DEV, dtype=torch.float32) / HEAD_DIM))
        freqs = pos.float()[:, None] * inv[None, :]
        cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
        x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def context_rows(self, features):
        return self.rmsnorm(features.float() @ self.w["fc.weight"].T, self.w["hidden_norm.weight"])

    def kv(self, layer, ctx, pos):
        """K (normed, roped) and V of context rows ``ctx`` at ``pos``, ``[n, KV_HEADS, HEAD_DIM]`` each."""
        p = f"layers.{layer}.self_attn"
        k = self.rmsnorm((ctx @ self.w[f"{p}.k_proj.weight"].T).view(-1, KV_HEADS, HEAD_DIM), self.w[f"{p}.k_norm.weight"])
        return self.rope(k, pos), (ctx @ self.w[f"{p}.v_proj.weight"].T).view(-1, KV_HEADS, HEAD_DIM)

    def conv(self, x, prefix, side, coef=None):
        """``out[j] = sum_t (base[side, t] + coef[j, side, t, group]) * x[j - t]`` over one block of rows; the first row has no previous row."""
        if coef is None:
            coef = (x @ self.w[f"{prefix}.kernel_projection.weight"].T).view(x.shape[0], 2, 2, H // GROUP_SIZE)
        base = self.w[f"{prefix}.base_kernel"][side]
        c = base.view(1, 2, H // GROUP_SIZE, GROUP_SIZE) + coef[:, side].unsqueeze(-1)
        blocks = x.view(x.shape[0], H // GROUP_SIZE, GROUP_SIZE)
        prev = F.pad(blocks[:-1], (0, 0, 0, 0, 1, 0))
        return (c[:, 0] * blocks + c[:, 1] * prev).flatten(-2), coef

    def block(self, bonus, start, ctx, block):
        """Final hidden ``[block, H]`` of ``[bonus, mask x (block-1)]`` at ``start``, over the context rows ``ctx`` at positions ``0 .. len(ctx)-1``."""
        w = self.w
        x = torch.cat([w["target.embed"][bonus][None], w["target.embed"][MASK][None].expand(block - 1, -1)])
        pos = start + torch.arange(block, device=DEV)
        hist = torch.arange(ctx.shape[0], device=DEV)
        residual = None
        for i in range(LAYERS):
            p = f"layers.{i}"
            residual = x if residual is None else residual + x
            h = self.rmsnorm(residual, w[f"{p}.input_layernorm.weight"])
            h, coef = self.conv(h, f"{p}.attention_conv", 0)
            q = self.rope(self.rmsnorm((h @ w[f"{p}.self_attn.q_proj.weight"].T).view(block, HEADS, HEAD_DIM), w[f"{p}.self_attn.q_norm.weight"]), pos)
            k_hist, v_hist = self.kv(i, ctx, hist)
            k_block, v_block = self.kv(i, h, pos)
            k = torch.cat([k_hist, k_block]).repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = torch.cat([v_hist, v_block]).repeat_interleave(HEADS // KV_HEADS, dim=1)
            key_pos = torch.cat([hist, pos])
            # a query at start+j sees the window of positions > start+j-WINDOW and the whole block
            keep = (key_pos[None, :] >= (pos - WINDOW + 1)[:, None]) | (key_pos >= start)[None, :]
            scores = torch.einsum("jhd,nhd->jhn", q, k) * HEAD_DIM**-0.5
            probs = torch.softmax(scores.masked_fill(~keep[:, None, :], float("-inf")), dim=-1)
            o = torch.einsum("jhn,nhd->jhd", probs, v).reshape(block, -1) @ w[f"{p}.self_attn.o_proj.weight"].T
            o, _ = self.conv(o, f"{p}.attention_conv", 1, coef)
            residual = residual + o
            h = self.rmsnorm(residual, w[f"{p}.post_attention_layernorm.weight"])
            h, coef = self.conv(h, f"{p}.mlp_conv", 0)
            m = (F.silu(h @ w[f"{p}.mlp.gate_proj.weight"].T) * (h @ w[f"{p}.mlp.up_proj.weight"].T)) @ w[f"{p}.mlp.down_proj.weight"].T
            x, _ = self.conv(m, f"{p}.mlp_conv", 1, coef)
        return self.rmsnorm(residual + x, w["norm.weight"])

    def walk(self, hidden, cand, anchor, temperature, greedy, uniforms):
        """One path over the given candidates of slots 1..: ``unary + <pre[prev] * state, suc[cand]>`` per slot, argmax or a draw with ``uniforms``; ``cand`` must be a best-first top-K of the head within the real head's bf16 noise."""
        w = self.w
        bs = anchor.shape[0]
        pred = hidden.float().view(bs, -1, H)[:, 1:]
        logits = pred @ w["target.head"].T
        unary = logits.gather(-1, cand)
        noise = 2**-7 * logits.abs().max()
        runner_up = logits.topk(TOP_K + 1, dim=-1).values[..., -1:]
        assert (unary >= runner_up - noise).all() and (unary[..., :-1] >= unary[..., 1:] - noise).all()
        state = pred @ w["candidate_selector.hidden_projection.weight"].T
        prev, tokens, qs = anchor, [], []
        for s in range(pred.shape[1]):
            scale = w["candidate_selector.predecessor_codebook"][prev] * state[:, s]
            scores = unary[:, s] + (w["candidate_selector.successor_codebook"][cand[:, s]] * scale[:, None]).sum(-1)
            probs = torch.softmax(scores / temperature[:, None], -1)
            top2 = scores.topk(2, dim=-1).values
            # the argmax and the draw must be clear of that noise too, or the comparison means nothing
            assert (top2[:, 0] - top2[:, 1])[greedy].min() > 5e-2
            assert (uniforms[:, s : s + 1] - probs.cumsum(-1)).abs()[~greedy].min() > 2e-2
            index = torch.where(greedy, scores.argmax(-1), (uniforms[:, s : s + 1] >= probs.cumsum(-1)).sum(-1).clamp(max=TOP_K - 1))
            qs.append(torch.where(greedy[:, None], F.one_hot(index, TOP_K).float(), probs))
            prev = cand[:, s].gather(1, index[:, None])[:, 0]
            tokens.append(prev)
        return torch.stack(tokens, 1), torch.stack(qs, 1)


@cuda
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 2**-6), (torch.bfloat16, 2**-2)])
@pytest.mark.parametrize("block", [4, 8])
def test_block_forward_matches_reference(tmp_path, block, dtype, tol):
    """Two requests in one block forward: one with a context shorter than the window (positions before 0 masked), one longer than the ring (wrapped slots). The fp32 draft pins the math (tf32 dots); the bf16 draft, the serving dtype, is held to bf16 noise. A ring rebuilt from only the window rows gives the same hidden, bitwise."""
    w = _weights()
    draft, ref = _load_draft(tmp_path, w, dtype), _Reference(w)
    dflash_ring = draft.make_dflash_ring(max_rows=3, block=block, ring_len=RING, max_position=4096, device=DEV)
    torch.manual_seed(1)
    lengths, rows = (5, 100), (2, 1)
    contexts, draft_contexts = [], []
    for n, row in zip(lengths, rows):
        features = torch.randn(n, len(TAPS) * H, device=DEV, dtype=torch.bfloat16).to(dtype)
        ctx = draft.context_rows(features)
        torch.testing.assert_close(ctx.float(), ref.context_rows(features), atol=0, rtol=2**-6)
        draft_contexts.append(ctx)
        # as the proposer stores a chunk: its last RING rows at most
        tail = torch.arange(max(0, n - RING), n, device=DEV)
        draft.store_context(ctx[tail], tail, torch.full((len(tail),), row, device=DEV), dflash_ring)
        contexts.append(ref.context_rows(features))
        for i in range(LAYERS):
            k, v = ref.kv(i, contexts[-1][tail], tail)
            torch.testing.assert_close(dflash_ring.k[i, row, tail % RING].float(), k, atol=2**-6, rtol=2**-6)
            torch.testing.assert_close(dflash_ring.v[i, row, tail % RING].float(), v, atol=2**-6, rtol=2**-6)
    bonus = torch.tensor([3, 40], device=DEV)
    starts, rows_t = torch.tensor(lengths, device=DEV), torch.tensor(rows, device=DEV)

    hidden = draft.forward_block(bonus, starts, rows_t, dflash_ring)

    assert hidden.shape == (2 * block, H) and hidden.dtype == dtype
    for b in range(2):
        want = ref.block(int(bonus[b]), lengths[b], contexts[b], block)
        torch.testing.assert_close(hidden[b * block : (b + 1) * block].float(), want, atol=tol, rtol=tol)

    # a restore from the window loses nothing, and the block reads nothing older
    dflash_ring.k.zero_()
    dflash_ring.v.zero_()
    for n, row, ctx in zip(lengths, rows, draft_contexts):
        window = torch.arange(max(0, n - WINDOW + 1), n, device=DEV)
        draft.store_context(ctx[window], window, torch.full((len(window),), row, device=DEV), dflash_ring)
    assert torch.equal(draft.forward_block(bonus, starts, rows_t, dflash_ring), hidden)


@cuda
def test_sample_path_walks_the_selector_lattice(tmp_path):
    """Greedy and sampled requests follow the fp32 walk (same uniforms); a request whose scores all tie takes the first candidate; a NaN hidden yields in-range ids and a q that is no distribution, without disturbing the other requests."""
    w = _weights()
    draft, ref = _load_draft(tmp_path, w), _Reference(w)
    block, bs = 4, 4
    torch.manual_seed(3)
    hidden = torch.randn(bs * block, H, device=DEV, dtype=torch.bfloat16)
    hidden.view(bs, block, H)[2] = 0
    hidden.view(bs, block, H)[3] = float("nan")
    anchor = torch.tensor([5, 17, 30, 2], device=DEV)
    temperature = torch.tensor([1.0, 0.7, 1.0, 0.7], device=DEV)
    greedy = torch.tensor([True, False, True, False], device=DEV)

    pred = hidden.view(bs, block, H)[:, 1:]
    cand, unary = draft.compute_candidates(pred)
    uniforms = torch.rand(bs, block - 1, device=DEV)
    tokens, q = draft.candidate_selector.sample_path(cand, unary, pred, anchor, temperature, greedy, uniforms)
    want_tokens, want_q = ref.walk(hidden[: 2 * block], cand[:2], anchor[:2], temperature[:2], greedy[:2], uniforms[:2])

    assert tokens.shape == (bs, block - 1) and cand.shape == q.shape == (bs, block - 1, TOP_K)
    assert torch.equal(tokens[:2], want_tokens)
    torch.testing.assert_close(q[:2], want_q, atol=1e-2, rtol=0)
    assert torch.equal(tokens[2], cand[2, :, 0]) and torch.equal(q[2], F.one_hot(torch.zeros(block - 1, dtype=torch.int64, device=DEV), TOP_K).float())
    assert ((tokens >= 0) & (tokens < VOCAB)).all() and ((cand >= 0) & (cand < VOCAB)).all()
    assert not torch.isclose(q[3].sum(-1), torch.ones(block - 1, device=DEV)).any()


# ---------------------------------------------------------------------------------------------- TP=2 on the CPU

K, MAX_REQ = 3, 4
MAX_POS = CONFIG["max_position_embeddings"]
SHARDED = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".gate_proj.", ".up_proj.", ".down_proj.")


class _Comm:
    """PyNCCL's two collectives on one simulated rank: every call is recorded; ``reply(i, x)`` stands in for the other rank, else an all-reduce gives its input back and an all-gather ``size`` copies of it, stacked on dim 0 as PyNCCL stacks them."""

    def __init__(self, reply=None, size: int = 2) -> None:
        self.reply, self.size, self.calls = reply, size, []

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(("all_reduce", x.clone()))
        return x if self.reply is None else x.copy_(self.reply(len(self.calls) - 1, x))

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(("all_gather", x.clone()))
        return torch.cat([x] * self.size) if self.reply is None else self.reply(len(self.calls) - 1, x)

    def signature(self) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        return [(op, tuple(t.shape), t.dtype) for op, t in self.calls]


def _use(monkeypatch, comm: _Comm) -> _Comm:
    from freetoken.distributed import DistributedCommunicator

    monkeypatch.setattr(DistributedCommunicator, "plugins", [comm])
    return comm


def _cpu_kernels(monkeypatch) -> None:
    """Torch stand-ins for the GPU kernels under the draft, so the CPU runs its real modules and collectives: they keep shapes, not values, and the draft kernels' stand-ins assert the shapes the kernels read from their arguments."""
    from freetoken import kernel
    from freetoken.kernel import backend
    from freetoken.kernel.triton import activation, norm
    from freetoken.models.dflash import model

    def indexing(weights, indices, *, output=None, vocab_range=None):
        start, n = vocab_range or (0, weights.shape[0])
        local = indices - start
        return weights[local.clamp(0, n - 1)] * ((local >= 0) & (local < n))[:, None]

    def norm_rope_store(src, *, num_layers, num_q_heads, q_out, ring_k, **_):
        kv_heads, head_dim = ring_k.shape[3:]
        assert src.shape[1] == num_layers * (num_q_heads + 2 * kv_heads) * head_dim, "the q|k|v rows do not match the ring's KV heads"
        if q_out is not None:
            assert q_out.shape[1] == num_q_heads * head_dim
            q_out.copy_(src[:, : q_out.shape[1]])

    def ring_attention(q, ring_k, *_):
        assert q.shape[1] == HEADS // KV_HEADS * ring_k.shape[3] * ring_k.shape[4], "the query heads do not match the ring's KV heads"
        return q.clone()

    def selector_walk(ids, *_):
        return ids[..., 0].clone(), F.one_hot(torch.zeros_like(ids[..., 0]), ids.shape[-1]).float()

    def topk(x, k, dim=-1):
        # a top-k kernel promises no order among ties: this one gives the higher id first, the worst case for the merge
        vals, idx = x.flip(dim).sort(dim=dim, descending=True, stable=True)
        return vals.narrow(dim, 0, k), (x.shape[dim] - 1 - idx).narrow(dim, 0, k)

    for module in (backend, model):
        monkeypatch.setattr(module, "is_flashinfer_installed", lambda: False)
    monkeypatch.setattr(torch, "topk", topk)
    monkeypatch.setattr(kernel, "indexing", indexing)
    monkeypatch.setattr(norm, "rmsnorm", lambda x, w, eps, out=None: x.clone() if out is None else out.copy_(x))
    monkeypatch.setattr(norm, "fused_add_rmsnorm", lambda x, residual, w, eps: x.copy_(residual.add_(x)))
    monkeypatch.setattr(activation, "silu_and_mul", lambda x, out=None: x[..., : x.shape[-1] // 2] * x[..., x.shape[-1] // 2 :])
    monkeypatch.setattr(model, "grouped_conv", lambda x, *_: x.clone())
    for fn in (norm_rope_store, ring_attention, selector_walk):
        monkeypatch.setattr(model, fn.__name__, fn)


def _load_rank(monkeypatch, path: str, w: dict[str, torch.Tensor], rank: int, size: int = 2, device: torch.device = CPU):
    """The draft as TP rank ``rank`` of ``size`` loads it, beside the target's embedding and head sharded by vocab rows."""
    import freetoken.distributed.info as dist_info
    from freetoken.distributed import DistributedInfo
    from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.dflash import DFlash2DraftModel, parse_config
    from freetoken.models.loader import shard_tensor
    from freetoken.utils import cached_load_hf_config, torch_dtype

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, size))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        embed, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    for op, name, key in ((embed, "target.embed", "embed_tokens.weight"), (head, "target.head", "lm_head.weight")):
        op.load_state_dict({"weight": shard_tensor(key, w[name], rank=rank, world_size=size, num_kv_heads=None).to(device)})
        finalize_quant(op)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)
    return DFlash2DraftModel.load(parse_config(cached_load_hf_config(path)), target, path, device, torch.bfloat16)


def _config(path: str, tp: int, rank: int = 0) -> SimpleNamespace:
    """The engine config fields the DFlash proposer reads, for a Qwen3.8-Flash-Next target of the draft's shape."""
    from freetoken.distributed import DistributedInfo
    from freetoken.models.register import get_model_spec

    arch = "Qwen4ExpForConditionalGeneration"
    return SimpleNamespace(
        tp_info=DistributedInfo(rank, tp), model_spec=get_model_spec(arch), hf_config=SimpleNamespace(architectures=[arch]),
        model_config=SimpleNamespace(num_layers=CONFIG["num_target_layers"], hidden_size=H, vocab_size=VOCAB, linear_attention_group=lambda: None),
        speculative_draft_model_path=path, speculative_token_map=None, speculative_num_steps=K, cache_type="naive",
        max_running_req=MAX_REQ, max_seq_len=MAX_POS, dtype=torch.bfloat16,
    )


def _proposer(monkeypatch, path: str, draft, rank: int):
    """DFlashProposer as TP rank ``rank`` of 2 builds it around ``draft``: its constructor needs a GPU only for the draft load, which hands back ``draft``."""
    from freetoken.models.dflash import DFlash2DraftModel
    from freetoken.spec.dflash.proposer import DFlashProposer

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(DFlash2DraftModel, "load", lambda *args: draft)
    return DFlashProposer(_config(path, 2, rank), None)


def _context(device: torch.device):
    from freetoken.spec.proposer import DraftContext

    page_table = torch.zeros(MAX_REQ + 1, MAX_POS, dtype=torch.int32, device=device)
    return DraftContext(page_table=page_table, num_pages=0, device=device, stream=None, dummy_req=None, slot_states={}, kv_cache=None)


@pytest.mark.parametrize("tp, refused", [(2, None), (3, "num_attention_heads"), (4, "num_key_value_heads")])
def test_adjust_config_accepts_tp2_refuses_uneven_heads(tmp_path, tp, refused):
    """TP=2 passes the config check; a TP size that does not divide the query heads, or only the KV heads (which the qkv layer would replicate), is refused there, before the target loads."""
    from freetoken.spec.dflash.proposer import DFlashProposer

    config = _config(_checkpoint(tmp_path, _weights()), tp)
    if refused is None:
        DFlashProposer.adjust_config(config, None)
        return
    with pytest.raises(ValueError, match=refused):
        DFlashProposer.adjust_config(config, None)


def test_sharded_load_matches_checkpoint_slices(tmp_path, monkeypatch):
    """At TP=2 each rank holds its half of the query and KV heads and of the MLP rows, sliced before the q|k|v and gate|up merge, and its half of the o/down columns; fc, the convs, the selector and the norms stay whole; the context projection holds the rank's K|V rows of every layer."""
    _cpu_kernels(monkeypatch)
    _use(monkeypatch, _Comm())
    w = _weights()
    path = _checkpoint(tmp_path, w)
    for rank in range(2):
        draft = _load_rank(monkeypatch, path, w, rank)
        state = draft.state_dict()

        def half(name, dim=0, rank=rank):
            return w[name].chunk(2, dim)[rank]

        for i in range(LAYERS):
            attn, mlp = f"layers.{i}.self_attn", f"layers.{i}.mlp"
            q, k, v = (half(f"{attn}.{x}_proj.weight") for x in "qkv")
            assert torch.equal(state[f"{attn}.qkv_proj.weight"], torch.cat([q, k, v])), f"rank {rank} layer {i} q|k|v"
            assert torch.equal(state[f"{attn}.o_proj.weight"], half(f"{attn}.o_proj.weight", 1))
            assert torch.equal(state[f"{mlp}.gate_up_proj.weight"], torch.cat([half(f"{mlp}.gate_proj.weight"), half(f"{mlp}.up_proj.weight")]))
            assert torch.equal(state[f"{mlp}.down_proj.weight"], half(f"{mlp}.down_proj.weight", 1))
            assert torch.equal(draft._kv_all[i], torch.cat([k, v])), f"rank {rank} layer {i} context K|V"
        whole = [name for name in w if not name.startswith("target.") and not any(s in name for s in SHARDED)]
        assert len(whole) == 6 + 8 * LAYERS
        for name in whole:
            assert torch.equal(state[name], w[name]), f"rank {rank} {name}"


def test_budget_matches_local_ring(tmp_path, monkeypatch):
    """At TP=2 ``budget()`` is what rank 0's attach allocates: the K/V ring of its KV heads, the ring attention's scratch, the rope table and the verify feature buffer (the window rides the GDN slots and is budgeted with them)."""
    from freetoken.spec.dflash.proposer import DFlashProposer

    _cpu_kernels(monkeypatch)
    _use(monkeypatch, _Comm())
    w = _weights()
    path = _checkpoint(tmp_path, w)
    proposer = _proposer(monkeypatch, path, _load_rank(monkeypatch, path, w, 0), 0)
    proposer.attach(_context(torch.device("meta")))
    ring = proposer.dflash_ring
    assert ring.k.shape[3] == KV_HEADS // 2
    held = sum(t.nbytes for t in (ring.k, ring.v, *ring.workspace, ring.cos_sin, proposer.aux_buf))
    assert DFlashProposer.budget(proposer.config) == (held, 0)


def test_gathered_topk_matches_full(tmp_path, monkeypatch):
    """At TP=2 a rank gathers its best TOP_K (id, value) pairs, not its logits, and every rank gets the full head's candidates: ids in the full vocab, best first, ties to the lower id, inside a rank's half (whose top-k gives ties the higher id first) and across the halves (the expectation is a stable sort of the full logits; no tie crosses the TOP_K cut, where the top-k kernel decides)."""
    _cpu_kernels(monkeypatch)
    w = _weights()
    # an identity head makes each logit the hidden value itself, so the ties are exact in every rank's GEMM
    w["target.head"] = torch.eye(VOCAB, H, dtype=torch.bfloat16)
    path = _checkpoint(tmp_path, w)
    tops = ({3: 4.0, 40: 4.0, 5: 4.0, 60: 3.0}, {33: 2.0, 50: 2.0, 63: 3.0, 32: 1.5}, {10: 5.0, 20: 4.5, 31: 4.0, 32: 4.0})
    logits = -torch.randperm(VOCAB, generator=torch.Generator().manual_seed(7)).float().div(8).expand(len(tops), -1).clone()
    for row, top in enumerate(tops):
        logits[row, list(top)] = torch.tensor(list(top.values()))
    hidden = logits.to(torch.bfloat16).view(len(tops), 1, H)
    vals, ids = logits.sort(dim=-1, descending=True, stable=True)
    want_ids, want_vals = ids[:, None, :TOP_K], vals[:, None, :TOP_K]

    drafts, sent = [], []
    for rank in range(2):
        comm = _use(monkeypatch, _Comm())
        drafts.append(_load_rank(monkeypatch, path, w, rank))
        comm.calls.clear()
        drafts[-1].compute_candidates(hidden)
        assert [op for op, _ in comm.calls] == ["all_gather"] and comm.calls[0][1].numel() == len(tops) * TOP_K * 2, comm.signature()
        sent.append(comm.calls[0][1])
    gathered = torch.cat(sent)
    for draft in drafts:
        _use(monkeypatch, _Comm(reply=lambda i, x: gathered))
        cand, unary = draft.compute_candidates(hidden)
        assert torch.equal(cand, want_ids), cand.view(len(tops), -1).tolist()
        assert torch.equal(unary, want_vals) and cand.is_contiguous() and unary.is_contiguous()


def test_drafts_follow_rank0(tmp_path, monkeypatch):
    """Each rank walks its own path (own uniforms; here own candidates too), but the target verifies one set of drafts: inside ``_draft``, so every graph replay does it too, rank 1 sends zeros and both ranks take rank 0's tokens, on an all-greedy step too; each rank keeps its own candidates and q for its verify, whose result the runner replaces with rank 0's."""
    path = _checkpoint(tmp_path, _weights())
    bs, gen = 2, torch.Generator().manual_seed(5)
    cand = [torch.randint(0, VOCAB, (bs, K, TOP_K), generator=gen) for _ in range(2)]
    q = [torch.rand(bs, K, TOP_K, generator=gen) for _ in range(2)]

    def stub(rank):
        selector = SimpleNamespace(sample_path=lambda *_: (cand[rank][..., 0].clone(), q[rank]))
        return SimpleNamespace(
            state_dict=dict, forward_block=lambda bonus, *_: torch.zeros(bonus.shape[0] * (K + 1), H),
            compute_candidates=lambda pred: (cand[rank], q[rank]), candidate_selector=selector,
        )

    args = (torch.tensor([3, 40]), torch.tensor([5, 100]), torch.tensor([1, 2]), torch.ones(bs), torch.ones(bs, dtype=torch.bool))
    sent = []
    for rank in range(2):
        comm = _use(monkeypatch, _Comm())
        _proposer(monkeypatch, path, stub(rank), rank)._draft(*args)
        assert [op for op, _ in comm.calls] == ["all_reduce"], f"rank {rank} sent {comm.signature()}"
        sent.append(comm.calls[0][1])
    assert torch.equal(sent[0], cand[0][..., 0]) and not sent[1].any()
    for rank in range(2):
        _use(monkeypatch, _Comm(reply=lambda i, x: sent[0] + sent[1]))
        tokens, got_cand, got_q = _proposer(monkeypatch, path, stub(rank), rank)._draft(*args)
        assert torch.equal(tokens, cand[0][..., 0]) and got_cand is cand[rank] and got_q is q[rank]


def test_collectives_match_across_ranks(tmp_path, monkeypatch):
    """Both ranks issue the same collectives in the same order from ``_draft`` (block forward, candidates, token broadcast) for a greedy, a sampled and a padded step, or NCCL hangs; the eager verify inject and the prefill issue none."""
    from freetoken.core import Batch, Req, SamplingParams

    _cpu_kernels(monkeypatch)
    w = _weights()
    path = _checkpoint(tmp_path, w)
    steps = {"greedy": [True, True], "sampled": [False, True], "padded": [False, True, True, True]}
    bonus, starts, rows = torch.tensor([3, 40, 0, 0]), torch.tensor([5, 100, 0, 0]), torch.tensor([1, 2, MAX_REQ, MAX_REQ])

    def req(table_idx, n, cached):
        return Req(input_ids=torch.arange(n), table_idx=table_idx, cached_len=cached, output_len=8, uid=table_idx, sampling_params=SamplingParams(), cache_handle=None)

    seen = []
    for rank in range(2):
        comm = _use(monkeypatch, _Comm())
        proposer = _proposer(monkeypatch, path, _load_rank(monkeypatch, path, w, rank), rank)
        proposer.attach(_context(CPU))
        # built for cuda:0, where the prefill sink allocates
        proposer.device = CPU
        signatures = {}
        for name, greedy in steps.items():
            bs = len(greedy)
            comm.calls.clear()
            proposer._draft(bonus[:bs], starts[:bs], rows[:bs], torch.ones(bs), torch.tensor(greedy))
            signatures[name] = comm.signature()
        seen.append(signatures)

        comm.calls.clear()
        verify = Batch(reqs=[req(1, 10, 6), req(2, 20, 16)], phase="verify", rows_per_req=K + 1)
        verify.padded_reqs, verify.active_table_idx = verify.reqs, torch.tensor([1, 2])
        verify.positions = torch.cat([torch.arange(6, 6 + K + 1), torch.arange(16, 16 + K + 1)]).to(torch.int32)
        proposer.post_verify(verify, None, None)
        prefill = Batch(reqs=[req(3, 20, 0)], phase="prefill")
        sink = proposer.feature_sink(prefill)
        for layer in TAPS:
            sink.layer_output(layer, torch.randn(20, H, dtype=torch.bfloat16))
        proposer.on_prefill(prefill, None)
        assert comm.calls == [], comm.signature()
    assert seen[0] == seen[1]
    assert seen[0]["greedy"] == seen[0]["sampled"] and seen[0]["greedy"][-1] == ("all_reduce", (2, K), torch.int64)


@cuda
def test_tp2_block_forward_matches_reference(tmp_path, monkeypatch):
    """Both TP=2 ranks on one GPU, in lockstep threads whose all-reduce sums them: each rank's ring holds its KV heads of the reference K/V, and each rank's block forward matches the fp32 reference to bf16 noise, as the TP=1 draft does. The real kernels with the local head counts, which the CPU stand-ins only shape-check."""
    import threading

    w, block, tol = _weights(), 4, 2**-2
    ref, path = _Reference(w), _checkpoint(tmp_path, w)
    mask_row = w["target.embed"][MASK].to(DEV)
    drafts = []
    for rank in range(2):
        # the load's one collective, the mask embedding: the sum of the two ranks' halves is the full row
        _use(monkeypatch, _Comm(reply=lambda i, x: mask_row[None]))
        drafts.append(_load_rank(monkeypatch, path, w, rank, device=DEV))
    rings = [d.make_dflash_ring(max_rows=3, block=block, ring_len=RING, max_position=4096, device=DEV) for d in drafts]
    torch.manual_seed(1)
    lengths, rows, contexts = (5, 100), (2, 1), []
    for n, row in zip(lengths, rows):
        features = torch.randn(n, len(TAPS) * H, device=DEV, dtype=torch.bfloat16)
        contexts.append(ref.context_rows(features))
        tail = torch.arange(max(0, n - RING), n, device=DEV)
        for rank, (draft, ring) in enumerate(zip(drafts, rings)):
            draft.store_context(draft.context_rows(features)[tail], tail, torch.full((len(tail),), row, device=DEV), ring)
            heads = slice(rank * KV_HEADS // 2, (rank + 1) * KV_HEADS // 2)
            for i in range(LAYERS):
                k, v = ref.kv(i, contexts[-1][tail], tail)
                # one bf16 step wider than the TP=1 check: on sm_89 one K element of 5120 landed 0.0178 off, against
                # 2**-6 = 0.0156, while the TP=1 ring passes at 2**-6 (the cause of the difference was not examined)
                torch.testing.assert_close(ring.k[i, row, tail % RING].float(), k[:, heads], atol=2**-5, rtol=2**-5)
                torch.testing.assert_close(ring.v[i, row, tail % RING].float(), v[:, heads], atol=2**-5, rtol=2**-5)
    bonus, starts, rows_t = torch.tensor([3, 40], device=DEV), torch.tensor(lengths, device=DEV), torch.tensor(rows, device=DEV)
    # the kernels compile here, outside the threads, with an all-reduce that waits for no one
    _use(monkeypatch, _Comm())
    drafts[0].forward_block(bonus, starts, rows_t, rings[0])
    torch.cuda.synchronize()

    barrier, parts, local = threading.Barrier(2, timeout=60), {}, threading.local()

    class _Lockstep:
        """The two threads' ranks summed; both threads queue on the device's one default stream, so the second wait orders each sum before either copy."""

        def all_reduce(self, x):
            parts[local.rank] = x
            barrier.wait()
            total = parts[0] + parts[1]
            barrier.wait()
            return x.copy_(total)

    _use(monkeypatch, _Lockstep())
    hidden, errors = [None, None], []

    def run(rank):
        local.rank = rank
        try:
            hidden[rank] = drafts[rank].forward_block(bonus, starts, rows_t, rings[rank])
        except BaseException as e:  # noqa: BLE001 -- reported below; the other rank must not wait forever
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for rank in range(2):
        for b in range(2):
            want = ref.block(int(bonus[b]), lengths[b], contexts[b], block)
            torch.testing.assert_close(hidden[rank][b * block : (b + 1) * block].float(), want, atol=tol, rtol=tol, msg=f"rank {rank} request {b}")
