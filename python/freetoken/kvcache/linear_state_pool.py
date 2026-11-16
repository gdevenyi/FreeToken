from __future__ import annotations

import math
from typing import List

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec, SlotVerify
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads, k_heads)`` for the GDN state tensors -- the single
    source of the sharding math shared by the pool allocation and the byte estimate."""
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
    return len(group.layer_ids), local_conv_dim, local_v_heads, local_k_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Indexed by ``Req.table_idx`` (0..max_running_req), the same per-request slot the
    page table uses, so the scheduler's existing admit/free of ``table_idx`` covers the
    state's lifetime. One fixed slot per running request; no paging, no eviction.

    A model (``ModelConfig.slot_states``) or a proposer (``Proposer.slot_states``) can declare
    extra per-request tensors on the same slots; the pool gets both as ``EngineConfig.slot_states``
    (see ``SlotStateSpec``). ``copy_from``, ``reset`` and ``rebuild`` cover them with the GDN state; their owner writes their prefill track snapshot. They are read back through ``slot_state(name, layer_id)``.
    Consumers must re-read them each forward: ``rebuild`` replaces the tensors.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
        slot_states: tuple[SlotStateSpec, ...] = (),
        host_slots: int = 0,
        engine_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size

        self._group = group
        self._num_slots = num_slots
        self._device = device
        self._conv_dtype = dtype

        n_layers, local_conv_dim, local_v_heads, self._local_k_heads = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}

        self._slot_specs = tuple(slot_states)
        names = [spec.name for spec in self._slot_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate slot_state names: {names}")
        self._state_layer_index = {
            spec.name: {lid: i for i, lid in enumerate(spec.layer_ids)}
            for spec in self._slot_specs
        }
        self.slot_states: dict[str, torch.Tensor] = self._alloc_slot_states(num_slots)

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as a padding sink,
        # sglang MambaPool convention). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single free-list, so memory
        # flows between them by demand. Unused by the op harness (which assigns slots by hand).
        self.padding_slot = 0
        # verify batches only read the live state; Engine attaches this when a decode step has more than one row
        self.verify_scratch: LinearVerifyScratch | None = None
        self._verify_handlers: dict[str, SlotVerify] = {}
        self._free_slots: list[int] = list(range(1, num_slots))
        self._alloc_host_bank(host_slots, engine_stream)

    def _slot_tensors(self) -> dict[str, torch.Tensor]:
        """Every tensor a slot holds, ``[layers, num_slots, ...]`` each (re-read: rebuild replaces them)."""
        return {"conv": self.conv_states, "recurrent": self.recurrent_states, **self.slot_states}

    def _alloc_host_bank(self, host_slots: int, engine_stream: torch.cuda.Stream | None) -> None:
        """--mamba-host-slots: one pinned host buffer per bank slot, so registration goes in pieces
        of one slot, holding each slot tensor's layer rows (KV host tier design 3.1, 3.7)."""
        from freetoken.moe.host_banks import HostBank

        self._num_host_slots = max(0, int(host_slots))
        self._free_host_slots: list[int] = list(range(self._num_host_slots))
        self._host_views: list[dict[str, torch.Tensor]] = []
        self.host_copies = _HostCopies(self._device, engine_stream)
        if not self._num_host_slots:
            return
        layout, offset = {}, 0
        for name, t in self._slot_tensors().items():
            nbytes = t[:, 0].numel() * t.element_size()
            layout[name] = (offset, t.dtype, (t.shape[0], *t.shape[2:]))
            offset += -(-nbytes // 16) * 16  # 16-byte aligned, so each dtype view is legal
        cuda = self._device.type == "cuda"
        for _ in range(self._num_host_slots):
            bank = HostBank((offset,), torch.uint8, backing=None if cuda else "mmap")
            if cuda:
                bank.pin()  # page-locked and device-mapped: the copies are async on the copy engine
            self._host_views.append({
                name: bank.tensor[o : o + math.prod(shape) * torch.empty((), dtype=dt).element_size()]
                .view(dt).view(shape)
                for name, (o, dt, shape) in layout.items()
            })

    @property
    def num_host_slots(self) -> int:
        return self._num_host_slots

    @property
    def num_free_host_slots(self) -> int:
        return len(self._free_host_slots)

    def host_bytes_per_slot(self) -> int:
        if not self._host_views:
            return 0
        return sum(t.numel() * t.element_size() for t in self._host_views[0].values())

    def alloc_host(self, n: int = 1) -> list[int]:
        if n > len(self._free_host_slots):
            raise RuntimeError(f"GDN host bank exhausted: need {n}, have {len(self._free_host_slots)}")
        return [self._free_host_slots.pop() for _ in range(n)]

    def free_host(self, slots) -> None:
        self._free_host_slots.extend(int(s) for s in slots)

    def reclaim_all_host_slots(self) -> None:
        """Idle-only, as ``reclaim_all_slots``: the caller discards the tree that owned the bank slots."""
        self._free_host_slots = list(range(self._num_host_slots))

    def to_host(self, src: int, dst: int) -> None:
        """Copy device slot ``src`` into bank slot ``dst`` (the caller then frees ``src``). Each layer row
        is contiguous on both sides, so every copy is a plain memcpy and no temporary is made."""
        self.host_copies.to_host([
            (self._host_views[dst][name][li], t[li, src])
            for name, t in self._slot_tensors().items() for li in range(t.shape[0])
        ])

    def from_host(self, src: int, dst: int) -> None:
        """Copy bank slot ``src`` into device slot ``dst``, on the caller's (engine) stream after ``fence_host_copies``."""
        self.host_copies.from_host([
            (t[li, dst], self._host_views[src][name][li])
            for name, t in self._slot_tensors().items() for li in range(t.shape[0])
        ])

    def fence_host_copies(self) -> None:
        """Make the caller's stream wait for every ``to_host`` so far. The scheduler calls it on the
        engine stream before each forward, so no slot a demotion freed is written before its copy."""
        self.host_copies.fence()

    def _alloc_slot_states(self, num_slots: int) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.full(
                (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                spec.fill_value,
                dtype=self._state_dtype(spec),
                device=self._device,
            )
            for spec in self._slot_specs
        }

    def _state_dtype(self, spec: SlotStateSpec) -> torch.dtype:
        return spec.dtype if spec.dtype is not None else self._conv_dtype

    def has_slot_state(self, name: str) -> bool:
        return name in self.slot_states

    def slot_state(self, name: str, layer_id: int | None = None) -> torch.Tensor:
        """One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks the layer row."""
        t = self.slot_states[name]
        if layer_id is None:
            assert not self._state_layer_index[name], (
                f"slot_state {name!r} is per-layer, pass layer_id"
            )
            return t[0]
        return t[self._state_layer_index[name][layer_id]]

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a
        CacheManager rebuild that discards the tree owning donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice."""
        self._free_slots = list(range(1, self._num_slots))

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only and destructive: every
        live/snapshot state is dropped, so the caller must guarantee no running request
        holds a slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = None
        self.recurrent_states = None
        self.slot_states = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        self.slot_states = self._alloc_slot_states(num_slots)
        self._num_slots = num_slots
        self._free_slots = list(range(1, num_slots))
        self.reclaim_all_host_slots()  # the bank keeps its geometry; its tree is discarded too

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, slots] = spec.fill_value

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot)."""
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, table_idx] = spec.fill_value

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def attach_verify_scratch(self, max_bs: int, rows_per_req: int) -> None:
        """Build the scratch for verify batches of up to ``max_bs`` requests with ``rows_per_req`` rows each, and the verify handler of every slot state; build_fla_metadata hands the GDN scratch to the GDN layers."""
        self.verify_scratch = LinearVerifyScratch(self, max_bs, rows_per_req)
        self._verify_handlers = {
            spec.name: spec.verify(spec, max_bs, rows_per_req, self._device, self._state_dtype(spec))
            for spec in self._slot_specs
            if spec.verify is not None
        }

    def verify_handler(self, name: str) -> SlotVerify:
        """The verify handler of slot state ``name``, for its owner's verify forward."""
        return self._verify_handlers[name]

    def commit_verify(self, slots: List[int], counts: List[int]) -> None:
        """Advance live slot slots[i] over verify row i's first counts[i] tokens."""
        self.verify_scratch.commit(self, slots, counts)
        if slots and self._verify_handlers:
            index = torch.tensor([slots, counts], dtype=torch.int64, pin_memory=self._device.type == "cuda")
            index = index.to(self._device, non_blocking=True)
            for handler in self._verify_handlers.values():
                handler.commit(self, *index)

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        for t in self.slot_states.values():
            per += t[:, 0].numel() * t.element_size()
        return int(per)


class LinearVerifyScratch:
    """Per-row scratch of a GDN verify: each row's recurrence inputs and conv window, keyed by verify-batch row.

    The verify leaves the live recurrent state untouched and rolls the live conv window as if every row were
    accepted; ``commit`` replays the accepted rows into the recurrent state and restores the conv window that ends
    at the last accepted row. Dense and indexed by batch row, never by pool slot.
    """

    def __init__(self, pool: LinearStatePool, max_bs: int, rows_per_req: int) -> None:
        num_layers, _, hv, k_dim, v_dim = pool.recurrent_states.shape
        _, _, conv_dim, km1 = pool.conv_states.shape
        hk = pool._local_k_heads
        device = pool.recurrent_states.device
        rows = (num_layers, max_bs, rows_per_req)
        f32 = dict(dtype=torch.float32, device=device)
        self.stash = (
            torch.zeros(*rows, hk, k_dim, **f32),
            torch.zeros(*rows, hv, v_dim, **f32),
            torch.zeros(*rows, hv, **f32),
            torch.zeros(*rows, hv, **f32),
        )
        self.conv_window = torch.zeros((*rows, conv_dim, km1), dtype=pool.conv_states.dtype, device=device)

    def input_stash(self, li: int) -> tuple:
        return tuple(t[li] for t in self.stash)

    def commit(self, pool: LinearStatePool, slots: List[int], counts: List[int]) -> None:
        """Advance live slot slots[i] over row i's first counts[i] verified tokens (count >= 1)."""
        from freetoken.kernel.fla import gdn_state_replay

        if not slots:
            return
        rows = list(range(len(slots)))
        meta = torch.tensor([rows, slots, counts], dtype=torch.int32, pin_memory=pool.recurrent_states.is_cuda)
        meta = meta.to(pool.recurrent_states.device, non_blocking=True)
        gdn_state_replay(pool.recurrent_states, self.stash, meta)
        row_idx, slot_idx, count = meta.long()
        pool.conv_states[:, slot_idx] = self.conv_window[:, row_idx, count - 1]


class _HostCopies:
    """The copies between device slots and the GDN host bank. CUDA: a device->host copy runs on a side
    stream (the copy engine) behind the caller's and the engine stream; ``fence`` makes the caller's
    stream wait for all of them. A host->device copy runs on the caller's stream. CPU: plain copies.
    No decision reads the state of a copy: the copies only order streams."""

    def __init__(self, device: torch.device, engine_stream: torch.cuda.Stream | None = None) -> None:
        self.side = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.engine_stream = engine_stream
        self._pending = False

    def to_host(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if self.side is None:
            for dst, src in pairs:
                dst.copy_(src)
            return
        cur = torch.cuda.current_stream()
        self.side.wait_stream(cur)
        if self.engine_stream is not None and self.engine_stream != cur:
            self.side.wait_stream(self.engine_stream)  # the forwards that wrote the snapshot
        with torch.cuda.stream(self.side):
            for dst, src in pairs:
                dst.copy_(src, non_blocking=True)
        self._pending = True

    def from_host(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for dst, src in pairs:
            dst.copy_(src, non_blocking=self.side is not None)

    def fence(self) -> None:
        if self._pending:
            torch.cuda.current_stream().wait_stream(self.side)
            self._pending = False


# KV host tier design 3.7: the host bank must leave this much MemAvailable
HOST_BANK_MIN_AVAILABLE = 32 * 2**30


def read_mem_available() -> int:
    """``MemAvailable`` of /proc/meminfo, in bytes."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def check_host_bank_memory(nbytes: int, mem_available: int | None = None) -> None:
    """Refuse a GDN host bank of ``nbytes`` (the sum over the local TP ranks) that would leave less
    than 32 GiB MemAvailable."""
    available = read_mem_available() if mem_available is None else mem_available
    if available - nbytes < HOST_BANK_MIN_AVAILABLE:
        raise RuntimeError(
            f"--mamba-host-slots asks for {nbytes / 2**30:.1f} GiB of pinned host memory with "
            f"MemAvailable {available / 2**30:.1f} GiB: less than "
            f"{HOST_BANK_MIN_AVAILABLE // 2**30} GiB would remain"
        )


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
    slot_states: tuple[SlotStateSpec, ...] = (),
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local), plus any
    declared slot_states."""
    n_layers, local_conv_dim, local_v_heads, _ = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    total = n_layers * (conv_bytes + rec_bytes)

    for spec in slot_states:
        item = (spec.dtype if spec.dtype is not None else dtype).itemsize
        total += max(1, len(spec.layer_ids)) * math.prod(spec.shape) * item
    return int(total)


__all__ = ["LinearStatePool", "check_host_bank_memory", "linear_state_bytes_per_req", "read_mem_available"]


_GDN_CHUNK_SIZE = 64  # keep in lockstep with kernel/fla/chunk_delta_h.py CHUNK_SIZE


def gdn_prefill_workspace_bytes(config) -> int:
    """Bytes of the GDN chunked-prefill buffers that scale with the extend length.

    ``chunk_gated_delta_rule_fwd_h`` materializes the per-chunk recurrent states for
    the WHOLE extend at once -- ``[NT, H, V, K]`` with ``NT = ceil(extend / 64)`` --
    plus the ``v_new`` value buffer (288 MiB at an 8192-token extend for qwen4_exp).
    This is a lower bound on the GDN prefill peak, not the peak: w/u/A/o and the
    in_proj/conv activations still come out of the ``(1 - memory_ratio)`` headroom.
    0 for non-GDN models."""
    linear_group = config.model_config.linear_attention_group()
    if linear_group is None:
        return 0
    # EngineConfig has no max_extend_tokens (the served config is a SchedulerConfig subclass)
    extent = min(getattr(config, "max_extend_tokens", 8192), config.max_seq_len)
    nt = -(-extent // _GDN_CHUNK_SIZE)
    states = (
        nt
        * linear_group.num_value_heads
        * linear_group.value_head_dim
        * linear_group.key_head_dim
    )
    v_new = extent * linear_group.num_value_heads * linear_group.value_head_dim
    return (states + v_new) * config.dtype.itemsize


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup
    slot count). The engine adds this to the KV family's fixed cost when budgeting --
    the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    slot_states = getattr(config, "slot_states", ())
    if linear_group is None:
        if slot_states:
            raise ValueError("slot_states ride the linear-state slots; model has no linear group")
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    per_req = linear_state_bytes_per_req(
        linear_group, config.tp_info.size, config.dtype, slot_states
    )
    return per_req * slots + verify_scratch_bytes(config, getattr(config, "decode_rows_per_req", 1))


def verify_scratch_bytes(config, rows_per_req: int) -> int:
    """Bytes of the LinearVerifyScratch the engine builds for verify batches of ``rows_per_req`` rows per request (0 when it is 1)."""
    group = config.model_config.linear_attention_group()
    if group is None or rows_per_req <= 1:
        return 0
    n_layers, conv_dim, hv, hk = _linear_local_dims(group, config.tp_info.size)
    rows = n_layers * config.max_running_req * rows_per_req
    stash = rows * (hk * group.key_head_dim + hv * group.value_head_dim + 2 * hv) * 4
    window = rows * conv_dim * (group.conv_kernel_dim - 1) * config.dtype.itemsize
    handlers = sum(
        spec.verify.nbytes(spec, config.max_running_req, rows_per_req, spec.dtype if spec.dtype is not None else config.dtype)
        for spec in getattr(config, "slot_states", ())
        if spec.verify is not None
    )
    return stash + window + handlers


def _linear_pool_num_slots(config) -> int:
    """LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request
    (1 live + 2 ping-pong + 1 committed snapshot locked through decode), plus a cross-request
    snapshot cache and a padding sink; naive GDN keeps the old (max_running_req + 1)."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    ratio = config.linear_state_cache_ratio
    n_cache = max(4, int(ratio * mr))
    return 4 * mr + n_cache + 1  # live + 2 ping-pong + locked committed snapshot + cache + padding


def _linear_pool_min_slots(config) -> int:
    """Floor on LinearStatePool slots that still runs: the non-evictable working set with a
    zero snapshot cache. Hybrid-radix needs 4 per running request (1 live + 2 ping-pong + 1
    committed snapshot locked through decode) + the padding sink; naive needs 1 per request +
    padding. Below this, a full max_running_req batch can't get its slots and admission
    deadlocks -- so a runtime rebuild rejects a smaller request."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    return 4 * mr + 1
