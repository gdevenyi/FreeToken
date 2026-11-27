from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class LogitsPlan:
    """Inputs of the logits processors for the rows of one batch that asked for any
    (logit_bias, min_tokens, min_p). The penalties are not here: the sampler applies
    them from per-request device state.

    Built on the host in ``Sampler.prepare`` from the requests' SamplingParams, applied
    on the device by ``apply_logits_processors`` before the sampling kernel. ``None``
    when no request in the batch uses a processor, so the default path stays exactly
    as it was. Row indices inside the plan are LOCAL (0..r-1, in the order of ``rows``);
    ``rows`` maps them back to the batch.
    """

    rows: torch.Tensor  # [r] int64 batch rows
    temps: torch.Tensor  # [r, 1] float32 temperatures (min_p works on probabilities)
    bias_rows: torch.Tensor | None = None  # [b] int64 local rows
    bias_ids: torch.Tensor | None = None  # [b] int64
    bias_vals: torch.Tensor | None = None  # [b] float32
    # Rows still under min_tokens and the ids (EOS + stop_token_ids, padded with
    # vocab_size, a scratch column that is dropped) they must not sample yet.
    min_rows: torch.Tensor | None = None  # [m] int64 local rows
    min_ids: torch.Tensor | None = None  # [m, S] int64
    min_left: torch.Tensor | None = None  # [m] int64 tokens the row must still generate
    min_left_host: list[int] | None = None
    min_p: torch.Tensor | None = None  # [r, 1] float32 (0 = off for that row)
    # Rows with allowed_token_ids and their ids, padded with vocab_size like min_ids.
    allow_rows: torch.Tensor | None = None  # [a] int64 local rows
    allow_ids: torch.Tensor | None = None  # [a, A] int64
    # Grammar-constrained rows and their packed int32 vocabulary bitmasks ([g, ceil(V/32)]).
    grammar_rows: torch.Tensor | None = None  # [g] int64 local rows
    grammar_bitmask: torch.Tensor | None = None  # [g, W] int32 on device
    # True when grammar_rows/grammar_bitmask are already expanded across verify slots 0..k
    # (so _verify_plan must pass them through instead of repeating slot 0).
    grammar_per_verify_row: bool = False


class RowRepetition(NamedTuple):
    """One batch row's repetition_penalty. ``counts`` are the generated-token counts that the
    row's ``penalties`` entry keeps current on the device; ``prompt_mask`` [V] bool marks the
    prompt's ids."""

    row: int
    counts: torch.Tensor
    penalty: float
    prompt_mask: torch.Tensor


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    penalties: list[tuple[int, torch.Tensor, float, float]] = field(
        default_factory=list
    )
    repetitions: list[RowRepetition] = field(default_factory=list)
    plan: LogitsPlan | None = None
    # Rows that asked for logprobs, as an int64 device index built on the host (no nonzero
    # sync). None with logprob_all_rows set when every row asks: no gather then.
    logprob_row_idx: torch.Tensor | None = None
    logprob_all_rows: bool = False
    max_top_logprobs: int = 0
    # Logprob rows restricted to their allowed_token_ids, indexing the logprob rows above.
    restrict_rows: torch.Tensor | None = None  # [q] int64
    restrict_ids: torch.Tensor | None = None  # [q, A] int64, padded with vocab_size
    # Logprob rows with an explicit candidate set (allowed_token_ids or logprob_token_ids),
    # indexing the logprob rows above so candidates are gathered directly without full-vocab topk.
    explicit_rows: torch.Tensor | None = None  # [e] int64
    explicit_ids: torch.Tensor | None = None  # [e, M] int64, padded with vocab_size
    explicit_include_sampled: torch.Tensor | None = None  # [e] bool
    need_full_topk: bool = False
    # Non-greedy rows with a deterministic seed: (batch_row, seed, gen_len, temp, top_k, top_p).
    seeded_rows: list[tuple[int, int, int, float, int, float]] = field(
        default_factory=list
    )


def mix_step_seed(seed: int, step: int) -> int:
    """SplitMix64 mix of ``(seed, step)`` into a non-negative 63-bit torch.Generator seed."""
    x = (
        (int(seed) & 0xFFFFFFFFFFFFFFFF) + (int(step) * 0x9E3779B97F4A7C15)
    ) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0x7FFFFFFFFFFFFFFF


def sample_seeded_row(
    row_logits: torch.Tensor,
    *,
    seed: int,
    step: int,
    temp: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Batch-invariant Gumbel-max sample for one 1D logits row using its per-step seed."""
    vocab_size = row_logits.shape[-1]
    scaled = row_logits.float() / max(temp, 1e-6)
    if 0 < top_k < vocab_size:
        vals, idx = torch.topk(scaled, k=top_k, dim=-1)
        masked = torch.full_like(scaled, float("-inf"))
        masked.scatter_(-1, idx, vals)
        scaled = masked
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = (cum - sorted_probs) >= top_p
        drop[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    gen = torch.Generator(device=row_logits.device)
    gen.manual_seed(mix_step_seed(seed, step))
    noise = torch.empty_like(probs).exponential_(1.0, generator=gen).clamp_min_(1e-30)
    return torch.argmax(probs / noise, dim=-1)


def make_device_tensor(
    data: list, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if device.type != "cuda":  # the unit tests build plans on the CPU
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


def _padded_rows(rows: list[list[int]], pad: int, device: torch.device) -> torch.Tensor:
    """[len(rows), max_len] int64 with ``pad`` on the right."""
    width = max(1, max((len(r) for r in rows), default=1))
    return make_device_tensor(
        [r + [pad] * (width - len(r)) for r in rows], torch.int64, device
    )


def disallowed_mask(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """[a, V] bool, True outside each row's ids (the vocab_size padding column is dropped)."""
    block = torch.ones(
        (ids.shape[0], vocab_size + 1), dtype=torch.bool, device=ids.device
    )
    block.scatter_(1, ids, False)
    return block[:, :vocab_size]


def apply_penalties(logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    """A float32 copy of ``logits`` with each row's penalties applied in vLLM's order:
    repetition (over prompt + generated ids), then frequency and presence (generated)."""
    logits = logits.float().clone()
    for row, counts, penalty, prompt_mask in args.repetitions:
        x = logits[row]
        # HF / vLLM semantics: divide positive logits, multiply negative ones.
        penalized = torch.where(x > 0, x / penalty, x * penalty)
        logits[row] = torch.where((counts > 0) | prompt_mask, penalized, x)
    for row, counts, presence, frequency in args.penalties:
        if presence == 0.0 and frequency == 0.0:
            continue
        logits[row] -= frequency * counts + presence * (counts > 0)
    return logits


def apply_logits_processors(
    logits: torch.Tensor, plan: LogitsPlan, vocab_size: int
) -> torch.Tensor:
    """A float32 copy of ``logits`` with the plan applied to its rows: logit_bias, the
    allowed_token_ids and min_tokens masks, then min_p. Rows outside the plan are copied
    unchanged."""
    out = logits.to(torch.float32, copy=True)
    sub = out[plan.rows]  # [r, V] (advanced indexing copies)
    device = sub.device
    neg_inf = float("-inf")

    if plan.bias_ids is not None:
        sub.index_put_((plan.bias_rows, plan.bias_ids), plan.bias_vals, accumulate=True)

    if plan.allow_ids is not None:
        rows = sub[plan.allow_rows]
        sub[plan.allow_rows] = rows.masked_fill(
            disallowed_mask(plan.allow_ids, vocab_size), neg_inf
        )

    if plan.grammar_bitmask is not None and plan.grammar_rows is not None:
        from freetoken.grammar.bitmask import apply_token_bitmask_inplace

        # Apply grammar bitmask BEFORE min_ids (whose has_valid guard ensures min_tokens
        # never masks out the sole grammar-valid token such as EOS after '}') and BEFORE
        # min_p (so softmax runs over grammar-valid tokens only).
        apply_token_bitmask_inplace(
            sub, plan.grammar_bitmask, plan.grammar_rows, vocab_size
        )

    if plan.min_ids is not None:
        block = torch.zeros(
            (plan.min_ids.shape[0], vocab_size + 1), dtype=torch.bool, device=device
        )
        block.scatter_(1, plan.min_ids, torch.ones_like(plan.min_ids, dtype=torch.bool))
        rows = sub[plan.min_rows]
        masked = torch.where(
            block[:, :vocab_size], torch.full_like(rows, neg_inf), rows
        )
        has_valid = (masked > neg_inf).any(dim=-1, keepdim=True)
        sub[plan.min_rows] = torch.where(has_valid, masked, rows)

    if plan.min_p is not None:
        probs = torch.softmax(sub / plan.temps, dim=-1)
        keep = probs >= plan.min_p * probs.amax(dim=-1, keepdim=True)
        keep = keep | (plan.min_p <= 0.0)  # rows without min_p stay untouched
        sub = torch.where(keep, sub, torch.full_like(sub, neg_inf))

    out[plan.rows] = sub
    return out


def _verify_plan(plan: LogitsPlan, width: int) -> LogitsPlan:
    """``plan`` over a verify batch of ``width`` rows per request. Row j of a request stands
    j tokens further along, so min_tokens releases its mask j rows early."""
    offsets = torch.arange(width, device=plan.rows.device)

    def rows(t: torch.Tensor) -> torch.Tensor:
        return (t.unsqueeze(1) * width + offsets).flatten()

    def repeat(t: torch.Tensor) -> torch.Tensor:
        return t.repeat_interleave(width, dim=0)

    out = LogitsPlan(rows=rows(plan.rows), temps=repeat(plan.temps))
    if plan.bias_ids is not None:
        out.bias_rows = rows(plan.bias_rows)
        out.bias_ids = repeat(plan.bias_ids)
        out.bias_vals = repeat(plan.bias_vals)
    if plan.min_ids is not None:
        if plan.min_left_host is not None:
            pairs = [
                (e_idx, j_idx)
                for e_idx, left in enumerate(plan.min_left_host)
                for j_idx in range(min(width, max(0, int(left))))
            ]
            if pairs:
                entry = make_device_tensor(
                    [p[0] for p in pairs], torch.int64, plan.rows.device
                )
                j = make_device_tensor(
                    [p[1] for p in pairs], torch.int64, plan.rows.device
                )
                out.min_rows = plan.min_rows[entry] * width + j
                out.min_ids = plan.min_ids[entry]
        else:
            entry, j = (offsets < plan.min_left.unsqueeze(1)).nonzero(as_tuple=True)
            out.min_rows = plan.min_rows[entry] * width + j
            out.min_ids = plan.min_ids[entry]
    if plan.allow_ids is not None:
        out.allow_rows = rows(plan.allow_rows)
        out.allow_ids = repeat(plan.allow_ids)
    if plan.grammar_bitmask is not None and plan.grammar_rows is not None:
        if plan.grammar_per_verify_row:
            out.grammar_rows = plan.grammar_rows
            out.grammar_bitmask = plan.grammar_bitmask
            out.grammar_per_verify_row = True
        else:
            out.grammar_rows = rows(plan.grammar_rows)
            out.grammar_bitmask = repeat(plan.grammar_bitmask)
    if plan.min_p is not None:
        out.min_p = repeat(plan.min_p)
    return out


def sampling_module(device: torch.device | None = None):
    """flashinfer.sampling when installed, else triton on sm_70+ CUDA or torch on CPU/older GPUs."""
    if device is not None and device.type != "cuda":
        from freetoken.kernel import torch_sampling as sampling

        return sampling
    import freetoken.utils.arch as arch
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import sampling
    elif not arch.is_sm70_supported():
        from freetoken.kernel import torch_sampling as sampling
    else:
        from freetoken.kernel.triton import sampling
    return sampling


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    sampling = sampling_module(logits.device)
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _has_active_grammar(req) -> bool:
    return getattr(req, "grammar_matcher", None) is not None and not getattr(
        req, "grammar_waiting_reasoning_end", False
    )


def _ends_with_ids(seq: list[int], suffix: tuple[int, ...]) -> bool:
    n = len(suffix)
    return n > 0 and len(seq) >= n and tuple(seq[-n:]) == suffix


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int
    _grammar_host_buf: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    def _get_grammar_host_buf(self, num_rows: int) -> torch.Tensor:
        from freetoken.grammar.bitmask import allocate_host_bitmask, bitmask_word_count

        words = bitmask_word_count(self.vocab_size)
        buf = self._grammar_host_buf
        if buf is None or buf.shape[0] < num_rows or buf.shape[1] != words:
            cap = max(num_rows, 16)
            buf = allocate_host_bitmask(
                cap, self.vocab_size, pin_memory=(self.device.type == "cuda")
            )
            self._grammar_host_buf = buf
        view = buf[:num_rows]
        view.fill_(-1)
        return view

    def _plan(self, batch: Batch, params) -> LogitsPlan | None:
        """The LogitsPlan for this batch, or None when no request asks for a processor."""
        MIN_T = 1e-6
        is_verify = getattr(batch, "phase", None) == "verify"
        rows = [
            i
            for i, p in enumerate(params)
            if (
                p.needs_logits_processing
                or (not is_verify and _has_active_grammar(batch.reqs[i]))
            )
            and getattr(batch.reqs[i], "can_decode", True)
        ]
        if not rows:
            return None
        picked = [(batch.reqs[i], params[i]) for i in rows]
        temps = [[max(0.0 if p.is_greedy else p.temperature, MIN_T)] for _, p in picked]
        plan = LogitsPlan(
            rows=make_device_tensor(rows, torch.int64, self.device),
            temps=make_device_tensor(temps, torch.float32, self.device),
        )
        pad = self.vocab_size

        bias_rows: list[int] = []
        bias_ids: list[int] = []
        bias_vals: list[float] = []
        for local, (_, p) in enumerate(picked):
            for tid, val in p.logit_bias or ():
                if 0 <= int(tid) < self.vocab_size:
                    bias_rows.append(local)
                    bias_ids.append(int(tid))
                    bias_vals.append(float(val))
        if bias_ids:
            plan.bias_rows = make_device_tensor(bias_rows, torch.int64, self.device)
            plan.bias_ids = make_device_tensor(bias_ids, torch.int64, self.device)
            plan.bias_vals = make_device_tensor(bias_vals, torch.float32, self.device)

        min_rows: list[int] = []
        min_ids: list[list[int]] = []
        min_left: list[int] = []
        for local, (req, p) in enumerate(picked):
            if p.min_tokens > 0 and p.min_tokens_stop_ids:
                # device_len, not the host ids: under overlap scheduling the previous token
                # reaches req.input_ids only after this batch is prepared
                generated = req.device_len - (req.max_device_len - req.output_len)
                if generated < p.min_tokens:
                    min_rows.append(local)
                    min_ids.append(
                        [t for t in p.min_tokens_stop_ids if 0 <= t < self.vocab_size]
                    )
                    min_left.append(p.min_tokens - generated)
        if min_rows:
            plan.min_rows = make_device_tensor(min_rows, torch.int64, self.device)
            plan.min_ids = _padded_rows(min_ids, pad, self.device)
            plan.min_left = make_device_tensor(min_left, torch.int64, self.device)
            plan.min_left_host = min_left

        allow = [(local, self._allowed(p)) for local, (_, p) in enumerate(picked)]
        allow = [(local, ids) for local, ids in allow if ids]
        if allow:
            plan.allow_rows = make_device_tensor(
                [local for local, _ in allow], torch.int64, self.device
            )
            plan.allow_ids = _padded_rows([ids for _, ids in allow], pad, self.device)

        if not is_verify:
            g_entries = [
                (local, req.grammar_matcher)
                for local, (req, _) in enumerate(picked)
                if _has_active_grammar(req)
            ]
            if g_entries:
                host_bm = self._get_grammar_host_buf(len(g_entries))
                for g_i, (_, matcher) in enumerate(g_entries):
                    matcher.fill_next_token_bitmask(host_bm, g_i)
                plan.grammar_rows = make_device_tensor(
                    [local for local, _ in g_entries], torch.int64, self.device
                )
                plan.grammar_bitmask = (
                    host_bm.to(self.device, non_blocking=True)
                    if self.device.type == "cuda"
                    else host_bm.clone()
                )

        if any(p.min_p > 0.0 for _, p in picked):
            plan.min_p = make_device_tensor(
                [[p.min_p] for _, p in picked], torch.float32, self.device
            )
        return plan

    def attach_verify_grammar(
        self, batch: Batch, args: BatchSamplingArgs, drafts: torch.Tensor
    ) -> None:
        """Build per-slot verify grammar bitmasks after drafting and attach them to ``args.plan``.

        Advances each constrained request's matcher along the linear draft chain
        ``[d_0, ..., d_{k-1}]`` (also tracking mid-block reasoning end tags), fills slots
        ``0..k``, and rolls the matcher cleanly back to its pre-verify state.
        """
        grammar_reqs = [
            (i, req)
            for i, req in enumerate(batch.reqs)
            if getattr(req, "grammar_matcher", None) is not None
        ]
        if not grammar_reqs:
            return
        k = int(drafts.shape[1])
        width = k + 1
        drafts_cpu = drafts.to("cpu") if k > 0 else None

        # Ensure every grammar request is present in args.plan (extending plan.rows if needed).
        MIN_T = 1e-6
        if args.plan is None:
            plan_rows_list = [i for i, _ in grammar_reqs]
            temps_list = [
                [
                    max(
                        0.0
                        if req.sampling_params.is_greedy
                        else req.sampling_params.temperature,
                        MIN_T,
                    )
                ]
                for _, req in grammar_reqs
            ]
            args.plan = LogitsPlan(
                rows=make_device_tensor(plan_rows_list, torch.int64, self.device),
                temps=make_device_tensor(temps_list, torch.float32, self.device),
            )
            req_to_local = {req_idx: loc for loc, req_idx in enumerate(plan_rows_list)}
        else:
            existing = args.plan.rows.to("cpu").tolist()
            req_to_local = {req_idx: loc for loc, req_idx in enumerate(existing)}
            extra_rows: list[int] = []
            extra_temps: list[list[float]] = []
            for req_idx, req in grammar_reqs:
                if req_idx not in req_to_local:
                    req_to_local[req_idx] = len(existing) + len(extra_rows)
                    extra_rows.append(req_idx)
                    sp = req.sampling_params
                    extra_temps.append(
                        [max(0.0 if sp.is_greedy else sp.temperature, MIN_T)]
                    )
            if extra_rows:
                args.plan.rows = torch.cat(
                    [
                        args.plan.rows,
                        make_device_tensor(extra_rows, torch.int64, self.device),
                    ],
                    dim=0,
                )
                args.plan.temps = torch.cat(
                    [
                        args.plan.temps,
                        make_device_tensor(extra_temps, torch.float32, self.device),
                    ],
                    dim=0,
                )
                if args.plan.min_p is not None:
                    args.plan.min_p = torch.cat(
                        [
                            args.plan.min_p,
                            torch.zeros(
                                (len(extra_rows), 1),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

        total_slots = len(grammar_reqs) * width
        host_bm = self._get_grammar_host_buf(total_slots)
        local_slot_rows: list[int] = []

        for g_idx, (req_idx, req) in enumerate(grammar_reqs):
            local_base = req_to_local[req_idx] * width
            matcher = req.grammar_matcher
            waiting = bool(getattr(req, "grammar_waiting_reasoning_end", False))
            end_ids = tuple(getattr(req, "grammar_think_end_ids", ()) or ())
            tail = (
                req.input_ids[-len(end_ids) :].tolist()
                if (waiting and end_ids and getattr(req, "input_ids", None) is not None)
                else []
            )
            spec_accepted = 0
            for slot in range(width):
                row_slot = g_idx * width + slot
                local_slot_rows.append(local_base + slot)
                if not waiting:
                    matcher.fill_next_token_bitmask(host_bm, row_slot)
                if slot < k and drafts_cpu is not None:
                    d_tok = int(drafts_cpu[req_idx, slot].item())
                    if waiting:
                        tail.append(d_tok)
                        if _ends_with_ids(tail, end_ids):
                            waiting = False
                    else:
                        if matcher.accept_token(d_tok):
                            spec_accepted += 1
                        else:
                            # Slot `slot` already masks out d_tok (p(d_tok) == 0), so chain
                            # rejection stops here; fill remaining local_slot_rows as unconstrained
                            # placeholders so tensor shapes match total_slots.
                            for rem in range(slot + 1, width):
                                local_slot_rows.append(local_base + rem)
                            break
            if spec_accepted > 0:
                matcher.rollback(spec_accepted)

        args.plan.grammar_rows = make_device_tensor(
            local_slot_rows, torch.int64, self.device
        )
        args.plan.grammar_bitmask = (
            host_bm.to(self.device, non_blocking=True)
            if self.device.type == "cuda"
            else host_bm.clone()
        )
        args.plan.grammar_per_verify_row = True

    def _allowed(self, p) -> list[int]:
        # an empty list after the filter leaves the row unmasked rather than all -inf
        return [t for t in p.allowed_token_ids or () if 0 <= t < self.vocab_size]

    def _logprob_targets(self, p) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for t in p.logprob_token_ids or ():
            ti = int(t)
            if 0 <= ti < self.vocab_size and ti not in seen:
                seen.add(ti)
                out.append(ti)
        return out

    def _prompt_mask(self, req) -> torch.Tensor:
        """[V] bool marking the request's prompt ids, uploaded once per request."""
        prompt_len = req.max_device_len - req.output_len
        # multimodal placeholders are pseudo-ids >= vocab_size (MM_PAD_SHIFT_VALUE);
        # send them to the scratch column, not past the scatter target
        ids = req.input_ids[:prompt_len].to(torch.int64).clamp(max=self.vocab_size)
        if self.device.type == "cuda":
            ids = ids.pin_memory().to(self.device, non_blocking=True)
        mask = torch.zeros(self.vocab_size + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        return mask[: self.vocab_size]

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        penalties = []
        repetitions = []
        for row, req in enumerate(batch.reqs):
            p = req.sampling_params
            repetition = p.repetition_penalty
            if not (p.presence_penalty or p.frequency_penalty or repetition != 1.0):
                continue
            if not getattr(req, "can_decode", True):
                continue
            if req.output_token_counts is None:
                req.output_token_counts = torch.zeros(
                    self.vocab_size, dtype=torch.int32, device=self.device
                )
            # also for a repetition-only row: this entry keeps its counts current
            penalties.append(
                (row, req.output_token_counts, p.presence_penalty, p.frequency_penalty)
            )
            if repetition != 1.0:
                if req.prompt_token_mask is None:
                    req.prompt_token_mask = self._prompt_mask(req)
                repetitions.append(
                    RowRepetition(
                        row, req.output_token_counts, repetition, req.prompt_token_mask
                    )
                )
        plan = self._plan(batch, params)
        want_logprobs = [
            bool(p.logprobs and getattr(batch.reqs[i], "can_decode", True))
            for i, p in enumerate(params)
        ]
        logprob_all_rows = bool(want_logprobs) and all(want_logprobs)
        logprob_row_idx = (
            make_device_tensor(
                [i for i, want in enumerate(want_logprobs) if want],
                torch.int64,
                self.device,
            )
            if any(want_logprobs) and not logprob_all_rows
            else None
        )
        logprob_params = [p for p, want in zip(params, want_logprobs) if want]
        restrict = [(i, self._allowed(p)) for i, p in enumerate(logprob_params)]
        restrict = [(i, ids) for i, ids in restrict if ids]
        restrict_rows = restrict_ids = None
        if restrict:
            restrict_rows = make_device_tensor(
                [i for i, _ in restrict], torch.int64, self.device
            )
            restrict_ids = _padded_rows(
                [ids for _, ids in restrict], self.vocab_size, self.device
            )

        explicit_entries: list[tuple[int, list[int], bool]] = []
        row_widths: list[int] = []
        need_full_topk = False
        for i, p in enumerate(logprob_params):
            targets = self._logprob_targets(p)
            allowed = self._allowed(p)
            if targets:
                if allowed:
                    allowed_set = set(allowed)
                    targets = [t for t in targets if t in allowed_set]
                # reserve one extra pad slot so compute_logprobs can append sampled_token if needed
                explicit_entries.append((i, targets + [self.vocab_size], True))
                row_widths.append(len(targets) + 1)
            elif allowed:
                k_req = min(p.top_logprobs, len(allowed))
                if k_req > 0:
                    explicit_entries.append((i, allowed, False))
                row_widths.append(k_req)
            else:
                k_req = min(p.top_logprobs, self.vocab_size)
                if k_req > 0:
                    need_full_topk = True
                row_widths.append(k_req)
        max_top_logprobs = max(row_widths, default=0)
        explicit_rows = explicit_ids = explicit_include_sampled = None
        if explicit_entries and max_top_logprobs > 0:
            explicit_rows = make_device_tensor(
                [i for i, _, _ in explicit_entries], torch.int64, self.device
            )
            explicit_ids = _padded_rows(
                [ids for _, ids, _ in explicit_entries], self.vocab_size, self.device
            )
            if any(inc for _, _, inc in explicit_entries):
                explicit_include_sampled = make_device_tensor(
                    [inc for _, _, inc in explicit_entries], torch.bool, self.device
                )
        if all(is_greedy):
            return BatchSamplingArgs(
                temperatures=None,
                penalties=penalties,
                repetitions=repetitions,
                plan=plan,
                logprob_row_idx=logprob_row_idx,
                logprob_all_rows=logprob_all_rows,
                max_top_logprobs=max_top_logprobs,
                restrict_rows=restrict_rows,
                restrict_ids=restrict_ids,
                explicit_rows=explicit_rows,
                explicit_ids=explicit_ids,
                explicit_include_sampled=explicit_include_sampled,
                need_full_topk=need_full_topk,
            )

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [
            1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)
        ]
        top_ks = [
            min(p.top_k, self.vocab_size) if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        seeded_rows: list[tuple[int, int, int, float, int, float]] = []
        for row, (req, p, g) in enumerate(zip(batch.reqs, params, is_greedy)):
            if g or p.seed is None or not getattr(req, "can_decode", True):
                continue
            prompt_len = req.max_device_len - req.output_len
            gen_len = max(0, req.device_len - prompt_len)
            seeded_rows.append(
                (row, int(p.seed), gen_len, ts[row], top_ks[row], top_ps[row])
            )
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device)
            if any(is_greedy)
            else None
        )
        return BatchSamplingArgs(
            temperatures,
            top_k=top_k,
            top_p=top_p,
            greedy_mask=greedy_mask,
            penalties=penalties,
            repetitions=repetitions,
            plan=plan,
            logprob_row_idx=logprob_row_idx,
            logprob_all_rows=logprob_all_rows,
            max_top_logprobs=max_top_logprobs,
            restrict_rows=restrict_rows,
            restrict_ids=restrict_ids,
            explicit_rows=explicit_rows,
            explicit_ids=explicit_ids,
            explicit_include_sampled=explicit_include_sampled,
            need_full_topk=need_full_topk,
            seeded_rows=seeded_rows,
        )

    @nvtx_annotate("Sampler")
    def sample(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        *,
        record_counts: bool = True,
    ) -> torch.Tensor:
        if args.penalties:
            logits = apply_penalties(logits, args)
        # after the penalties: min_p must see the final distribution
        if args.plan is not None:
            logits = apply_logits_processors(logits, args.plan, self.vocab_size)
        if args.temperatures is None:  # greedy sampling
            tokens = torch.argmax(logits, dim=-1)
        else:
            tokens = sample_impl(
                logits.float(), args.temperatures, args.top_k, args.top_p
            )
            if args.greedy_mask is not None:
                # Mixed batches still run probability sampling for all rows, but
                # greedy rows must follow argmax's deterministic tie-breaking.
                greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            for row, seed, gen_len, temp, tk, tp in args.seeded_rows:
                tokens[row] = sample_seeded_row(
                    logits[row],
                    seed=seed,
                    step=gen_len,
                    temp=temp,
                    top_k=tk,
                    top_p=tp,
                ).to(tokens.dtype)
        if record_counts:
            self.record_tokens(tokens, args)
        return tokens

    def record_tokens(self, tokens: torch.Tensor, args: BatchSamplingArgs) -> None:
        # Update on the sampling stream: overlapped scheduling can prepare the next
        # batch before the previous token reaches Req.input_ids on the CPU.
        for row, counts, _, _ in args.penalties:
            counts.scatter_add_(0, tokens[row : row + 1].long(), counts.new_ones(1))

    def compute_logprobs(
        self,
        logits: torch.Tensor,
        sampled_tokens: torch.Tensor,
        args: BatchSamplingArgs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """(chosen [bs], top ids [bs, K], top logprobs [bs, K]) as non-blocking host copies,
        K the batch's widest top_logprobs; None when no row asked. Rows that did not ask are
        NaN / -1 / -inf."""
        rows = args.logprob_row_idx
        if not args.logprob_all_rows and (rows is None or rows.numel() == 0):
            return None

        request_logits = (
            logits if rows is None else logits.index_select(0, rows)
        ).float()
        if args.restrict_rows is not None:
            # not in place: on an fp32 batch request_logits can be the engine's logits view
            masked = request_logits[args.restrict_rows].masked_fill(
                disallowed_mask(args.restrict_ids, self.vocab_size), float("-inf")
            )
            request_logits = request_logits.index_put((args.restrict_rows,), masked)
        # Reported values are raw model logprobs (pre-temperature log_softmax over logits),
        # renormalized over allowed_token_ids on the restricted rows.
        request_logprobs = torch.log_softmax(request_logits, dim=-1)

        request_tokens = sampled_tokens.to(dtype=torch.long, device=logits.device)
        if rows is not None:
            request_tokens = request_tokens.index_select(0, rows)
        request_row_idx = torch.arange(request_logprobs.shape[0], device=logits.device)
        request_chosen_logprobs = request_logprobs[request_row_idx, request_tokens]

        if rows is None:
            chosen_logprobs = request_chosen_logprobs
        else:
            chosen_logprobs = torch.full(
                (logits.shape[0],),
                float("nan"),
                dtype=torch.float32,
                device=logits.device,
            )
            chosen_logprobs.index_copy_(0, rows, request_chosen_logprobs)

        if args.max_top_logprobs > 0:
            n_req = request_logprobs.shape[0]
            k_out = args.max_top_logprobs
            request_top_ids = torch.full(
                (n_req, k_out), -1, dtype=torch.int64, device=logits.device
            )
            request_top_logprobs = torch.full(
                (n_req, k_out), float("-inf"), dtype=torch.float32, device=logits.device
            )
            if args.need_full_topk or args.explicit_rows is None:
                k_vocab = min(k_out, self.vocab_size)
                tk_lps, tk_ids = torch.topk(request_logprobs, k=k_vocab, dim=-1)
                request_top_ids[:, :k_vocab] = tk_ids
                request_top_logprobs[:, :k_vocab] = tk_lps
            if args.explicit_rows is not None and args.explicit_ids is not None:
                cand_ids = args.explicit_ids.clone()
                if args.explicit_include_sampled is not None:
                    samp = request_tokens[args.explicit_rows].unsqueeze(1)
                    already = (cand_ids == samp).any(dim=1)
                    need = args.explicit_include_sampled & (~already)
                    cand_ids[:, -1] = torch.where(
                        need, samp.squeeze(1), cand_ids[:, -1]
                    )
                scratch = torch.full(
                    (cand_ids.shape[0], 1),
                    float("-inf"),
                    dtype=torch.float32,
                    device=logits.device,
                )
                padded_lps = torch.cat(
                    [request_logprobs[args.explicit_rows], scratch], dim=1
                )
                cand_lps = padded_lps.gather(1, cand_ids)
                valid = cand_ids < self.vocab_size
                # Keep valid -inf candidates ahead of vocab_size padding slots.
                sort_key = torch.where(
                    valid,
                    cand_lps.clamp_min(-1e38),
                    torch.full_like(cand_lps, float("-inf")),
                )
                order = torch.argsort(sort_key, dim=1, descending=True)
                sorted_ids = cand_ids.gather(1, order)
                sorted_lps = cand_lps.gather(1, order)
                sorted_ids = torch.where(
                    sorted_ids < self.vocab_size,
                    sorted_ids,
                    torch.full_like(sorted_ids, -1),
                )
                take = min(k_out, sorted_ids.shape[1])
                exp_ids = torch.full(
                    (cand_ids.shape[0], k_out),
                    -1,
                    dtype=torch.int64,
                    device=logits.device,
                )
                exp_lps = torch.full(
                    (cand_ids.shape[0], k_out),
                    float("-inf"),
                    dtype=torch.float32,
                    device=logits.device,
                )
                exp_ids[:, :take] = sorted_ids[:, :take]
                exp_lps[:, :take] = sorted_lps[:, :take]
                request_top_ids[args.explicit_rows] = exp_ids
                request_top_logprobs[args.explicit_rows] = exp_lps

            if rows is None:
                top_ids = request_top_ids.to(torch.int32)
                top_logprobs = request_top_logprobs
            else:
                top_ids = torch.full(
                    (logits.shape[0], k_out),
                    -1,
                    dtype=torch.int32,
                    device=logits.device,
                )
                top_logprobs = torch.full(
                    (logits.shape[0], k_out),
                    float("-inf"),
                    dtype=torch.float32,
                    device=logits.device,
                )
                top_ids[rows] = request_top_ids.to(torch.int32)
                top_logprobs[rows] = request_top_logprobs
        else:
            top_ids = torch.empty(
                (logits.shape[0], 0), dtype=torch.int32, device=logits.device
            )
            top_logprobs = torch.empty(
                (logits.shape[0], 0), dtype=torch.float32, device=logits.device
            )

        return (
            chosen_logprobs.to("cpu", non_blocking=True),
            top_ids.to("cpu", non_blocking=True),
            top_logprobs.to("cpu", non_blocking=True),
        )

    def compute_verify_logprobs(
        self,
        logits: torch.Tensor,
        sampled_tokens: torch.Tensor,
        args: BatchSamplingArgs,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Per-row logprobs across a speculative verify batch of ``width`` rows per request,
        returning ``(chosen [bs, width], top_ids [bs, width, K], top_logprobs [bs, width, K])`` on CPU."""
        if not args.logprob_all_rows and (
            args.logprob_row_idx is None or args.logprob_row_idx.numel() == 0
        ):
            return None
        offsets = torch.arange(width, device=logits.device)

        def expand_rows(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else (t.unsqueeze(1) * width + offsets).flatten()

        def repeat_rows(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.repeat_interleave(width, dim=0)

        verify_args = BatchSamplingArgs(
            temperatures=None,
            logprob_row_idx=expand_rows(args.logprob_row_idx),
            logprob_all_rows=args.logprob_all_rows,
            max_top_logprobs=args.max_top_logprobs,
            restrict_rows=expand_rows(args.restrict_rows),
            restrict_ids=repeat_rows(args.restrict_ids),
            explicit_rows=expand_rows(args.explicit_rows),
            explicit_ids=repeat_rows(args.explicit_ids),
            explicit_include_sampled=repeat_rows(args.explicit_include_sampled),
            need_full_topk=args.need_full_topk,
        )
        res = self.compute_logprobs(logits, sampled_tokens.reshape(-1), verify_args)
        if res is None:
            return None
        bs = logits.shape[0] // width
        chosen_cpu, top_ids_cpu, top_lps_cpu = res
        return (
            chosen_cpu.view(bs, width),
            top_ids_cpu.view(bs, width, -1),
            top_lps_cpu.view(bs, width, -1),
        )

    def sample_verify(
        self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor
    ) -> torch.Tensor:
        """One token per verify row as decode would sample it, ``drafts`` [bs, k] giving k+1 rows per request.

        Row j of a request is penalized as if its drafts d1..dj were already committed; the penalty counts are not updated.
        """
        bs, k = drafts.shape
        width = k + 1
        if args.penalties:
            logits = self._penalize_rows(logits, args, drafts)
        if args.plan is not None:
            logits = apply_logits_processors(
                logits, _verify_plan(args.plan, width), self.vocab_size
            )
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)

        def rows(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.repeat_interleave(width)

        tokens = sample_impl(
            logits.float(), rows(args.temperatures), rows(args.top_k), rows(args.top_p)
        )
        if args.greedy_mask is not None:
            greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
            tokens = torch.where(rows(args.greedy_mask), greedy_tokens, tokens)
        for row, seed, gen_len, temp, tk, tp in args.seeded_rows:
            base = row * width
            for j in range(width):
                tokens[base + j] = sample_seeded_row(
                    logits[base + j],
                    seed=seed,
                    step=gen_len + j,
                    temp=temp,
                    top_k=tk,
                    top_p=tp,
                ).to(tokens.dtype)
        return tokens

    def probs(
        self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor
    ) -> torch.Tensor | None:
        """[bs*(k+1), V] fp32 distribution each verify row samples from (softmax(T), then top-k, then top-p, as the fused samplers apply them), penalized like sample_verify and one-hot at the argmax on greedy rows; None when every request is greedy."""
        if args.temperatures is None:
            return None
        sampling = sampling_module(logits.device)
        width = drafts.shape[1] + 1
        if args.penalties:
            logits = self._penalize_rows(logits, args, drafts)
        if args.plan is not None:
            logits = apply_logits_processors(
                logits, _verify_plan(args.plan, width), self.vocab_size
            )
        top = (
            None
            if args.greedy_mask is None
            else torch.argmax(logits, dim=-1, keepdim=True)
        )
        probs = sampling.softmax(
            logits.float(),
            args.temperatures.repeat_interleave(width),
            enable_pdl=is_sm90_supported(),
        )
        # frees the penalized copy before the renorms allocate theirs
        del logits
        if args.top_k is not None:
            probs = sampling.top_k_renorm_probs(
                probs, args.top_k.repeat_interleave(width)
            )
        if args.top_p is not None:
            probs = sampling.top_p_renorm_probs(
                probs, args.top_p.repeat_interleave(width)
            )
        if top is not None:
            greedy = args.greedy_mask.repeat_interleave(width).unsqueeze(1)
            hot = torch.where(greedy, 1.0, probs.gather(1, top))
            probs.masked_fill_(greedy, 0.0).scatter_(1, top, hot)
        return probs

    def _penalize_rows(
        self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor
    ) -> torch.Tensor:
        k = drafts.shape[1]
        width = k + 1
        logits = logits.float().clone()
        # (row j, draft i) pairs with i < j: row j has seen drafts 0..j-1
        pair_rows, pair_drafts = torch.tril_indices(
            width, k, offset=-1, device=logits.device
        )
        earlier = torch.ones(k, k, dtype=torch.bool, device=logits.device).tril(-1)
        for req_row, counts, penalty, prompt_mask in args.repetitions:
            block = logits[req_row * width : (req_row + 1) * width]
            seen = ((counts > 0) | prompt_mask).repeat(width, 1)
            if k > 0:
                seen[pair_rows, drafts[req_row].long()[pair_drafts]] = True
            penalized = torch.where(block > 0, block / penalty, block * penalty)
            block.copy_(torch.where(seen, penalized, block))
        for req_row, counts, presence, frequency in args.penalties:
            if presence == 0.0 and frequency == 0.0:
                continue
            block = logits[req_row * width : (req_row + 1) * width]
            block -= frequency * counts + presence * (counts > 0)
            if k == 0:
                continue
            d = drafts[req_row].long()
            # presence applies once per token: only to a draft that is new to both the counts and the earlier drafts
            first = ~((d.unsqueeze(1) == d.unsqueeze(0)) & earlier).any(dim=1)
            extra = frequency + presence * ((counts[d] == 0) & first).float()
            block.index_put_(
                (pair_rows, d[pair_drafts]), -extra[pair_drafts], accumulate=True
            )
        return logits
