from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple

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
    min_p: torch.Tensor | None = None  # [r, 1] float32 (0 = off for that row)
    # Rows with allowed_token_ids and their ids, padded with vocab_size like min_ids.
    allow_rows: torch.Tensor | None = None  # [a] int64 local rows
    allow_ids: torch.Tensor | None = None  # [a, A] int64


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
    penalties: list[tuple[int, torch.Tensor, float, float]] = field(default_factory=list)
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


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":  # the unit tests build plans on the CPU
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _padded_rows(rows: list[list[int]], pad: int, device: torch.device) -> torch.Tensor:
    """[len(rows), max_len] int64 with ``pad`` on the right."""
    width = max(1, max((len(r) for r in rows), default=1))
    return make_device_tensor([r + [pad] * (width - len(r)) for r in rows], torch.int64, device)


def disallowed_mask(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """[a, V] bool, True outside each row's ids (the vocab_size padding column is dropped)."""
    block = torch.ones((ids.shape[0], vocab_size + 1), dtype=torch.bool, device=ids.device)
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
        sub[plan.allow_rows] = rows.masked_fill(disallowed_mask(plan.allow_ids, vocab_size), neg_inf)

    if plan.min_ids is not None:
        block = torch.zeros((plan.min_ids.shape[0], vocab_size + 1), dtype=torch.bool, device=device)
        block.scatter_(1, plan.min_ids, torch.ones_like(plan.min_ids, dtype=torch.bool))
        rows = sub[plan.min_rows]
        sub[plan.min_rows] = torch.where(block[:, :vocab_size], torch.full_like(rows, neg_inf), rows)

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
        entry, j = (offsets < plan.min_left.unsqueeze(1)).nonzero(as_tuple=True)
        out.min_rows = plan.min_rows[entry] * width + j
        out.min_ids = plan.min_ids[entry]
    if plan.allow_ids is not None:
        out.allow_rows = rows(plan.allow_rows)
        out.allow_ids = repeat(plan.allow_ids)
    if plan.min_p is not None:
        out.min_p = repeat(plan.min_p)
    return out


def sampling_module():
    """flashinfer.sampling when installed, else the triton kernels with the same API."""
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling
    return sampling


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    sampling = sampling_module()
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


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def _plan(self, batch: Batch, params) -> LogitsPlan | None:
        """The LogitsPlan for this batch, or None when no request asks for a processor."""
        MIN_T = 1e-6
        rows = [i for i, p in enumerate(params) if p.needs_logits_processing]
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
                    min_ids.append([t for t in p.min_tokens_stop_ids if 0 <= t < self.vocab_size])
                    min_left.append(p.min_tokens - generated)
        if min_rows:
            plan.min_rows = make_device_tensor(min_rows, torch.int64, self.device)
            plan.min_ids = _padded_rows(min_ids, pad, self.device)
            plan.min_left = make_device_tensor(min_left, torch.int64, self.device)

        allow = [(local, self._allowed(p)) for local, (_, p) in enumerate(picked)]
        allow = [(local, ids) for local, ids in allow if ids]
        if allow:
            plan.allow_rows = make_device_tensor([local for local, _ in allow], torch.int64, self.device)
            plan.allow_ids = _padded_rows([ids for _, ids in allow], pad, self.device)

        if any(p.min_p > 0.0 for _, p in picked):
            plan.min_p = make_device_tensor([[p.min_p] for _, p in picked], torch.float32, self.device)
        return plan

    def _allowed(self, p) -> list[int]:
        # an empty list after the filter leaves the row unmasked rather than all -inf
        return [t for t in p.allowed_token_ids or () if 0 <= t < self.vocab_size]

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
            if not req.can_decode:
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
                    RowRepetition(row, req.output_token_counts, repetition, req.prompt_token_mask)
                )
        plan = self._plan(batch, params)
        want_logprobs = [p.logprobs for p in params]
        logprob_all_rows = bool(want_logprobs) and all(want_logprobs)
        logprob_row_idx = (
            make_device_tensor(
                [i for i, want in enumerate(want_logprobs) if want], torch.int64, self.device
            )
            if any(want_logprobs) and not logprob_all_rows
            else None
        )
        max_top_logprobs = min(
            max((p.top_logprobs for p in params if p.logprobs), default=0), self.vocab_size
        )
        restrict = [(i, self._allowed(p)) for i, p in enumerate(p for p in params if p.logprobs)]
        restrict = [(i, ids) for i, ids in restrict if ids]
        restrict_rows = restrict_ids = None
        if restrict:
            restrict_rows = make_device_tensor([i for i, _ in restrict], torch.int64, self.device)
            restrict_ids = _padded_rows([ids for _, ids in restrict], self.vocab_size, self.device)
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
            )

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            min(p.top_k, self.vocab_size) if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
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
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.penalties:
            logits = apply_penalties(logits, args)
        # after the penalties: min_p must see the final distribution
        if args.plan is not None:
            logits = apply_logits_processors(logits, args.plan, self.vocab_size)
        if args.temperatures is None:  # greedy sampling
            tokens = torch.argmax(logits, dim=-1)
        else:
            tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
            if args.greedy_mask is not None:
                # Mixed batches still run probability sampling for all rows, but
                # greedy rows must follow argmax's deterministic tie-breaking.
                greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
        # Update on the sampling stream: overlapped scheduling can prepare the next
        # batch before the previous token reaches Req.input_ids on the CPU.
        for row, counts, _, _ in args.penalties:
            counts.scatter_add_(
                0, tokens[row : row + 1].long(), counts.new_ones(1)
            )
        return tokens

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

        request_logits = (logits if rows is None else logits.index_select(0, rows)).float()
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
                (logits.shape[0],), float("nan"), dtype=torch.float32, device=logits.device
            )
            chosen_logprobs.index_copy_(0, rows, request_chosen_logprobs)

        if args.max_top_logprobs > 0:
            request_top_logprobs, request_top_ids = torch.topk(
                request_logprobs, k=args.max_top_logprobs, dim=-1
            )
            if rows is None:
                top_ids = request_top_ids.to(torch.int32)
                top_logprobs = request_top_logprobs
            else:
                top_ids = torch.full(
                    (logits.shape[0], args.max_top_logprobs),
                    -1,
                    dtype=torch.int32,
                    device=logits.device,
                )
                top_logprobs = torch.full(
                    (logits.shape[0], args.max_top_logprobs),
                    float("-inf"),
                    dtype=torch.float32,
                    device=logits.device,
                )
                top_ids[rows] = request_top_ids.to(torch.int32)
                top_logprobs[rows] = request_top_logprobs
        else:
            top_ids = torch.empty((logits.shape[0], 0), dtype=torch.int32, device=logits.device)
            top_logprobs = torch.empty((logits.shape[0], 0), dtype=torch.float32, device=logits.device)

        return (
            chosen_logprobs.to("cpu", non_blocking=True),
            top_ids.to("cpu", non_blocking=True),
            top_logprobs.to("cpu", non_blocking=True),
        )

    def sample_verify(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor:
        """One token per verify row as decode would sample it, ``drafts`` [bs, k] giving k+1 rows per request.

        Row j of a request is penalized as if its drafts d1..dj were already committed; the penalty counts are not updated.
        """
        bs, k = drafts.shape
        width = k + 1
        if args.penalties:
            logits = self._penalize_rows(logits, args, drafts)
        if args.plan is not None:
            logits = apply_logits_processors(logits, _verify_plan(args.plan, width), self.vocab_size)
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)

        def rows(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.repeat_interleave(width)

        tokens = sample_impl(logits.float(), rows(args.temperatures), rows(args.top_k), rows(args.top_p))
        if args.greedy_mask is not None:
            greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
            tokens = torch.where(rows(args.greedy_mask), greedy_tokens, tokens)
        return tokens

    def probs(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor | None:
        """[bs*(k+1), V] fp32 distribution each verify row samples from (softmax(T), then top-k, then top-p, as the fused samplers apply them), penalized like sample_verify and one-hot at the argmax on greedy rows; None when every request is greedy."""
        if args.temperatures is None:
            return None
        sampling = sampling_module()
        width = drafts.shape[1] + 1
        if args.penalties:
            logits = self._penalize_rows(logits, args, drafts)
        if args.plan is not None:
            logits = apply_logits_processors(logits, _verify_plan(args.plan, width), self.vocab_size)
        top = None if args.greedy_mask is None else torch.argmax(logits, dim=-1, keepdim=True)
        probs = sampling.softmax(logits.float(), args.temperatures.repeat_interleave(width), enable_pdl=is_sm90_supported())
        # frees the penalized copy before the renorms allocate theirs
        del logits
        if args.top_k is not None:
            probs = sampling.top_k_renorm_probs(probs, args.top_k.repeat_interleave(width))
        if args.top_p is not None:
            probs = sampling.top_p_renorm_probs(probs, args.top_p.repeat_interleave(width))
        if top is not None:
            greedy = args.greedy_mask.repeat_interleave(width).unsqueeze(1)
            hot = torch.where(greedy, 1.0, probs.gather(1, top))
            probs.masked_fill_(greedy, 0.0).scatter_(1, top, hot)
        return probs

    def _penalize_rows(self, logits: torch.Tensor, args: BatchSamplingArgs, drafts: torch.Tensor) -> torch.Tensor:
        k = drafts.shape[1]
        width = k + 1
        logits = logits.float().clone()
        # (row j, draft i) pairs with i < j: row j has seen drafts 0..j-1
        pair_rows, pair_drafts = torch.tril_indices(width, k, offset=-1, device=logits.device)
        earlier = torch.ones(k, k, dtype=torch.bool, device=logits.device).tril(-1)
        for req_row, counts, penalty, prompt_mask in args.repetitions:
            block = logits[req_row * width : (req_row + 1) * width]
            seen = ((counts > 0) | prompt_mask).repeat(width, 1)
            if k > 0:
                seen[pair_rows, drafts[req_row].long()[pair_drafts]] = True
            penalized = torch.where(block > 0, block / penalty, block * penalty)
            block.copy_(torch.where(seen, penalized, block))
        for req_row, counts, presence, frequency in args.penalties:
            block = logits[req_row * width : (req_row + 1) * width]
            block -= frequency * counts + presence * (counts > 0)
            if k == 0:
                continue
            d = drafts[req_row].long()
            # presence applies once per token: only to a draft that is new to both the counts and the earlier drafts
            first = ~((d.unsqueeze(1) == d.unsqueeze(0)) & earlier).any(dim=1)
            extra = frequency + presence * ((counts[d] == 0) & first).float()
            block.index_put_((pair_rows, d[pair_drafts]), -extra[pair_drafts], accumulate=True)
        return logits
