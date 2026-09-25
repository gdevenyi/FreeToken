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
    them from per-request device state (``RowPenalty``).

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
    min_p: torch.Tensor | None = None  # [r, 1] float32 (0 = off for that row)


class RowPenalty(NamedTuple):
    """One batch row's presence / frequency / repetition penalty. ``counts`` are the
    request's generated-token counts ([V] int32), updated on the device after each draw so
    overlap scheduling never sees them late; ``prompt_mask`` ([V] bool) marks the prompt's
    ids for the repetition penalty and is None when that is off."""

    row: int
    counts: torch.Tensor
    presence: float
    frequency: float
    repetition: float = 1.0
    prompt_mask: torch.Tensor | None = None


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    penalties: list[RowPenalty] = field(default_factory=list)
    plan: LogitsPlan | None = None
    logprob_rows: torch.Tensor | None = None
    max_top_logprobs: int = 0


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":  # the unit tests build plans on the CPU
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _padded_rows(rows: list[list[int]], pad: int, device: torch.device) -> torch.Tensor:
    """[len(rows), max_len] int64 with ``pad`` on the right."""
    width = max(1, max((len(r) for r in rows), default=1))
    return make_device_tensor([r + [pad] * (width - len(r)) for r in rows], torch.int64, device)


def apply_penalties(logits: torch.Tensor, penalties: list[RowPenalty]) -> torch.Tensor:
    """A float32 copy of ``logits`` with each row's penalties applied in vLLM's order:
    repetition (over prompt + generated ids), then frequency and presence (generated)."""
    logits = logits.float().clone()
    for row, counts, presence, frequency, repetition, prompt_mask in penalties:
        if repetition != 1.0:
            x = logits[row]
            seen = counts > 0
            if prompt_mask is not None:
                seen = seen | prompt_mask
            # HF / vLLM semantics: divide positive logits, multiply negative ones.
            penalized = torch.where(x > 0, x / repetition, x * repetition)
            logits[row] = torch.where(seen, penalized, x)
        if presence or frequency:
            logits[row] -= frequency * counts + presence * (counts > 0)
    return logits


def apply_logits_processors(
    logits: torch.Tensor, plan: LogitsPlan, vocab_size: int
) -> torch.Tensor:
    """A float32 copy of ``logits`` with the plan applied to its rows: logit_bias, the
    min_tokens mask, then min_p. Rows outside the plan are copied unchanged."""
    out = logits.to(torch.float32, copy=True)
    sub = out[plan.rows]  # [r, V] (advanced indexing copies)
    device = sub.device
    neg_inf = float("-inf")

    if plan.bias_ids is not None:
        sub.index_put_((plan.bias_rows, plan.bias_ids), plan.bias_vals, accumulate=True)

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


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed
    from freetoken.utils.arch import is_sm70_supported

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    elif not is_sm70_supported():
        # The triton top-p/top-k threshold search accumulates through tl.atomic_*, which
        # triton can only lower to sm_70+ encodings. Sort with torch instead.
        import freetoken.kernel.torch_sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

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
        for local, (req, p) in enumerate(picked):
            if p.min_tokens > 0 and p.min_tokens_stop_ids:
                # device_len, not the host ids: under overlap scheduling the previous token
                # reaches req.input_ids only after this batch is prepared
                generated = req.device_len - (req.max_device_len - req.output_len)
                if generated < p.min_tokens:
                    min_rows.append(local)
                    min_ids.append([t for t in p.min_tokens_stop_ids if 0 <= t < self.vocab_size])
        if min_rows:
            plan.min_rows = make_device_tensor(min_rows, torch.int64, self.device)
            plan.min_ids = _padded_rows(min_ids, pad, self.device)

        if any(p.min_p > 0.0 for _, p in picked):
            plan.min_p = make_device_tensor([[p.min_p] for _, p in picked], torch.float32, self.device)
        return plan

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
            prompt_mask = None
            if repetition != 1.0:
                if req.prompt_token_mask is None:
                    req.prompt_token_mask = self._prompt_mask(req)
                prompt_mask = req.prompt_token_mask
            penalties.append(
                RowPenalty(
                    row,
                    req.output_token_counts,
                    p.presence_penalty,
                    p.frequency_penalty,
                    repetition,
                    prompt_mask,
                )
            )
        plan = self._plan(batch, params)
        want_logprobs = [p.logprobs for p in params]
        logprob_rows = (
            make_device_tensor(want_logprobs, torch.bool, self.device)
            if any(want_logprobs)
            else None
        )
        max_top_logprobs = min(
            max((p.top_logprobs for p in params if p.logprobs), default=0), self.vocab_size
        )
        if all(is_greedy):
            return BatchSamplingArgs(
                temperatures=None,
                penalties=penalties,
                plan=plan,
                logprob_rows=logprob_rows,
                max_top_logprobs=max_top_logprobs,
            )

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            p.top_k if not g and p.top_k >= 1 else self.vocab_size
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
            plan=plan,
            logprob_rows=logprob_rows,
            max_top_logprobs=max_top_logprobs,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.penalties:
                logits = apply_penalties(logits, args.penalties)
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
            for row, counts, *_ in args.penalties:
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
        if args.logprob_rows is None:
            return None

        requested_rows = torch.nonzero(args.logprob_rows, as_tuple=False).flatten()
        if requested_rows.numel() == 0:
            return None

        request_logits = logits.index_select(0, requested_rows).float()
        # Reported values are raw model logprobs (pre-temperature log_softmax over logits).
        request_logprobs = torch.log_softmax(request_logits, dim=-1)

        request_tokens = sampled_tokens.to(dtype=torch.long, device=logits.device).index_select(
            0, requested_rows
        )
        request_row_idx = torch.arange(requested_rows.numel(), device=logits.device)
        request_chosen_logprobs = request_logprobs[request_row_idx, request_tokens]

        chosen_logprobs = torch.full(
            (logits.shape[0],), float("nan"), dtype=torch.float32, device=logits.device
        )
        chosen_logprobs.index_copy_(0, requested_rows, request_chosen_logprobs)

        if args.max_top_logprobs > 0:
            request_top_logprobs, request_top_ids = torch.topk(
                request_logprobs, k=args.max_top_logprobs, dim=-1
            )
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
            top_ids[requested_rows] = request_top_ids.to(torch.int32)
            top_logprobs[requested_rows] = request_top_logprobs
        else:
            top_ids = torch.empty((logits.shape[0], 0), dtype=torch.int32, device=logits.device)
            top_logprobs = torch.empty((logits.shape[0], 0), dtype=torch.float32, device=logits.device)

        return (
            chosen_logprobs.to("cpu", non_blocking=True),
            top_ids.to("cpu", non_blocking=True),
            top_logprobs.to("cpu", non_blocking=True),
        )
