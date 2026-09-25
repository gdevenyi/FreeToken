"""The Triton sampling softmax, which samples when flashinfer is not installed."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_softmax_survives_a_fully_masked_prefix():
    """min_p / min_tokens write -inf over most of the vocabulary. A column chunk or leading
    block that is -inf throughout gave exp(-inf - -inf) = NaN and a NaN row."""
    from freetoken.kernel.triton.sampling import softmax

    V = 32768  # several column chunks per row
    torch.manual_seed(0)
    logits = torch.full((2, V), float("-inf"), device="cuda")
    logits[0, [20000, 30000, 32000]] = torch.tensor([1.0, 2.0, 0.5], device="cuda")
    logits[1] = torch.randn(V, device="cuda")
    logits[1, :8192] = float("-inf")
    temps = torch.tensor([0.7, 1.3], device="cuda")
    probs = softmax(logits, temps)
    assert not probs.isnan().any()
    torch.testing.assert_close(probs, torch.softmax(logits / temps[:, None], dim=-1))
