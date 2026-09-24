"""resolve_sampling rejects non-finite penalties: NaN passes a `<= 0` check and would
turn every already-seen token's logit into NaN."""

import pytest

from freetoken.server.generation import resolve_sampling


def _resolve(**extra):
    return resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=8, ignore_eos=False,
        model_sampling={}, **extra,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_repetition_penalty_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="repetition_penalty"):
        _resolve(repetition_penalty=value)


def test_a_normal_repetition_penalty_is_kept():
    assert _resolve(repetition_penalty=1.1).repetition_penalty == pytest.approx(1.1)
