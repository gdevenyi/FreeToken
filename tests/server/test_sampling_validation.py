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


def _resolve_with(**overrides):
    kwargs = dict(
        temperature=None, top_k=None, top_p=None, max_tokens=8, ignore_eos=False,
        model_sampling={},
    )
    kwargs.update(overrides)
    return resolve_sampling(**kwargs)


@pytest.mark.parametrize("value", [0, -1, -5])
def test_a_non_positive_top_k_means_disabled(value):
    # vLLM's rule: any top_k <= 0 is "no top-k filter"; clients send 0 for "off"
    assert _resolve_with(top_k=value).top_k == -1


def test_a_checkpoint_default_top_k_of_zero_means_disabled():
    assert _resolve_with(model_sampling={"top_k": 0}).top_k == -1


def test_a_positive_top_k_is_kept():
    assert _resolve_with(top_k=20).top_k == 20


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", float("nan")), ("temperature", float("inf")), ("temperature", -0.5),
     ("top_p", 0.0), ("top_p", 1.5), ("top_p", float("nan"))],
)
def test_the_other_sampling_ranges_are_still_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _resolve_with(**{field: value})
