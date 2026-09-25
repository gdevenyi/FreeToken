"""The optional-package probes must also reject a GPU the package ships no kernels for."""

import pytest

import freetoken.kernel.backend as backend


@pytest.fixture
def probes(monkeypatch):
    monkeypatch.setattr(backend, "_importable", lambda name: True)
    cached = (backend.is_flashinfer_installed, backend.is_sgl_kernel_installed)
    for probe in cached:
        probe.cache_clear()
    yield
    for probe in cached:
        probe.cache_clear()


@pytest.mark.parametrize(
    "cap, flashinfer, sgl_kernel",
    [((6, 1), False, False), ((7, 5), True, False), ((8, 0), True, True), ((0, 0), True, True)],
    ids=["pascal", "turing", "ampere", "no-cuda"],
)
def test_installed_probes_gate_on_device_capability(probes, monkeypatch, cap, flashinfer, sgl_kernel):
    monkeypatch.setattr(backend, "device_capability", lambda: cap)
    assert backend.is_flashinfer_installed() is flashinfer
    assert backend.is_sgl_kernel_installed() is sgl_kernel


def test_missing_package_is_unavailable_on_any_gpu(probes, monkeypatch):
    monkeypatch.setattr(backend, "_importable", lambda name: False)
    monkeypatch.setattr(backend, "device_capability", lambda: (9, 0))
    assert not backend.is_flashinfer_installed()
    assert not backend.is_sgl_kernel_installed()
