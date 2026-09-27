"""Package-wide runtime hygiene: TP info set once, the global ctx never leaks across tests."""

import pytest


@pytest.fixture(autouse=True)
def _runtime():
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        import gc

        # torch keeps a cuBLAS workspace per stream for good: each offload cache's dedicated
        # streams would otherwise pin ~16 MiB for the rest of the session
        gc.collect()
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
