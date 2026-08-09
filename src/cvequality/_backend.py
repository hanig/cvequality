"""Torch plumbing shared by the batched test implementations.

Everything here is deliberately device-agnostic: the whole library runs on CPU (which is
how the test suite validates it against R) and on CUDA unchanged. Both special functions
the tests need exist in torch, so no computation ever leaves the device:

* ``chi2_sf``  -- ``torch.special.gammaincc(df/2, x/2)``, verified to agree with
  ``scipy.stats.chi2.sf`` to full float64 at df = 1, 2 and 1965.
* ``chi2_rvs`` -- ``2 * torch._standard_gamma(df/2)``, verified correct in mean and variance
  at df up to ~2e5, the size of a large single-cell reference group.
"""

from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "Status",
    "chi2_rvs",
    "chi2_sf",
    "make_generator",
    "pick_chunk",
    "resolve_device",
    "resolve_dtype",
    "standard_normal",
    "to_tk",
]


class Status(IntEnum):
    """Per-test outcome. R silently returns ``NaN``/``Inf``; we say why."""

    OK = 0
    #: Some group mean was <= 0. Both tests divide by the mean and take ``log(tau*u)``, so
    #: they are undefined -- common for lowly expressed genes with an all-zero group.
    NONPOSITIVE_MEAN = 1
    #: Some group had zero sample variance (every cell identical, usually all zero).
    ZERO_VARIANCE = 2
    #: Some group had fewer than 2 observations, so its sample SD is undefined.
    TOO_FEW_OBS = 3
    #: The common-CV MLE did not reach the requested tolerance.
    NOT_CONVERGED = 4
    #: The statistic came out non-finite for some other reason.
    NON_FINITE = 5


def resolve_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """``None``/``"auto"`` -> CUDA when available, else CPU. Explicit values are honoured."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    return dev


def resolve_dtype(dtype: Union[None, torch.dtype] = None) -> torch.dtype:
    """Default float64.

    float32 is allowed but warns: the literal R form of the LRT statistic is a difference
    of two ``O(n log n)`` quantities, so at n ~ 2e5 it loses every significant digit. The
    collapsed form (used whenever the MLE is solved to convergence) is well conditioned, but
    the bootstrap moments still accumulate over ``nr`` replicates.
    """
    if dtype is None:
        return torch.float64
    if dtype == torch.float32:
        warnings.warn(
            "float32 loses precision in the LRT statistic at single-cell group sizes; "
            "float64 is the default and costs little on H100 (1:2 fp64 throughput).",
            RuntimeWarning,
            stacklevel=2,
        )
    elif dtype != torch.float64:
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    return dtype


def make_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    """Seeded per-device generator, so results are reproducible and shard-independent."""
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def to_tk(
    *arrays,
    device: torch.device,
    dtype: torch.dtype,
    k: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Coerce inputs to a common broadcast ``(T, k)`` shape on ``device``.

    Accepts anything broadcastable: ``(k,)`` for statistics shared across all tests (e.g. a
    fixed group-size vector), ``(T, 1)``, or full ``(T, k)``.
    """
    tensors = []
    for a in arrays:
        t = a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 0:
            raise ValueError("summary statistics must have at least one dimension (k groups)")
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim > 2:
            raise ValueError(f"expected 1-D or 2-D summary statistics, got shape {tuple(t.shape)}")
        tensors.append(t)
    shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    if k is not None and shape[-1] != k:
        raise ValueError(f"expected k={k} groups, got {shape[-1]}")
    if shape[-1] < 2:
        raise ValueError("need at least k=2 groups to test equality of CVs")
    return tuple(t.expand(shape) for t in tensors)


def chi2_sf(x: torch.Tensor, df: Union[float, torch.Tensor]) -> torch.Tensor:
    """Upper tail of chi-square -- R's ``pchisq(x, df, lower.tail=FALSE)``, on device.

    Negative ``x`` clamps to 0 (p = 1), which is what R's ``1 - pchisq(negative, df)``
    gives. MSLRT statistics *can* be negative when the observed LRT falls below the
    bootstrap mean.
    """
    a = torch.as_tensor(df, dtype=x.dtype, device=x.device) / 2.0
    return torch.special.gammaincc(a, x.clamp_min(0.0) / 2.0)


def standard_normal(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    out = torch.empty(tuple(shape), device=device, dtype=dtype)
    return out.normal_(generator=generator)


def chi2_rvs(
    df: torch.Tensor,
    shape: Sequence[int],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw chi-square variates as ``2 * Gamma(df/2, 1)``.

    ``df`` is broadcast to ``shape``. Uses ``torch._standard_gamma`` (the sampler behind
    ``torch.distributions.Gamma``) because it accepts a per-element shape parameter, which a
    ``Gamma`` object would rebuild on every call.
    """
    conc = (df / 2.0).expand(tuple(shape)).contiguous()
    return 2.0 * torch._standard_gamma(conc, generator)


#: Live temporaries in the bootstrap inner loop, measured from the implementation in
#: :mod:`cvequality.mslrt` (u, F, Fp, x*, s*, vsq, plus Newton scratch). Used only to size
#: chunks; being off by a factor of two costs memory headroom, not correctness.
_BOOTSTRAP_TEMPORARIES = 12


def pick_chunk(
    n_tests: int,
    nr: int,
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fraction: float = 0.25,
    requested: Optional[int] = None,
) -> int:
    """Number of tests to process per bootstrap step.

    The bootstrap materializes ``(chunk, nr, k)`` arrays, so this is what keeps memory
    bounded. Targets ``fraction`` of *free* device memory across
    ``_BOOTSTRAP_TEMPORARIES`` live buffers.
    """
    if requested is not None:
        if requested < 1:
            raise ValueError("chunk must be >= 1")
        return min(int(requested), n_tests)
    itemsize = torch.finfo(dtype).bits // 8
    per_test = nr * k * itemsize * _BOOTSTRAP_TEMPORARIES
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        budget = int(free * fraction)
    else:
        budget = 2 * 1024**3  # CPU: keep the working set cache-friendly rather than maximal
    return int(max(1, min(n_tests, budget // max(per_test, 1))))
