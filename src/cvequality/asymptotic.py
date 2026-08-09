"""Feltz & Miller (1996) asymptotic test for equality of coefficients of variation.

Closed form, so the batched version is a handful of elementwise ops and effectively free
even at tens of millions of tests:

.. math::
    D = \\frac{\\sum_j m_j CV_j}{\\sum_j m_j}, \\qquad
    D_{AD} = \\frac{\\sum_j m_j (CV_j - D)^2}{D^2 (0.5 + D^2)} \\sim \\chi^2_{k-1}

with :math:`m_j = n_j - 1` and :math:`CV_j = s_j / \\bar{x}_j`.

Feltz CJ, Miller GE (1996) An asymptotic test for the equality of coefficients of variation
from k populations. *Stat Med* 15:647-658.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from ._backend import Status, chi2_sf, resolve_device, resolve_dtype, to_tk

__all__ = ["AsymptoticResult", "asymptotic_test", "asymptotic_test2", "asymptotic_test2_batch"]


class AsymptoticResult(NamedTuple):
    """Batched result. All tensors are 1-D of length ``T`` except ``status``, also ``(T,)``."""

    D_AD: torch.Tensor
    p_value: torch.Tensor
    #: Pooled CV under H0 (R's ``D``), useful as a covariate when interpreting results.
    cv_pooled: torch.Tensor
    status: torch.Tensor


def asymptotic_test2_batch(
    *,
    n,
    s,
    x,
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> AsymptoticResult:
    """Batched Feltz-Miller test over ``T`` independent tests of ``k`` groups each.

    Parameters
    ----------
    n, s, x : array_like or torch.Tensor
        Per-group sample size, sample SD (``ddof=1``) and mean, each broadcastable to
        ``(T, k)``. A ``(k,)`` input is shared across all tests -- e.g. pass ``n`` as
        ``(k,)`` when every test in the batch uses the same group sizes.
    device, dtype
        Default: CUDA when available, float64. See :func:`cvequality._backend.resolve_dtype`.

    Returns
    -------
    AsymptoticResult
        Degenerate tests yield ``NaN`` statistics and a non-zero
        :class:`~cvequality._backend.Status`; they are never silently dropped.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    n_t, s_t, x_t = to_tk(n, s, x, device=device, dtype=dtype)
    k = n_t.shape[-1]

    status = torch.full(n_t.shape[:-1], int(Status.OK), device=device, dtype=torch.int8)
    bad_n = (n_t < 2).any(dim=-1)
    bad_mean = (x_t <= 0).any(dim=-1) | ~torch.isfinite(x_t).all(dim=-1)
    bad_sd = (s_t <= 0).any(dim=-1) | ~torch.isfinite(s_t).all(dim=-1)
    # Assigned narrowest-cause-last so the reported status is the most specific one.
    status = torch.where(bad_sd, torch.tensor(int(Status.ZERO_VARIANCE), device=device, dtype=torch.int8), status)
    status = torch.where(bad_mean, torch.tensor(int(Status.NONPOSITIVE_MEAN), device=device, dtype=torch.int8), status)
    status = torch.where(bad_n, torch.tensor(int(Status.TOO_FEW_OBS), device=device, dtype=torch.int8), status)
    ok = status == int(Status.OK)

    # Neutralize degenerate rows before the arithmetic so they cannot emit warnings or
    # poison neighbours; their outputs are overwritten with NaN below.
    safe = ok.unsqueeze(-1)
    n_c = torch.where(safe, n_t, torch.full_like(n_t, 10.0))
    x_c = torch.where(safe, x_t, torch.ones_like(x_t))
    s_c = torch.where(safe, s_t, torch.ones_like(s_t))

    m = n_c - 1.0
    cv = s_c / x_c
    m_sum = m.sum(dim=-1)
    D = (m * cv).sum(dim=-1) / m_sum
    D_AD = (m * (cv - D.unsqueeze(-1)) ** 2).sum(dim=-1) / (D**2 * (0.5 + D**2))
    p = chi2_sf(D_AD, float(k - 1))

    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    non_finite = ok & ~torch.isfinite(D_AD)
    status = torch.where(non_finite, torch.tensor(int(Status.NON_FINITE), device=device, dtype=torch.int8), status)
    ok = status == int(Status.OK)
    return AsymptoticResult(
        D_AD=torch.where(ok, D_AD, nan),
        p_value=torch.where(ok, p, nan),
        cv_pooled=torch.where(ok, D, nan),
        status=status,
    )


def asymptotic_test2(*, k: Optional[int] = None, n, s, x, **kwargs) -> dict:
    """Single Feltz-Miller test from summary statistics -- R's ``asymptotic_test2``.

    Note the keyword-only signature: R's ``asymptotic_test2(k, n, s, x)`` and
    ``mslr_test2(nr, n, x, s)`` disagree on the order of ``s`` and ``x``, so positional
    calls are refused here rather than silently transposed.
    """
    res = asymptotic_test2_batch(n=n, s=s, x=x, **kwargs)
    if res.D_AD.numel() != 1:
        raise ValueError(
            f"asymptotic_test2 expects one test; got {res.D_AD.numel()}. "
            "Use asymptotic_test2_batch for many."
        )
    if k is not None and int(k) != int(torch.as_tensor(x).shape[-1]):
        raise ValueError(f"k={k} disagrees with the number of groups in x")
    return {"D_AD": float(res.D_AD), "p_value": float(res.p_value), "status": Status(int(res.status))}


def asymptotic_test(x, y, **kwargs) -> dict:
    """Single Feltz-Miller test from raw measurements -- R's ``asymptotic_test(x, y)``.

    Parameters
    ----------
    x : array_like
        Measurement values.
    y : array_like
        Grouping variable of the same length.
    """
    from .reference import group_summary

    n, mean, sd = group_summary(x, y)
    return asymptotic_test2(n=n, s=sd, x=mean, **kwargs)
