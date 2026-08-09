"""Equality of standard deviations across k groups -- a sibling test, not part of the R port.

Motivation. Because ``cv = sd/mean``,

.. math:: \\log_2 \\frac{cv_g}{cv_r} = \\log_2 \\frac{sd_g}{sd_r} - \\log_2 \\frac{\\mu_g}{\\mu_r}

so "the CV change with the differential-expression component removed" **is** the log SD ratio.
Testing it directly avoids the post-hoc correction that :mod:`cvequality.adapter` documents as
worse than useless, and it does so without a threshold to tune. On simulated screens with known
ground truth it separates genuine dispersion changes from pure mean changes substantially
better than the CV test does, at a comparable hit count.

Statistic. With :math:`l_j = \\log sd_j`, the sampling variance of a log standard deviation is

.. math:: \\mathrm{Var}(l_j) \\approx \\frac{\\gamma_{4,j} - 1}{4 n_j}

(:math:`\\gamma_4` = kurtosis; the familiar :math:`1/(2(n_j-1))` is the special case
:math:`\\gamma_4 = 3`). Inverse-variance weights :math:`w_j = 1/\\mathrm{Var}(l_j)` then give a
Wald statistic for equality of all k standard deviations,

.. math:: \\sum_j w_j (l_j - \\bar{l})^2 \\sim \\chi^2_{k-1}, \\qquad
          \\bar{l} = \\frac{\\sum_j w_j l_j}{\\sum_j w_j}

which for k=2 is the square of the usual two-sample log-SD z.

**The kurtosis correction is what makes this usable**, and it is only available because the
calibrated transform is ``log1p``. Assuming normality (``kurt = 3``) leaves the test
noticeably anti-conservative; substituting the sample kurtosis brings it close to nominal.
Under ``tp10k`` the same correction is impossible -- its kurtosis estimate is barely
reproducible between random halves of the same cells, whereas under ``log1p`` it is stable.

Caveat: like the CV, a log-SD ratio computed on ``log1p`` values is **not** invariant to
``target_sum``, because ``log1p`` is only logarithmic for ``x >> 1`` and single-cell data is
dominated by zeros and small values -- so the location-shift argument that would make an SD
ratio scale-free does not hold here. Pin ``target_sum`` as you would for the CV tests.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from ._backend import Status, chi2_sf, resolve_device, resolve_dtype, to_tk

__all__ = ["SdRatioResult", "sd_ratio_test", "sd_ratio_test_batch"]

#: Kurtosis of a normal distribution -- the fallback when none is supplied.
NORMAL_KURTOSIS = 3.0


class SdRatioResult(NamedTuple):
    """Batched result. All tensors are ``(T,)``."""

    #: Wald statistic, chi-square with k-1 df.
    stat: torch.Tensor
    p_value: torch.Tensor
    #: ``log2(sd_last / sd_first)`` -- the effect size, meaningful for k=2.
    log2_sd_ratio: torch.Tensor
    #: Inverse-variance weighted mean of log(sd), i.e. the pooled SD under H0.
    log_sd_pooled: torch.Tensor
    status: torch.Tensor


def sd_ratio_test_batch(
    *,
    n,
    sd,
    kurtosis=None,
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SdRatioResult:
    """Batched test for equality of standard deviations across ``k`` groups.

    Parameters
    ----------
    n, sd : array_like or torch.Tensor
        Per-group sample size and sample SD (``ddof=1``), broadcastable to ``(T, k)``.
    kurtosis : array_like or torch.Tensor, optional
        Per-group kurtosis of the values (non-excess). Default 3.0, i.e. normal theory, which
        is measurably anti-conservative on real data -- pass
        :attr:`cvequality.sufficient.GroupStats.kurtosis` instead.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    if kurtosis is None:
        kurtosis = NORMAL_KURTOSIS
    if not isinstance(kurtosis, torch.Tensor) and not hasattr(kurtosis, "__len__"):
        kurtosis = torch.full_like(torch.as_tensor(sd, dtype=dtype), float(kurtosis))
    n_t, sd_t, k4 = to_tk(n, sd, kurtosis, device=device, dtype=dtype)
    k = n_t.shape[-1]

    i8 = lambda v: torch.tensor(int(v), device=device, dtype=torch.int8)  # noqa: E731
    status = torch.full(n_t.shape[:-1], int(Status.OK), device=device, dtype=torch.int8)
    status = torch.where((sd_t <= 0).any(-1) | ~torch.isfinite(sd_t).all(-1), i8(Status.ZERO_VARIANCE), status)
    status = torch.where((n_t < 2).any(-1), i8(Status.TOO_FEW_OBS), status)
    ok = status == int(Status.OK)
    safe = ok.unsqueeze(-1)
    sd_c = torch.where(safe, sd_t, torch.ones_like(sd_t))
    n_c = torch.where(safe, n_t, torch.full_like(n_t, 10.0))
    # Kurtosis is >= 1 for any distribution; clamp just above so the variance stays positive.
    k4_c = torch.where(safe & torch.isfinite(k4), k4, torch.full_like(k4, NORMAL_KURTOSIS))
    k4_c = k4_c.clamp_min(1.0 + 1e-6)

    l = torch.log(sd_c)
    var = (k4_c - 1.0) / (4.0 * n_c)
    w = 1.0 / var
    w_sum = w.sum(dim=-1)
    lbar = (w * l).sum(dim=-1) / w_sum
    stat = (w * (l - lbar.unsqueeze(-1)) ** 2).sum(dim=-1)
    p = chi2_sf(stat, float(k - 1))

    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    status = torch.where(ok & ~torch.isfinite(stat), i8(Status.NON_FINITE), status)
    ok = status == int(Status.OK)
    return SdRatioResult(
        stat=torch.where(ok, stat, nan),
        p_value=torch.where(ok, p, nan),
        log2_sd_ratio=torch.where(ok, (l[..., -1] - l[..., 0]) / torch.log(torch.tensor(2.0, device=device, dtype=dtype)), nan),
        log_sd_pooled=torch.where(ok, lbar, nan),
        status=status,
    )


def sd_ratio_test(*, n, sd, kurtosis=None, **kwargs) -> dict:
    """Single test for equality of standard deviations, from summary statistics."""
    res = sd_ratio_test_batch(n=n, sd=sd, kurtosis=kurtosis, **kwargs)
    if res.stat.numel() != 1:
        raise ValueError(
            f"sd_ratio_test expects one test; got {res.stat.numel()}. Use sd_ratio_test_batch."
        )
    return {
        "stat": float(res.stat),
        "p_value": float(res.p_value),
        "log2_sd_ratio": float(res.log2_sd_ratio),
        "status": Status(int(res.status)),
    }
