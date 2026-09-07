"""Krishnamoorthy & Lee (2014) modified signed-likelihood ratio test, batched for GPUs.

Krishnamoorthy K, Lee M (2014) Improved tests for the equality of normal coefficients of
variation. *Comput Stat* 29:215-232.

The test statistic is a signed LRT whose null distribution is estimated **per test** by a
parametric bootstrap, then used to recentre and rescale::

    statm = sqrt(2(k-1)) * (stat0 - mean(stat*)) / sd(stat*) + (k-1)   ~ chi2_{k-1}

so the cost is ``T * nr`` MLE solves. Two things make that tractable here.

**A properly converged MLE, in 2-6 Newton steps instead of ~300.**
``u_j(t)`` is the positive root of ``t u^2 + x u - (v + x^2) = 0``, hence
``(v + (x-u)^2)/u^2 == (t+1) - x/u`` exactly, and R's fixed point ``t <- G(t)`` collapses to
the scalar equation ``sum_j n_j x_j/u_j(t) == sum_j n_j``. Since ``du_j/dt < 0``, the left
side is strictly increasing whenever all ``x_j > 0``, so the root is unique and Newton is
unconditionally safe. R instead iterates ``t <- G(t)`` directly, which converges only
linearly at rate ~0.91 for CV~2.3 -- so R's cap of 31 iterations with an *absolute*
tolerance of 1e-7 returns an unconverged MLE at exactly single-cell CVs. See
:func:`cvequality.reference.solve_common_cv`.

**An exact, cancellation-free statistic.** At the root the quadratic terms of the
constrained log-likelihood sum to precisely ``N/2``, cancelling ``elf``'s ``-N/2``, so
``stat = 2 * sum_j n_j log(tauh u_j / v_j^{1/2})``. The literal R form subtracts two
``O(n log n)`` quantities and is unusable in float32 at ``n ~ 2e5``.

``solver="fixedpoint"`` reproduces R exactly -- same 31-iteration cap, same absolute
tolerance, same literal statistic, same ``u``/``tau`` off-by-one -- and is what the test
suite compares against the R fixtures. ``solver="newton"`` is the default everywhere else.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from ._backend import (
    Status,
    chi2_rvs,
    chi2_sf,
    make_generator,
    pick_chunk,
    resolve_device,
    resolve_dtype,
    standard_normal,
    to_tk,
)

__all__ = [
    "CommonCvFitBatch",
    "MslrtResult",
    "lrt_stat_batch",
    "mslr_test",
    "mslr_test2",
    "mslr_test2_batch",
    "solve_common_cv_batch",
]

#: Newton needs 2-6 steps in practice; the cap only guards pathological bootstrap draws.
NEWTON_MAX_ITER = 40
#: Float64 relative step tolerance for the Newton solver. Other dtypes use the same multiple
#: of machine epsilon.
NEWTON_TOL = 1e-13
#: R's absolute tolerance on ``t`` and its iteration cap, for ``solver="fixedpoint"``.
R_TOL = 1e-7
R_MAX_ITER = 31
#: Float64 bound on ``|sum(n x/u)/sum(n) - 1|``. Other dtypes use the same multiple
#: of machine epsilon.
RESIDUAL_TOL = 1e-8


class CommonCvFitBatch(NamedTuple):
    """Batched common-CV (H0) MLE. Leading dims are arbitrary; ``k`` is the last axis."""

    #: ``(..., k)`` constrained per-group means.
    u: torch.Tensor
    #: ``(...)`` common CV.
    tauh: torch.Tensor
    #: ``(...)`` LRT statistic.
    stat: torch.Tensor
    #: ``(...)`` bool -- met the tolerance.
    converged: torch.Tensor
    #: ``(...)`` scale-free residual of the MLE equation.
    residual: torch.Tensor


def _u_of_t(t: torch.Tensor, x: torch.Tensor, vsq: torch.Tensor) -> torch.Tensor:
    """Positive root of ``t u^2 + x u - (vsq + x^2) = 0``; R's closed form for ``uh``.

    Stays positive even for ``x < 0`` (which bootstrap draws can produce), because
    ``sqrt(x^2 + 4t(vsq+x^2)) > |x|``.
    """
    return (-x + torch.sqrt(x * x + 4.0 * t * (vsq + x * x))) / 2.0 / t


def _collapsed_stat(n: torch.Tensor, u: torch.Tensor, tauh: torch.Tensor, vsq: torch.Tensor):
    return 2.0 * (n * torch.log(tauh.unsqueeze(-1) * u / torch.sqrt(vsq))).sum(dim=-1)


def solve_common_cv_batch(
    *,
    n: torch.Tensor,
    x: torch.Tensor,
    s: torch.Tensor,
    tol: Optional[float] = None,
    max_iter: int = NEWTON_MAX_ITER,
    residual_tol: Optional[float] = None,
) -> CommonCvFitBatch:
    """Solve the common-CV MLE by Newton on ``F(t) = sum n_j x_j/u_j(t) - sum n_j``.

    ``F'(t) = sum n_j x_j / (2 t u_j + x_j)``. Elements converge independently; iteration
    stops as soon as every element's step is below ``tol``. Returns the exact collapsed
    statistic.

    The default step and residual tolerances are scaled from their float64 values by machine
    epsilon. All tensors are ``(..., k)`` and must already be float and broadcast to a common
    shape.
    """
    vsq = (n - 1.0) * s * s / n
    N = n.sum(dim=-1, keepdim=True)

    t = (n * vsq / (x * x)).sum(dim=-1, keepdim=True) / N  # R's starting value
    eps_scale = torch.finfo(t.dtype).eps / torch.finfo(torch.float64).eps
    if tol is None:
        tol = NEWTON_TOL * eps_scale
    if residual_tol is None:
        residual_tol = RESIDUAL_TOL * eps_scale
    active = torch.ones(t.shape[:-1], dtype=torch.bool, device=t.device)
    for _ in range(max_iter):
        u = _u_of_t(t, x, vsq)
        F = (n * x / u).sum(dim=-1, keepdim=True) - N
        Fp = (n * x / (2.0 * t * u + x)).sum(dim=-1, keepdim=True)
        t_new = t - F / Fp
        # F is monotone increasing for x>0, so retreating toward 0 is always a safe repair
        # for a step that overshoots into non-positive t (or goes non-finite).
        t_new = torch.where(t_new > 0, t_new, t * 0.5)
        step_small = ((t_new - t).abs() <= tol * t.clamp_min(1.0)).squeeze(-1)
        t = torch.where(active.unsqueeze(-1), t_new, t)
        active = active & ~step_small
        if not bool(active.any()):
            break

    u = _u_of_t(t, x, vsq)
    tauh = torch.sqrt(t).squeeze(-1)
    residual = ((n * x / u).sum(dim=-1) / N.squeeze(-1) - 1.0).abs()
    converged = torch.isfinite(residual) & (residual <= residual_tol)
    stat = _collapsed_stat(n, u, tauh, vsq)
    converged = converged & torch.isfinite(stat)
    return CommonCvFitBatch(u=u, tauh=tauh, stat=stat, converged=converged, residual=residual)


def _solve_common_cv_batch_r(
    *,
    n: torch.Tensor,
    x: torch.Tensor,
    s: torch.Tensor,
    tol: float = R_TOL,
    max_iter: int = R_MAX_ITER,
) -> CommonCvFitBatch:
    """R's ``LRT_STAT`` fixed point, elementwise-identical -- including its quirks.

    Replicates: the linear iteration ``t <- G(t)``, the *absolute* tolerance on ``t``, the
    31-iteration cap, the fact that on exit ``u`` corresponds to the pre-update ``t`` while
    ``tauh`` is the post-update value, and the literal (cancellation-prone) statistic.

    Each element freezes the moment its own break condition fires, which is what makes this
    match R's scalar loop rather than merely approximate it.
    """
    vsq = (n - 1.0) * s * s / n
    v = torch.sqrt(vsq)
    N = n.sum(dim=-1, keepdim=True)

    t0 = (n * vsq / (x * x)).sum(dim=-1, keepdim=True) / N
    u_out = torch.zeros_like(x)
    tau_out = torch.zeros_like(t0)
    done = torch.zeros(t0.shape[:-1], dtype=torch.bool, device=t0.device)
    conv_out = torch.zeros_like(done)

    for l in range(1, max_iter + 1):
        u = _u_of_t(t0, x, vsq)
        tau = (n * (vsq + (x - u) ** 2) / (u * u)).sum(dim=-1, keepdim=True) / N
        conv = ((tau - t0).abs() <= tol).squeeze(-1)
        newly = (conv | (l > max_iter - 1)) & ~done
        nb = newly.unsqueeze(-1)
        u_out = torch.where(nb, u, u_out)
        tau_out = torch.where(nb, tau, tau_out)
        conv_out = conv_out | (newly & conv)
        done = done | newly
        if bool(done.all()):
            break
        t0 = torch.where(done.unsqueeze(-1), t0, tau)

    tauh = torch.sqrt(tau_out).squeeze(-1)
    th = tauh.unsqueeze(-1)
    clf = (-n * torch.log(th * u_out) - (n * (vsq + (x - u_out) ** 2)) / (2.0 * th**2 * u_out**2)).sum(dim=-1)
    elf = (-n * torch.log(v) - n / 2.0).sum(dim=-1)
    stat = 2.0 * (elf - clf)
    residual = ((n * x / u_out).sum(dim=-1) / N.squeeze(-1) - 1.0).abs()
    return CommonCvFitBatch(u=u_out, tauh=tauh, stat=stat, converged=conv_out, residual=residual)


def lrt_stat_batch(
    *,
    n,
    x,
    s,
    solver: str = "newton",
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> CommonCvFitBatch:
    """Batched LRT statistic and H0 MLE.

    Parameters
    ----------
    n, x, s : array_like or torch.Tensor
        Per-group size, mean, sample SD (``ddof=1``), broadcastable to ``(T, k)``.
    solver : {"newton", "fixedpoint"}
        ``"newton"`` solves the MLE to convergence and returns the exact collapsed
        statistic. ``"fixedpoint"`` reproduces R's ``LRT_STAT`` bit for bit, including its
        non-convergence at large CV; use it only for R comparison.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    n_t, x_t, s_t = to_tk(n, x, s, device=device, dtype=dtype)
    if solver == "newton":
        return solve_common_cv_batch(n=n_t, x=x_t, s=s_t)
    if solver == "fixedpoint":
        return _solve_common_cv_batch_r(n=n_t, x=x_t, s=s_t)
    raise ValueError(f"solver must be 'newton' or 'fixedpoint', got {solver!r}")


class MslrtResult(NamedTuple):
    """Batched MSLRT result. All tensors are ``(T,)``."""

    MSLRT: torch.Tensor
    p_value: torch.Tensor
    #: Observed LRT statistic before recentring.
    stat0: torch.Tensor
    #: Bootstrap mean of the LRT under H0.
    null_mean: torch.Tensor
    #: Bootstrap SD of the LRT under H0 (``ddof=1``, matching R's ``sd``).
    null_sd: torch.Tensor
    #: Common CV under H0.
    tauh: torch.Tensor
    #: Bootstrap replicates actually used (``nr`` minus discarded draws).
    n_valid: torch.Tensor
    #: Whether the observed MLE met its tolerance. Under ``solver="fixedpoint"`` this is
    #: routinely ``False`` at single-cell CVs -- that is R's behaviour, reported rather than
    #: treated as a failure -- so inspect it instead of assuming ``status`` covers it.
    converged: torch.Tensor
    status: torch.Tensor


def mslr_test2_batch(
    *,
    n,
    x,
    s,
    nr: int = 1000,
    seed: Optional[int] = 0,
    solver: str = "newton",
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    chunk: Optional[int] = None,
    share_draws: bool = False,
    progress: bool = False,
) -> MslrtResult:
    """Batched MSLRT over ``T`` independent tests of ``k`` groups each.

    Parameters
    ----------
    n, x, s : array_like or torch.Tensor
        Per-group size, mean, sample SD (``ddof=1``), broadcastable to ``(T, k)``.
    nr : int
        Parametric bootstrap replicates per test. R's default is 1000.
    seed : int or None
        An integer seeds a private generator, so a sharded run is reproducible and each
        shard is independent of the others' scheduling. ``None`` uses PyTorch's global RNG.
    solver : {"newton", "fixedpoint"}
        Passed to the MLE for both the observed statistic and every bootstrap replicate.
    chunk : int, optional
        Tests per bootstrap step. Defaults to a memory-aware size derived from free device
        memory; the bootstrap materializes ``(chunk, nr, k)`` buffers. With independent
        draws, each chunk draws from a generator seeded by ``(seed, start position in the
        compacted batch of OK tests)``. Changing which neighbouring rows are degenerate can
        therefore shift an OK row's compacted position and change its result by Monte Carlo
        error. The same inputs with the same ``seed`` and ``chunk`` reproduce bit for bit.
        Different ``chunk`` values consume independent draws differently and therefore agree
        only up to Monte Carlo error -- pass ``chunk`` explicitly if you need independent-draw
        results that do not depend on how much memory happened to be free. Shared-draw results
        are independent of chunk boundaries.
    share_draws : bool
        Reuse one ``(nr, k)`` set of draws across every test in the batch (common random
        numbers). Requires all tests to share the same ``n``. Each test's bootstrap sample
        is still correctly distributed -- this only correlates Monte Carlo error across
        tests, which *reduces* noise when comparing them (e.g. ranking genes within one
        perturbation). Default ``False``, which is faithful to R's independent draws.
    progress : bool
        Print chunk progress to stdout; useful in long sharded runs.

    Notes
    -----
    Bootstrap draws can put a group mean at or below zero (``x* = u + z tau u/sqrt(n)``, so
    for one group at ``n=30``, ``Phi(-sqrt(n)/tau)`` is 0.31% of draws at ``tau=2``, 3.4% at
    ``tau=3``, and 8.5% at ``tau=4``; the fixture grid reaches CV 4. Such replicates are
    **discarded** from the null moments rather than contributing a garbage statistic;
    ``n_valid`` reports how many survived. The observed statistic is not subject to this
    truncation, so at small ``n`` and high CV the bootstrap null is slightly mismatched. R has
    no such guard.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    n_t, x_t, s_t = to_tk(n, x, s, device=device, dtype=dtype)
    T, k = n_t.shape
    nr = int(nr)
    if nr < 2:
        raise ValueError("nr must be >= 2 (the bootstrap SD needs two replicates)")
    if solver not in ("newton", "fixedpoint"):
        raise ValueError(f"solver must be 'newton' or 'fixedpoint', got {solver!r}")
    solve = solve_common_cv_batch if solver == "newton" else _solve_common_cv_batch_r
    # Newton failing to converge is a real error. R's fixed point failing to converge is
    # R's normal behaviour at single-cell CVs, so it is reported, not nullified -- otherwise
    # the R-parity mode would return NaN precisely where R and Newton disagree.
    require_convergence = solver == "newton"

    # ---- degenerate-input screening (same policy as the asymptotic test) ----
    i8 = lambda v: torch.tensor(int(v), device=device, dtype=torch.int8)  # noqa: E731
    status = torch.full((T,), int(Status.OK), device=device, dtype=torch.int8)
    status = torch.where((s_t <= 0).any(-1) | ~torch.isfinite(s_t).all(-1), i8(Status.ZERO_VARIANCE), status)
    status = torch.where((x_t <= 0).any(-1) | ~torch.isfinite(x_t).all(-1), i8(Status.NONPOSITIVE_MEAN), status)
    status = torch.where((n_t < 2).any(-1), i8(Status.TOO_FEW_OBS), status)
    ok = status == int(Status.OK)
    safe = ok.unsqueeze(-1)
    n_c = torch.where(safe, n_t, torch.full_like(n_t, 10.0))
    x_c = torch.where(safe, x_t, torch.ones_like(x_t))
    s_c = torch.where(safe, s_t, torch.ones_like(s_t))

    # ---- observed statistic ----
    fit0 = solve(n=n_c, x=x_c, s=s_c)
    if require_convergence:
        status = torch.where(ok & ~fit0.converged, i8(Status.NOT_CONVERGED), status)
    else:
        status = torch.where(ok & ~torch.isfinite(fit0.stat), i8(Status.NON_FINITE), status)

    if share_draws and T:
        spread = (n_t - n_t[:1]).abs().max()
        if bool(spread > 0):
            raise ValueError("share_draws=True requires every test in the batch to share the same n")

    ok = status == int(Status.OK)
    ok_idx = torch.nonzero(ok, as_tuple=False).squeeze(-1)
    n_ok = n_t[ok_idx]
    df = (n_ok - 1.0).unsqueeze(1)  # (T_ok,1,k)
    u0 = fit0.u[ok_idx]
    sh0 = fit0.tauh[ok_idx].unsqueeze(-1) * u0
    se0 = sh0 / torch.sqrt(n_ok)
    if share_draws and ok_idx.numel():
        shared_df = (n_ok[:1] - 1.0).unsqueeze(1)

    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    null_mean = torch.full((T,), nan, device=device, dtype=dtype)
    null_sd = torch.full((T,), nan, device=device, dtype=dtype)
    n_valid = torch.zeros(T, device=device, dtype=torch.int64)

    T_ok = ok_idx.numel()
    step = pick_chunk(T_ok, nr, k, device=device, dtype=dtype, requested=chunk) if T_ok else 1
    if share_draws and T_ok:
        # Common random numbers are call-wide, not chunk-local. Keeping the bases at
        # (1, nr, k) and expanding them as views preserves bounded chunk memory.
        gen = make_generator(None if seed is None else seed * 1_000_003, device)
        shared_z = standard_normal((1, nr, k), device=device, dtype=dtype, generator=gen)
        shared_ch = chi2_rvs(shared_df, (1, nr, k), generator=gen)
    for start in range(0, T_ok, step):
        stop = min(start + step, T_ok)
        c = stop - start
        dfc = df[start:stop]
        if share_draws:
            z = shared_z.expand(c, nr, k)
            ch = shared_ch.expand(c, nr, k)
        else:
            # Seed from the chunk's start in the compacted batch so a given `chunk` value is
            # bit-reproducible regardless of iteration order or preceding chunks.
            gen = make_generator(None if seed is None else seed * 1_000_003 + start, device)
            z = standard_normal((c, nr, k), device=device, dtype=dtype, generator=gen)
            ch = chi2_rvs(dfc, (c, nr, k), generator=gen)

        xb = u0[start:stop].unsqueeze(1) + z * se0[start:stop].unsqueeze(1)
        sb = sh0[start:stop].unsqueeze(1) * torch.sqrt(ch / dfc)
        nb = n_ok[start:stop].unsqueeze(1).expand(c, nr, k)

        # Replicates outside the model's support: neutralize before solving, drop after.
        good = (xb > 0).all(dim=-1)
        xb = torch.where(good.unsqueeze(-1), xb, torch.ones_like(xb))
        fb = solve(n=nb, x=xb, s=sb)
        valid = good & torch.isfinite(fb.stat)
        if require_convergence:
            valid = valid & fb.converged

        w = valid.to(dtype)
        cnt = w.sum(dim=1)
        st = torch.where(valid, fb.stat, torch.zeros_like(fb.stat))
        mean_c = st.sum(dim=1) / cnt
        # sum((st-mean)^2) restricted to valid replicates, then ddof=1 as R's sd()
        var_c = ((st - mean_c.unsqueeze(1)) ** 2 * w).sum(dim=1) / (cnt - 1.0)
        dest = ok_idx[start:stop]
        null_mean[dest] = mean_c
        null_sd[dest] = torch.sqrt(var_c)
        n_valid[dest] = cnt.to(torch.int64)
        if progress:
            print(f"  mslrt {stop}/{T_ok} tests", flush=True)
        del z, ch, xb, sb, nb, fb, st, w

    statm = torch.sqrt(torch.tensor(2.0 * (k - 1), device=device, dtype=dtype)) * (
        fit0.stat - null_mean
    ) / null_sd + (k - 1)
    p = chi2_sf(statm, float(k - 1))

    ok = status == int(Status.OK)
    bad = ok & (~torch.isfinite(statm) | (n_valid < 2))
    status = torch.where(bad, i8(Status.NON_FINITE), status)
    ok = status == int(Status.OK)
    return MslrtResult(
        MSLRT=torch.where(ok, statm, nan),
        p_value=torch.where(ok, p, nan),
        stat0=torch.where(ok, fit0.stat, nan),
        null_mean=torch.where(ok, null_mean, nan),
        null_sd=torch.where(ok, null_sd, nan),
        tauh=torch.where(ok, fit0.tauh, nan),
        n_valid=n_valid,
        converged=fit0.converged,
        status=status,
    )


def mslr_test2(*, n, x, s, nr: int = 1000, seed: Optional[int] = 0, **kwargs) -> dict:
    """Single MSLRT from summary statistics -- R's ``mslr_test2(nr, n, x, s)``.

    Keyword-only: R's ``asymptotic_test2(k, n, s, x)`` and ``mslr_test2(nr, n, x, s)``
    disagree on the order of ``s`` and ``x``.
    """
    res = mslr_test2_batch(n=n, x=x, s=s, nr=nr, seed=seed, **kwargs)
    if res.MSLRT.numel() != 1:
        raise ValueError(
            f"mslr_test2 expects one test; got {res.MSLRT.numel()}. Use mslr_test2_batch for many."
        )
    return {
        "MSLRT": float(res.MSLRT),
        "p_value": float(res.p_value),
        "stat0": float(res.stat0),
        "null_mean": float(res.null_mean),
        "null_sd": float(res.null_sd),
        "tauh": float(res.tauh),
        "n_valid": int(res.n_valid),
        "converged": bool(res.converged),
        "status": Status(int(res.status)),
    }


def mslr_test(x, y, *, nr: int = 1000, seed: Optional[int] = 0, **kwargs) -> dict:
    """Single MSLRT from raw measurements -- R's ``mslr_test(nr, x, y)``."""
    from .reference import group_summary

    n, mean, sd = group_summary(x, y)
    return mslr_test2(n=n, x=mean, s=sd, nr=nr, seed=seed, **kwargs)
