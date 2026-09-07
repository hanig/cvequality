"""Line-by-line NumPy transcription of the ``cvequality`` R package (CRAN 0.2.0).

This module is deliberately **scalar and slow**. It exists to be obviously correct by
inspection against ``R/functions.R`` upstream, and it is the oracle that the vectorized
GPU implementations in :mod:`cvequality.asymptotic` and :mod:`cvequality.mslrt` are tested
against. Do not optimize it.

Upstream source: https://github.com/benmarwick/cvequality/blob/master/R/functions.R

Two faithfulness details that matter and are easy to get wrong:

1. ``LRT_STAT``'s fixed-point loop leaves ``uh`` and ``tauh`` **one iteration apart**. R's
   loop body computes ``uh`` from ``tau0``, then ``tau`` from ``uh``, then breaks. So on
   exit ``uh`` corresponds to the *pre-update* ``tau0`` while ``tauh = sqrt(tau)`` is the
   *post-update* value. Iterating to full convergence instead changes the statistic at the
   ~1e-7 level. We reproduce R.
2. R's ``sd()`` uses the ``n-1`` denominator, so every sample SD here is ``ddof=1``.

R's own argument orders are inconsistent -- ``asymptotic_test2(k, n, s, x)`` versus
``mslr_test2(nr, n, x, s)`` swap ``s`` and ``x``. All summary-statistic arguments in this
package are keyword-only so that hazard cannot bite.

.. warning::
   **R's MLE does not converge at single-cell scale.** ``LRT_STAT``'s fixed-point iteration
   converges only *linearly*, with a rate that approaches 1 as the common CV grows: rate
   ~0.15 at CV~0.3 (converges in 6 iterations) but ~0.91 at CV~2.3, which needs ~300
   iterations. R caps at 31 with an *absolute* tolerance of 1e-7, so for perturb-seq CVs
   (typically 1-3) it silently returns an unconverged MLE -- e.g. 358163.859 where the true
   value is 358163.189. See :func:`solve_common_cv` for the properly converged solver used
   by the vectorized code paths, and :mod:`cvequality.mslrt` for how the two are exposed.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

__all__ = [
    "MLE_MAX_ITER",
    "MLE_TOL",
    "LrtStat",
    "asymptotic_test",
    "asymptotic_test2",
    "chi2_sf",
    "group_summary",
    "lrt_stat",
    "mslr_test",
    "mslr_test2",
    "solve_common_cv",
]

#: R breaks on ``l > 30`` *after* running the body, so the body runs at most 31 times.
MLE_MAX_ITER = 31
#: R's ``abs(tau - tau0) <= 1.0e-7``.
MLE_TOL = 1.0e-7


def chi2_sf(x: float, df: float) -> float:
    """Upper tail of the chi-square distribution -- R's ``pchisq(x, df, lower=FALSE)``.

    Uses the regularized upper incomplete gamma function so the reference module has the
    same closed form the torch backend uses (``gammaincc(df/2, x/2)``).
    """
    from scipy.special import gammaincc

    return float(gammaincc(df / 2.0, np.asarray(x, dtype=np.float64) / 2.0))


def group_summary(x, y):
    """Per-group ``(n, mean, sd)``, mirroring R's ``table``/``aggregate`` pair.

    R derives group order from factor levels (``table(y)``) and ``aggregate(by=list(y))``,
    both of which sort. We sort too, though it is immaterial: every statistic in this
    package is a sum over groups and therefore invariant to their order.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if np.isnan(x).any():
        raise ValueError("x cannot contain any NA values")
    groups = np.unique(y)
    n = np.empty(groups.size, dtype=np.float64)
    mean = np.empty(groups.size, dtype=np.float64)
    sd = np.empty(groups.size, dtype=np.float64)
    for j, g in enumerate(groups):
        xs = x[y == g]
        n[j] = xs.size
        mean[j] = xs.mean()
        sd[j] = xs.std(ddof=1)  # R's sd()
    return n, mean, sd


# ---------------------------------------------------------------------------
# Feltz & Miller (1996) asymptotic test
# ---------------------------------------------------------------------------


def asymptotic_test(x, y) -> dict:
    """Feltz & Miller (1996) asymptotic test for equality of CVs, from raw measurements.

    Transcribes R ``asymptotic_test(x, y)``.

    Parameters
    ----------
    x : array_like
        Individual measurement values.
    y : array_like
        Grouping variable, same length as ``x``.

    Returns
    -------
    dict
        ``{"D_AD": float, "p_value": float}``.
    """
    n, mean, sd = group_summary(x, y)
    return asymptotic_test2(k=n.size, n=n, s=sd, x=mean)


def asymptotic_test2(*, k: Optional[int] = None, n, s, x) -> dict:
    """Feltz & Miller (1996) asymptotic test from per-group summary statistics.

    Transcribes R ``asymptotic_test2(k, n, s, x)``::

        m_j  = n_j - 1
        D    = sum(m_j * s_j/x_j) / sum(m_j)
        D_AD = sum(m_j * (s_j/x_j - D)^2) / (D^2 * (0.5 + D^2))   ~ chi2_{k-1}

    Parameters
    ----------
    k : int, optional
        Number of groups. R takes it as an argument; here it defaults to ``len(x)`` and is
        only used for the degrees of freedom. Passing a value inconsistent with ``len(x)``
        raises rather than silently producing the wrong df.
    n, s, x : array_like
        Per-group sample size, sample SD (``ddof=1``), and mean.
    """
    n = np.asarray(n, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not (n.shape == s.shape == x.shape):
        raise ValueError(f"n, s, x must have the same shape; got {n.shape}, {s.shape}, {x.shape}")
    if k is None:
        k = int(x.size)
    elif int(k) != int(x.size):
        raise ValueError(f"k={k} disagrees with len(x)={x.size}")

    m_j = n - 1.0
    cv = s / x
    D = float(np.sum(m_j * cv) / np.sum(m_j))
    D_AD = float(np.sum(m_j * (cv - D) ** 2) / (D**2 * (0.5 + D**2)))
    return {"D_AD": D_AD, "p_value": chi2_sf(D_AD, k - 1)}


# ---------------------------------------------------------------------------
# Krishnamoorthy & Lee (2014) modified signed-likelihood ratio test
# ---------------------------------------------------------------------------


class LrtStat(NamedTuple):
    """Return value of :func:`lrt_stat`. R returns ``c(uh, tauh, stat)``."""

    uh: np.ndarray
    tauh: float
    stat: float
    #: True if the fixed point met ``MLE_TOL``; False if it exhausted ``MLE_MAX_ITER``.
    #: Extra information -- R does not report this.
    converged: bool
    #: Iterations actually run (R's ``l`` at break).
    n_iter: int


def lrt_stat(*, n, x, s, tol: float = MLE_TOL, max_iter: int = MLE_MAX_ITER) -> LrtStat:
    """Signed log-likelihood ratio statistic for equality of CVs.

    Transcribes R ``LRT_STAT(n, x, s)``. Fits the common-CV MLE ``(u_j, tau)`` under H0 by
    fixed-point iteration, then returns twice the difference between the unconstrained and
    constrained log-likelihoods.

    Note ``tau0``/``tau`` inside the iteration are **tau-squared**; only the returned
    ``tauh`` is the CV itself (R takes ``sqrt`` at the end).

    Parameters
    ----------
    tol, max_iter :
        Defaults reproduce R exactly. Passing ``tol=0.0`` with a larger ``max_iter`` drives
        the fixed point to full convergence, which is how the test suite demonstrates that
        the residual gap to :func:`lrt_stat_collapsed` is entirely R's early break.
    """
    n = np.asarray(n, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if not (n.shape == x.shape == s.shape):
        raise ValueError(f"n, x, s must have the same shape; got {n.shape}, {x.shape}, {s.shape}")

    k = x.size
    df = n - 1.0
    ssq = s**2
    vsq = df * ssq / n  # MLE variance (biased), R's vsq
    v = np.sqrt(vsq)
    sn = float(np.sum(n))

    # --- MLEs under H0 ---
    tau0 = float(np.sum(n * vsq / x**2) / sn)
    l = 1
    while True:
        uh = (-x + np.sqrt(x**2 + 4.0 * tau0 * (vsq + x**2))) / 2.0 / tau0
        tau = float(np.sum(n * (vsq + (x - uh) ** 2) / uh**2) / sn)
        converged = abs(tau - tau0) <= tol
        if converged or l > max_iter - 1:
            break
        l += 1
        tau0 = tau
    tauh = float(np.sqrt(tau))
    # NB: uh is the value computed from the *pre-update* tau0 -- see module docstring.
    # --- END MLEs ---

    clf = 0.0
    elf = 0.0
    for j in range(k):
        clf = (
            clf
            - n[j] * np.log(tauh * uh[j])
            - (n[j] * (vsq[j] + (x[j] - uh[j]) ** 2)) / (2.0 * tauh**2 * uh[j] ** 2)
        )
        elf = elf - n[j] * np.log(v[j]) - n[j] / 2.0
    stat = float(2.0 * (elf - clf))
    return LrtStat(uh=uh, tauh=tauh, stat=stat, converged=bool(converged), n_iter=l)


def lrt_stat_collapsed(*, n, x, s, tol: float = MLE_TOL, max_iter: int = MLE_MAX_ITER) -> float:
    """The same statistic in cancellation-free form: ``2 * sum(n_j log(tauh u_j / v_j))``.

    At the fixed point the MLE equation forces
    ``sum n_j (vsq_j + (x_j-u_j)^2)/u_j^2 == sum(n_j) * tauh^2``, so the quadratic terms of
    ``clf`` and the ``-n_j/2`` terms of ``elf`` cancel exactly. The literal form in
    :func:`lrt_stat` subtracts two ``O(n log n)`` quantities, which loses all precision in
    float32 at single-cell group sizes; this form does not.

    The two forms are identical **only at the converged fixed point**. With R's default
    early break (``tol=1e-7``, and ``uh`` one iteration behind ``tauh``) they differ by the
    fixed-point residual -- empirically up to ~4e-7 relative on the fixture grid. Drive
    ``tol=0.0, max_iter=200`` and they agree to ~1e-15. Both facts are asserted in the test
    suite; the literal form remains the default so the package reproduces R.
    """
    n = np.asarray(n, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    fit = lrt_stat(n=n, x=x, s=s, tol=tol, max_iter=max_iter)
    vsq = (n - 1.0) * s**2 / n
    return float(2.0 * np.sum(n * np.log(fit.tauh * fit.uh / np.sqrt(vsq))))


class CommonCvFit(NamedTuple):
    """Converged H0 (common-CV) MLE from :func:`solve_common_cv`."""

    #: Per-group constrained mean estimates.
    u: np.ndarray
    #: The common CV itself (``sqrt`` of the solved ``t``).
    tauh: float
    #: ``t = tauh**2``, the quantity the equation is solved in.
    t: float
    #: LRT statistic from the exact collapsed form ``2*sum(n log(tauh u / v^0.5))``.
    stat: float
    n_iter: int
    converged: bool
    #: ``|sum(n x/u)/sum(n) - 1|`` -- the scale-free residual of the MLE equation.
    residual: float


def solve_common_cv(*, n, x, s, tol: float = 1e-13, max_iter: int = 60) -> CommonCvFit:
    r"""Solve the common-CV MLE properly, by Newton on a scalar monotone equation.

    R's fixed point ``t <- G(t)`` converges linearly and far too slowly at single-cell CVs
    (see the module warning). The system reduces to something much better behaved.

    ``u_j(t)`` is defined as the positive root of

    .. math:: t u_j^2 + x_j u_j - (v_j + x_j^2) = 0

    which is exactly R's closed form. Substituting ``v_j = t u_j^2 + x_j u_j - x_j^2``:

    .. math:: \frac{v_j + (x_j-u_j)^2}{u_j^2} = (t+1) - \frac{x_j}{u_j}

    so R's update ``t = (1/N) sum n_j (v_j + (x_j-u_j)^2)/u_j^2`` has fixed point

    .. math:: \sum_j n_j \frac{x_j}{u_j(t)} = \sum_j n_j =: N

    Because ``du_j/dt = -u_j^2/(2 t u_j + x_j) < 0``, the left side is strictly increasing
    in ``t``, so the root is **unique** and Newton is unconditionally safe with

    .. math:: F(t) = \sum_j n_j x_j/u_j - N, \qquad F'(t) = \sum_j n_j \frac{x_j}{2 t u_j + x_j}

    Two further consequences at the root:

    * the quadratic terms of the constrained log-likelihood sum to exactly ``N/2``, which
      cancels ``elf``'s ``-N/2``, making ``stat = 2 sum n_j log(tauh u_j / v_j^{1/2})``
      **exact** rather than merely a good approximation;
    * ``residual`` below is a scale-free convergence check, unlike R's absolute tolerance
      on ``t`` (which is why R's criterion degrades as the CV grows).

    Converges in 2-6 iterations on the whole fixture grid, versus ~300 for R's iteration.
    """
    n = np.asarray(n, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if not (n.shape == x.shape == s.shape):
        raise ValueError(f"n, x, s must have the same shape; got {n.shape}, {x.shape}, {s.shape}")

    vsq = (n - 1.0) * s**2 / n
    N = float(np.sum(n))
    t = float(np.sum(n * vsq / x**2) / N)  # R's starting value

    a = vsq + x**2

    def terms(t_):
        root = np.sqrt(x**2 + 4.0 * t_ * a)
        # Unlike the literal R oracle above, this properly-converged oracle uses the
        # rationalized positive root and a centered residual so it remains useful at low CV.
        u = 2.0 * a / (x + root)
        centered = 2.0 * x * t_ / (root + x) - vsq / a
        derivative = x / root
        return u, centered, derivative

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        _, centered, derivative = terms(t)
        residual_signed = float(np.sum(n * centered) / N)
        derivative_scaled = float(np.sum(n * derivative) / N)
        delta = residual_signed / derivative_scaled
        t_new = t - delta
        if not (t_new > 0.0 and np.isfinite(t_new)):
            t_new = t / 2.0
        step_ok = abs(delta) <= tol * abs(t)
        t = t_new
        if step_ok and abs(residual_signed) <= 1e-12:
            converged = True
            break

    u, centered, derivative = terms(t)
    tauh = float(np.sqrt(t))
    stat = float(2.0 * np.sum(n * np.log(tauh * u / np.sqrt(vsq))))
    residual_signed = float(np.sum(n * centered) / N)
    correction = abs(residual_signed / float(np.sum(n * derivative) / N))
    residual = abs(residual_signed)
    converged = bool(
        np.isfinite(residual)
        and np.isfinite(correction)
        and residual <= 1e-12
        and correction <= tol * abs(t)
    )
    return CommonCvFit(
        u=u, tauh=tauh, t=t, stat=stat, n_iter=it, converged=converged, residual=residual
    )


def mslr_test(x, y, *, nr: int = 1000, seed: Optional[int] = None) -> dict:
    """Krishnamoorthy & Lee (2014) MSLRT for equality of CVs, from raw measurements.

    Transcribes R ``mslr_test(nr, x, y)``.

    The p-value is Monte Carlo: the parametric bootstrap estimates the mean and SD of the
    LRT statistic under H0, which are used to recentre and rescale ``stat0``. Two runs with
    different seeds will not agree exactly; at ``nr=1e4`` the statistic moves by ~1%.
    """
    n, mean, sd = group_summary(x, y)
    return mslr_test2(n=n, x=mean, s=sd, nr=nr, seed=seed)


def mslr_test2(*, n, x, s, nr: int = 1000, seed: Optional[int] = None) -> dict:
    """Krishnamoorthy & Lee (2014) MSLRT from per-group summary statistics.

    Transcribes R ``mslr_test2(nr, n, x, s)``::

        stat0                = LRT_STAT(n, x, s)
        x* = uh0 + z * tauh0 uh0/sqrt(n)      z ~ N(0,1)
        s* = tauh0 uh0 * sqrt(chi2_df / df)
        statm = sqrt(2(k-1)) (stat0 - mean(stat*)) / sd(stat*) + (k-1)   ~ chi2_{k-1}

    Parameters
    ----------
    n, x, s : array_like
        Per-group sample size, mean, and sample SD (``ddof=1``).
    nr : int
        Parametric bootstrap replicates. R's default is 1e3.
    seed : int, optional
        Seed for a private ``numpy.random.Generator``. R seeds the global RNG instead, and
        R's Mersenne-Twister stream differs from NumPy's PCG64 regardless, so this cannot
        reproduce a specific R run -- only its distribution.
    """
    n = np.asarray(n, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    nr = int(nr)
    if nr < 2:
        raise ValueError("nr must be >= 2 (the bootstrap SD needs two replicates)")

    k = x.size
    df = n - 1.0

    xst0 = lrt_stat(n=n, x=x, s=s)
    uh0, tauh0, stat0 = xst0.uh, xst0.tauh, xst0.stat
    sh0 = tauh0 * uh0
    se0 = tauh0 * uh0 / np.sqrt(n)

    rng = np.random.default_rng(seed)
    gv = np.empty(nr, dtype=np.float64)
    for ii in range(nr):
        z = rng.standard_normal(k)
        xb = uh0 + z * se0
        ch = rng.chisquare(df)
        sb = sh0 * np.sqrt(ch / df)
        gv[ii] = lrt_stat(n=n, x=xb, s=sb).stat

    am = float(gv.mean())
    sdv = float(gv.std(ddof=1))  # R's sd()
    statm = float(np.sqrt(2.0 * (k - 1)) * (stat0 - am) / sdv + (k - 1))
    return {
        "MSLRT": statm,
        "p_value": chi2_sf(statm, k - 1),
        # Extras beyond R, used by the test suite to assert on the null moments rather
        # than on a single Monte Carlo draw.
        "stat0": stat0,
        "null_mean": am,
        "null_sd": sdv,
        "tauh": tauh0,
    }
