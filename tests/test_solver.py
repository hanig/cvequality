"""Properties of the Newton MLE solver that the R fixed point does not have.

The key claims from :func:`cvequality.reference.solve_common_cv`:

* the MLE system reduces to ``sum n_j x_j/u_j(t) == sum n_j``, strictly increasing in ``t``;
* Newton reaches machine precision in a handful of iterations, where R's linear iteration
  needs hundreds and gives up at 31;
* at the converged root the collapsed statistic ``2 sum n_j log(tauh u_j / v_j^0.5)`` equals
  the literal log-likelihood difference exactly.
"""

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality import reference as ref

from .conftest import as_case, relerr

pytestmark = pytest.mark.rfixtures


def _fixedpoint_t(n, x, s, iters):
    """R's linear iteration, run far past its cap, as an independent root estimate."""
    v = (n - 1) * s**2 / n
    sn = n.sum()
    t = (n * v / x**2).sum() / sn
    for _ in range(iters):
        u = (-x + np.sqrt(x**2 + 4.0 * t * (v + x**2))) / 2.0 / t
        t = (n * (v + (x - u) ** 2) / u**2).sum() / sn
    return float(t)


def test_newton_finds_the_same_root_as_a_long_fixedpoint_run(fx):
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        fit = ref.solve_common_cv(n=n, x=x, s=s)
        slow = _fixedpoint_t(n, x, s, 20_000)
        assert fit.converged, c["label"]
        assert relerr(fit.t, slow) < 1e-9, f"{c['label']}: newton {fit.t!r} vs fixedpoint {slow!r}"


def test_newton_converges_everywhere_and_fast(fx):
    worst_iter, worst_resid = 0, 0.0
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        fit = ref.solve_common_cv(n=n, x=x, s=s)
        assert fit.converged, c["label"]
        worst_iter = max(worst_iter, fit.n_iter)
        worst_resid = max(worst_resid, fit.residual)
    assert worst_iter <= 12, f"newton needed {worst_iter} iterations"
    assert worst_resid < 1e-12, f"worst MLE residual {worst_resid:.3e}"


def test_r_fixedpoint_does_not_converge_at_single_cell_cvs(fx):
    """Documents the defect the Newton solver exists to fix.

    If a future R release fixes this, this test fails loudly rather than silently keeping a
    stale justification in the docs.
    """
    stalled = [
        c["label"] for c in fx["deterministic"]
        if not ref.lrt_stat(n=as_case(c)[0], x=as_case(c)[1], s=as_case(c)[2]).converged
    ]
    assert stalled, "expected R's 31-iteration cap to be hit by the high-CV fixture cases"
    assert any("pseq" in s or "omnibus" in s for s in stalled)


def _cancellation_bound(n, x, s):
    """Floating-point error floor of the *literal* statistic form.

    ``stat = 2(elf - clf)`` differences two sums of magnitude ``O(n log n)``, so its absolute
    error is bounded by roughly ``2 * eps * |elf|``. At n~2e5 that is ~1e-10 absolute, which
    is larger than a near-null statistic of ~1e-4 in *relative* terms -- hence this test
    asserts on absolute agreement against the bound, not on relative agreement.
    """
    vsq = (n - 1) * s**2 / n
    elf = abs(float((-n * np.log(np.sqrt(vsq)) - n / 2.0).sum()))
    return 4.0 * np.finfo(np.float64).eps * max(elf, 1.0)


def test_collapsed_statistic_is_exact_at_the_converged_root(fx):
    """The quadratic terms provably cancel, so the two forms coincide to the noise floor."""
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        literal = ref.lrt_stat(n=n, x=x, s=s, tol=0.0, max_iter=20_000)
        collapsed = ref.lrt_stat_collapsed(n=n, x=x, s=s, tol=0.0, max_iter=20_000)
        bound = _cancellation_bound(n, x, s)
        assert abs(collapsed - literal.stat) < 10 * bound, (
            f"{c['label']}: |{collapsed:.12g} - {literal.stat:.12g}| exceeds 10x the "
            f"cancellation bound {bound:.3e}"
        )


def test_collapsed_and_literal_differ_only_by_the_fixed_point_residual(fx):
    """With R's early break the two forms differ -- bounded, and by the residual's size."""
    worst = 0.0
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        lit = ref.lrt_stat(n=n, x=x, s=s).stat
        col = ref.lrt_stat_collapsed(n=n, x=x, s=s)
        worst = max(worst, abs(col - lit) / max(abs(lit), 1e-300))
    assert 1e-12 < worst < 1e-4, f"unexpected collapsed/literal gap {worst:.3e}"


def test_batch_newton_matches_scalar_reference(fx, device):
    """Both solve the same equation; only the reduction order over groups differs.

    ``tauh`` is a clean quantity and must agree to ~1e-12. ``stat = 2 sum n log(...)`` has
    mixed-sign terms, so for a near-null statistic the reordering shows up relatively even
    though it is ~1e-10 absolute.
    """
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        one = ref.solve_common_cv(n=n, x=x, s=s)
        got = cvq.lrt_stat_batch(n=n, x=x, s=s, solver="newton", device=device)
        assert bool(got.converged), c["label"]
        assert relerr(float(got.tauh), one.tauh) < 1e-12, c["label"]
        assert float(got.stat) == pytest.approx(one.stat, rel=1e-9, abs=1e-8), c["label"]


def test_mle_equation_is_monotone_in_t(fx):
    """``F(t) = sum n x/u(t) - N`` is strictly increasing, which is why Newton is safe."""
    for c in fx["deterministic"][:8]:
        n, x, s = as_case(c)
        v = (n - 1) * s**2 / n
        grid = np.geomspace(1e-4, 1e3, 200)
        F = np.array(
            [float((n * x / ((-x + np.sqrt(x**2 + 4 * t * (v + x**2))) / 2.0 / t)).sum() - n.sum())
             for t in grid]
        )
        assert np.all(np.diff(F) > 0), c["label"]


def test_solver_choice_barely_moves_the_studentized_statistic(fx):
    """R's non-convergence largely cancels in the MSLRT.

    Both ``stat0`` and the bootstrap null mean are biased in the same direction, so the
    recentred statistic is insensitive to it. This is why the Newton solver is a correctness
    and speed improvement rather than a change in scientific conclusions -- worth pinning so
    the claim in the README stays true.
    """
    for c in fx["mslrt"]:
        n, x, s = as_case(c)
        a = cvq.mslr_test2(n=n, x=x, s=s, nr=4000, seed=5, solver="fixedpoint", device="cpu")
        b = cvq.mslr_test2(n=n, x=x, s=s, nr=4000, seed=5, solver="newton", device="cpu")
        assert abs(a["MSLRT"] - b["MSLRT"]) < 1e-3, (
            f"{c['label']}: {a['MSLRT']:.6f} vs {b['MSLRT']:.6f}"
        )
        # ...while the underlying statistic really does differ where R stalls
        if not a["converged"]:
            assert a["stat0"] != b["stat0"]
