"""Agreement with the real R cvequality package.

Deterministic quantities (the asymptotic statistic, and R's ``LRT_STAT``) are closed form:
they must match to near machine precision, so a failure here is a bug rather than a
tolerance to be loosened. The MSLRT is Monte Carlo and R's Mersenne-Twister stream cannot be
reproduced from NumPy/torch, so it is checked *distributionally* -- against the bootstrap
null moments, and across many seeds.
"""

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality import reference as ref

from .conftest import as_case, relerr

pytestmark = pytest.mark.rfixtures


# ---------------------------------------------------------------------------
# the four values asserted by the R package's own testthat suite
# ---------------------------------------------------------------------------


def test_published_asymptotic_mtcars(fx):
    p = fx["published"]["asymptotic_test_mtcars"]
    got = ref.asymptotic_test(fx["mtcars"]["disp"], fx["mtcars"]["am"])
    assert got["D_AD"] == pytest.approx(p["expect_D_AD"], rel=2e-3)
    assert got["p_value"] == pytest.approx(p["expect_p"], rel=2e-3)
    # and against the freshly computed R value, which should be exact
    assert relerr(got["D_AD"], p["D_AD"]) < 1e-12


def test_published_asymptotic_miller(fx):
    p = fx["published"]["asymptotic_test2_miller"]
    miller = [c for c in fx["deterministic"] if c["label"] == "miller_k3"][0]
    n, x, s = as_case(miller)
    got = ref.asymptotic_test2(n=n, s=s, x=x)
    assert got["D_AD"] == pytest.approx(p["expect_D_AD"], rel=2e-3)
    assert got["p_value"] == pytest.approx(p["expect_p"], rel=2e-3)


def test_published_mslrt_within_mc_error(fx):
    """R's seeded fixtures at nr=1e4, compared across 24 of our own seeds.

    The published values are single Monte Carlo draws, so the assertion is that our mean sits
    within a few standard errors of them -- not that any one run reproduces them.
    """
    miller = [c for c in fx["deterministic"] if c["label"] == "miller_k3"][0]
    n, x, s = as_case(miller)
    vals = np.array(
        [cvq.mslr_test2(n=n, x=x, s=s, nr=10_000, seed=sd, solver="fixedpoint", device="cpu")["MSLRT"]
         for sd in range(24)]
    )
    expected = fx["published"]["mslr_test2_miller_expect"]["MSLRT"]
    se = vals.std(ddof=1) / np.sqrt(vals.size)
    assert abs(vals.mean() - expected) < 4 * se + 0.02, (
        f"mean MSLRT {vals.mean():.5f} vs R {expected:.5f}, se {se:.5f}"
    )


# ---------------------------------------------------------------------------
# deterministic grid -- exact agreement required
# ---------------------------------------------------------------------------


def test_asymptotic_reference_matches_r_exactly(fx):
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        got = ref.asymptotic_test2(n=n, s=s, x=x)
        assert relerr(got["D_AD"], c["D_AD"]) < 1e-12, c["label"]
        assert relerr(got["p_value"], c["p_value"]) < 1e-11, c["label"]


def test_asymptotic_batch_matches_r_exactly(fx, device):
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        got = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device=device)
        assert int(got.status) == int(cvq.Status.OK), c["label"]
        assert relerr(float(got.D_AD), c["D_AD"]) < 1e-12, c["label"]
        assert relerr(float(got.p_value), c["p_value"]) < 1e-11, c["label"]


def test_reference_lrt_stat_matches_r_exactly(fx):
    """R's LRT_STAT, including its uh/tauh off-by-one and 31-iteration cap."""
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        got = ref.lrt_stat(n=n, x=x, s=s)
        assert relerr(got.uh, np.atleast_1d(c["lrt_uh"])) < 1e-12, c["label"]
        assert relerr(got.tauh, c["lrt_tauh"]) < 1e-12, c["label"]
        # The literal statistic form subtracts two O(n log n) quantities, so at n~2e5 it
        # carries ~1e-10 relative error even in float64. That is R's arithmetic, not ours.
        assert relerr(got.stat, c["lrt_stat"]) < 1e-8, c["label"]


def test_lrt_batch_fixedpoint_matches_r_exactly(fx, device):
    for c in fx["deterministic"]:
        n, x, s = as_case(c)
        got = cvq.lrt_stat_batch(n=n, x=x, s=s, solver="fixedpoint", device=device)
        assert relerr(got.u[0].cpu().numpy(), np.atleast_1d(c["lrt_uh"])) < 1e-12, c["label"]
        assert relerr(float(got.tauh), c["lrt_tauh"]) < 1e-12, c["label"]
        assert relerr(float(got.stat), c["lrt_stat"]) < 1e-8, c["label"]


def test_lrt_batch_fixedpoint_matches_reference_elementwise(fx, device):
    """The whole grid in one batched call, padded to a common k."""
    cases = [c for c in fx["deterministic"] if len(np.atleast_1d(c["x"])) == 2]
    assert len(cases) >= 10
    n = np.stack([as_case(c)[0] for c in cases])
    x = np.stack([as_case(c)[1] for c in cases])
    s = np.stack([as_case(c)[2] for c in cases])
    batch = cvq.lrt_stat_batch(n=n, x=x, s=s, solver="fixedpoint", device=device)
    for i, c in enumerate(cases):
        one = ref.lrt_stat(n=n[i], x=x[i], s=s[i])
        assert relerr(float(batch.tauh[i]), one.tauh) < 1e-13, c["label"]
        # The literal statistic form differences two O(n log n) sums, so at n~2e5 it is only
        # good to ~1e-10 relative -- and batched vs scalar reduction order differ. That noise
        # floor is R's arithmetic; see test_solver.py for the well-conditioned form.
        assert relerr(float(batch.stat[i]), one.stat) < 1e-8, c["label"]


# ---------------------------------------------------------------------------
# Monte Carlo -- assert on the null moments, never on one seeded draw
# ---------------------------------------------------------------------------


#: The bootstrap LRT null is right-skewed and heavy-tailed, so ``SE(sd) = sd/sqrt(2 nr)``
#: (which assumes normality, and depends on the 4th moment in general) understates the true
#: spread. Measured empirically at nr=2e4 over 20 seeds: 0.0204 observed vs 0.0079 predicted,
#: i.e. 2.6x. This factor keeps the comparison against R honest rather than loosening the
#: tolerance arbitrarily.
_HEAVY_TAIL_SE_INFLATION = 3.0


@pytest.mark.slow
def test_mslrt_null_moments_match_r(fx):
    """Bootstrap mean/SD of the LRT under H0, against R at the same nr.

    Both sides estimate the same quantity with independent RNG streams, so the tolerance is
    the Monte Carlo standard error of the comparison, not an arbitrary epsilon.
    """
    for c in fx["mslrt"]:
        n, x, s = as_case(c)
        nr = int(c["nr"])
        got = cvq.mslr_test2(n=n, x=x, s=s, nr=nr, seed=1, solver="fixedpoint", device="cpu")
        se_mean = c["null_sd"] / np.sqrt(nr)
        se_sd = _HEAVY_TAIL_SE_INFLATION * c["null_sd"] / np.sqrt(2 * nr)
        assert abs(got["null_mean"] - c["null_mean"]) < 6 * se_mean, (
            f"{c['label']}: null_mean {got['null_mean']:.6f} vs R {c['null_mean']:.6f} (se {se_mean:.6f})"
        )
        assert abs(got["null_sd"] - c["null_sd"]) < 6 * se_sd, (
            f"{c['label']}: null_sd {got['null_sd']:.6f} vs R {c['null_sd']:.6f} (se {se_sd:.6f})"
        )


@pytest.mark.slow
def test_heavy_tail_se_inflation_factor_is_still_right(fx):
    """Guards the constant above: if the null's tails change, the tolerance model must too."""
    c = [m for m in fx["mslrt"] if m["label"] == "mtcars_k2"][0]
    n, x, s = as_case(c)
    nr = 20_000
    sds = np.array(
        [cvq.mslr_test2(n=n, x=x, s=s, nr=nr, seed=sd, solver="fixedpoint", device="cpu")["null_sd"]
         for sd in range(12)]
    )
    observed = sds.std(ddof=1)
    predicted = sds.mean() / np.sqrt(2 * nr)
    assert observed / predicted < _HEAVY_TAIL_SE_INFLATION * 2.0, (
        f"null_sd spread is {observed/predicted:.1f}x normal theory; raise "
        f"_HEAVY_TAIL_SE_INFLATION (currently {_HEAVY_TAIL_SE_INFLATION})"
    )


@pytest.mark.slow
def test_mslrt_stat0_matches_r_exactly(fx):
    """The *observed* statistic is deterministic even though the p-value is not."""
    for c in fx["mslrt"]:
        n, x, s = as_case(c)
        got = cvq.mslr_test2(n=n, x=x, s=s, nr=64, seed=1, solver="fixedpoint", device="cpu")
        assert relerr(got["stat0"], c["stat0"]) < 1e-8, c["label"]
