"""Regression tests for cancellation in the common-CV Newton solver."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality import reference as ref
from cvequality._backend import Status


N = np.array([100.0, 100.0])
X = np.array([1.0, 1.0])
S = np.array([0.0001, 0.00015])


def _decimal_common_cv(n, x, s):
    """Independent 80-digit solution for the exact binary64 inputs.

    Decimal arithmetic is intentionally kept in this test rather than shared with either
    production solver.  Its Newton equation is evaluated at 80 digits and the statistic is
    formed only after a 70-digit relative correction has been reached.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        to_decimal = lambda value: Decimal.from_float(float(value))  # noqa: E731
        nd = [to_decimal(value) for value in n]
        xd = [to_decimal(value) for value in x]
        sd = [to_decimal(value) for value in s]
        total_n = sum(nd)
        vsq = [(nj - 1) * sj * sj / nj for nj, sj in zip(nd, sd)]
        a = [vj + xj * xj for vj, xj in zip(vsq, xd)]
        t = sum(nj * vj / (xj * xj) for nj, vj, xj in zip(nd, vsq, xd)) / total_n

        for _ in range(100):
            root = [(xj * xj + 4 * t * aj).sqrt() for xj, aj in zip(xd, a)]
            centered = [
                2 * xj * t / (rj + xj) - vj / aj
                for xj, rj, vj, aj in zip(xd, root, vsq, a)
            ]
            derivative = [xj / rj for xj, rj in zip(xd, root)]
            delta = (
                sum(nj * value for nj, value in zip(nd, centered))
                / sum(nj * value for nj, value in zip(nd, derivative))
            )
            t -= delta
            if abs(delta) <= abs(t) * Decimal("1e-70"):
                break
        else:  # pragma: no cover - a broken test oracle should fail conspicuously
            raise AssertionError("Decimal common-CV oracle did not converge")

        root = [(xj * xj + 4 * t * aj).sqrt() for xj, aj in zip(xd, a)]
        u = [2 * aj / (xj + rj) for aj, xj, rj in zip(a, xd, root)]
        stat = 2 * sum(
            nj * (t.sqrt() * uj / vj.sqrt()).ln()
            for nj, uj, vj in zip(nd, u, vsq)
        )
        return t, stat


def test_low_cv_matches_independent_high_precision_solution():
    expected_t, expected_stat = _decimal_common_cv(N, X, S)
    fit = cvq.lrt_stat_batch(n=N, x=X, s=S, solver="newton", device="cpu")

    # The published reproducer used decimal input literals.  Exact binary64 inputs move the
    # final digits only, which the independent oracle above accounts for.
    literal_reference = Decimal(
        "16.00854105874576145093069016392322791404600412163472651408481079829042"
    )
    assert abs(expected_stat - literal_reference) < Decimal("2e-14")
    assert float(fit.stat) == pytest.approx(float(expected_stat), rel=0, abs=3e-13)
    assert float(fit.tauh) ** 2 == pytest.approx(float(expected_t), rel=3e-16, abs=0)
    assert bool(fit.converged)
    assert float(fit.residual) < 1e-20


def test_adjacent_float_inputs_match_their_high_precision_solutions():
    variants = []
    for index in range(S.size):
        for direction in (0.0, np.inf):
            adjacent = S.copy()
            adjacent[index] = np.nextafter(adjacent[index], direction)
            variants.append(adjacent)
    variants = np.asarray(variants)
    fit = cvq.lrt_stat_batch(
        n=np.broadcast_to(N, variants.shape).copy(),
        x=np.broadcast_to(X, variants.shape).copy(),
        s=variants,
        solver="newton",
        device="cpu",
    )
    expected = np.array([float(_decimal_common_cv(N, X, row)[1]) for row in variants])

    np.testing.assert_allclose(fit.stat.numpy(), expected, rtol=0, atol=3e-13)
    assert bool(fit.converged.all())
    # One-ULP input changes have one-ULP-scale effects, rather than changing the scientific
    # conclusion through cancellation in the quadratic root.
    assert np.ptp(fit.stat.numpy()) < 1e-10


def test_low_cv_is_rejected_when_no_newton_step_is_allowed():
    fit = cvq.solve_common_cv_batch(
        n=torch.as_tensor(N).unsqueeze(0),
        x=torch.as_tensor(X).unsqueeze(0),
        s=torch.as_tensor(S).unsqueeze(0),
        max_iter=0,
    )

    assert not bool(fit.converged)


def test_low_cv_bootstrap_uses_the_accurate_observed_statistic():
    result = cvq.mslr_test2(n=N, x=X, s=S, nr=10_000, seed=0, device="cpu")

    assert result["status"] == Status.OK
    assert result["converged"]
    assert result["n_valid"] == 10_000
    assert result["stat0"] == pytest.approx(16.00854105874575, rel=0, abs=3e-13)
    assert result["p_value"] == pytest.approx(7.210062954174236e-05, rel=1e-10)


def test_low_cv_fixedpoint_path_remains_the_literal_r_formula():
    expected = ref.lrt_stat(n=N, x=X, s=S)
    fit = cvq.lrt_stat_batch(n=N, x=X, s=S, solver="fixedpoint", device="cpu")

    np.testing.assert_allclose(fit.u.numpy()[0], expected.uh, rtol=0, atol=2e-16)
    assert float(fit.tauh) == expected.tauh
    assert float(fit.stat) == pytest.approx(expected.stat, rel=0, abs=1e-12)
    assert bool(fit.converged) == expected.converged
