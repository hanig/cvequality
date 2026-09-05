"""Batched-API behaviour: broadcasting, chunking, degenerate rows, determinism, devices."""

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality._backend import Status


@pytest.fixture
def grid():
    """A batch of well-behaved k=2 tests at perturb-seq-like group sizes."""
    rng = np.random.default_rng(7)
    T = 96
    n = np.stack([np.full(T, 209_760.0), rng.choice([30.0, 120.0, 722.0, 3000.0], T)], axis=-1)
    x = rng.uniform(0.05, 40.0, (T, 2))
    cv = rng.uniform(0.3, 3.5, (T, 2))
    return n, x, cv * x


def test_broadcasting_k_vector_equals_explicit_tk(grid):
    _n, x, s = grid
    T = x.shape[0]
    n_shared = np.array([500.0, 700.0])
    a = cvq.asymptotic_test2_batch(n=n_shared, s=s, x=x, device="cpu")
    b = cvq.asymptotic_test2_batch(n=np.tile(n_shared, (T, 1)), s=s, x=x, device="cpu")
    torch.testing.assert_close(a.D_AD, b.D_AD)
    torch.testing.assert_close(a.p_value, b.p_value)


def test_batch_equals_one_at_a_time(grid):
    n, x, s = grid
    batch = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    for i in range(x.shape[0]):
        one = cvq.asymptotic_test2_batch(n=n[i], s=s[i], x=x[i], device="cpu")
        torch.testing.assert_close(batch.D_AD[i], one.D_AD[0])


def test_chunk_size_is_bit_reproducible(grid):
    """A given chunk value must reproduce exactly, so a rerun is verifiable."""
    n, x, s = grid
    a = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=200, seed=3, device="cpu", chunk=7)
    b = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=200, seed=3, device="cpu", chunk=7)
    torch.testing.assert_close(a.MSLRT, b.MSLRT, rtol=0, atol=0)


def test_chunking_agrees_within_monte_carlo_error(grid):
    """Different chunk sizes consume the RNG differently, so only the estimand is shared.

    ``stat0`` is deterministic and must be identical; the bootstrap moments are two
    independent Monte Carlo estimates of the same quantity, so they are compared against the
    standard error of a difference of two such estimates.
    """
    n, x, s = grid
    nr = 4000
    full = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=nr, seed=3, device="cpu", chunk=10**6)
    for chunk in (1, 7, 31):
        part = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=nr, seed=3, device="cpu", chunk=chunk)
        torch.testing.assert_close(full.stat0, part.stat0, rtol=0, atol=0)
        # SE of one mean is null_sd/sqrt(nr); a difference of two independent means is
        # sqrt(2)x that. Allow 6 sigma across the whole batch.
        tol = 6 * np.sqrt(2.0) * full.null_sd.numpy() / np.sqrt(nr)
        diff = (full.null_mean - part.null_mean).abs().numpy()
        assert np.all(diff < tol), f"chunk={chunk}: max {diff.max():.4f} vs tol {tol.min():.4f}"


def test_seed_is_reproducible(grid):
    n, x, s = grid
    a = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=128, seed=11, device="cpu")
    b = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=128, seed=11, device="cpu")
    torch.testing.assert_close(a.MSLRT, b.MSLRT)
    c = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=128, seed=12, device="cpu")
    assert not torch.allclose(a.MSLRT, c.MSLRT)


def test_share_draws_agrees_with_independent_draws_on_average():
    """Common random numbers must not bias the null moments."""
    rng = np.random.default_rng(1)
    T = 64
    n = np.tile(np.array([209_760.0, 722.0]), (T, 1))
    x = rng.uniform(1.0, 5.0, (T, 2))
    s = rng.uniform(0.8, 2.5, (T, 2)) * x
    indep = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=4000, seed=2, device="cpu", share_draws=False)
    shared = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=4000, seed=2, device="cpu", share_draws=True)
    # Same estimand, independent Monte Carlo error: means agree to a few percent.
    assert abs(float(indep.null_mean.mean()) - float(shared.null_mean.mean())) < 0.1
    assert abs(float(indep.null_sd.mean()) - float(shared.null_sd.mean())) < 0.1


def test_share_draws_rejects_heterogeneous_n(grid):
    n, x, s = grid
    with pytest.raises(ValueError, match="share_draws"):
        cvq.mslr_test2_batch(n=n, x=x, s=s, nr=32, device="cpu", share_draws=True)


@pytest.mark.parametrize(
    "n,x,s,expect",
    [
        ([100.0, 100.0], [1.0, 0.0], [1.0, 1.0], Status.NONPOSITIVE_MEAN),
        ([100.0, 100.0], [1.0, -2.0], [1.0, 1.0], Status.NONPOSITIVE_MEAN),
        ([100.0, 100.0], [1.0, 1.0], [1.0, 0.0], Status.ZERO_VARIANCE),
        ([1.0, 100.0], [1.0, 1.0], [1.0, 1.0], Status.TOO_FEW_OBS),
        ([100.0, 100.0], [1.0, np.nan], [1.0, 1.0], Status.NONPOSITIVE_MEAN),
    ],
)
def test_degenerate_rows_get_status_and_nan(n, x, s, expect):
    a = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    assert int(a.status) == int(expect)
    assert np.isnan(float(a.D_AD)) and np.isnan(float(a.p_value))
    m = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=16, device="cpu")
    assert int(m.status) == int(expect)
    assert np.isnan(float(m.MSLRT))


def test_degenerate_rows_do_not_poison_neighbours(grid):
    n, x, s = grid
    x = x.copy()
    x[3, 1] = 0.0  # one bad row in the middle of a good batch
    a = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    assert int(a.status[3]) == int(Status.NONPOSITIVE_MEAN)
    good = np.setdiff1d(np.arange(x.shape[0]), [3])
    assert bool(torch.isfinite(a.D_AD[good]).all())
    clean = cvq.asymptotic_test2_batch(n=n[good], s=s[good], x=x[good], device="cpu")
    torch.testing.assert_close(a.D_AD[good], clean.D_AD)


def test_p_values_are_in_range(grid):
    n, x, s = grid
    a = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    m = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=256, seed=0, device="cpu")
    for p in (a.p_value, m.p_value):
        ok = torch.isfinite(p)
        assert bool(((p[ok] >= 0) & (p[ok] <= 1)).all())


def test_negative_mslrt_gives_p_one():
    """A statistic below the bootstrap mean is legitimate and must not produce NaN."""
    n = np.array([[209_760.0, 722.0]])
    x = np.array([[1.7, 1.72]])
    s = x * 1.2  # CVs essentially equal -> statistic near zero, often negative
    m = cvq.mslr_test2_batch(n=n, x=x, s=s, nr=4000, seed=11, device="cpu")
    assert bool(torch.isfinite(m.p_value).all())
    assert float(m.p_value) <= 1.0


def test_asymptotic_scales_to_millions_of_tests():
    rng = np.random.default_rng(0)
    T = 2_000_000
    n = np.stack([np.full(T, 209_760.0), np.full(T, 722.0)], axis=-1)
    x = rng.uniform(0.1, 20.0, (T, 2))
    s = rng.uniform(0.5, 3.0, (T, 2)) * x
    a = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    assert a.D_AD.shape == (T,)
    assert bool((a.status == int(Status.OK)).all())


def test_float32_warns():
    n = np.array([[500.0, 700.0]])
    x = np.array([[3.0, 3.2]])
    s = x * 1.4
    with pytest.warns(RuntimeWarning, match="float32"):
        cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu", dtype=torch.float32)


def test_float32_newton_tolerances_track_machine_epsilon():
    """A representative float32 batch must not be judged by float64's noise floor."""
    rng = np.random.default_rng(0)
    T = 200
    n = np.tile(np.array([2000.0, 700.0]), (T, 1))
    x = rng.uniform(0.5, 5.0, (T, 2))
    s = x * rng.uniform(0.5, 2.5, (T, 2))

    with pytest.warns(RuntimeWarning, match="three significant digits"):
        single = cvq.mslr_test2_batch(
            n=n, x=x, s=s, nr=16, seed=0, device="cpu", dtype=torch.float32
        )
    double = cvq.mslr_test2_batch(
        n=n, x=x, s=s, nr=16, seed=0, device="cpu", dtype=torch.float64
    )

    assert bool((single.status == int(Status.OK)).all())
    assert bool(single.converged.all())
    torch.testing.assert_close(single.tauh.double(), double.tauh, rtol=1e-4, atol=0)


def test_rejects_k_less_than_two():
    with pytest.raises(ValueError, match="k=2"):
        cvq.asymptotic_test2_batch(n=[10.0], s=[1.0], x=[1.0], device="cpu")


@pytest.mark.gpu
def test_cuda_matches_cpu_in_float64(grid):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    n, x, s = grid
    a_cpu = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cpu")
    a_gpu = cvq.asymptotic_test2_batch(n=n, s=s, x=x, device="cuda")
    torch.testing.assert_close(a_cpu.D_AD, a_gpu.D_AD.cpu(), rtol=1e-13, atol=0)
    torch.testing.assert_close(a_cpu.p_value, a_gpu.p_value.cpu(), rtol=1e-13, atol=0)

    l_cpu = cvq.lrt_stat_batch(n=n, x=x, s=s, device="cpu")
    l_gpu = cvq.lrt_stat_batch(n=n, x=x, s=s, device="cuda")
    # tauh is a clean quantity and must agree essentially exactly.
    torch.testing.assert_close(l_cpu.tauh, l_gpu.tauh.cpu(), rtol=1e-12, atol=0)
    # stat = 2*sum(n log(...)) has mixed-sign terms, and CPU and CUDA use different reduction
    # trees. For a near-null statistic (~1e-3) that shows up as ~1e-8 relative even though it
    # is only ~1e-10 absolute -- the same summation noise floor as batch-vs-scalar.
    torch.testing.assert_close(l_cpu.stat, l_gpu.stat.cpu(), rtol=1e-9, atol=1e-9)


@pytest.mark.gpu
def test_cuda_chi2_sampling_at_huge_df():
    """torch._standard_gamma on CUDA at df = the non-targeting group size."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from cvequality._backend import chi2_rvs, make_generator

    dev = torch.device("cuda")
    df = torch.tensor([[4.0, 209_759.0]], dtype=torch.float64, device=dev)
    g = make_generator(0, dev)
    draws = chi2_rvs(df, (1, 200_000, 2), generator=g)
    mean_ratio = (draws.mean(dim=1) / df).squeeze()
    var_ratio = (draws.var(dim=1) / (2 * df)).squeeze()
    assert torch.allclose(mean_ratio, torch.ones_like(mean_ratio), rtol=0.02)
    assert torch.allclose(var_ratio, torch.ones_like(var_ratio), rtol=0.05)
