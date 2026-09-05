"""The SD-ratio sibling test: the identity it rests on, its variance model, and its purpose."""

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality._backend import Status


def test_reduces_to_the_two_sample_log_sd_z():
    """For k=2 the Wald statistic is the square of the usual z on log(sd)."""
    n = np.array([[500.0, 700.0]])
    sd = np.array([[1.3, 1.8]])
    g4 = np.array([[3.0, 3.0]])
    got = cvq.sd_ratio_test(n=n, sd=sd, kurtosis=g4, device="cpu")
    var = (3.0 - 1) / (4 * 500) + (3.0 - 1) / (4 * 700)
    z = (np.log(1.8) - np.log(1.3)) / np.sqrt(var)
    assert got["stat"] == pytest.approx(z**2, rel=1e-12)
    assert got["log2_sd_ratio"] == pytest.approx(np.log2(1.8 / 1.3), rel=1e-12)


def test_normal_theory_variance_matches_the_classical_form():
    """(kurt-1)/(4n) with kurt=3 is 1/(2n), the textbook Var(log s)."""
    n = np.array([[1000.0, 1000.0]])
    sd = np.array([[2.0, 2.0]])
    res = cvq.sd_ratio_test_batch(n=n, sd=sd, device="cpu")  # default kurtosis = 3
    assert float(res.stat) == pytest.approx(0.0, abs=1e-12)
    assert float(res.p_value) == pytest.approx(1.0, rel=1e-12)


def test_higher_kurtosis_widens_the_null():
    """Heavier tails must make the same SD difference less significant, not more."""
    n = np.array([[400.0, 400.0]])
    sd = np.array([[1.0, 1.25]])
    p_norm = cvq.sd_ratio_test(n=n, sd=sd, kurtosis=np.array([[3.0, 3.0]]), device="cpu")["p_value"]
    p_heavy = cvq.sd_ratio_test(n=n, sd=sd, kurtosis=np.array([[9.0, 9.0]]), device="cpu")["p_value"]
    assert p_heavy > p_norm


def test_pooled_kurtosis_is_weighted_by_group_size():
    n = np.array([[1000.0, 100.0]])
    sd = np.array([[1.0, 1.2]])
    kurtosis = np.array([[3.0, 9.0]])
    pooled = np.full_like(kurtosis, (1000 * 3.0 + 100 * 9.0) / 1100)
    got = cvq.sd_ratio_test_batch(n=n, sd=sd, kurtosis=kurtosis, device="cpu")
    expected = cvq.sd_ratio_test_batch(
        n=n, sd=sd, kurtosis=pooled, kurtosis_shrinkage="none", device="cpu"
    )
    torch.testing.assert_close(got.stat, expected.stat)


def test_equal_sds_give_a_uniformly_distributed_statistic():
    """Simulated normal data: the k=2 test must be calibrated when H0 holds."""
    rng = np.random.default_rng(0)
    T, n = 4000, 300
    a = rng.normal(0, 1, (T, n))
    b = rng.normal(0, 1, (T, n))
    sd = np.stack([a.std(axis=1, ddof=1), b.std(axis=1, ddof=1)], axis=-1)
    ns = np.full_like(sd, float(n))
    p = cvq.sd_ratio_test_batch(n=ns, sd=sd, device="cpu").p_value.numpy()
    assert 0.7 < (p < 0.05).mean() / 0.05 < 1.3, (p < 0.05).mean() / 0.05
    assert 0.4 < p.mean() < 0.6


def test_kurtosis_shrinkage_calibrates_a_small_group_against_a_large_reference():
    """Pooling removes the small group's noisy fourth moment from its null variance."""
    rng = np.random.default_rng(123)
    T, n_ref, n_grp = 20_000, 200_000, 100
    sd_ref = np.sqrt(rng.chisquare(n_ref - 1, T) / (n_ref - 1))
    group = rng.normal(size=(T, n_grp))
    sd_grp = group.std(axis=1, ddof=1)
    group -= group.mean(axis=1, keepdims=True)
    m2 = (group**2).mean(axis=1)
    kurtosis_grp = (group**4).mean(axis=1) / m2**2

    n = np.column_stack([np.full(T, n_ref), np.full(T, n_grp)])
    sd = np.column_stack([sd_ref, sd_grp])
    kurtosis = np.column_stack([np.full(T, 3.0), kurtosis_grp])
    default = cvq.sd_ratio_test_batch(
        n=n, sd=sd, kurtosis=kurtosis, device="cpu"
    ).p_value.numpy()
    raw = cvq.sd_ratio_test_batch(
        n=n, sd=sd, kurtosis=kurtosis, kurtosis_shrinkage="none", device="cpu"
    ).p_value.numpy()

    default_inflation = np.array([(default < a).mean() / a for a in (0.05, 0.01)])
    raw_inflation = np.array([(raw < a).mean() / a for a in (0.05, 0.01)])
    assert (default_inflation <= [1.15, 1.30]).all(), default_inflation
    assert raw_inflation[0] == pytest.approx(1.30, abs=0.08)
    assert raw_inflation[1] == pytest.approx(1.67, abs=0.15)


def test_degenerate_rows_flagged():
    r = cvq.sd_ratio_test_batch(n=[[100.0, 100.0]], sd=[[1.0, 0.0]], device="cpu")
    assert int(r.status) == int(Status.ZERO_VARIANCE)
    assert np.isnan(float(r.stat))
    r = cvq.sd_ratio_test_batch(n=[[1.0, 100.0]], sd=[[1.0, 1.0]], device="cpu")
    assert int(r.status) == int(Status.TOO_FEW_OBS)


def test_pi_score_uses_the_primary_tests_effect(toy_adata):
    common = dict(
        group_key="target_gene_name", reference="ntc", min_cells=30,
        device="cpu", progress=False,
    )
    sd_ratio = cvq.vs_reference(toy_adata, test="sd_ratio", **common)
    asymptotic = cvq.vs_reference(toy_adata, test="asymptotic", **common)

    sd_expected = (
        -np.log10(np.clip(sd_ratio["pval_sd_ratio"], 1e-300, 1.0))
        * sd_ratio["log2_sd_ratio"].abs()
    )
    cv_expected = (
        -np.log10(np.clip(asymptotic["pval_asymptotic"], 1e-300, 1.0))
        * asymptotic["log2_cv_ratio"].abs()
    )
    np.testing.assert_allclose(sd_ratio["pi_score"], sd_expected)
    np.testing.assert_allclose(asymptotic["pi_score"], cv_expected)


# ---------------------------------------------------------------------------
# integration: the identity that motivates the test, and what it buys
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def screen_stats():
    """Screen with DE-only and dispersion-only perturbations, plus both transforms."""
    anndata = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    from cvequality.sufficient import group_sufficient_stats

    rng = np.random.default_rng(3)
    n_per, g = 400, 500
    names = ["non-targeting"] + [f"null{i}" for i in range(6)] + ["DE1", "DE2", "DISP1", "DISP2"]
    kind = {n: ("de" if n.startswith("DE") else "disp" if n.startswith("DISP") else "null")
            for n in names}
    base = rng.gamma(1.5, 3.0, g)
    blocks, groups = [], []
    for n in names:
        fold = np.ones(g)
        if kind[n] == "de":
            idx = rng.choice(g, int(0.3 * g), replace=False)
            fold[idx] = np.exp(rng.normal(0, 0.7, idx.size))
            fold /= (fold * base).sum() / base.sum()
        lam = np.outer(rng.gamma(20, 1 / 20, n_per), base * fold)
        if kind[n] == "disp":
            lam = lam * rng.gamma(2.0, 1 / 2.0, (n_per, g))
        blocks.append(rng.poisson(lam).astype(np.float32))
        groups += [n] * n_per
    ad = anndata.AnnData(X=sp.csr_matrix(np.vstack(blocks)),
                         obs={"target_gene_name": np.array(groups, dtype=object)})
    ad.var_names = [f"g{i}" for i in range(g)]
    kw = dict(group_key="target_gene_name", device="cpu", progress=False)
    return (group_sufficient_stats(ad, transform="log1p", **kw),
            group_sufficient_stats(ad, transform="tp10k", **kw), kind)


@pytest.fixture(scope="module")
def called(screen_stats):
    st_log, st_tp, _ = screen_stats
    df = cvq.vs_reference(stats=st_log, mean_stats=st_tp, reference="non-targeting",
                          transform="log1p", test=("asymptotic", "sd_ratio"),
                          min_cells=30, min_frac_expressed=0.5, device="cpu", progress=False)
    return df[df["status"] == int(Status.OK)]


def test_log_sd_ratio_is_the_de_corrected_cv_ratio(called):
    """The identity the whole test rests on: log2 sd ratio == log2 cv ratio + log2 mean ratio.

    Note both terms must be on the *same* transform -- ``mean_ref``/``mean_grp`` are the log1p
    means that ``cv_ref``/``cv_grp`` were formed from, not the tp10k means in
    ``log2_mean_ratio``.
    """
    same_scale = np.log2(called["mean_grp"] / called["mean_ref"])
    np.testing.assert_allclose(
        called["log2_sd_ratio"], called["log2_cv_ratio"] + same_scale, atol=1e-12
    )


def test_sd_ratio_is_less_confounded_with_expression_than_the_cv(called):
    cv_r = np.corrcoef(called["log2_cv_ratio"], called["log2_mean_ratio"])[0, 1]
    sd_r = np.corrcoef(called["log2_sd_ratio"], called["log2_mean_ratio"])[0, 1]
    assert abs(sd_r) < abs(cv_r), f"cv {cv_r:+.3f} vs sd {sd_r:+.3f}"


def test_sd_ratio_separates_dispersion_from_de_better_than_the_cv(called, screen_stats):
    """The reason this test exists. Guards the direction of the README's comparison."""
    _, _, kind = screen_stats
    k = np.array([kind[p] for p in called["perturbation"]])
    def sep(col):
        de = (called[k == "de"][col] < 0.1).mean()
        disp = (called[k == "disp"][col] < 0.1).mean()
        return disp / max(de, 1e-9)
    assert sep("fdr_sd_ratio") > sep("fdr_asymptotic"), (
        f"sd_ratio {sep('fdr_sd_ratio'):.2f}x vs cv {sep('fdr_asymptotic'):.2f}x"
    )


def test_kurtosis_columns_are_reported(called):
    for col in ("kurtosis_ref", "kurtosis_grp", "kurtosis_shrinkage", "log2_sd_ratio",
                "stat_sd_ratio", "pval_sd_ratio", "fdr_sd_ratio"):
        assert col in called.columns
    assert (called["kurtosis_ref"] >= 1.0).all()
    assert (called["kurtosis_shrinkage"] == "pooled").all()


def test_requires_moments(screen_stats):
    """Statistics computed without higher moments must fail loudly, not silently assume 3.0."""
    from cvequality.sufficient import GroupStats

    st_log, _, _ = screen_stats
    stripped = GroupStats(
        group_names=st_log.group_names, var_names=st_log.var_names, n=st_log.n,
        sum=st_log.sum, sumsq=st_log.sumsq, n_expressed=st_log.n_expressed,
        transform=st_log.transform, target_sum=st_log.target_sum,
    )
    with pytest.raises(ValueError, match="moments=True"):
        cvq.vs_reference(stats=stripped, reference="non-targeting", transform="log1p",
                         test="sd_ratio", min_cells=30, device="cpu", progress=False)


def test_moments_match_a_direct_computation():
    """Power-sum moments are exact for sparse input; check against dense numpy."""
    anndata = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    from cvequality.sufficient import group_sufficient_stats

    rng = np.random.default_rng(0)
    n, g = 2000, 80
    X = rng.poisson(np.outer(rng.gamma(20, 1 / 20, n), rng.gamma(1.5, 3.0, g))).astype(np.float32)
    ad = anndata.AnnData(X=sp.csr_matrix(X), obs={"target_gene_name": np.array(["a"] * n, dtype=object)})
    ad.var_names = [f"g{i}" for i in range(g)]
    st = group_sufficient_stats(ad, group_key="target_gene_name", transform="log1p",
                                device="cpu", progress=False)
    A = np.log1p(1e4 * X / X.sum(axis=1, keepdims=True))
    c = A - A.mean(0)
    m2 = (c**2).mean(0)
    np.testing.assert_allclose(st.kurtosis[0].numpy(), (c**4).mean(0) / m2**2, rtol=1e-3)
    np.testing.assert_allclose(st.skewness[0].numpy(), (c**3).mean(0) / m2**1.5, atol=1e-3)
