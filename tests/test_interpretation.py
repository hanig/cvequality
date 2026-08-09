"""Pins the mean-CV coupling documented in the README.

These are not tests of correctness -- the statistics are right either way. They pin the
*interpretation*: on count data a pure mean shift genuinely changes the CV, so a significant
CV difference is not evidence of a dispersion change. If a future change to normalization or
defaults alters this, the README's central caveat needs rewriting, and these fail first.
"""

import numpy as np
import pytest

import cvequality as cvq


@pytest.fixture(scope="module")
def screen():
    """A screen with known ground truth, built to reproduce the real mean-CV coupling.

    Critically, ``DE_UP``/``DE_DN`` apply **gene-specific** fold changes to a subset of genes
    and renormalize to constant total expression. Multiplying *every* gene by a constant
    instead would be a sequencing-depth change, which normalization removes -- it would still
    move the CV (through counting noise) while leaving ``log2_mean_ratio`` at zero, and so
    would not exercise the mean-CV coupling at all.
    """
    anndata = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(3)
    n_per, g = 400, 300
    base = rng.gamma(1.5, 3.0, g)
    names = ["non-targeting", "DE_UP", "DE_DN", "DISP_UP"]
    blocks, groups = [], []
    for name in names:
        fold = np.ones(g)
        if name.startswith("DE_"):
            idx = rng.choice(g, int(0.3 * g), replace=False)
            fold[idx] = 1.8 if name == "DE_UP" else 1 / 1.8
            fold /= (fold * base).sum() / base.sum()      # hold total expression fixed
        depth = rng.gamma(20, 1 / 20, n_per)
        lam = np.outer(depth, base * fold)
        if name == "DISP_UP":
            lam = lam * rng.gamma(2.0, 1 / 2.0, (n_per, g))  # dispersion up, mean unchanged
        blocks.append(rng.poisson(lam).astype(np.float32))
        groups += [name] * n_per
    ad = anndata.AnnData(
        X=sp.csr_matrix(np.vstack(blocks)),
        obs={"target_gene_name": np.array(groups, dtype=object)},
    )
    ad.var_names = [f"g{i}" for i in range(g)]
    return ad


@pytest.fixture(scope="module")
def stats_pair(screen):
    """log1p statistics for testing, tp10k statistics for the mean effect."""
    from cvequality.sufficient import group_sufficient_stats

    kw = dict(group_key="target_gene_name", device="cpu", progress=False)
    return (group_sufficient_stats(screen, transform="log1p", **kw),
            group_sufficient_stats(screen, transform="tp10k", **kw))


@pytest.fixture(scope="module")
def called(stats_pair):
    st_log, st_tp = stats_pair
    df = cvq.vs_reference(
        stats=st_log, mean_stats=st_tp, reference="non-targeting", transform="log1p",
        test="asymptotic", min_cells=30, device="cpu", progress=False,
    )
    return df[df["status"] == int(cvq.Status.OK)]


def test_pure_mean_shift_is_detected_as_a_cv_change(called):
    """The documented hazard: differential expression alone still gets called as a CV change."""
    t1 = called[(called["perturbation"] == "DE_UP") & (called["log2_mean_ratio"].abs() > 0.3)]
    assert (t1["fdr_asymptotic"] < 0.1).mean() > 0.5, (
        "a pure mean shift is no longer detected as a CV change -- the README's central "
        "caveat may be stale"
    )


def test_mean_increase_lowers_the_cv_and_dispersion_increase_raises_it(called):
    """Sign is informative even though the p-value is not: CV^2 ~ 1/mu + phi."""
    # Genes whose own mean moved: up -> CV down, down -> CV up.
    up = called[(called["perturbation"] == "DE_UP") & (called["log2_mean_ratio"] > 0.3)]
    dn = called[(called["perturbation"] == "DE_DN") & (called["log2_mean_ratio"] < -0.3)]
    assert up["log2_cv_ratio"].median() < 0, up["log2_cv_ratio"].median()
    assert dn["log2_cv_ratio"].median() > 0, dn["log2_cv_ratio"].median()
    # a genuine dispersion increase raises the CV at unchanged mean
    disp = called[called["perturbation"] == "DISP_UP"]
    assert disp["log2_cv_ratio"].median() > 0.1, disp["log2_cv_ratio"].median()


def test_dispersion_only_perturbation_leaves_the_mean_alone(called):
    """The mean effect must be clean, which is why it comes from tp10k, not log1p.

    Under log1p the "mean" is the mean of logged values; Jensen's inequality makes it fall
    when dispersion rises at constant expression (measured -0.14 here), so conditioning on a
    log1p mean ratio would throw away exactly the hits we want.
    """
    assert (called["mean_ratio_transform"] == "tp10k").all()
    t4 = called[called["perturbation"] == "DISP_UP"]
    lfc = t4["log2_mean_ratio"].median()
    assert abs(lfc) < 0.05, f"DISP_UP median log2 mean ratio {lfc:.4f} should be ~0"


def test_the_output_carries_what_is_needed_to_disentangle_them(called):
    """mean_ref/mean_grp must be present, since the p-value alone cannot be interpreted."""
    for col in ("mean_ref", "mean_grp", "cv_ref", "cv_grp", "log2_cv_ratio"):
        assert col in called.columns
    assert called[["mean_ref", "mean_grp"]].gt(0).all().all()


def test_tp10k_cv_is_invariant_to_target_sum(screen):
    """CV is scale-free on a ratio scale, so the normalization constant cancels exactly."""
    a = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                         transform="tp10k", target_sum=1e4, test="asymptotic",
                         min_cells=30, device="cpu", progress=False)
    b = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                         transform="tp10k", target_sum=1e6, test="asymptotic",
                         min_cells=30, device="cpu", progress=False)
    np.testing.assert_allclose(a["cv_ref"], b["cv_ref"], rtol=1e-10)
    np.testing.assert_allclose(a["log2_cv_ratio"], b["log2_cv_ratio"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a["pval_asymptotic"], b["pval_asymptotic"], rtol=1e-9)


def test_log1p_cv_depends_on_target_sum(screen):
    """The counterpart hazard: log has no non-arbitrary zero, so its CV is not scale-free.

    ``log1p(k*x) ~ log(x) + log(k)`` for x >> 1 is a location shift: ``sd`` is unmoved but
    ``mean`` is, so ``sd/mean`` changes. Documented in TRANSFORMS; pinned here so the claim
    cannot silently become false.
    """
    a = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                         transform="log1p", target_sum=1e4, test="asymptotic",
                         min_cells=30, device="cpu", progress=False)
    b = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                         transform="log1p", target_sum=1e6, test="asymptotic",
                         min_cells=30, device="cpu", progress=False)
    assert a["cv_ref"].median() != pytest.approx(b["cv_ref"].median(), rel=0.05)
    # larger target_sum -> larger mean of the logged values -> smaller CV
    assert b["cv_ref"].median() < a["cv_ref"].median()


def test_conditioning_on_the_mean_beats_regressing_it_out(stats_pair):
    """Pins the measured recipe: filter on the mean effect, do not residualize it.

    Regressing ``log2_cv_ratio`` on ``log2_mean_ratio`` and testing the residual was tried
    and measured **worse** than doing nothing (disp/DE separation 1.37-1.53x vs 1.64x raw),
    because there is no null reference at a non-zero mean change to calibrate against.
    Filtering to genes whose mean did not move improves it (2.04x). This test guards the
    direction of that comparison on a screen with known ground truth.
    """
    st_log, st_tp = stats_pair
    df = cvq.vs_reference(stats=st_log, mean_stats=st_tp, reference="non-targeting",
                          transform="log1p", test="asymptotic", min_cells=30,
                          device="cpu", progress=False)
    df = df[df["status"] == int(cvq.Status.OK)]
    assert "log2_mean_ratio" in df.columns
    kept = df[df["log2_mean_ratio"].abs() < 0.15]
    # DISP_UP changes dispersion at constant mean; DE_UP changes the mean.
    def rate(d, pert):
        s = d[d["perturbation"] == pert]
        return (s["fdr_asymptotic"] < 0.1).mean() if len(s) else np.nan

    raw_sep = rate(df, "DISP_UP") / max(rate(df, "DE_UP"), 1e-9)
    filt_sep = rate(kept, "DISP_UP") / max(rate(kept, "DE_UP"), 1e-9)
    assert filt_sep >= raw_sep, (
        f"conditioning on the mean should not hurt separation: raw {raw_sep:.2f} -> "
        f"filtered {filt_sep:.2f}"
    )


def test_output_records_its_own_provenance(screen):
    """A parquet must be interpretable on its own: log1p CVs mean nothing without both."""
    df = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                          transform="log1p", target_sum=1e4, test="asymptotic",
                          min_cells=30, device="cpu", progress=False)
    assert (df["cv_transform"] == "log1p").all()
    assert (df["target_sum"] == 1e4).all()
    omni = cvq.omnibus(screen, group_key="target_gene_name", test="asymptotic",
                       min_cells=30, device="cpu", progress=False)
    assert (omni["cv_transform"] == "log1p").all()
    assert (omni["target_sum"] == 1e4).all()


def test_tp10k_warns_that_its_p_values_are_not_usable(screen):
    with pytest.warns(RuntimeWarning, match="p-values are unreliable"):
        cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                         transform="tp10k", test="asymptotic", min_cells=30,
                         device="cpu", progress=False)


def test_default_transform_is_the_calibrated_one(screen):
    """Guards the decision: log1p is the default because it is the calibrated option."""
    import inspect

    for fn in (cvq.vs_reference, cvq.omnibus, cvq.null_ntc_split):
        assert inspect.signature(fn).parameters["transform"].default == "log1p"
    df = cvq.vs_reference(screen, group_key="target_gene_name", reference="non-targeting",
                          test="asymptotic", min_cells=30, device="cpu", progress=False)
    assert (df["cv_transform"] == "log1p").all()


def test_reusing_stats_with_a_different_target_sum_is_refused(screen):
    """Silently mixing two normalization constants would be a subtle, invisible error."""
    from cvequality.sufficient import group_sufficient_stats

    st = group_sufficient_stats(screen, group_key="target_gene_name", transform="log1p",
                                target_sum=1e4, device="cpu", progress=False)
    assert st.target_sum == 1e4
    with pytest.raises(ValueError, match="target_sum"):
        cvq.vs_reference(stats=st, reference="non-targeting", transform="log1p",
                         target_sum=1e6, device="cpu", progress=False)


@pytest.mark.slow
def test_high_detection_is_power_not_inflation(screen, stats_pair):
    """The counterpart claim: detection of DE perturbations far exceeds the true-null rate.

    Deliberately *not* asserting near-nominal calibration here. Transform calibration is
    data-dependent -- on near-normal synthetic counts log1p runs inflated while tp10k is
    conservative, the reverse of the real screen (see the README). The claim this pins is the
    one that holds either way: a reference-vs-itself split detects far less than a genuine
    perturbation, so the high rates elsewhere in this module are power.
    """
    st_log, st_tp = stats_pair
    null = cvq.null_ntc_split(
        screen, group_key="target_gene_name", reference="non-targeting", n_splits=3,
        match_size=150, transform="log1p", test="asymptotic", device="cpu", progress=False,
    )
    null_rate = (null["fdr_asymptotic"] < 0.1).mean()

    real = cvq.vs_reference(stats=st_log, mean_stats=st_tp, reference="non-targeting",
                            transform="log1p", test="asymptotic", min_cells=30,
                            device="cpu", progress=False)
    real = real[real["status"] == int(cvq.Status.OK)]
    de_rate = (real[real["perturbation"] == "DE_UP"]["fdr_asymptotic"] < 0.1).mean()
    assert de_rate > 4 * max(null_rate, 1e-9), f"null {null_rate:.3f} vs DE {de_rate:.3f}"
