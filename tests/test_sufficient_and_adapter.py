"""Sufficient statistics against a pandas/numpy reference, and adapter end-to-end behaviour."""

import numpy as np
import pytest
import torch

import cvequality as cvq
from cvequality._backend import Status
from cvequality.sufficient import TRANSFORMS, group_sufficient_stats


def _dense_reference(ad, transform):
    """Independent per-group mean/sd, computed densely with numpy -- no shared code."""
    X = np.asarray(ad.X.todense(), dtype=np.float64)
    if transform != "counts":
        tot = X.sum(axis=1, keepdims=True)
        X = 1e4 * X / np.where(tot == 0, 1.0, tot)
        if transform == "log1p":
            X = np.log1p(X)
    groups = np.asarray(ad.obs["target_gene_name"].values)
    names = np.unique(groups)
    out = {}
    for g in names:
        sub = X[groups == g]
        out[g] = (sub.shape[0], sub.mean(axis=0), sub.std(axis=0, ddof=1), (sub != 0).sum(axis=0))
    return names, out


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_group_stats_match_dense_reference(toy_adata, transform):
    st = group_sufficient_stats(
        toy_adata, group_key="target_gene_name", transform=transform, device="cpu", progress=False
    )
    names, want = _dense_reference(toy_adata, transform)
    np.testing.assert_array_equal(st.group_names, names)
    for g in names:
        i = st.group_index(g)
        n, mean, sd, nexp = want[g]
        assert float(st.n[i]) == n
        np.testing.assert_allclose(st.mean[i].numpy(), mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(st.sd[i].numpy(), sd, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(st.n_expressed[i].numpy(), nexp, rtol=0, atol=0)


def test_group_stats_blocking_is_irrelevant(toy_adata):
    a = group_sufficient_stats(toy_adata, group_key="target_gene_name", target_nnz=10**9,
                               device="cpu", progress=False)
    b = group_sufficient_stats(toy_adata, group_key="target_gene_name", target_nnz=500,
                               device="cpu", progress=False)
    torch.testing.assert_close(a.sum, b.sum)
    torch.testing.assert_close(a.sumsq, b.sumsq)
    torch.testing.assert_close(a.n, b.n)


def test_group_stats_from_h5ad_file_matches_anndata(toy_adata, tmp_path):
    p = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(p)
    a = group_sufficient_stats(toy_adata, group_key="target_gene_name", device="cpu", progress=False)
    b = group_sufficient_stats(str(p), group_key="target_gene_name", device="cpu", progress=False)
    np.testing.assert_array_equal(a.group_names, b.group_names)
    np.testing.assert_array_equal(a.var_names, b.var_names)
    torch.testing.assert_close(a.sum, b.sum)
    torch.testing.assert_close(a.n_expressed, b.n_expressed)


def test_cell_mask_equals_prefiltering(toy_adata):
    groups = np.asarray(toy_adata.obs["target_gene_name"].values)
    mask = np.isin(groups, ["ntc", "A"])
    masked = group_sufficient_stats(toy_adata, group_key="target_gene_name", cell_mask=mask,
                                    device="cpu", progress=False)
    subset = group_sufficient_stats(toy_adata[mask], group_key="target_gene_name",
                                    device="cpu", progress=False)
    for g in ("ntc", "A"):
        i, j = masked.group_index(g), subset.group_index(g)
        assert float(masked.n[i]) == float(subset.n[j])
        torch.testing.assert_close(masked.sum[i], subset.sum[j])
    # groups excluded by the mask survive as empty rows rather than vanishing
    assert float(masked.n[masked.group_index("B")]) == 0.0


def test_sparse_selection_gather_matches_full_stream(toy_adata, tmp_path, monkeypatch):
    """The h5ad fast path reads only the wanted rows; it must agree with reading everything.

    Exercises the branch that makes null_ntc_split cheap (~1.4k of 1.7M cells on the real
    screen), by comparing it against the same computation with the fast path disabled.
    """
    from cvequality import sufficient

    p = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(p)
    groups = np.asarray(toy_adata.obs["target_gene_name"].values)

    # a deliberately sparse, scattered selection: 12 cells out of 400
    rng = np.random.default_rng(0)
    mask = np.zeros(len(groups), dtype=bool)
    mask[rng.choice(np.flatnonzero(groups == "ntc"), 12, replace=False)] = True
    assert mask.mean() < sufficient.GATHER_FRACTION

    kw = dict(group_key="target_gene_name", cell_mask=mask, device="cpu", progress=False)
    gathered = sufficient.group_sufficient_stats(str(p), **kw)
    monkeypatch.setattr(sufficient, "GATHER_FRACTION", 0.0)  # force the full stream
    streamed = sufficient.group_sufficient_stats(str(p), **kw)

    torch.testing.assert_close(gathered.sum, streamed.sum)
    torch.testing.assert_close(gathered.sumsq, streamed.sumsq)
    torch.testing.assert_close(gathered.n, streamed.n)
    torch.testing.assert_close(gathered.n_expressed, streamed.n_expressed)
    assert float(gathered.n[gathered.group_index("ntc")]) == 12.0


def test_gather_path_handles_a_single_selected_cell(toy_adata, tmp_path):
    """Edge case for the nnz-batching arithmetic in the gather branch."""
    from cvequality import sufficient

    p = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(p)
    mask = np.zeros(toy_adata.n_obs, dtype=bool)
    mask[7] = True
    st = sufficient.group_sufficient_stats(
        str(p), group_key="target_gene_name", cell_mask=mask, device="cpu",
        transform="counts", progress=False,
    )
    i = st.group_index(np.asarray(toy_adata.obs["target_gene_name"].values)[7])
    assert float(st.n[i]) == 1.0
    row = np.asarray(toy_adata.X[7].todense(), dtype=np.float64).ravel()
    np.testing.assert_allclose(st.sum[i].numpy(), row, rtol=1e-12)


def test_stats_roundtrip(toy_adata, tmp_path):
    st = group_sufficient_stats(toy_adata, group_key="target_gene_name", device="cpu", progress=False)
    p = tmp_path / "stats.pt"
    st.save(p)
    back = cvq.GroupStats.load(p, device="cpu")
    torch.testing.assert_close(st.sum, back.sum)
    np.testing.assert_array_equal(st.group_names, back.group_names)
    assert back.transform == st.transform


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------


def test_vs_reference_shape_and_columns(toy_adata):
    df = cvq.vs_reference(
        toy_adata, group_key="target_gene_name", reference="ntc", transform="tp10k",
        test="both", nr=64, min_cells=30, device="cpu", progress=False,
    )
    n_genes = toy_adata.n_vars
    assert set(df["perturbation"]) == {"A", "B", "tiny"}
    assert len(df) == 3 * n_genes
    for col in ("cv_ref", "cv_grp", "log2_cv_ratio", "pval_asymptotic", "pval_mslrt",
                "fdr_asymptotic", "fdr_mslrt", "pi_score", "status"):
        assert col in df.columns
    assert (df["n_ref"] == 200).all()


def test_vs_reference_reference_columns_are_shared_across_targets(toy_adata):
    """cv_ref for a given gene must not depend on which target it is compared against."""
    df = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                          test="asymptotic", device="cpu", progress=False, min_cells=30)
    piv = df.pivot_table(index="gene", columns="perturbation", values="cv_ref")
    spread = np.nanmax(np.abs(piv.to_numpy() - piv.to_numpy()[:, :1]))
    assert spread == 0.0


def test_all_zero_gene_is_flagged_not_crashed(toy_adata):
    """Gene g0 is zero in every cell -- mean 0, so both tests are undefined."""
    df = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                          test="both", nr=32, device="cpu", progress=False, min_cells=30)
    zero = df[df["gene"] == "g0"]
    assert len(zero) == 3
    assert (zero["status"] == int(Status.NONPOSITIVE_MEAN)).all()
    assert zero["pval_asymptotic"].isna().all()
    assert zero["pval_mslrt"].isna().all()
    # ...and the rest of the table is unaffected
    assert df[df["gene"] != "g0"]["pval_asymptotic"].notna().all()


def test_min_cells_and_max_targets(toy_adata):
    df = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                          test="asymptotic", min_cells=80, device="cpu", progress=False)
    assert set(df["perturbation"]) == {"A"}  # B has 70 cells, tiny has 40
    df2 = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                           test="asymptotic", min_cells=30, max_targets=1,
                           device="cpu", progress=False)
    assert df2["perturbation"].nunique() == 1


def test_min_frac_expressed_filters_genes(toy_adata):
    loose = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                             test="asymptotic", device="cpu", progress=False, min_cells=30)
    strict = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                              test="asymptotic", min_frac_expressed=0.5,
                              device="cpu", progress=False, min_cells=30)
    assert len(strict) < len(loose)
    assert "g0" not in set(strict["gene"])  # the all-zero gene cannot pass


def test_shards_partition_the_targets(toy_adata):
    whole = cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                             test="asymptotic", device="cpu", progress=False, min_cells=30)
    parts = [
        cvq.vs_reference(toy_adata, group_key="target_gene_name", reference="ntc",
                         test="asymptotic", shard=(r, 3), device="cpu", progress=False, min_cells=30)
        for r in range(3)
    ]
    got = sorted(set().union(*[set(p["perturbation"]) for p in parts]))
    assert got == sorted(set(whole["perturbation"]))
    assert sum(len(p) for p in parts) == len(whole)
    # disjoint
    seen = [set(p["perturbation"]) for p in parts]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])


def test_precomputed_stats_are_reused(toy_adata):
    st = group_sufficient_stats(toy_adata, group_key="target_gene_name", transform="log1p",
                                device="cpu", progress=False)
    df = cvq.vs_reference(stats=st, reference="ntc", transform="log1p", test="asymptotic",
                          device="cpu", progress=False, min_cells=30)
    assert len(df) > 0
    with pytest.raises(ValueError, match="transform"):
        cvq.vs_reference(stats=st, reference="ntc", transform="tp10k", device="cpu", progress=False)


def test_omnibus(toy_adata):
    df = cvq.omnibus(toy_adata, group_key="target_gene_name", test="both", nr=64,
                     min_cells=30, device="cpu", progress=False)
    assert len(df) == toy_adata.n_vars
    assert (df["k_groups"] == 4).all()
    ok = df[df["status"] == int(Status.OK)]
    assert len(ok) > 0
    assert ok["pval_asymptotic"].between(0, 1).all()
    assert ok["pval_mslrt"].between(0, 1).all()


def test_omnibus_exclude(toy_adata):
    df = cvq.omnibus(toy_adata, group_key="target_gene_name", exclude=("ntc",),
                     test="asymptotic", min_cells=30, device="cpu", progress=False)
    assert (df["k_groups"] == 3).all()


def test_null_ntc_split_is_null_by_construction(toy_adata):
    """The two halves are the same cells, so nothing should be discoverable."""
    df = cvq.null_ntc_split(
        toy_adata, group_key="target_gene_name", reference="ntc", n_splits=2,
        transform="tp10k", test="asymptotic", device="cpu", progress=False,
    )
    assert set(df["split"]) == {0, 1}
    p = df["pval_asymptotic"].to_numpy()
    p = p[np.isfinite(p)]
    assert p.size > 50
    # Only a sanity band here -- real calibration is measured on real data, where the
    # answer is a finding rather than an assertion.
    assert 0.2 < p.mean() < 0.8


def test_bh_fdr_matches_statsmodels(toy_adata):
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    from cvequality.adapter import _bh_fdr

    rng = np.random.default_rng(3)
    p = rng.uniform(size=500)
    got = _bh_fdr(p)
    want = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_bh_fdr_is_nan_safe():
    from cvequality.adapter import _bh_fdr

    p = np.array([0.01, np.nan, 0.5, np.nan, 0.001])
    got = _bh_fdr(p)
    assert np.isnan(got[[1, 3]]).all()
    assert np.isfinite(got[[0, 2, 4]]).all()
    finite = _bh_fdr(np.array([0.01, 0.5, 0.001]))
    np.testing.assert_allclose(got[[0, 2, 4]], finite)
