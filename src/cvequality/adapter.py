"""Perturb-seq entry points: CV-equality testing as a differential-*variability* metric.

Where a DE method asks "did this perturbation move the gene's mean?", these ask "did it move
the gene's noise?". The output is deliberately DE-shaped -- one row per (perturbation, gene)
with an effect size, a p-value and an FDR -- so it can be scored by the same downstream
machinery as a DE table, with ``log2_cv_ratio`` standing in for ``log2FoldChange``.

Two designs:

* :func:`vs_reference` -- k=2, each perturbation against a shared reference group
  (e.g. non-targeting), one test per (perturbation, gene).
* :func:`omnibus` -- k = all perturbations at once, one test per gene: which genes vary in
  CV across the whole screen.

and one diagnostic:

* :func:`null_ntc_split` -- split the reference group in half at random and run
  :func:`vs_reference` on it. Both halves are the same cells, so every p-value is null by
  construction and they should be uniform. This is the cheapest available read on whether
  the test is calibrated on your data, which matters because the CV of count data is tied to
  the mean (see :data:`cvequality.sufficient.TRANSFORMS`).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ._backend import Status, resolve_device, resolve_dtype
from .asymptotic import asymptotic_test2_batch
from .mslrt import mslr_test2_batch
from .sdratio import sd_ratio_test_batch
from .sufficient import GroupStats, group_sufficient_stats

__all__ = ["null_ntc_split", "omnibus", "vs_reference"]

_PFLOOR = 1e-300

#: Which tests a ``test=`` value expands to.
_TEST_SETS = {
    "asymptotic": ("asymptotic",),
    "mslrt": ("mslrt",),
    "sd_ratio": ("sd_ratio",),
    "both": ("asymptotic", "mslrt"),
    "all": ("asymptotic", "mslrt", "sd_ratio"),
}


def _resolve_tests(test):
    """Accept a keyword, or any sequence of test names."""
    if isinstance(test, str):
        if test not in _TEST_SETS:
            raise ValueError(f"test must be one of {sorted(_TEST_SETS)} or a sequence, got {test!r}")
        return _TEST_SETS[test]
    names = tuple(test)
    unknown = set(names) - {"asymptotic", "mslrt", "sd_ratio"}
    if unknown:
        raise ValueError(f"unknown test(s) {sorted(unknown)}")
    return names


def _warn_if_uncalibrated(transform: str) -> None:
    """Warn when testing on a transform measured to be miscalibrated on single-cell data.

    ``tp10k`` is the interpretable choice -- its CV is scale-free -- but its p-values are not
    usable: the tests' inferential machinery rests on the sampling variance of the CV, which
    for TP10K expression is governed by a 4th moment that does not converge -- its sample
    estimate is barely reproducible between random halves of the same cells and still grows
    with n. Effect sizes are fine; thresholding is not.
    """
    if transform == "tp10k":
        warnings.warn(
            "transform='tp10k' gives a scale-free, interpretable CV, but its p-values are "
            "unreliable on single-cell counts: the CV's sampling variance depends on a "
            "kurtosis that does not converge for depth-normalized expression, so the "
            "false-positive rate is inflated and worsens with group size. Use it for effect "
            "sizes only; transform='log1p' is the calibrated default. Measure this on your "
            "own data with cvequality.null_ntc_split().",
            RuntimeWarning,
            stacklevel=3,
        )


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, NaN-safe: NaN p-values stay NaN and are excluded from the count."""
    out = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    if not m.any():
        return out
    q = p[m]
    order = np.argsort(q, kind="stable")
    ranked = q[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1.0)
    res = np.empty(n)
    res[order] = adj
    out[m] = res
    return out


def _resolve_stats(
    source,
    *,
    stats: Optional[GroupStats],
    group_key: str,
    groups,
    transform: str,
    layer: Optional[str],
    device,
    dtype,
    progress: bool,
    target_sum: float = 1e4,
    **stats_kwargs,
) -> GroupStats:
    if stats is not None:
        if stats.transform != transform:
            raise ValueError(
                f"precomputed stats use transform={stats.transform!r} but transform={transform!r} "
                "was requested; pass the matching stats or drop the transform argument"
            )
        if stats.transform == "log1p" and float(stats.target_sum) != float(target_sum):
            raise ValueError(
                f"precomputed stats use target_sum={stats.target_sum:g} but target_sum="
                f"{target_sum:g} was requested. For log1p this changes the CV itself, not just "
                "the scale, so the two are not interchangeable."
            )
        return stats.to(device)
    if source is None:
        raise ValueError("pass either source= or stats=")
    return group_sufficient_stats(
        source,
        groups=groups,
        group_key=group_key,
        layer=layer,
        transform=transform,
        target_sum=target_sum,
        device=device,
        dtype=dtype,
        progress=progress,
        **stats_kwargs,
    )


def vs_reference(
    source=None,
    *,
    stats: Optional[GroupStats] = None,
    mean_stats: Optional[GroupStats] = None,
    group_key: str = "target_gene_name",
    groups=None,
    reference: str = "non-targeting",
    transform: str = "log1p",
    target_sum: float = 1e4,
    layer: Optional[str] = None,
    test: str = "both",
    nr: int = 1000,
    seed: Optional[int] = 0,
    solver: str = "newton",
    min_cells: int = 30,
    min_frac_expressed: float = 0.0,
    targets: Optional[Sequence[str]] = None,
    max_targets: Optional[int] = None,
    shard: Tuple[int, int] = (0, 1),
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    chunk: Optional[int] = None,
    share_draws: bool = False,
    out: Optional[str] = None,
    progress: bool = True,
    return_stats: bool = False,
):
    """Test each perturbation against a shared reference group, per gene (k=2).

    Parameters
    ----------
    source : str or PathLike or AnnData, optional
        Expression matrix. Omit if ``stats`` is given.
    stats : GroupStats, optional
        Precomputed sufficient statistics (see
        :func:`cvequality.sufficient.group_sufficient_stats`). Reusing them avoids
        re-streaming the matrix for a second transform/test/design.
    mean_stats : GroupStats, optional
        Statistics from a **ratio-scale** transform (i.e. ``tp10k``) used *only* to compute
        ``log2_mean_ratio``. Strongly recommended when ``transform="log1p"``: the log1p mean
        is the mean of logged values, which by Jensen's inequality drops when dispersion rises
        even at constant expression -- so conditioning on a log1p mean ratio discards the
        genuine dispersion hits you are trying to keep.
    group_key, groups
        Where perturbation labels come from -- an ``obs`` column name, or an explicit array.
    reference : str
        Label of the control group every other group is tested against.
    transform : {"counts", "tp10k", "log1p"}
        What the CV is computed on; ``log1p`` is the default because it is the only option
        measured to give calibrated p-values on single-cell data. ``tp10k`` gives a
        scale-free, interpretable CV but unusable p-values, and warns. See
        :data:`cvequality.sufficient.TRANSFORMS`.
    target_sum : float
        Depth-normalization constant. It cancels out of a ``tp10k`` CV but **not** a ``log1p``
        one, where it is a real analysis choice -- see :data:`cvequality.sufficient.TRANSFORMS`.
    test : {"both", "asymptotic", "mslrt"}
        ``asymptotic`` is closed-form and effectively free; ``mslrt`` costs ``nr`` MLE solves
        per test and dominates the runtime.
    nr, seed, solver, chunk, share_draws
        Passed to :func:`cvequality.mslrt.mslr_test2_batch`.
    min_cells : int
        Skip perturbations with fewer cells. The tests need at least 2; 30 is the
        conventional floor for a Perturb-seq screen.
    min_frac_expressed : float
        Require this fraction of non-zero cells in **both** groups. Genes failing it are
        dropped before testing, which is the main lever on MSLRT runtime.
    targets, max_targets
        Restrict which perturbations are tested (``max_targets`` keeps the first N after
        filtering -- useful for smoke tests).
    shard : (rank, world)
        Process only perturbations ``rank::world``. Shards are independent: run one process
        per GPU and concatenate. No collective communication is used.
    out : str, optional
        Write the result to this parquet path as well as returning it.
    return_stats : bool
        Also return the :class:`GroupStats`, so a caller can reuse them.

    Returns
    -------
    pandas.DataFrame
        Columns: ``perturbation, gene, n_ref, n_grp, mean_ref, mean_grp, sd_ref, sd_grp,
        frac_expressed_ref, frac_expressed_grp, cv_ref, cv_grp, log2_cv_ratio,
        log2_mean_ratio``, then per
        requested test ``stat_*``, ``pval_*``, ``fdr_*``, plus ``pi_score``, ``status``, and
        the provenance pair ``cv_transform`` / ``target_sum``. ``fdr_*`` is BH within each
        perturbation, matching how per-perturbation DE tables are usually thresholded.
        ``pi_score`` combines the primary test's p-value with its effect size: the absolute
        ``log2_sd_ratio`` for SD-ratio, or the absolute ``log2_cv_ratio`` for MSLRT and the
        asymptotic test.

        The provenance columns are not decoration: under ``log1p`` the CV depends on
        ``target_sum``, so a table without both is not interpretable on its own. The column is
        ``cv_transform`` rather than ``transform`` because ``df.transform`` is a pandas
        method -- attribute access would silently return the method, not the column.
    """
    import pandas as pd

    tests = _resolve_tests(test)
    _warn_if_uncalibrated(transform)
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    st = _resolve_stats(
        source, stats=stats, group_key=group_key, groups=groups, transform=transform,
        target_sum=target_sum, layer=layer, device=device, dtype=dtype, progress=progress,
    )

    # The mean effect used for conditioning should come from a ratio-scale transform. Under
    # log1p the "mean" is the mean of logged values, which by Jensen's inequality falls when
    # dispersion rises at fixed expression -- so a log1p mean ratio is contaminated by the
    # very thing we want to condition away.
    mst = mean_stats.to(device) if mean_stats is not None else None
    if mst is not None:
        if list(mst.var_names) != list(st.var_names) or list(mst.group_names) != list(st.group_names):
            raise ValueError("mean_stats must describe the same genes and groups as stats")

    ref_i = st.group_index(reference)
    n_ref = st.n[ref_i]
    mean_ref, sd_ref = st.mean[ref_i], st.sd[ref_i]
    frac_ref = st.frac_expressed[ref_i]

    keep = [
        g for g in range(len(st.group_names))
        if g != ref_i and float(st.n[g]) >= min_cells
        and (targets is None or st.group_names[g] in set(targets))
    ]
    if max_targets is not None:
        keep = keep[:max_targets]
    rank, world = shard
    if world < 1 or not (0 <= rank < world):
        raise ValueError(f"invalid shard {shard!r}")
    keep = keep[rank::world]
    if not keep:
        raise ValueError("no perturbations left after filtering; check min_cells/targets/shard")

    genes = st.var_names
    frames = []
    t0 = time.time()
    for i, g in enumerate(keep):
        n_g = st.n[g]
        sel = torch.ones(len(genes), dtype=torch.bool, device=device)
        if min_frac_expressed > 0:
            sel = (frac_ref >= min_frac_expressed) & (st.frac_expressed[g] >= min_frac_expressed)
        idx = torch.nonzero(sel, as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        n_tk = torch.stack([n_ref.expand(idx.numel()), n_g.expand(idx.numel())], dim=-1)
        x_tk = torch.stack([mean_ref[idx], st.mean[g][idx]], dim=-1)
        s_tk = torch.stack([sd_ref[idx], st.sd[g][idx]], dim=-1)

        cols = {
            "perturbation": np.repeat(st.group_names[g], idx.numel()),
            "gene": genes[idx.cpu().numpy()],
            "n_ref": float(n_ref),
            "n_grp": float(n_g),
            "mean_ref": x_tk[:, 0].cpu().numpy(),
            "mean_grp": x_tk[:, 1].cpu().numpy(),
            "sd_ref": s_tk[:, 0].cpu().numpy(),
            "sd_grp": s_tk[:, 1].cpu().numpy(),
            # Fraction of cells with a non-zero value: the natural axis for expression
            # filtering, and for stratifying calibration (normality of per-cell values fails
            # hardest for sparsely expressed genes).
            "frac_expressed_ref": frac_ref[idx].cpu().numpy(),
            "frac_expressed_grp": st.frac_expressed[g][idx].cpu().numpy(),
        }
        cv = (s_tk / x_tk).cpu().numpy()
        cols["cv_ref"], cols["cv_grp"] = cv[:, 0], cv[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["log2_cv_ratio"] = np.log2(cv[:, 1] / cv[:, 0])
            # The CV effect is strongly entangled with the mean effect on count data, so the
            # mean effect ships alongside it: conditioning on it is the most reliable way to
            # enrich for genuine dispersion changes (residualizing is measurably worse).
            if mst is None:
                cols["log2_mean_ratio"] = np.log2(cols["mean_grp"] / cols["mean_ref"])
                cols["mean_ratio_transform"] = st.transform
            else:
                mr = (mst.mean[g][idx] / mst.mean[ref_i][idx]).cpu().numpy()
                cols["log2_mean_ratio"] = np.log2(mr)
                cols["mean_ratio_transform"] = mst.transform

        status = None
        if "sd_ratio" in tests:
            k4 = torch.stack([st.kurtosis[ref_i][idx], st.kurtosis[g][idx]], dim=-1)
            sr = sd_ratio_test_batch(n=n_tk, sd=s_tk, kurtosis=k4, device=device, dtype=dtype)
            cols["log2_sd_ratio"] = sr.log2_sd_ratio.cpu().numpy()
            cols["kurtosis_ref"] = k4[:, 0].cpu().numpy()
            cols["kurtosis_grp"] = k4[:, 1].cpu().numpy()
            cols["stat_sd_ratio"] = sr.stat.cpu().numpy()
            cols["pval_sd_ratio"] = sr.p_value.cpu().numpy()
            status = sr.status.cpu().numpy()
        if "asymptotic" in tests:
            a = asymptotic_test2_batch(n=n_tk, s=s_tk, x=x_tk, device=device, dtype=dtype)
            cols["stat_asymptotic"] = a.D_AD.cpu().numpy()
            cols["pval_asymptotic"] = a.p_value.cpu().numpy()
            st_a = a.status.cpu().numpy()
            status = st_a if status is None else np.maximum(status, st_a)
        if "mslrt" in tests:
            m = mslr_test2_batch(
                n=n_tk, x=x_tk, s=s_tk, nr=nr, seed=None if seed is None else seed + i,
                solver=solver, device=device, dtype=dtype, chunk=chunk, share_draws=share_draws,
            )
            cols["stat_mslrt"] = m.MSLRT.cpu().numpy()
            cols["pval_mslrt"] = m.p_value.cpu().numpy()
            cols["n_valid_mslrt"] = m.n_valid.cpu().numpy()
            status = m.status.cpu().numpy() if status is None else np.maximum(status, m.status.cpu().numpy())
        cols["status"] = status

        cols["cv_transform"] = st.transform
        cols["target_sum"] = float(st.target_sum)
        df = pd.DataFrame(cols)
        primary = ("pval_mslrt" if "mslrt" in tests else
                   "pval_asymptotic" if "asymptotic" in tests else "pval_sd_ratio")
        for name in ("asymptotic", "mslrt", "sd_ratio"):
            if f"pval_{name}" in df:
                df[f"fdr_{name}"] = _bh_fdr(df[f"pval_{name}"].to_numpy())
        effect = "log2_sd_ratio" if primary == "pval_sd_ratio" else "log2_cv_ratio"
        df["pi_score"] = -np.log10(np.clip(df[primary], _PFLOOR, 1.0)) * df[effect].abs()
        frames.append(df)

        if progress and (i % 25 == 0 or i == len(keep) - 1):
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-9)
            print(
                f"  vs_reference {i+1}/{len(keep)} perturbations  {el:7.1f}s "
                f"({rate:.2f}/s, eta {(len(keep)-i-1)/max(rate,1e-9):7.1f}s)",
                flush=True,
            )

    if not frames:
        raise ValueError(
            "every perturbation was empty after gene filtering; min_frac_expressed="
            f"{min_frac_expressed} is likely too strict"
        )
    result = pd.concat(frames, ignore_index=True)
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        result.to_parquet(out, index=False)
        if progress:
            print(f"  wrote {out} ({len(result):,} rows)", flush=True)
    return (result, st) if return_stats else result


def omnibus(
    source=None,
    *,
    stats: Optional[GroupStats] = None,
    group_key: str = "target_gene_name",
    groups=None,
    exclude: Sequence[str] = (),
    transform: str = "log1p",
    target_sum: float = 1e4,
    layer: Optional[str] = None,
    test: str = "asymptotic",
    nr: int = 1000,
    seed: Optional[int] = 0,
    solver: str = "newton",
    min_cells: int = 30,
    min_frac_expressed: float = 0.0,
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    chunk: Optional[int] = None,
    out: Optional[str] = None,
    progress: bool = True,
    return_stats: bool = False,
):
    """One test per gene across **all** groups at once (k = number of groups).

    Answers a different question from :func:`vs_reference`: not "did this perturbation change
    this gene's noise?" but "is this gene's noise heterogeneous across the screen?".

    Cheap for the asymptotic test (18k tests). For MSLRT the bootstrap is
    ``n_genes * nr * k`` elements, so ``chunk`` must be small -- the default sizing accounts
    for this, but expect it to cost the same order as the k=2 design.

    Parameters
    ----------
    exclude : sequence of str
        Groups to leave out, e.g. ``("non-targeting",)`` to ask about heterogeneity among
        the actual perturbations only.
    test : {"asymptotic", "mslrt", "both"}
        Defaults to ``asymptotic`` here, since it is the natural screening statistic at this
        scale.

    Returns
    -------
    pandas.DataFrame
        One row per gene: ``gene, k_groups, n_cells_total, cv_pooled``, then ``stat_*``,
        ``pval_*``, ``fdr_*`` for each requested test, plus ``status``.
    """
    import pandas as pd

    tests = _resolve_tests(test)
    _warn_if_uncalibrated(transform)
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    st = _resolve_stats(
        source, stats=stats, group_key=group_key, groups=groups, transform=transform,
        target_sum=target_sum, layer=layer, device=device, dtype=dtype, progress=progress,
    )

    ex = set(exclude)
    keep = [
        g for g in range(len(st.group_names))
        if st.group_names[g] not in ex and float(st.n[g]) >= min_cells
    ]
    if len(keep) < 2:
        raise ValueError(f"need >=2 groups for an omnibus test, got {len(keep)}")
    gi = torch.as_tensor(keep, device=device)

    # (V, k): genes are the batch axis, groups the test axis -- the transpose of vs_reference.
    n_tk = st.n[gi].unsqueeze(0).expand(len(st.var_names), len(keep))
    x_tk = st.mean[gi].T.contiguous()
    s_tk = st.sd[gi].T.contiguous()

    sel = torch.ones(len(st.var_names), dtype=torch.bool, device=device)
    if min_frac_expressed > 0:
        sel = (st.frac_expressed[gi] >= min_frac_expressed).all(dim=0)
    idx = torch.nonzero(sel, as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError("no genes passed min_frac_expressed in all groups")
    n_tk, x_tk, s_tk = n_tk[idx], x_tk[idx], s_tk[idx]

    cols = {
        "gene": st.var_names[idx.cpu().numpy()],
        "k_groups": len(keep),
        "n_cells_total": float(st.n[gi].sum()),
    }
    status = None
    if "sd_ratio" in tests:
        k4 = st.kurtosis[gi].T.contiguous()[idx]
        sr = sd_ratio_test_batch(n=n_tk, sd=s_tk, kurtosis=k4, device=device, dtype=dtype)
        cols["stat_sd_ratio"] = sr.stat.cpu().numpy()
        cols["pval_sd_ratio"] = sr.p_value.cpu().numpy()
        status = sr.status.cpu().numpy()
    if "asymptotic" in tests:
        a = asymptotic_test2_batch(n=n_tk, s=s_tk, x=x_tk, device=device, dtype=dtype)
        cols["stat_asymptotic"] = a.D_AD.cpu().numpy()
        cols["pval_asymptotic"] = a.p_value.cpu().numpy()
        cols["cv_pooled"] = a.cv_pooled.cpu().numpy()
        st_a = a.status.cpu().numpy()
        status = st_a if status is None else np.maximum(status, st_a)
    if "mslrt" in tests:
        m = mslr_test2_batch(
            n=n_tk, x=x_tk, s=s_tk, nr=nr, seed=seed, solver=solver, device=device,
            dtype=dtype, chunk=chunk, progress=progress,
        )
        cols["stat_mslrt"] = m.MSLRT.cpu().numpy()
        cols["pval_mslrt"] = m.p_value.cpu().numpy()
        cols.setdefault("cv_pooled", m.tauh.cpu().numpy())
        status = m.status.cpu().numpy() if status is None else np.maximum(status, m.status.cpu().numpy())
    cols["status"] = status
    cols["cv_transform"] = st.transform
    cols["target_sum"] = float(st.target_sum)

    df = pd.DataFrame(cols)
    for name in ("asymptotic", "mslrt", "sd_ratio"):
        if f"pval_{name}" in df:
            df[f"fdr_{name}"] = _bh_fdr(df[f"pval_{name}"].to_numpy())
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        df.to_parquet(out, index=False)
    return (df, st) if return_stats else df


def null_ntc_split(
    source=None,
    *,
    groups=None,
    group_key: str = "target_gene_name",
    reference: str = "non-targeting",
    n_splits: int = 1,
    split_seed: int = 0,
    transform: str = "log1p",
    target_sum: float = 1e4,
    layer: Optional[str] = None,
    test: str = "both",
    nr: int = 1000,
    seed: Optional[int] = 0,
    solver: str = "newton",
    match_size: Optional[int] = None,
    min_frac_expressed: float = 0.0,
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    progress: bool = True,
):
    """Calibration diagnostic: test the reference group against **itself**.

    Splits the reference cells into two disjoint random halves and runs the k=2 test between
    them. H0 is true by construction for every gene, so the p-values must be uniform. Any
    departure is miscalibration of the test *on this data* rather than biology -- which is
    the thing worth knowing before spending GPU hours, because both tests assume normally
    distributed measurements and single-cell counts are anything but.

    Parameters
    ----------
    n_splits : int
        Independent random splits to run; each contributes a full set of null p-values.
    match_size : int, optional
        Make each half this many cells, to match the size of a real perturbation group
        (the median group size in your design). Calibration depends
        strongly on group size, so comparing like with like matters. Default: split all
        reference cells in half.
    Other parameters
        As :func:`vs_reference`.

    Returns
    -------
    pandas.DataFrame
        The usual per-gene table with ``split`` and ``perturbation`` set to the synthetic
        half labels, plus a printed summary of the p-value distribution when
        ``progress=True``.
    """
    import pandas as pd

    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    if groups is None:
        from .sufficient import _open_matrix, _read_obs_column

        handle, _kind, _shape, _vn, obs_src = _open_matrix(source, layer)
        try:
            groups = _read_obs_column(obs_src, group_key)
        finally:
            if handle is not None:
                handle.close()
    groups = np.asarray(groups)

    ref_cells = np.flatnonzero(groups == reference)
    if ref_cells.size < 4:
        raise ValueError(f"reference {reference!r} has only {ref_cells.size} cells")

    frames = []
    for sp in range(n_splits):
        rng = np.random.default_rng(split_seed + sp)
        perm = rng.permutation(ref_cells)
        half = ref_cells.size // 2 if match_size is None else int(match_size)
        if 2 * half > ref_cells.size:
            raise ValueError(
                f"match_size={half} needs {2*half} reference cells but only {ref_cells.size} exist"
            )
        a, b = perm[:half], perm[half : 2 * half]

        synth = np.full(groups.shape, "__unused__", dtype=object)
        synth[a] = f"{reference}__A"
        synth[b] = f"{reference}__B"
        mask = np.zeros(groups.shape, dtype=bool)
        mask[a] = True
        mask[b] = True

        st = group_sufficient_stats(
            source, groups=synth, layer=layer, transform=transform, target_sum=target_sum,
            device=device, dtype=dtype, cell_mask=mask, progress=progress,
        )
        df = vs_reference(
            stats=st, reference=f"{reference}__A", transform=transform,
            target_sum=target_sum, test=test, nr=nr,
            seed=seed, solver=solver, min_cells=2, min_frac_expressed=min_frac_expressed,
            device=device, dtype=dtype, progress=False,
        )
        df.insert(0, "split", sp)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    if progress:
        _report_calibration(result, transform=transform, test=test)
    return result


def _report_calibration(df, *, transform: str, test: str) -> None:
    """Print the p-value distribution and inflation for a null run."""
    print(f"\n=== null calibration (transform={transform}, {len(df):,} null tests) ===")
    for name in ("asymptotic", "mslrt"):
        col = f"pval_{name}"
        if col not in df:
            continue
        p = df[col].to_numpy()
        p = p[np.isfinite(p)]
        if p.size == 0:
            print(f"  {name:11s} no finite p-values")
            continue
        frac = {a: float((p < a).mean()) for a in (0.05, 0.01, 0.001)}
        print(
            f"  {name:11s} n={p.size:>9,}  mean={p.mean():.4f} (want 0.5)  "
            f"P(p<0.05)={frac[0.05]:.4f}  P(p<0.01)={frac[0.01]:.4f}  P(p<0.001)={frac[0.001]:.5f}"
        )
        print(f"              inflation at 0.05: {frac[0.05]/0.05:6.2f}x   at 0.01: {frac[0.01]/0.01:6.2f}x")
    bad = df["status"].to_numpy()
    if (bad != int(Status.OK)).any():
        vals, cnt = np.unique(bad, return_counts=True)
        detail = ", ".join(f"{Status(int(v)).name}={c:,}" for v, c in zip(vals, cnt))
        print(f"  status: {detail}")
