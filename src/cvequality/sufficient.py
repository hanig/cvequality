"""Per-(group, gene) sufficient statistics, streamed from disk onto the GPU.

Both CV tests consume only ``(n, mean, sd)`` per group, so a single pass over an expression
matrix produces everything needed for every test and every comparison design. That is the
whole reason this scales: the 142 GB pass is paid once, and the resulting
``(n_groups, n_genes)`` tables are a few hundred megabytes.

Accumulation is a scatter-add of the CSR non-zeros into ``(G*V,)`` float64 buffers keyed on
``group_of_cell * V + gene``. Zeros contribute nothing to either the sum or the sum of
squares, so working only on stored values is exact, not an approximation -- but note that
``n`` is the number of *cells in the group*, including the zeros, which is what makes the
resulting mean and SD the right ones.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ._backend import resolve_device, resolve_dtype

__all__ = ["GroupStats", "TRANSFORMS", "group_sufficient_stats"]

#: Built-in value transforms, applied to the stored non-zeros during accumulation.
#:
#: ``counts``
#:     Raw values, untouched. The CV then mixes biological variability with per-cell
#:     sequencing-depth variation, since a deeply sequenced cell has larger counts for every
#:     gene.
#: ``tp10k``
#:     ``target_sum * count / total_counts(cell)`` -- depth-normalized, the literal "CV of
#:     expression", and the only transform whose CV is scale-free.
#:
#:     **But its p-values are not usable on single-cell counts.** Both tests rest on the
#:     sampling variance of the CV, which by the delta method depends on the kurtosis of the
#:     values. For depth-normalized single-cell expression that 4th moment is dominated by a
#:     handful of extreme cells: the sample estimate barely correlates between random halves
#:     of the same cells and still grows with sample size, i.e. it has not converged. No
#:     variance formula can be built on it, so the false-positive rate is inflated -- and
#:     inflated *worse* as cells increase, which is the tell that this is model
#:     misspecification rather than a finite-sample approximation error. Use it for effect
#:     sizes; do not threshold its p-values.
#: ``log1p`` (default)
#:     ``log1p(target_sum * count / total_counts(cell))`` -- depth-normalized, then
#:     variance-stabilized. Compressing the tail brings the kurtosis close to the normal value
#:     and makes it stably estimable, which is what restores calibration. This is the default
#:     for that reason.
#:
#:     **Its CV is not scale-free.** ``CV = sd/mean`` is only a meaningful dispersion measure
#:     on a ratio scale, and a log-transformed value has no non-arbitrary zero: for ``x >> 1``,
#:     ``log1p(k*x) ~ log(x) + log(k)`` is a pure location shift, which leaves ``sd`` alone but
#:     moves ``mean``, hence moves the CV. So ``target_sum`` is a real analysis choice under
#:     ``log1p``, not a cosmetic one -- it is recorded on :class:`GroupStats` and a reuse under
#:     a different value raises. Read the effect as "relative variability of log-expression",
#:     not "CV of expression". ``tp10k`` has no such issue.
#:
#: Which transform is calibrated is a property of *your* data, and the ranking above does not
#: hold on every dataset -- on near-normal synthetic counts it comes out the other way round.
#: Measure it with :func:`cvequality.null_ntc_split` before trusting any p-value.
TRANSFORMS = ("counts", "tp10k", "log1p")


@dataclass
class GroupStats:
    """Sufficient statistics for every (group, gene) pair.

    Attributes
    ----------
    group_names, var_names : np.ndarray
        Labels for the two axes.
    n : torch.Tensor
        ``(G,)`` cells per group -- *including* cells where a gene is zero.
    sum, sumsq : torch.Tensor
        ``(G, V)`` float64 accumulators over the transformed values.
    n_expressed : torch.Tensor
        ``(G, V)`` number of cells with a stored non-zero, used for expression filtering.
    transform : str
        Which transform produced these.
    target_sum : float
        Depth-normalization constant used (irrelevant for ``counts``, and cancelling for
        ``tp10k``, but a real analysis choice for ``log1p`` -- see
        :func:`group_sufficient_stats`).
    """

    group_names: np.ndarray
    var_names: np.ndarray
    n: torch.Tensor
    sum: torch.Tensor
    sumsq: torch.Tensor
    n_expressed: torch.Tensor
    transform: str
    target_sum: float = 1e4
    #: Raw 3rd/4th power sums, present when ``moments=True``. Needed by the SD-ratio test,
    #: whose variance depends on the kurtosis of the transformed values.
    sum3: Optional[torch.Tensor] = None
    sum4: Optional[torch.Tensor] = None

    @property
    def mean(self) -> torch.Tensor:
        """``(G, V)`` per-group mean."""
        return self.sum / self.n.unsqueeze(-1)

    @property
    def var(self) -> torch.Tensor:
        """``(G, V)`` sample variance with the ``n-1`` denominator, matching R's ``sd``.

        Computed as ``(sumsq - sum^2/n)/(n-1)`` and clamped at zero: the subtraction can go
        slightly negative for a gene that is constant within a group, and a negative
        variance would propagate as NaN through the tests.
        """
        n = self.n.unsqueeze(-1)
        return ((self.sumsq - self.sum**2 / n) / (n - 1.0)).clamp_min(0.0)

    @property
    def sd(self) -> torch.Tensor:
        """``(G, V)`` sample SD (``ddof=1``)."""
        return torch.sqrt(self.var)

    @property
    def cv(self) -> torch.Tensor:
        """``(G, V)`` coefficient of variation. NaN/Inf where the mean is non-positive."""
        return self.sd / self.mean

    def _require_moments(self) -> None:
        if self.sum3 is None or self.sum4 is None:
            raise ValueError(
                "these statistics were computed without higher moments; recompute with "
                "group_sufficient_stats(..., moments=True)"
            )

    @property
    def m2(self) -> torch.Tensor:
        """``(G, V)`` second central moment (population, i.e. ddof=0)."""
        n = self.n.unsqueeze(-1)
        return (self.sumsq / n - (self.sum / n) ** 2).clamp_min(0.0)

    @property
    def skewness(self) -> torch.Tensor:
        """``(G, V)`` sample skewness of the transformed values."""
        self._require_moments()
        n = self.n.unsqueeze(-1)
        mu = self.sum / n
        m3 = self.sum3 / n - 3 * mu * self.sumsq / n + 2 * mu**3
        return m3 / self.m2.clamp_min(1e-300) ** 1.5

    @property
    def kurtosis(self) -> torch.Tensor:
        """``(G, V)`` sample kurtosis (non-excess; 3.0 for a normal distribution).

        This is what sets the sampling variance of ``log(sd)`` in the SD-ratio test. Computed
        from raw power sums, which is exact for sparse input because zeros contribute nothing
        to any power sum while still counting in ``n``.
        """
        self._require_moments()
        n = self.n.unsqueeze(-1)
        mu = self.sum / n
        m4 = (
            self.sum4 / n
            - 4 * mu * self.sum3 / n
            + 6 * mu**2 * self.sumsq / n
            - 3 * mu**4
        )
        # m4/m2^2 >= 1 always; clamp guards float noise for near-constant genes.
        return (m4 / self.m2.clamp_min(1e-300) ** 2).clamp_min(1.0)

    @property
    def frac_expressed(self) -> torch.Tensor:
        """``(G, V)`` fraction of cells in the group with a non-zero value."""
        return self.n_expressed / self.n.unsqueeze(-1)

    def group_index(self, name) -> int:
        """Row index of a named group; raises with the available names if absent."""
        hits = np.flatnonzero(self.group_names == name)
        if hits.size != 1:
            preview = ", ".join(map(str, self.group_names[:5]))
            raise KeyError(f"group {name!r} not found among {len(self.group_names)} groups ({preview}, ...)")
        return int(hits[0])

    def to(self, device) -> "GroupStats":
        dev = resolve_device(device)
        return GroupStats(
            group_names=self.group_names,
            var_names=self.var_names,
            n=self.n.to(dev),
            sum=self.sum.to(dev),
            sumsq=self.sumsq.to(dev),
            n_expressed=self.n_expressed.to(dev),
            transform=self.transform,
            target_sum=self.target_sum,
            sum3=None if self.sum3 is None else self.sum3.to(dev),
            sum4=None if self.sum4 is None else self.sum4.to(dev),
        )

    def save(self, path) -> None:
        """Persist so a second analysis need not re-stream the matrix."""
        torch.save(
            {
                "group_names": self.group_names,
                "var_names": self.var_names,
                "n": self.n.cpu(),
                "sum": self.sum.cpu(),
                "sumsq": self.sumsq.cpu(),
                "n_expressed": self.n_expressed.cpu(),
                "transform": self.transform,
                "target_sum": self.target_sum,
                "sum3": None if self.sum3 is None else self.sum3.cpu(),
                "sum4": None if self.sum4 is None else self.sum4.cpu(),
            },
            os.fspath(path),
        )

    @classmethod
    def load(cls, path, device=None) -> "GroupStats":
        d = torch.load(os.fspath(path), weights_only=False)
        return cls(**d).to(device)


# ---------------------------------------------------------------------------
# CSR streaming
# ---------------------------------------------------------------------------


def _open_matrix(source, layer: Optional[str] = None):
    """Return ``(handle, kind, shape, var_names, obs_getter)`` for a supported source.

    Supported: path to ``.h5ad``, an :class:`anndata.AnnData` (in-memory or backed), or a
    ``(matrix, var_names)`` tuple for tests.
    """
    if isinstance(source, (str, os.PathLike)):
        import h5py

        f = h5py.File(os.fspath(source), "r")
        grp = f["X"] if layer is None else f["layers"][layer]
        enc = grp.attrs.get("encoding-type", "") if hasattr(grp, "attrs") else ""
        if enc != "csr_matrix":
            raise NotImplementedError(
                f"only CSR (cell-major) h5ad matrices are supported; {layer or 'X'} is {enc!r}. "
                "Load it with anndata and pass the AnnData instead."
            )
        shape = tuple(int(v) for v in grp.attrs["shape"])
        var_names = _read_h5ad_array(f["var"][f["var"].attrs["_index"]])
        return f, ("h5ad", grp), shape, var_names, f
    try:
        import anndata  # noqa: F401

        if hasattr(source, "obs") and hasattr(source, "var"):
            X = source.X if layer is None else source.layers[layer]
            return None, ("anndata", X), tuple(source.shape), np.asarray(source.var_names), source
    except ImportError:
        pass
    raise TypeError(f"unsupported source type {type(source)!r}")


def _decode(arr) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in arr], dtype=object)


def _read_h5ad_array(node) -> np.ndarray:
    """Read a plain or pandas-nullable array from an h5ad node."""
    if node.attrs.get("encoding-type", "") == "nullable-string-array":
        values = _decode(node["values"][:])
        missing = np.asarray(node["mask"][:], dtype=bool)
        if missing.any():
            values[missing] = np.nan
        return values
    if not hasattr(node, "shape"):
        raise NotImplementedError(
            f"unsupported h5ad array encoding {node.attrs.get('encoding-type', '')!r}"
        )
    values = node[:]
    return _decode(values) if values.dtype.kind in "SO" else values


#: Below this selected fraction, an h5ad is read row-by-row for just the wanted cells rather
#: than streamed whole. Per-row h5py reads cost ~100 us of overhead each, so they win only
#: when they avoid far more bytes than they add calls -- true for the ``null_ntc_split``
#: diagnostic, which touches a tiny fraction of cells; false for a whole-matrix pass.
GATHER_FRACTION = 0.05


def _iter_row_blocks(
    kind, shape, target_nnz: int, selected: Optional[np.ndarray] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(cell_ids, indices, data, row_lengths)`` CSR blocks of ~``target_nnz`` nnz.

    ``cell_ids`` are the *global* row indices present in the block, so a block need not be
    contiguous -- that is what lets the sparse-selection path gather scattered rows.

    Blocking on non-zeros rather than a fixed cell count keeps peak memory flat: the int64
    index array alone is 8 bytes per non-zero, and cells vary several-fold in depth.

    Parameters
    ----------
    selected : array_like of bool, optional
        Cells that will actually be accumulated. When only a small fraction is selected (see
        :data:`GATHER_FRACTION`) and the source is an h5ad, only those rows are read.
    """
    tag, obj = kind
    n_cells = shape[0]

    if tag == "h5ad":
        indptr = obj["indptr"][:]
        data_ds, idx_ds = obj["data"], obj["indices"]
        lengths = np.diff(indptr)

        if selected is not None and selected.mean() < GATHER_FRACTION:
            wanted = np.flatnonzero(selected)
            # Batch so each yielded block holds ~target_nnz non-zeros.
            cuts = np.searchsorted(
                np.cumsum(lengths[wanted]), np.arange(1, len(wanted)) * target_nnz, side="left"
            )
            for batch in np.array_split(wanted, np.unique(cuts[cuts < len(wanted)])):
                if batch.size == 0:
                    continue
                idx = np.concatenate([idx_ds[indptr[c] : indptr[c + 1]] for c in batch])
                dat = np.concatenate([data_ds[indptr[c] : indptr[c + 1]] for c in batch])
                yield batch, idx, dat, lengths[batch]
            return

        start = 0
        while start < n_cells:
            budget = indptr[start] + target_nnz
            stop = int(np.searchsorted(indptr, budget, side="right")) - 1
            stop = min(max(stop, start + 1), n_cells)
            lo, hi = int(indptr[start]), int(indptr[stop])
            yield (
                np.arange(start, stop),
                idx_ds[lo:hi],
                data_ds[lo:hi],
                lengths[start:stop],
            )
            start = stop
        return

    import scipy.sparse as sp

    X = obj
    if sp.issparse(X):
        X = X.tocsr()  # once, not per block
        per = max(1, int(target_nnz / max(X.nnz / max(n_cells, 1), 1)))
        for start in range(0, n_cells, per):
            stop = min(start + per, n_cells)
            blk = X[start:stop]
            yield np.arange(start, stop), blk.indices, blk.data, np.diff(blk.indptr)
    else:
        A = np.asarray(X)
        per = max(1, int(target_nnz / max(shape[1], 1)))
        for start in range(0, n_cells, per):
            stop = min(start + per, n_cells)
            blk = A[start:stop]
            rows, cols = np.nonzero(blk)
            order = np.lexsort((cols, rows))
            rows, cols = rows[order], cols[order]
            yield (
                np.arange(start, stop),
                cols.astype(np.int64),
                blk[rows, cols],
                np.bincount(rows, minlength=stop - start),
            )


def group_sufficient_stats(
    source,
    groups=None,
    *,
    group_key: Optional[str] = None,
    layer: Optional[str] = None,
    transform: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "log1p",
    target_sum: float = 1e4,
    device: Union[None, str, torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cell_mask=None,
    target_nnz: int = 40_000_000,
    max_cells: Optional[int] = None,
    size_factors=None,
    moments: bool = True,
    progress: bool = True,
) -> GroupStats:
    """Stream an expression matrix once and return per-(group, gene) sufficient statistics.

    Parameters
    ----------
    source : str or PathLike or AnnData
        Path to a CSR ``.h5ad`` (read with h5py, never loaded whole) or an AnnData.
    groups : array_like, optional
        Per-cell group label. Either this or ``group_key`` must be given.
    group_key : str, optional
        Column of ``obs`` holding the group label; read directly from the file for paths, so
        no AnnData object is constructed.
    layer : str, optional
        Layer name; default ``X``.
    transform : {"counts", "tp10k", "log1p"} or callable
        See :data:`TRANSFORMS`. A callable receives the block's stored values (already
        depth-normalized to TP10K) and must return a tensor of the same shape.
    cell_mask : array_like of bool, optional
        Restrict to a subset of cells. Excluded cells are routed to a discarded sentinel
        group, so this costs no extra pass.
    target_nnz : int
        Non-zeros per streamed block. Controls peak memory: roughly
        ``target_nnz * 12`` bytes host-side plus the same again on device.
    max_cells : int, optional
        Stop after this many cells. For smoke-testing a pipeline against a large file
        without paying for the whole pass -- the resulting statistics describe only the
        leading cells, so never use it for analysis.
    target_sum : float
        Depth-normalization constant: values become ``target_sum * count / total(cell)``.
        Irrelevant for ``counts``, and it cancels out of the ``tp10k`` CV entirely -- but for
        ``log1p`` it materially changes the CV and which genes are called, because the log has
        no non-arbitrary zero. See :data:`TRANSFORMS`. Recorded on the returned
        :class:`GroupStats` so a reuse cannot silently mix two choices.
    moments : bool
        Also accumulate 3rd/4th power sums, which the SD-ratio test needs for its kurtosis
        correction. Costs two more scatter-adds per block and two more ``(G, V)`` float64
        accumulators; the pass is I/O bound, so the wall-clock difference is small.
    size_factors : array_like, optional
        Per-cell divisor for ``tp10k``/``log1p``. Defaults to each cell's total over the
        matrix itself, computed in the same pass -- deliberately not read from
        ``obs["total_counts"]``, which may have been computed on a different layer.

    Returns
    -------
    GroupStats
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)
    handle, kind, shape, var_names, obs_src = _open_matrix(source, layer)
    try:
        n_cells, n_genes = shape
        n_stream = n_cells if max_cells is None else min(int(max_cells), n_cells)

        if groups is None:
            if group_key is None:
                raise ValueError("pass either groups= or group_key=")
            groups = _read_obs_column(obs_src, group_key)
        groups = np.asarray(groups)
        if groups.shape[0] != n_cells:
            raise ValueError(f"groups has {groups.shape[0]} entries but the matrix has {n_cells} cells")

        missing_labels = _missing_label_mask(groups)
        n_missing = int(missing_labels.sum())
        if n_missing:
            warnings.warn(
                f"Dropped {n_missing} cell{'s' if n_missing != 1 else ''} with missing group labels.",
                UserWarning,
                stacklevel=2,
            )
        group_names, valid_codes = np.unique(groups[~missing_labels], return_inverse=True)
        G = len(group_names)
        codes = np.full(n_cells, G, dtype=np.int64)
        codes[~missing_labels] = valid_codes
        if cell_mask is not None:
            cell_mask = np.asarray(cell_mask, dtype=bool)
            if cell_mask.shape[0] != n_cells:
                raise ValueError("cell_mask length must equal the number of cells")
            codes = np.where(cell_mask, codes, G)  # sentinel row, dropped at the end
        if n_stream < n_cells:
            codes = codes.copy()
            codes[n_stream:] = G  # unstreamed cells must not count toward n either

        if isinstance(transform, str):
            if transform not in TRANSFORMS:
                raise ValueError(f"transform must be one of {TRANSFORMS} or a callable, got {transform!r}")
        elif not callable(transform):
            raise TypeError("transform must be a string or a callable")
        needs_sf = (transform != "counts") if isinstance(transform, str) else True

        codes_t = torch.from_numpy(codes).to(device)
        sf_t = None
        if size_factors is not None:
            sf_t = torch.as_tensor(np.asarray(size_factors, dtype=np.float64)).to(device=device, dtype=dtype)
            if sf_t.numel() != n_cells:
                raise ValueError("size_factors length must equal the number of cells")

        rows = G + 1  # +1 for the sentinel
        acc_sum = torch.zeros(rows * n_genes, device=device, dtype=torch.float64)
        acc_sqr = torch.zeros(rows * n_genes, device=device, dtype=torch.float64)
        acc_cnt = torch.zeros(rows * n_genes, device=device, dtype=torch.float64)
        acc_c3 = torch.zeros(rows * n_genes, device=device, dtype=torch.float64) if moments else None
        acc_c4 = torch.zeros(rows * n_genes, device=device, dtype=torch.float64) if moments else None

        # Cells that will actually contribute. Passing this down lets the reader skip rows it
        # does not need, which turns the null_ntc_split diagnostic from a full 142 GB pass
        # into a gather of ~1.4k rows.
        selected = codes < G
        if n_stream < n_cells:
            selected = selected.copy()
            selected[n_stream:] = False
        n_target = int(selected.sum())

        t0 = time.time()
        seen = 0
        for cells, idx, data, lens in _iter_row_blocks(kind, shape, target_nnz, selected):
            keep = selected[cells]
            if not keep.any():
                continue  # whole block is unwanted -- skip the GPU work entirely
            gene = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device, non_blocking=True)
            vals = torch.from_numpy(np.ascontiguousarray(data)).to(
                device=device, dtype=dtype, non_blocking=True
            )
            lens_t = torch.from_numpy(np.asarray(lens, dtype=np.int64)).to(device)
            cells_t = torch.from_numpy(np.asarray(cells, dtype=np.int64)).to(device)
            # local row index of each non-zero within this block
            local = torch.repeat_interleave(
                torch.arange(len(cells), device=device), lens_t
            )

            if needs_sf:
                if sf_t is None:
                    # Per-cell total over this block, from the block itself.
                    tot = torch.zeros(len(cells), device=device, dtype=dtype)
                    tot.index_add_(0, local, vals)
                    denom = tot
                else:
                    denom = sf_t[cells_t]
                scale = target_sum / denom.clamp_min(torch.finfo(dtype).tiny)
                vals = vals * scale[local]
                if transform == "log1p":
                    vals = torch.log1p(vals)
                elif callable(transform):
                    vals = transform(vals)

            # Unwanted rows route to the sentinel group, which is dropped on return.
            flat = codes_t[cells_t[local]] * n_genes + gene
            acc_sum.index_add_(0, flat, vals.double())
            acc_sqr.index_add_(0, flat, (vals.double() ** 2))
            acc_cnt.index_add_(0, flat, torch.ones_like(vals, dtype=torch.float64))
            if moments:
                v2 = vals.double() ** 2
                acc_c3.index_add_(0, flat, v2 * vals.double())
                acc_c4.index_add_(0, flat, v2 * v2)

            seen += int(keep.sum())
            if progress:
                el = time.time() - t0
                print(
                    f"  stats {seen}/{n_target} cells ({100*seen/max(n_target,1):5.1f}%) "
                    f"{el:7.1f}s  {seen/max(el,1e-9):,.0f} cells/s",
                    flush=True,
                )
            del gene, vals, local, flat, lens_t, cells_t

        counts = np.bincount(codes, minlength=rows).astype(np.float64)
        n_t = torch.from_numpy(counts).to(device=device, dtype=dtype)[:G]
        return GroupStats(
            group_names=group_names,
            var_names=np.asarray(var_names),
            n=n_t,
            sum=acc_sum.view(rows, n_genes)[:G].to(dtype),
            sumsq=acc_sqr.view(rows, n_genes)[:G].to(dtype),
            n_expressed=acc_cnt.view(rows, n_genes)[:G].to(dtype),
            transform=transform if isinstance(transform, str) else getattr(transform, "__name__", "callable"),
            target_sum=float(target_sum),
            sum3=None if acc_c3 is None else acc_c3.view(rows, n_genes)[:G].to(dtype),
            sum4=None if acc_c4 is None else acc_c4.view(rows, n_genes)[:G].to(dtype),
        )
    finally:
        if handle is not None:
            handle.close()


def _read_obs_column(obs_src, key: str) -> np.ndarray:
    """Read one obs column, handling h5ad categorical encoding without anndata."""
    if hasattr(obs_src, "obs"):  # AnnData
        return np.asarray(obs_src.obs[key].values)
    obs = obs_src["obs"]
    if key not in obs:
        cols = list(obs.attrs.get("column-order", []))
        raise KeyError(f"obs column {key!r} not found; available: {cols}")
    node = obs[key]
    if hasattr(node, "keys") and "categories" in node:
        cats = _read_h5ad_array(node["categories"])
        codes = node["codes"][:]
        missing = codes == -1
        if not missing.any():
            return cats[codes]
        vals = np.empty(codes.shape, dtype=object)
        vals[missing] = np.nan
        vals[~missing] = cats[codes[~missing]]
        return vals
    return _read_h5ad_array(node)


def _missing_label_mask(labels: np.ndarray) -> np.ndarray:
    """Return missing scalar labels without requiring pandas in the core package."""
    if labels.dtype.kind in "fc":
        return np.isnan(labels)
    if labels.dtype.kind in "mM":
        return np.isnat(labels)
    if labels.dtype.kind != "O":
        return np.zeros(labels.shape, dtype=bool)

    def is_missing(value) -> bool:
        if value is None:
            return True
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            # Handles pandas.NA, whose truth value is intentionally ambiguous.
            return True

    return np.fromiter((is_missing(value) for value in labels), dtype=bool, count=len(labels))
