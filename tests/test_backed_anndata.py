"""Regression coverage for bounded streaming from backed AnnData objects."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData, read_h5ad
from scipy import sparse

from cvequality.sufficient import TRANSFORMS, group_sufficient_stats


def _adata(matrix) -> AnnData:
    values = np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix)
    return AnnData(
        matrix,
        obs=pd.DataFrame(
            {"group": pd.Categorical(np.resize(["a", "b"], values.shape[0]))},
            index=[f"cell-{i}" for i in range(values.shape[0])],
        ),
        var=pd.DataFrame(index=[f"gene-{i}" for i in range(values.shape[1])]),
    )


def _matrix(format_: str):
    values = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 2.0],
            [0.0, 5.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 6.0, 2.0],
            [3.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 4.0, 5.0],
        ]
    )
    if format_ == "csr":
        return sparse.csr_matrix(values)
    if format_ == "csc":
        return sparse.csc_matrix(values)
    return values


def _assert_stats_equal(actual, expected):
    np.testing.assert_array_equal(actual.group_names, expected.group_names)
    np.testing.assert_array_equal(actual.var_names, expected.var_names)
    for name in ("n", "sum", "sumsq", "n_expressed", "sum3", "sum4"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("format_", ["csr", "dense"])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_backed_matrix_matches_in_memory(tmp_path, format_, transform):
    adata = _adata(_matrix(format_))
    path = tmp_path / f"{format_}.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        expected = group_sufficient_stats(
            adata, group_key="group", transform=transform, device="cpu", progress=False
        )
        actual = group_sufficient_stats(
            backed,
            group_key="group",
            transform=transform,
            target_nnz=3,  # smaller than a row: exercise repeated one-row reads
            device="cpu",
            progress=False,
        )
        _assert_stats_equal(actual, expected)

        # group_sufficient_stats must not close a file handle owned by its caller.
        assert backed.isbacked
        assert backed.X[0, 0] == adata.X[0, 0]
    finally:
        backed.file.close()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_in_memory_csc_remains_supported(transform):
    csc = _adata(_matrix("csc"))
    dense = _adata(_matrix("dense"))
    actual = group_sufficient_stats(
        csc, group_key="group", transform=transform, device="cpu", progress=False
    )
    expected = group_sufficient_stats(
        dense, group_key="group", transform=transform, device="cpu", progress=False
    )
    _assert_stats_equal(actual, expected)


@pytest.mark.parametrize("format_", ["csr", "dense"])
def test_backed_reads_are_bounded_row_slices(tmp_path, monkeypatch, format_):
    adata = _adata(_matrix(format_))
    path = tmp_path / f"bounded-{format_}.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    matrix = backed.X
    matrix_type = type(matrix)
    original_getitem = matrix_type.__getitem__
    row_counts = []

    def recording_getitem(self, index):
        # Record only reads of the 2-D expression dataset. This matters for h5py.Dataset,
        # whose class is also used by unrelated arrays in the same file.
        if self.shape == matrix.shape:
            row_index = index[0] if isinstance(index, tuple) else index
            assert isinstance(row_index, slice)
            start, stop, step = row_index.indices(matrix.shape[0])
            count = len(range(start, stop, step))
            assert count <= 2
            row_counts.append(count)
        return original_getitem(self, index)

    monkeypatch.setattr(matrix_type, "__getitem__", recording_getitem)
    try:
        group_sufficient_stats(
            backed,
            group_key="group",
            transform="counts",
            target_nnz=2 * adata.n_vars,
            device="cpu",
            progress=False,
        )
        assert row_counts == [2, 2, 2, 2]
    finally:
        backed.file.close()


def test_backed_csc_is_rejected_before_data_read(tmp_path, monkeypatch):
    values = _matrix("csc")
    adata = _adata(values)
    path = tmp_path / "csc.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    matrix = backed.X

    def unexpected_read(*args, **kwargs):
        raise AssertionError("backed CSC data was read before rejection")

    monkeypatch.setattr(type(matrix), "__getitem__", unexpected_read)
    monkeypatch.setattr(type(matrix), "to_memory", unexpected_read)
    try:
        with pytest.raises(NotImplementedError, match=r"backed CSC.*convert it to CSR"):
            group_sufficient_stats(
                backed,
                group_key="group",
                transform="counts",
                device="cpu",
                progress=False,
            )
    finally:
        backed.file.close()


def test_in_memory_csc_layer_on_backed_object_remains_supported(tmp_path):
    values = _matrix("csc")
    adata = _adata(sparse.csr_matrix(values.shape))
    adata.layers["counts"] = values
    path = tmp_path / "in-memory-layer.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        assert sparse.isspmatrix_csc(backed.layers["counts"])
        actual = group_sufficient_stats(
            backed,
            group_key="group",
            layer="counts",
            transform="counts",
            device="cpu",
            progress=False,
        )
        expected = group_sufficient_stats(
            adata,
            group_key="group",
            layer="counts",
            transform="counts",
            device="cpu",
            progress=False,
        )
        _assert_stats_equal(actual, expected)
    finally:
        backed.file.close()


def test_backed_sparse_selection_and_layer(tmp_path):
    values = sparse.vstack([_matrix("csr")] * 6, format="csr")
    adata = _adata(sparse.csr_matrix(values.shape))
    adata.layers["counts"] = values
    path = tmp_path / "layer.h5ad"
    adata.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[[0, adata.n_obs - 1]] = True
    assert mask.mean() < 0.05  # exercise the scattered-row gather path
    try:
        expected = group_sufficient_stats(
            adata,
            group_key="group",
            layer="counts",
            cell_mask=mask,
            transform="log1p",
            device="cpu",
            progress=False,
        )
        actual = group_sufficient_stats(
            backed,
            group_key="group",
            layer="counts",
            cell_mask=mask,
            transform="log1p",
            target_nnz=4,
            device="cpu",
            progress=False,
        )
        _assert_stats_equal(actual, expected)
    finally:
        backed.file.close()
