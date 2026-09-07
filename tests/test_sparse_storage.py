"""Regression tests for legal, non-canonical CSR storage."""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from cvequality.sufficient import TRANSFORMS, group_sufficient_stats


def _adata(matrix, groups):
    return AnnData(
        matrix,
        obs=pd.DataFrame({"group": groups}, index=[f"c{i}" for i in range(matrix.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    )


def _assert_same_stats(actual, expected):
    np.testing.assert_array_equal(actual.group_names, expected.group_names)
    np.testing.assert_array_equal(actual.var_names, expected.var_names)
    for name in ("n", "sum", "sumsq", "n_expressed", "sum3", "sum4"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_duplicate_coordinates_and_explicit_zeros_match_dense_and_h5ad(transform, tmp_path):
    # The first row stores g0 twice (1 + 2 == 3) and stores an explicit zero for g1.
    # Powers and expression counts must operate on the logical value, not these entries.
    matrix = sparse.csr_matrix(
        (
            np.array([1.0, 2.0, 0.0, 4.0, 2.0, 3.0]),
            np.array([0, 0, 1, 0, 0, 1]),
            np.array([0, 3, 4, 6]),
        ),
        shape=(3, 2),
    )
    assert not matrix.has_canonical_format
    source = _adata(matrix, ["a", "a", "a"])
    dense = _adata(matrix.toarray(), ["a", "a", "a"])
    path = tmp_path / f"duplicates-{transform}.h5ad"
    source.write_h5ad(path)

    # Ensure this test really sends non-canonical storage through the direct h5ad reader.
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["X/indices"][:3], [0, 0, 1])
        assert len(handle["X/data"]) == 6

    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
    kwargs = dict(group_key="group", transform=transform, device="cpu", progress=False)
    expected = group_sufficient_stats(dense, **kwargs)
    in_memory = group_sufficient_stats(source, **kwargs)
    streamed = group_sufficient_stats(path, **kwargs)

    _assert_same_stats(in_memory, expected)
    _assert_same_stats(streamed, expected)
    for observed, original in zip((matrix.data, matrix.indices, matrix.indptr), before):
        np.testing.assert_array_equal(observed, original)

    if transform == "counts":
        row = in_memory.group_index("a")
        np.testing.assert_array_equal(in_memory.n_expressed[row].numpy(), [3.0, 1.0])
        np.testing.assert_allclose(in_memory.var[row].numpy(), [1.0, 3.0])


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_explicit_zero_in_zero_total_row_is_finite_and_not_expressed(transform, tmp_path):
    # Cell 0 has no logical counts, but its zero is explicitly present in CSR storage.
    matrix = sparse.csr_matrix(
        (
            np.array([0.0, 1.0, 3.0, 2.0, 1.0, 4.0]),
            np.array([0, 0, 1, 0, 1, 1]),
            np.array([0, 1, 3, 5, 6]),
        ),
        shape=(4, 2),
    )
    source = _adata(matrix, ["a", "a", "b", "b"])
    dense = _adata(matrix.toarray(), ["a", "a", "b", "b"])
    path = tmp_path / f"zero-total-{transform}.h5ad"
    source.write_h5ad(path)

    kwargs = dict(group_key="group", transform=transform, device="cpu", progress=False)
    expected = group_sufficient_stats(dense, **kwargs)
    in_memory = group_sufficient_stats(source, **kwargs)
    streamed = group_sufficient_stats(path, **kwargs)

    _assert_same_stats(in_memory, expected)
    _assert_same_stats(streamed, expected)
    for stats in (in_memory, streamed):
        assert torch.isfinite(stats.sum).all()
        assert torch.isfinite(stats.sumsq).all()
        a = stats.group_index("a")
        np.testing.assert_array_equal(stats.n_expressed[a].numpy(), [1.0, 1.0])
