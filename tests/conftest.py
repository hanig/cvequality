import json
import os

import numpy as np
import pytest
import torch

FIXTURES = os.path.join(os.path.dirname(__file__), "r", "fixtures.json")


@pytest.fixture(scope="session")
def fx():
    """Ground truth from the real R cvequality package."""
    if not os.path.exists(FIXTURES):
        pytest.skip(
            "tests/r/fixtures.json missing -- generate it with\n"
            "  Rscript tests/r/generate_fixtures.R"
        )
    with open(FIXTURES) as fh:
        return json.load(fh)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return request.param


def as_case(c):
    """Fixture case -> ``(n, x, s)`` as 1-D float64 arrays."""
    return (
        np.atleast_1d(np.asarray(c["n"], dtype=np.float64)),
        np.atleast_1d(np.asarray(c["x"], dtype=np.float64)),
        np.atleast_1d(np.asarray(c["s"], dtype=np.float64)),
    )


def relerr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


@pytest.fixture(scope="session")
def toy_adata():
    """Small AnnData with known group structure, for sufficient-statistics tests."""
    anndata = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    n_cells, n_genes = 400, 60
    # Poisson counts with per-cell depth variation and per-group rate shifts, so that the
    # three transforms genuinely differ.
    depth = rng.gamma(shape=6.0, scale=1 / 6.0, size=n_cells)
    base = rng.gamma(shape=1.2, scale=2.0, size=n_genes)
    groups = np.array(["ntc"] * 200 + ["A"] * 90 + ["B"] * 70 + ["tiny"] * 40, dtype=object)
    shift = {"ntc": 1.0, "A": 1.6, "B": 0.7, "tiny": 1.0}
    lam = np.outer(depth * np.array([shift[g] for g in groups]), base)
    X = rng.poisson(lam).astype(np.float32)
    X[:, 0] = 0.0  # a gene that is zero everywhere -> must be flagged, not crash
    ad = anndata.AnnData(
        X=sp.csr_matrix(X),
        obs={"target_gene_name": groups},
        var={"gene_ids": [f"g{i}" for i in range(n_genes)]},
    )
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    return ad
