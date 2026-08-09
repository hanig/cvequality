"""GPU-accelerated Python port of the ``cvequality`` R package.

Tests for the equality of coefficients of variation across k groups, batched so that tens of
millions of tests (e.g. every perturbation x gene pair in a Perturb-seq screen) run on a
GPU in one call.

Two tests, both consuming only per-group ``(n, mean, sd)``:

* :func:`asymptotic_test` / :func:`asymptotic_test2` -- Feltz & Miller (1996). Closed form.
* :func:`mslr_test` / :func:`mslr_test2` -- Krishnamoorthy & Lee (2014) modified
  signed-likelihood ratio test, with a per-test parametric bootstrap.

Layers, from thinnest to widest:

>>> import cvequality as cvq
>>> cvq.asymptotic_test2(n=[5, 5, 5], s=[0.61, 3.93, 2.04], x=[6.8, 8.5, 6.0])   # one test
>>> cvq.asymptotic_test2_batch(n=n_tk, s=s_tk, x=x_tk)                            # (T,k) batch
>>> cvq.vs_reference("screen.h5ad", group_key="target_gene_name",
...                  reference="non-targeting", transform="tp10k")                # perturb-seq

All summary-statistic arguments are **keyword-only**, because R's own signatures disagree on
the order of ``s`` and ``x`` (``asymptotic_test2(k, n, s, x)`` vs ``mslr_test2(nr, n, x, s)``).

See :mod:`cvequality.reference` for the literal NumPy transcription of the R package used as
the validation oracle, and :mod:`cvequality.mslrt` for why the default MLE solver departs
from R's (R's fixed point does not converge at single-cell CVs).
"""

from ._backend import Status
from .asymptotic import (
    AsymptoticResult,
    asymptotic_test,
    asymptotic_test2,
    asymptotic_test2_batch,
)
from .sdratio import SdRatioResult, sd_ratio_test, sd_ratio_test_batch
from .mslrt import (
    CommonCvFitBatch,
    MslrtResult,
    lrt_stat_batch,
    mslr_test,
    mslr_test2,
    mslr_test2_batch,
    solve_common_cv_batch,
)

__version__ = "0.1.0"

__all__ = [
    "AsymptoticResult",
    "CommonCvFitBatch",
    "MslrtResult",
    "SdRatioResult",
    "Status",
    "__version__",
    "asymptotic_test",
    "asymptotic_test2",
    "asymptotic_test2_batch",
    "lrt_stat_batch",
    "mslr_test",
    "mslr_test2",
    "mslr_test2_batch",
    "sd_ratio_test",
    "sd_ratio_test_batch",
    "solve_common_cv_batch",
]


def __getattr__(name):
    # Adapter entry points are imported lazily: they pull in pandas/h5py/anndata, which the
    # pure-statistics layers do not need.
    if name in ("vs_reference", "omnibus", "null_ntc_split", "group_sufficient_stats", "GroupStats"):
        from . import adapter, sufficient  # noqa: F401

        mod = {"group_sufficient_stats": sufficient, "GroupStats": sufficient}.get(name, adapter)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
