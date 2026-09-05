"""Packaging metadata needed to run the test suite from a clean install."""

import ast
from pathlib import Path


def _extra(name):
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    line = next(
        line for line in pyproject.read_text().splitlines() if line.startswith(f"{name} = ")
    )
    return ast.literal_eval(line.partition("=")[2].strip())


def _requirement(extra, package):
    return next(requirement for requirement in extra if requirement.startswith(package))


def test_io_extras_use_pandas_3_compatible_anndata():
    for name in ("adapter", "test"):
        extra = _extra(name)
        assert "pyarrow>=10" in extra
        assert _requirement(extra, "pandas") == "pandas>=1.5"
        assert _requirement(extra, "anndata") == "anndata>=0.13.0"
