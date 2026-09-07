"""Regression checks for quantitative claims in the user-facing documentation."""

import inspect
import json
from pathlib import Path

from cvequality import reference as ref
from cvequality.mslrt import mslr_test2_batch

from .conftest import as_case, relerr
from .test_solver import _fixedpoint_t


ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text()


def test_readme_reports_measured_solver_agreement_and_assertion():
    with (ROOT / "tests/r/fixtures.json").open() as fh:
        cases = json.load(fh)["deterministic"]

    worst = 0.0
    for case in cases:
        n, x, s = as_case(case)
        newton_t = ref.solve_common_cv(n=n, x=x, s=s).t
        fixedpoint_t = _fixedpoint_t(n, x, s, 20_000)
        worst = max(worst, float(relerr(newton_t, fixedpoint_t)))

    assert f"{worst:.2e}" == "8.97e-15"
    assert README.count("8.97e-15") == 2
    assert README.count("1e-9 relative") == 2


def test_bootstrap_truncation_caveat_is_quantified_in_both_places():
    notes = inspect.getdoc(mslr_test2_batch)
    for text in (README, notes):
        text = " ".join(text.split())
        assert "0.31%" in text
        assert "3.4%" in text
        assert "8.5%" in text
        assert "fixture grid reaches CV 4" in text
        assert "observed statistic is not subject to this truncation" in text
        assert "bootstrap null is slightly mismatched" in text


def test_readme_describes_observed_transform_calibration():
    assert "calibrated default" not in README
    assert "90 pass, 6 GPU-skipped" in README
    for claim in (
        "not inflated",
        "0.022 to 0.031",
        "0.45x to 0.6x",
        "mean p-value about 0.61",
        "2.9x at n=722",
        "4.6x at n=3000",
    ):
        assert claim in README
