# cvequality

Tests for the **equality of coefficients of variation** across *k* groups — an independent
Python port of the R package [`cvequality`](https://github.com/benmarwick/cvequality) (CRAN
0.2.0), batched in PyTorch so that tens of millions of tests run in one call on a GPU, plus a
sibling test for differential variability that the R package does not contain.

The motivating use is Perturb-seq, where a differential-expression method asks *"did this
perturbation move the gene's mean?"* and these ask *"did it move the gene's variability?"* —
one test per (perturbation, gene).

| | R name | here | cost |
|---|---|---|---|
| Feltz & Miller (1996) asymptotic | `asymptotic_test`, `asymptotic_test2` | same, `+ _batch` | closed form |
| Krishnamoorthy & Lee (2014) MSLRT | `mslr_test`, `mslr_test2` | same, `+ _batch` | `nr` MLE solves per test |
| Equality of SDs | — *(not in R)* | `sd_ratio_test`, `+ _batch` | closed form |

```bash
pip install -e .    # pure Python; needs numpy + torch, plus pandas/h5py/anndata for the adapter
```

---

## Quickstart

```python
import cvequality as cvq

# --- one test from summary statistics (drop-in for the R functions) ---
cvq.asymptotic_test2(n=[5, 5, 5], s=[0.612, 3.927, 2.04], x=[6.8, 8.5, 6.0])
# {'D_AD': 5.530452423813215, 'p_value': 0.06296185501569039, 'status': <Status.OK: 0>}

cvq.mslr_test2(n=[5, 5, 5], x=[6.8, 8.5, 6.0], s=[0.612, 3.927, 2.04], nr=10_000)

# --- batched: T tests x k groups, on the GPU ---
cvq.asymptotic_test2_batch(n=n_tk, s=s_tk, x=x_tk)     # (T,k) -> (T,)
cvq.mslr_test2_batch(n=n_tk, x=x_tk, s=s_tk, nr=1000)
cvq.sd_ratio_test_batch(n=n_tk, sd=s_tk, kurtosis=k_tk)

# --- single-cell: one test per (group, gene) ---
df = cvq.vs_reference("screen.h5ad", group_key="target_gene_name",
                      reference="non-targeting", test="sd_ratio")

# --- one test per gene across every group at once (k = n_groups) ---
df = cvq.omnibus("screen.h5ad", exclude=("non-targeting",))

# --- is any of this calibrated on YOUR data? ---
null = cvq.null_ntc_split("screen.h5ad", match_size=700, n_splits=2)
```

> **Argument order.** All summary-statistic arguments are **keyword-only**, deliberately. R's
> own signatures disagree — `asymptotic_test2(k, n, s, x)` versus `mslr_test2(nr, n, x, s)`
> swap `s` and `x` — and its docs only avoid the trap by naming arguments. Positional calls
> are refused here rather than silently transposed.

---

## Why it scales

Every test consumes only per-group `(n, mean, sd)` — and the SD-ratio test additionally a
kurtosis. So **one streaming pass** over an expression matrix produces sufficient statistics
for every test, transform and comparison design:

```
h5ad (CSR)  ──stream row blocks──▶  (n_groups × n_genes) sum / sumsq / sum³ / sum⁴ / n_expressed
                                            │
                                            ├──▶ vs_reference   (T = groups × genes, k = 2)
                                            └──▶ omnibus        (T = genes, k = groups)
```

Accumulation is a scatter-add keyed on `group_of_cell * n_genes + gene`. Zeros contribute
nothing to any power sum, so working on stored values only is exact — but `n` is the number of
*cells in the group* including zeros, which is what makes the mean and SD correct.

Cache the statistics once and every later analysis is nearly free:

```bash
python -m cvequality stats        --source screen.h5ad --transform log1p --out stats.pt
python -m cvequality vs-reference --source screen.h5ad --stats stats.pt --out hits.parquet
```

On a single H100, a genome-wide screen (~2,000 groups × ~18,000 genes ≈ 36M tests) runs the
asymptotic test in well under a second and MSLRT at `nr=1000` in about a minute; streaming the
matrix takes a few minutes and dominates. `scripts/benchmark.py` measures throughput on your
own hardware before you commit to a long run. `shard=(rank, world)` splits the group axis for
multi-GPU, with no collective communication — one process per GPU, each writing its own
parquet — though for most screens one GPU is plenty and the headroom is better spent on a
larger `nr`.

**float64 is the default and matters.** The literal R form of the LRT statistic subtracts two
`O(n log n)` quantities, so at n ~ 2e5 it retains only a few digits even in float64 and is
unusable in float32. float32 is accepted but warns.

---

## Two MLE solvers, and why the default is not R's

MSLRT's inner loop fits the common-CV MLE. R iterates `t ← G(t)` directly, which converges
only **linearly**, at a rate that approaches 1 as the CV grows: fast at CV ≈ 0.3, but needing
several hundred iterations at CV ≈ 2.3. R caps at 31 with an *absolute* tolerance of 1e-7, so
at single-cell CVs it returns an MLE that has not converged.

The system has a much better form. `u_j(t)` is the positive root of `t·u² + x·u − (v + x²) = 0`,
hence `(v + (x−u)²)/u² = (t+1) − x/u` exactly, and R's fixed point collapses to a scalar
equation:

```
F(t) = Σ nⱼ·xⱼ/uⱼ(t) − Σ nⱼ = 0        F′(t) = Σ nⱼ·xⱼ / (2t·uⱼ + xⱼ)
```

Since `duⱼ/dt < 0`, the left side is strictly increasing whenever all means are positive: the
root is **unique** and Newton is unconditionally safe. It converges in **2–6 iterations**,
agreeing with a 20,000-iteration fixed-point run to 1e-15. At the root the quadratic terms
provably cancel, so `stat = 2·Σ nⱼ·log(τ̂uⱼ/vⱼ^½)` is exact and cancellation-free.

| `solver=` | behaviour | use for |
|---|---|---|
| `"newton"` (default) | converged MLE, exact collapsed statistic | everything |
| `"fixedpoint"` | reproduces R bit for bit — the 31-iteration cap, the absolute tolerance, the `u`/`τ` off-by-one, the literal statistic | R comparison |

**How much does it change answers? Almost nothing**, and that is worth stating. The MSLRT is
studentized, so R's convergence error biases the observed statistic and the bootstrap null
mean in the same direction and largely cancels. Newton is a correctness and speed improvement,
not a change in conclusions.

---

## Transforms, and calibration

`transform=` controls what the CV is computed on, applied during accumulation:

| value | definition | note |
|---|---|---|
| `"counts"` | raw values | CV inflated by per-cell sequencing-depth variation |
| `"tp10k"` | `target_sum · count / total(cell)` | the literal CV of expression, and the only **scale-free** option |
| `"log1p"` (default) | `log1p(tp10k)` | variance-stabilized; the calibrated default |
| callable | your function on the stored values | bring your own |

The default is `log1p` for a specific reason. Both CV tests rest on the sampling variance of
the CV, which by the delta method depends on the kurtosis of the values:

```
Var(ĉv) ≈ (cv²/n)·[ cv² + (γ₄−1)/4 − cv·γ₃ ]
```

collapsing to Feltz–Miller's `D²(0.5 + D²)` exactly at `γ₃=0, γ₄=3`. For depth-normalized
single-cell expression that 4th moment is dominated by a handful of extreme cells: the sample
estimate is barely reproducible between random halves of the same cells and still grows with
sample size, so it has not converged and no variance formula can be built on it. The result is
an inflated false-positive rate that gets **worse with more cells** — the signature of model
misspecification rather than finite-sample error. `log1p` compresses the tail, brings the
kurtosis near the normal value, and makes it stably estimable.

Two caveats follow:

- **`log1p` CVs are not scale-free.** A CV is only meaningful on a ratio scale, and a log has
  no non-arbitrary zero: `log1p(k·x) ≈ log(x) + log(k)` is a location shift, which leaves `sd`
  alone but moves `mean`. So `target_sum` is a real analysis choice here. It is recorded on
  `GroupStats`, reusing statistics under a different value raises, and it appears in every
  output row. Read the effect as *relative variability of log-expression*, not *CV of
  expression*. `tp10k` has no such issue.
- **`tp10k` warns.** It stays available because its CV means what it says, but testing on it
  emits a `RuntimeWarning`: use it for effect sizes, not for thresholding.

**Which transform is calibrated is a property of your data.** The ranking above does not hold
universally — on near-normal synthetic counts it comes out the other way round. Measure it:

```python
null = cvq.null_ntc_split("screen.h5ad", match_size=700, n_splits=4)
```

This splits the reference group into two random halves and tests them against each other, so
H0 holds for every gene by construction and the p-values must be uniform. Any departure is
miscalibration on your data. It reads only the cells it needs, so a full transform × group-size
grid takes a couple of minutes.

---

## The SD-ratio test

Not part of the R package. Because `cv = sd/mean`:

```
log₂(cv_g / cv_r) = log₂(sd_g / sd_r) − log₂FC
```

so **the CV change with the differential-expression component removed is exactly the log SD
ratio**. Testing it directly needs no post-hoc correction and no threshold to tune. The
statistic is an inverse-variance Wald on `log(sd)` using `Var(log s) ≈ (γ₄−1)/(4n)`, which
generalizes to any *k* and reduces to the two-sample log-SD z at k=2.

The kurtosis correction is what makes it usable — assuming normality leaves it
anti-conservative — and it is only available under `log1p`, where the kurtosis is stably
estimable. On simulated screens with known ground truth it separates genuine dispersion changes
from pure mean changes substantially better than the CV tests, at a comparable hit count.

It needs `moments=True` statistics, which is the default:

```python
df = cvq.vs_reference("screen.h5ad", test="sd_ratio")   # or test="all" for all three
```

adding `log2_sd_ratio`, `stat_sd_ratio`, `pval_sd_ratio`, `fdr_sd_ratio`, `kurtosis_ref`,
`kurtosis_grp`.

---

## Interpreting a result

Two hazards are worth knowing before thresholding anything.

**A significant CV change is not evidence of a dispersion change.** On count data
`CV² ≈ 1/μ + φ`, so a perturbation that only shifts a gene's mean *genuinely* changes its CV
and the test correctly reports it. The two effects are strongly correlated in practice.

The fix is to **condition, not residualize**. Regressing the CV effect on the mean effect and
testing the residual was implemented, benchmarked, and removed: it performs worse than doing
nothing, because to know what CV change is expected at a given mean change you need
observations with that mean change and no dispersion change — and none exist, since mean
changes only occur in perturbed cells. Filtering on the mean effect does work:

```python
from cvequality.sufficient import group_sufficient_stats

st_log = group_sufficient_stats("screen.h5ad", group_key="target_gene_name", transform="log1p")
st_tp  = group_sufficient_stats("screen.h5ad", group_key="target_gene_name", transform="tp10k")

df = cvq.vs_reference(stats=st_log, mean_stats=st_tp, reference="non-targeting",
                      test="sd_ratio", min_frac_expressed=0.5)
hits = df[(df["fdr_sd_ratio"] < 0.1) & (df["log2_mean_ratio"].abs() < 0.08)]
```

`mean_stats` must come from a **ratio-scale** transform. Under `log1p` the "mean" is the mean
of logged values, which by Jensen's inequality falls when dispersion rises even at constant
expression — conditioning on it would discard the hits you are hunting.

**`min_frac_expressed` is load-bearing, not cosmetic.** A gene with a single non-zero cell has
a positive mean and a positive SD, so it passes every status check, and its kurtosis is the
theoretical maximum. Such genes can dominate the recurrence tail entirely. Filter them.

Finally, when comparing runs, **compare directions rather than gene lists**. Requiring
significance twice conflates reproducibility with power; a gene just under threshold in a
replicate has replicated.

---

## Status codes

R returns bare `NaN`/`Inf`. Every result here carries a `status` (`cvequality.Status`) so
failures are countable rather than silent, and never poison their neighbours in a batch:

| code | name | meaning |
|---|---|---|
| 0 | `OK` | |
| 1 | `NONPOSITIVE_MEAN` | a group mean ≤ 0 — both CV tests divide by it |
| 2 | `ZERO_VARIANCE` | a group had zero sample SD |
| 3 | `TOO_FEW_OBS` | a group had n < 2 |
| 4 | `NOT_CONVERGED` | the MLE missed its tolerance (`solver="newton"` only; under `"fixedpoint"` this is R's normal behaviour and is reported in `converged` instead) |
| 5 | `NON_FINITE` | non-finite for another reason |

Bootstrap draws can put a group mean ≤ 0, outside the model's support. Those replicates are
**discarded** from the null moments rather than contributing a garbage statistic; `n_valid`
reports how many survived. R has no such guard.

---

## Validation

`tests/r/generate_fixtures.R` runs the real R package over the four values in its own testthat
suite plus a randomized `(k, n, mean, sd)` grid — k=2 to k≈2000, n up to 2e5, CVs from 0.05 to
4, including near-null cases:

```bash
Rscript scripts/install_r_cvequality.R      # pure base R, no dependencies
Rscript tests/r/generate_fixtures.R
pytest tests/ -q                            # 86 pass, 6 GPU-skipped without a device
```

| quantity | tolerance | note |
|---|---|---|
| `D_AD`, its p-value | 1e-12 relative | closed form; exact agreement expected |
| `LRT_STAT`'s `u`, `τ̂` | 1e-12 relative | including R's off-by-one |
| `LRT_STAT`'s statistic | 1e-8 relative | R's own cancellation floor at large n |
| Newton vs a 20,000-iteration fixed point | 1e-9 | same root |
| collapsed vs literal statistic | within the literal form's cancellation bound, computed per case | asserts the exactness claim honestly |
| MSLRT null mean / SD vs R | 6 × the Monte Carlo SE | the null is heavy-tailed, so normal theory understates the SE — measured, and itself guarded by a test |
| MSLRT statistic vs R's published value | within MC error over 24 seeds | R's RNG stream is not reproducible from torch |
| sufficient statistics | vs an independent dense NumPy groupby, all transforms | |

`test_solver.py::test_r_fixedpoint_does_not_converge_at_single_cell_cvs` deliberately fails if
a future R release fixes the convergence issue, so the justification above cannot go stale.

---

## Layout

```
src/cvequality/
  reference.py   literal NumPy transcription of the R package — the validation oracle
  asymptotic.py  Feltz–Miller, scalar + batched
  mslrt.py       Krishnamoorthy–Lee: MLE solvers + chunked parametric bootstrap
  sdratio.py     equality of SDs, with a kurtosis-corrected variance (not in R)
  sufficient.py  streaming per-(group, gene) statistics incl. 3rd/4th moments
  adapter.py     vs_reference / omnibus / null_ntc_split
  _backend.py    device & dtype, chi-square sf and sampling, chunk sizing
  __main__.py    CLI
scripts/         install_r_cvequality.R, benchmark.py
tests/           pytest suite + tests/r/generate_fixtures.R
```

## References

- Feltz CJ, Miller GE (1996). An asymptotic test for the equality of coefficients of variation
  from k populations. *Statistics in Medicine* 15:647–658.
- Krishnamoorthy K, Lee M (2014). Improved tests for the equality of normal coefficients of
  variation. *Computational Statistics* 29:215–232.
- Marwick B, Krishnamoorthy K. *cvequality*: Tests for the Equality of Coefficients of
  Variation from Multiple Groups. R package version 0.2.0. MIT licensed; this is an
  independent port, not affiliated with the original authors.

## License

MIT.
