#!/usr/bin/env Rscript
# Emit tests/r/fixtures.json: ground truth from the real R cvequality package.
#
#   Rscript tests/r/generate_fixtures.R
#
# Contents
#   published     - the four values asserted in the package's own testthat suite
#   mtcars        - disp/am columns, so Python can exercise the raw-data entry points
#   deterministic - a randomized (k, n, mean, sd) grid with R's exact D_AD / p_value and
#                   LRT_STAT (uh, tauh, stat). These are closed form: Python must match to
#                   ~1e-12, any disagreement is a bug rather than tolerance.
#   mslrt         - Monte Carlo cases. Includes the bootstrap null moments (mean/sd of the
#                   LRT over replicates) computed with a large nr, because those are what
#                   can be compared across RNG implementations -- a single seeded MSLRT
#                   draw cannot be.

suppressPackageStartupMessages({
  stopifnot(requireNamespace("cvequality", quietly = TRUE))
})

out_path <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "fixtures.json")
if (is.na(out_path) || !nzchar(out_path)) out_path <- "tests/r/fixtures.json"

LRT_STAT <- cvequality:::LRT_STAT

# --- tiny JSON writer (no jsonlite dependency in this lib) ------------------
jnum <- function(v) {
  # 17 significant digits round-trips a double exactly
  s <- vapply(as.numeric(v), function(z) {
    if (is.na(z)) "null" else if (is.infinite(z)) (if (z > 0) "1e999" else "-1e999")
    else formatC(z, digits = 17, format = "g")
  }, character(1))
  if (length(s) == 1L) s else paste0("[", paste(s, collapse = ","), "]")
}
jstr <- function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jobj <- function(...) {
  kv <- list(...)
  paste0("{", paste(sprintf("%s:%s", jstr(names(kv)), unlist(kv, use.names = FALSE)), collapse = ","), "}")
}
jarr <- function(x) paste0("[", paste(x, collapse = ","), "]")

# --- published testthat values ---------------------------------------------
at_mt  <- cvequality::asymptotic_test(mtcars$disp, mtcars$am)
miller <- data.frame(Mean = c(6.8, 8.5, 6.0), CV = c(0.090, 0.462, 0.340), N = c(5, 5, 5))
miller$SD <- miller$CV * miller$Mean
at_mil <- cvequality::asymptotic_test2(k = nrow(miller), n = miller$N, s = miller$SD, x = miller$Mean)

published <- jobj(
  asymptotic_test_mtcars      = jobj(D_AD = jnum(at_mt$D_AD), p_value = jnum(at_mt$p_value),
                                     expect_D_AD = jnum(2.344839), expect_p = jnum(0.1256985)),
  asymptotic_test2_miller     = jobj(D_AD = jnum(at_mil$D_AD), p_value = jnum(at_mil$p_value),
                                     expect_D_AD = jnum(5.530452), expect_p = jnum(0.06296186)),
  mslr_test_mtcars_expect     = jobj(MSLRT = jnum(2.046478), p_value = jnum(0.1525588), nr = jnum(1e4)),
  mslr_test2_miller_expect    = jobj(MSLRT = jnum(6.64317),  p_value = jnum(0.03609557), nr = jnum(1e4))
)

mtcars_fx <- jobj(disp = jnum(mtcars$disp), am = jnum(mtcars$am))

# --- deterministic grid ----------------------------------------------------
set.seed(20260806)
cases <- list()
add_case <- function(label, n, x, s) {
  a <- cvequality::asymptotic_test2(k = length(x), n = n, s = s, x = x)
  L <- LRT_STAT(n, x, s)
  k <- length(x)
  cases[[length(cases) + 1L]] <<- jobj(
    label = jstr(label), n = jnum(n), x = jnum(x), s = jnum(s),
    D_AD = jnum(a$D_AD), p_value = jnum(a$p_value),
    lrt_uh = jnum(L[1:k]), lrt_tauh = jnum(L[k + 1]), lrt_stat = jnum(L[k + 2])
  )
}

# Feltz & Miller's own summary data
add_case("miller_k3", miller$N, miller$Mean, miller$SD)

# small-k randomized
for (i in 1:12) {
  k  <- sample(2:6, 1)
  n  <- sample(5:80, k, replace = TRUE)
  x  <- runif(k, 1, 20)
  cv <- runif(k, 0.05, 0.8)
  add_case(sprintf("rand_smallk_%02d", i), n, x, cv * x)
}

# k=2 at single-cell scale: a very large reference group against a small target group
for (i in 1:8) {
  n  <- c(209760, sample(c(30, 120, 700, 3000), 1))
  x  <- runif(2, 0.05, 50)
  cv <- runif(2, 0.3, 4.0)          # single-cell CVs are large
  add_case(sprintf("rand_k2_large_%02d", i), n, x, cv * x)
}

# near-null k=2 (CVs almost equal) -- the regime where cancellation bites
for (i in 1:5) {
  n   <- c(209760, 722)
  x   <- runif(2, 0.5, 5)
  cv0 <- runif(1, 0.5, 2.5)
  cv  <- cv0 * c(1, 1 + rnorm(1, 0, 0.002))
  add_case(sprintf("rand_k2_nearnull_%02d", i), n, x, cv * x)
}

# omnibus scale: k = 1966 groups
for (i in 1:2) {
  k  <- 1966
  n  <- sample(30:3000, k, replace = TRUE)
  x  <- runif(k, 0.1, 20)
  cv <- runif(k, 0.4, 3.0)
  add_case(sprintf("rand_omnibus_k1966_%02d", i), n, x, cv * x)
}

# --- Monte Carlo cases: report the bootstrap null moments ------------------
# Reimplements mslr_test2's loop verbatim so mean(gv)/sd(gv) can be exported; these are
# the quantities comparable across RNG implementations.
null_moments <- function(n, x, s, nr, seed) {
  set.seed(seed)
  k <- length(x); df <- n - 1
  xst0 <- LRT_STAT(n, x, s)
  uh0 <- xst0[1:k]; tauh0 <- xst0[k + 1]; stat0 <- xst0[k + 2]
  sh0 <- tauh0 * uh0; se0 <- tauh0 * uh0 / sqrt(n)
  gv <- numeric(nr)
  for (ii in 1:nr) {
    z  <- rnorm(k)
    xb <- uh0 + z * se0
    ch <- rchisq(k, df)
    sb <- sh0 * sqrt(ch / df)
    gv[ii] <- LRT_STAT(n, xb, sb)[k + 2]
  }
  am <- mean(gv); sdv <- sd(gv)
  statm <- sqrt(2 * (k - 1)) * (stat0 - am) / sdv + (k - 1)
  list(stat0 = stat0, null_mean = am, null_sd = sdv, MSLRT = statm,
       p_value = 1 - pchisq(statm, k - 1), nr = nr)
}

mc <- list()
add_mc <- function(label, n, x, s, nr, seed) {
  r <- null_moments(n, x, s, nr, seed)
  mc[[length(mc) + 1L]] <<- jobj(
    label = jstr(label), n = jnum(n), x = jnum(x), s = jnum(s), nr = jnum(r$nr),
    stat0 = jnum(r$stat0), null_mean = jnum(r$null_mean), null_sd = jnum(r$null_sd),
    MSLRT = jnum(r$MSLRT), p_value = jnum(r$p_value)
  )
}
add_mc("miller_k3", miller$N, miller$Mean, miller$SD, 2e5, 1L)
add_mc("mtcars_k2", c(sum(mtcars$am == 0), sum(mtcars$am == 1)),
       c(mean(mtcars$disp[mtcars$am == 0]), mean(mtcars$disp[mtcars$am == 1])),
       c(sd(mtcars$disp[mtcars$am == 0]), sd(mtcars$disp[mtcars$am == 1])), 2e5, 1L)
add_mc("k2_pseq_like", c(209760, 722), c(3.1, 3.4), c(3.1 * 1.6, 3.4 * 1.9), 5e4, 7L)
add_mc("k2_pseq_nearnull", c(209760, 722), c(1.7, 1.72), c(1.7 * 1.2, 1.72 * 1.2), 5e4, 11L)
add_mc("k5_moderate", c(40, 55, 30, 61, 47), c(5, 7, 6, 9, 8),
       c(5, 7, 6, 9, 8) * c(0.2, 0.5, 0.35, 0.3, 0.45), 1e5, 3L)

json <- jobj(
  r_version           = jstr(R.version.string),
  cvequality_version  = jstr(as.character(utils::packageVersion("cvequality"))),
  mle_tol             = jnum(1e-7),
  published           = published,
  mtcars              = mtcars_fx,
  deterministic       = jarr(unlist(cases, use.names = FALSE)),
  mslrt               = jarr(unlist(mc, use.names = FALSE))
)
writeLines(json, out_path)
cat("wrote", out_path, "\n")
cat("  deterministic cases:", length(cases), "\n")
cat("  monte carlo cases:  ", length(mc), "\n")
