#!/usr/bin/env Rscript
# Install the reference R package used ONLY to generate validation fixtures.
#
# cvequality 0.2.0 has zero `Imports` (pure base R), so this pulls nothing else in.
#
#   Rscript scripts/install_r_cvequality.R

lib <- .libPaths()[1]
cat("installing into:", lib, "\n")

if (requireNamespace("cvequality", quietly = TRUE)) {
  cat("cvequality already installed, version",
      as.character(utils::packageVersion("cvequality")), "\n")
} else {
  install.packages("cvequality", lib = lib, repos = "https://cloud.r-project.org")
}

stopifnot(requireNamespace("cvequality", quietly = TRUE))
cat("OK cvequality", as.character(utils::packageVersion("cvequality")), "\n")
