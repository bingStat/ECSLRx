# test_benchmark.R  --  Performance checks for ECSLRx internals (eigen cache).

library(testthat)
library(Matrix)

skip_on_cran()
skip_if_not_installed("microbenchmark")

make_fused_bench_data <- function(n = 60, p = 8, seed = 77) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), nrow = n)
  y <- as.numeric(x[, 1] - x[, 2] + rnorm(n) > 0)
  cost_matrix <- matrix(c(0, 2, 1, 0), nrow = n, ncol = 4, byrow = TRUE)
  Group_Matrix <- methods::as(Matrix::Diagonal(p), "dgCMatrix")
  D <- matrix(0, nrow = p - 1, ncol = p)
  for (i in seq_len(p - 1)) { D[i, i] <- 1; D[i, i + 1] <- -1 }
  fused_penmat <- Matrix::Matrix(D, sparse = TRUE)
  am_vals <- matrix(0, nrow = p, ncol = p - 1)
  for (i in seq_len(p - 1)) am_vals[i, i] <- 1
  Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(am_vals, sparse = TRUE), "dgCMatrix")
  list(x = x, y = y, cost_matrix = cost_matrix,
       Group_Matrix = Group_Matrix,
       fused_penmat = fused_penmat,
       Adjust_Matrix_Fused = Adjust_Matrix_Fused, p = p)
}

test_that("precomputed eigen cache supports repeated fused-lasso calls", {
  d <- make_fused_bench_data()
  eigen_cache <- compute_block_eigen(d$Group_Matrix, d$Adjust_Matrix_Fused, d$fused_penmat)

  run_args <- list(
    x = d$x, y = d$y, cost_matrix = d$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, d$p), Group_Matrix = d$Group_Matrix, w_group = rep(1, d$p),
    Adjust_Matrix_Fused = d$Adjust_Matrix_Fused, fused_penmat = d$fused_penmat,
    alpha = 0, gamma = 0.5, lambda_sparsity = 0.1, lambda_diversity = 0,
    tolerance = 1e-4, obj_tol_rel = 1e-4, max_iter = 20L, max_iter_admm = 50L,
    model = "ECSLRx"
  )

  bm <- microbenchmark::microbenchmark(
    with_cache = do.call(ECSLR_Compute_Coef_combine, c(run_args, list(eigen_cache_ = eigen_cache))),
    times = 10L
  )

  medians <- summary(bm)[, "median"]
  names(medians) <- summary(bm)[, "expr"]
  message(sprintf("Cache OPT-3 repeated calls: with=%.1f ms",
                  medians["with_cache"] / 1e6))

  expect_true(is.finite(medians["with_cache"]))
})
