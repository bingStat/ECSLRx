# test_reproducibility.R  --  Numerical reproducibility tests

library(testthat)
library(Matrix)

# Tiny reproducible dataset
make_repro_data <- function(n = 30, p = 5, seed = 42) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), nrow = n)
  lp <- x[, 1] - 0.5 * x[, 2]
  y <- as.numeric(lp + rnorm(n, sd = 0.5) > 0)
  cost_matrix <- matrix(
    c(0, 2, 1, 0), nrow = n, ncol = 4, byrow = TRUE
  )
  Group_Matrix <- methods::as(Matrix::Diagonal(p), "dgCMatrix")
  empty_fused  <- Matrix::Matrix(0, nrow = 0, ncol = p, sparse = TRUE)
  empty_adjust <- Matrix::Matrix(0, nrow = p, ncol = 0, sparse = TRUE)

  list(x = x, y = y, cost_matrix = cost_matrix,
       Group_Matrix = Group_Matrix,
       empty_fused = empty_fused, empty_adjust = empty_adjust, p = p)
}

# Determine default model name dynamically
default_model <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"

# Helper: run ECSLR_Compute_Coef_combine with standard args
run_fit <- function(d, lambda_sparsity = 0.05, M = 1L, eigen_cache_ = NULL,
                    alpha = 1, gamma = 1) {
  Adjust_Matrix_Fused <- if (!is.null(d$Adjust_Matrix_Fused)) {
    d$Adjust_Matrix_Fused
  } else {
    d$empty_adjust
  }
  fused_penmat <- if (!is.null(d$fused_penmat)) d$fused_penmat else d$empty_fused

  args <- list(
    x = d$x, y = d$y, cost_matrix = d$cost_matrix,
    M = M, include_intercept = TRUE,
    w_lasso = rep(1, d$p), Group_Matrix = d$Group_Matrix,
    w_group = rep(1, nrow(d$Group_Matrix)),
    Adjust_Matrix_Fused = Adjust_Matrix_Fused, fused_penmat = fused_penmat,
    alpha = alpha, gamma = gamma,
    lambda_sparsity = lambda_sparsity, lambda_diversity = 0,
    tolerance = 1e-5, obj_tol_rel = 1e-5,
    max_iter = 50L, max_iter_admm = 10L,
    model = default_model,
    x0_ = NULL
  )
  if ("eigen_cache_" %in% formalArgs(ECSLR_Compute_Coef_combine)) {
    args$eigen_cache_ <- eigen_cache_
  }
  do.call(ECSLR_Compute_Coef_combine, args)
}

test_that("ECSLR_Compute_Coef_combine is deterministic across two identical calls", {
  d <- make_repro_data()
  fit1 <- run_fit(d)
  fit2 <- run_fit(d)

  expect_equal(fit1$betas_est,     fit2$betas_est,     tolerance = 1e-14)
  expect_equal(fit1$intercept_est, fit2$intercept_est, tolerance = 1e-14)
  expect_equal(fit1$objective,     fit2$objective,     tolerance = 1e-14)
})

test_that("objective decreases as lambda_sparsity increases (stronger regularization)", {
  d <- make_repro_data()
  fit_low  <- run_fit(d, lambda_sparsity = 0.001)
  fit_high <- run_fit(d, lambda_sparsity = 0.5)

  # Higher sparsity -> sparser model -> fewer non-zeros
  nnz_low  <- sum(abs(fit_low$betas_est) > 1e-6)
  nnz_high <- sum(abs(fit_high$betas_est) > 1e-6)
  expect_gte(nnz_low, nnz_high)
})

test_that("M=2 run with eigen_cache matches M=2 run without (OPT-3 consistency)", {
  d <- make_repro_data()
  d$Group_Matrix <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3),
    j = 1:5,
    x = 1,
    dims = c(3, 5)
  )
  d$Adjust_Matrix_Fused <- Matrix::sparseMatrix(
    i = c(1, 2),
    j = c(1, 2),
    x = 1,
    dims = c(3, 2)
  )
  d$fused_penmat <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2),
    j = c(1, 2, 3, 4),
    x = c(1, -1, 1, -1),
    dims = c(2, d$p)
  )
  eigen_cache <- compute_block_eigen(
    methods::as(d$Group_Matrix, "dgCMatrix"),
    methods::as(d$Adjust_Matrix_Fused, "dgCMatrix"),
    methods::as(d$fused_penmat, "dgCMatrix")
  )

  fit_nocache <- ECSLR_multi_initial(
    x = d$x, y = d$y, cost_matrix = d$cost_matrix,
    M = 2L, include_intercept = TRUE,
    w_lasso = rep(1, d$p), Group_Matrix = d$Group_Matrix,
    w_group = rep(1, nrow(d$Group_Matrix)),
    Adjust_Matrix_Fused = d$Adjust_Matrix_Fused, fused_penmat = d$fused_penmat,
    alpha = 0, gamma = 0.5,
    lambda_sparsity = 0.05, lambda_diversity = 0,
    tolerance = 1e-5, obj_tol_rel = 1e-5,
    max_iter = 50L, max_iter_admm = 10L,
    model = default_model,
    x0 = NULL, start0 = TRUE,
    eigen_cache = eigen_cache
  )
  fit_cached  <- run_fit(d, M = 2L, eigen_cache_ = eigen_cache, alpha = 0, gamma = 0.5)

  expect_equal(fit_nocache$betas_scaled, fit_cached$betas_scaled, tolerance = 1e-14)
})

test_that("ECSLRx result agrees with ECSLRmulti to high precision if both installed", {
  current_pkg <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"
  other_pkg <- if (current_pkg == "ECSLRx") "ECSLRmulti" else "ECSLRx"

  skip_if_not_installed(other_pkg)

  d <- make_repro_data()
  fit_current <- run_fit(d, lambda_sparsity = 0.05)

  other_func <- getExportedValue(other_pkg, "ECSLR_Compute_Coef_combine")

  other_args <- list(
    x = d$x, y = d$y, cost_matrix = d$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, d$p), Group_Matrix = d$Group_Matrix, w_group = rep(1, d$p),
    Adjust_Matrix_Fused = d$empty_adjust, fused_penmat = d$empty_fused,
    alpha = 1, gamma = 1,
    lambda_sparsity = 0.05, lambda_diversity = 0,
    tolerance = 1e-5, obj_tol_rel = 1e-5,
    max_iter = 50L, max_iter_admm = 10L,
    model = other_pkg
  )

  fit_other <- do.call(other_func, other_args)

  expect_equal(fit_current$betas_est,     fit_other$betas_est,     tolerance = 1e-6)
  expect_equal(fit_current$intercept_est, fit_other$intercept_est, tolerance = 1e-6)
  expect_equal(fit_current$objective,     fit_other$objective,     tolerance = 1e-6)
})
