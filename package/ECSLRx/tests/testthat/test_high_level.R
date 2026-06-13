library(testthat)
library(Matrix)
library(dplyr)

context("High-level cross-validation and grid search functions")

# Setup tiny deterministic dataset
set.seed(42)
n <- 20
p <- 4
x <- matrix(as.numeric(rnorm(n * p)), n, p)
colnames(x) <- paste0("V", 1:p)
y <- as.numeric(rbinom(n, 1, 0.4))
cost_matrix <- matrix(0.0, n, 4)
cost_matrix[, 2] <- as.numeric(runif(n, 2, 4))
cost_matrix[, 3] <- as.numeric(runif(n, 1, 2))

Group_Matrix <- methods::as(Matrix::Matrix(
  c(1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0),
  nrow = 2, byrow = TRUE, sparse = TRUE
), "dgCMatrix")

fused_penmat <- methods::as(Matrix::Matrix(
  c(1.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, -1.0),
  nrow = 2, byrow = TRUE, sparse = TRUE
), "dgCMatrix")

Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(
  c(1.0, 0.0,
    0.0, 1.0),
  nrow = 2, ncol = 2, byrow = TRUE, sparse = TRUE
), "dgCMatrix")

w_lasso <- rep(1.0, p)
w_group <- rep(1.0, 2)

default_model <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"

test_that("ECSLR_multi_initial runs successfully", {
  eigen_cache <- compute_block_eigen(Group_Matrix, Adjust_Matrix_Fused, fused_penmat)
  fit <- ECSLR_multi_initial(
    x = x, y = y, cost_matrix = cost_matrix,
    M = 2L, include_intercept = TRUE,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix, w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused, fused_penmat = fused_penmat,
    alpha = 0.5, gamma = 0.5,
    lambda_sparsity = 0.05, lambda_diversity = 0.01,
    tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
    model = default_model, x0 = NULL, start0 = TRUE,
    eigen_cache = eigen_cache
  )

  expect_type(fit, "list")
  expect_true("objective" %in% names(fit))
  expect_true(is.numeric(fit$objective))
})

test_that("CV.ECSLR.alternating search works", {
  cv_fit <- cv.ECSLRx(
    x, y, cost_matrix,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    config = list(
      cv_method = "alternating",
      M = c(1L, 2L), include_intercept = TRUE,
      alpha = 0.5, gamma = 0.5,
      n_lambda_sparsity = 3L, n_lambda_diversity = 3L, n_folds = 2L,
      tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
      model = default_model, CV_criterion = "cost", start0 = TRUE, n_threads = 1L
    )
  )

  expect_type(cv_fit, "list")
  expect_equal(cv_fit$cv_method, "alternating")
  expect_true(all(c("M", "lambda_sparsity_opt", "lambda_diversity_opt", "ECSLR_model_opt") %in% names(cv_fit)))
})

test_that("cv.ECSLRx defaults w_lasso to all ones when omitted", {
  explicit_fit <- cv.ECSLRx(
    x, y, cost_matrix,
    w_lasso = rep(1, p),
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    config = list(
      cv_method = "alternating",
      M = 1L, include_intercept = TRUE,
      alpha = 0.5, gamma = 0.5,
      n_lambda_sparsity = 3L, n_lambda_diversity = 3L, n_folds = 2L,
      tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
      model = default_model, CV_criterion = "cost", start0 = TRUE, n_threads = 1L
    )
  )

  default_fit <- cv.ECSLRx(
    x, y, cost_matrix,
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    config = list(
      cv_method = "alternating",
      M = 1L, include_intercept = TRUE,
      alpha = 0.5, gamma = 0.5,
      n_lambda_sparsity = 3L, n_lambda_diversity = 3L, n_folds = 2L,
      tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
      model = default_model, CV_criterion = "cost", start0 = TRUE, n_threads = 1L
    )
  )

  expect_equal(default_fit$M, explicit_fit$M)
  expect_equal(default_fit$lambda_sparsity_opt, explicit_fit$lambda_sparsity_opt)
  expect_equal(default_fit$lambda_diversity_opt, explicit_fit$lambda_diversity_opt)
  expect_equal(default_fit$ECSLR_model_opt$betas_est, explicit_fit$ECSLR_model_opt$betas_est)
  expect_equal(default_fit$ECSLR_model_opt$intercept_est, explicit_fit$ECSLR_model_opt$intercept_est)
})

test_that("cv.ECSLRx random-search methods work", {
  ParamList <- expand.grid(
    alpha = c(0.3, 0.7),
    gamma = c(0.3, 0.7),
    M = c(1L, 2L),
    lambda_sparsity = c(0.01, 0.1),
    lambda_diversity = c(0.01, 0.1)
  )

  res_rand <- cv.ECSLRx(
    x, y, cost_matrix,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    config = list(
      cv_method = "random_search",
      include_intercept = TRUE, n_folds = 2L,
      ParamList = ParamList, n_params = 2L,
      refit = TRUE, model = default_model, CV_criterion = "cost",
      start0 = TRUE, n_threads = 1L,
      tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
      seed = 1234L
    )
  )

  expect_type(res_rand, "list")
  expect_equal(res_rand$cv_method, "random_search")
  expect_true("lambda_sparsity_opt" %in% names(res_rand))

  res_rand_imp <- cv.ECSLRx(
    x, y, cost_matrix,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    config = list(
      cv_method = "nested_sparsity",
      include_intercept = TRUE, n_folds = 2L,
      ParamList = ParamList, n_params = 4L,
      refit = TRUE, model = default_model, CV_criterion = "cost",
      start0 = TRUE, n_threads = 1L,
      tolerance = 1e-3, obj_tol_rel = 1e-3, max_iter = 5L, max_iter_admm = 3L,
      seed = 1234L
    )
  )

  expect_type(res_rand_imp, "list")
  expect_equal(res_rand_imp$cv_method, "nested_sparsity")
  expect_true("lambda_sparsity_opt" %in% names(res_rand_imp))
})
