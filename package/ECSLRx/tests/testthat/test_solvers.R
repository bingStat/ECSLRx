library(testthat)
library(Matrix)

context("Optimization solvers (ADMM and Coordinate Descent)")

# Helper to dynamically call admm_flasso
call_admm_flasso <- function(vv, Group_Matrix, Adjust_Matrix_Fused, lambda_3, fused_penmat, Q, eigval, max_iter_admm, betas_old, numfused, df, iter_count = 0L) {
  args <- list(
    vv = vv,
    Group_Matrix = Group_Matrix,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    lambda_3 = lambda_3,
    fused_penmat = fused_penmat,
    Q = Q,
    eigval = eigval,
    max_iter_admm = max_iter_admm,
    betas_old = betas_old,
    numfused = numfused,
    df = df
  )
  if ("iter_count" %in% formalArgs(admm_flasso)) {
    args$iter_count <- iter_count
  }
  do.call(admm_flasso, args)
}

test_that("admm_flasso and update_betas run with deterministic inputs", {
  # Create a tiny setup
  p <- 3
  J <- 2
  Group_Matrix <- methods::as(Matrix::Matrix(
    c(1.0, 1.0, 0.0,
      0.0, 0.0, 1.0),
    nrow = J, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")
  
  fused_penmat <- methods::as(Matrix::Matrix(
    c(1.0, -1.0, 0.0),
    nrow = 1, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")
  
  Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(
    c(1.0, 0.0),
    nrow = J, ncol = 1, sparse = TRUE
  ), "dgCMatrix")

  # Calculate Q and eigval in R to match C++ solver eigenvalues (increasing)
  A <- as.matrix(Matrix::t(fused_penmat) %*% fused_penmat)
  eig <- eigen(A)
  ord <- order(eig$values)
  eigval <- eig$values[ord]
  Q <- methods::as(eig$vectors[, ord], "dgCMatrix")

  numfused <- as.numeric(Adjust_Matrix_Fused %*% rep(1, 1))
  df <- as.numeric(Group_Matrix %*% rep(1, p))

  vv <- c(0.5, -0.2, 1.2)
  betas_old <- c(0.1, -0.1, 0.5)
  lambda_3 <- c(0.1, 0.1)

  # Test admm_flasso dynamically
  res_admm <- call_admm_flasso(
    vv = vv,
    Group_Matrix = Group_Matrix,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    lambda_3 = lambda_3,
    fused_penmat = fused_penmat,
    Q = Q,
    eigval = eigval,
    max_iter_admm = 5L,
    betas_old = betas_old,
    numfused = numfused,
    df = df,
    iter_count = 0L
  )
  expect_true(is.numeric(res_admm))
  expect_length(res_admm, p)

  # Test update_betas
  lambda_lasso <- 0.1
  w_lasso <- rep(1, p)
  lambda_fused <- 0.05
  group_soft_penalty <- c(0.1, 0.1)
  eta_m <- c(1.5, 1.5)
  exist_grouplasso <- TRUE

  res_update <- update_betas(
    vv = vv,
    betas_old = betas_old,
    Group_Matrix = Group_Matrix,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    Q = Q,
    eigval = eigval,
    numfused = numfused,
    df = df,
    max_iter_admm = 5L,
    lambda_lasso = lambda_lasso,
    w_lasso = w_lasso,
    lambda_fused = lambda_fused,
    group_soft_penalty = group_soft_penalty,
    eta_m = eta_m,
    exist_grouplasso = exist_grouplasso,
    lambda_1w = rep(0, p),
    lambda_2 = rep(0, J),
    lambda_3 = rep(0, J)
  )
  expect_true(is.numeric(res_update))
  expect_length(res_update, p)
})

test_that("ECSLR_Compute_Coef_combine runs and converges on deterministic data", {
  x <- matrix(c(
    -1.0, -1.0,
    -0.5, 0.0,
    0.5, 0.0,
    1.0, 1.0
  ), ncol = 2, byrow = TRUE)
  y <- c(0.0, 0.0, 1.0, 1.0)
  cost_matrix <- matrix(c(
    0.0, 2.0, 1.0, 0.0,
    0.0, 2.0, 1.0, 0.0,
    0.0, 2.0, 1.0, 0.0,
    0.0, 2.0, 1.0, 0.0
  ), nrow = 4, byrow = TRUE)

  Group_Matrix <- Matrix::sparseMatrix(i = c(1, 1), j = c(1, 2), x = 1, dims = c(1, 2))
  fused_penmat <- methods::as(Matrix::Matrix(c(1.0, -1.0), nrow = 1, sparse = TRUE), "dgCMatrix")
  Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(1.0, nrow = 1, ncol = 1, sparse = TRUE), "dgCMatrix")

  default_model <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"
  eigen_cache <- build_eigen_cache(
    Group_Matrix, Adjust_Matrix_Fused, fused_penmat,
    lambda_fused = 0.05 * (1 - 0.5)
  )

  fit <- ECSLR_Compute_Coef_combine(
    x = x,
    y = y,
    cost_matrix = cost_matrix,
    M = 2L,
    include_intercept = TRUE,
    w_lasso = rep(1.0, 2),
    Group_Matrix = Group_Matrix,
    w_group = 1.0,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat,
    alpha = 0.5,
    gamma = 0.5,
    lambda_sparsity = 0.05,
    lambda_diversity = 0.01,
    tolerance = 1e-4,
    obj_tol_rel = 1e-4,
    max_iter = 10L,
    max_iter_admm = 5L,
    model = default_model,
    eigen_cache_ = eigen_cache
  )

  expect_type(fit, "list")
  expect_true(all(c("betas_scaled", "intercept_scaled", "betas_est", "intercept_est", "objective") %in% names(fit)))
  expect_equal(dim(fit$betas_scaled), c(2L, 2L))
  expect_equal(length(fit$betas_est), 2L)
  expect_true(is.numeric(fit$objective))
})
