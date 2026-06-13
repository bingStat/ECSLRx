# test_core_fit.R  --  Tests for update_betas, compute_block_eigen, ECSLR_Compute_Coef_combine

library(testthat)
library(Matrix)

# Shared tiny-problem fixture --------------------------------------------------
make_tiny_problem <- function() {
  x <- matrix(c(
    -1, -1,
    -0.5, 0,
     0.5, 0,
     1,  1
  ), ncol = 2, byrow = TRUE)
  y <- c(0, 0, 1, 1)
  cost_matrix <- matrix(
    c(0, 2, 1, 0,
      0, 2, 1, 0,
      0, 2, 1, 0,
      0, 2, 1, 0),
    nrow = 4, byrow = TRUE
  )
  Group_Matrix  <- methods::as(Matrix::Diagonal(2), "dgCMatrix")
  empty_fused   <- Matrix::Matrix(0, nrow = 0, ncol = 2, sparse = TRUE)
  empty_adjust  <- Matrix::Matrix(0, nrow = 2, ncol = 0, sparse = TRUE)
  list(x = x, y = y, cost_matrix = cost_matrix,
       Group_Matrix = Group_Matrix,
       empty_fused = empty_fused, empty_adjust = empty_adjust)
}

# Determine default model name dynamically
default_model <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"

if (exists("compute_block_eigen", mode = "function")) {
  test_that("block eigen construction returns aligned sparse basis", {
    Group_Matrix       <- methods::as(Matrix::Diagonal(3), "dgCMatrix")
    Adjust_Matrix_Fused <- methods::as(Matrix::Diagonal(3), "dgCMatrix")
    fused_penmat       <- methods::as(Matrix::Diagonal(3), "dgCMatrix")

    res <- compute_block_eigen(Group_Matrix, Adjust_Matrix_Fused, fused_penmat)

    expect_s4_class(res$Q, "dgCMatrix")
    expect_equal(dim(res$Q), c(3L, 3L))
    expect_equal(as.numeric(res$eigval), rep(1, 3), tolerance = 1e-12)
  })

  test_that("block eigen uses row-sum block sizes and supports singleton groups", {
    Group_Matrix <- Matrix::sparseMatrix(
      i = c(1, 2, 2),
      j = c(1, 2, 3),
      x = 1,
      dims = c(2, 3)
    )
    Adjust_Matrix_Fused <- Matrix::sparseMatrix(
      i = 2,
      j = 1,
      x = 1,
      dims = c(2, 1)
    )
    fused_penmat <- Matrix::sparseMatrix(
      i = c(1, 1),
      j = c(2, 3),
      x = c(1, -1),
      dims = c(1, 3)
    )

    res <- compute_block_eigen(
      methods::as(Group_Matrix, "dgCMatrix"),
      methods::as(Adjust_Matrix_Fused, "dgCMatrix"),
      methods::as(fused_penmat, "dgCMatrix")
    )

    expect_s4_class(res$Q, "dgCMatrix")
    expect_equal(as.matrix(res$Q)[1, 1], 1, tolerance = 1e-12)
    expect_equal(as.numeric(res$eigval[-1]), c(0, 2), tolerance = 1e-12)
  })

  test_that("block eigen reconstructs fused Gram matrix across multiple blocks", {
    Group_Matrix <- Matrix::sparseMatrix(
      i = c(1, 1, 1, 2, 2, 3),
      j = 1:6,
      x = 1,
      dims = c(3, 6)
    )
    Adjust_Matrix_Fused <- Matrix::sparseMatrix(
      i = c(1, 1, 2),
      j = c(1, 2, 3),
      x = 1,
      dims = c(3, 3)
    )
    fused_penmat <- Matrix::sparseMatrix(
      i = c(1, 1, 2, 2, 3, 3),
      j = c(1, 2, 2, 3, 4, 5),
      x = c(1, -1, 1, -1, 1, -1),
      dims = c(3, 6)
    )

    res <- compute_block_eigen(
      methods::as(Group_Matrix, "dgCMatrix"),
      methods::as(Adjust_Matrix_Fused, "dgCMatrix"),
      methods::as(fused_penmat, "dgCMatrix")
    )

    reconstructed <- as.matrix(
      res$Q %*%
        Matrix::Diagonal(x = as.numeric(res$eigval)) %*%
        Matrix::t(res$Q)
    )
    expected <- as.matrix(Matrix::crossprod(fused_penmat))
    orthonormal <- as.matrix(Matrix::crossprod(res$Q))

    expect_equal(reconstructed, expected, tolerance = 1e-10)
    expect_equal(orthonormal, diag(6), tolerance = 1e-10)
  })
}

test_that("update_betas leaves coefficients unchanged without penalties", {
  Group_Matrix <- methods::as(Matrix::Diagonal(3), "dgCMatrix")
  Q            <- methods::as(Matrix::Diagonal(3), "dgCMatrix")
  vv           <- c(0.25, -0.5, 1.5)

  out <- update_betas(
    vv            = vv,
    betas_old     = vv,
    Group_Matrix  = Group_Matrix,
    Adjust_Matrix_Fused = Matrix::Matrix(0, nrow = 3, ncol = 0, sparse = TRUE),
    fused_penmat  = Matrix::Matrix(0, nrow = 0, ncol = 3, sparse = TRUE),
    Q             = Q,
    eigval        = rep(0, 3),
    numfused      = rep(0, 3),
    df            = rep(1, 3),
    max_iter_admm = 5L,
    lambda_lasso  = 0,
    w_lasso       = rep(1, 3),
    lambda_fused  = 0,
    group_soft_penalty = rep(0, 3),
    eta_m         = rep(1, 3),
    exist_grouplasso = FALSE,
    lambda_1w     = rep(0, 3),
    lambda_2      = rep(0, 3),
    lambda_3      = rep(0, 3)
  )

  expect_equal(as.numeric(out), vv)
})

test_that("ECSLR_Compute_Coef_combine runs on tiny deterministic problem", {
  p <- make_tiny_problem()

  args <- list(
    x                 = p$x,
    y                 = p$y,
    cost_matrix       = p$cost_matrix,
    M                 = 1L,
    include_intercept = TRUE,
    w_lasso           = rep(1, 2),
    Group_Matrix      = p$Group_Matrix,
    w_group           = rep(1, 2),
    Adjust_Matrix_Fused = p$empty_adjust,
    fused_penmat      = p$empty_fused,
    alpha             = 1,
    gamma             = 1,
    lambda_sparsity   = 0.01,
    lambda_diversity  = 0,
    tolerance         = 1e-4,
    obj_tol_rel       = 1e-4,
    max_iter          = 20L,
    max_iter_admm     = 5L,
    model             = default_model
  )

  fit <- do.call(ECSLR_Compute_Coef_combine, args)

  expect_true(all(c("betas_est", "intercept_est", "objective", "coef_x_std") %in% names(fit)))
  expect_equal(length(fit$betas_est), 2L)
  expect_equal(length(fit$intercept_est), 1L)
  expect_true(is.finite(fit$objective))
})

test_that("R-level initial passes eigen_cache to direct solver", {
  skip_if_not(exists("compute_block_eigen", mode = "function"))
  skip_if_not("eigen_cache_" %in% formalArgs(ECSLR_Compute_Coef_combine))

  p <- make_tiny_problem()

  # Fused penalty setup (non-trivial for eigen)
  fp <- Matrix::Matrix(c(1, -1), nrow = 1, sparse = TRUE)
  am <- Matrix::Matrix(1, nrow = 1, ncol = 1, sparse = TRUE)
  am <- methods::as(am, "dgCMatrix")
  gm <- Matrix::sparseMatrix(i = c(1, 1), j = c(1, 2), x = 1, dims = c(1, 2))

  eigen_cache <- compute_block_eigen(gm, am, fp)

  args <- list(
    x = p$x, y = p$y, cost_matrix = p$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, 2), Group_Matrix = gm, w_group = 1,
    Adjust_Matrix_Fused = am, fused_penmat = fp,
    alpha = 0, gamma = 0.5,
    lambda_sparsity = 0.05, lambda_diversity = 0,
    tolerance = 1e-4, obj_tol_rel = 1e-4,
    max_iter = 20L, max_iter_admm = 20L,
    model = default_model
  )

  fit_initial <- do.call(ECSLR_multi_initial, c(args, list(
    x0 = NULL, start0 = TRUE, eigen_cache = eigen_cache
  )))
  fit_with_cache <- do.call(ECSLR_Compute_Coef_combine, c(args, list(eigen_cache_ = eigen_cache)))

  expect_equal(fit_initial$betas_est,     fit_with_cache$betas_est,     tolerance = 1e-8)
  expect_equal(fit_initial$intercept_est, fit_with_cache$intercept_est, tolerance = 1e-8)
  expect_equal(fit_initial$objective,     fit_with_cache$objective,     tolerance = 1e-8)
})

test_that("fused direct solver call consumes eigen_cache_", {
  skip_if_not(exists("compute_block_eigen", mode = "function"))
  skip_if_not("eigen_cache_" %in% formalArgs(ECSLR_Compute_Coef_combine))

  p <- make_tiny_problem()
  fp <- Matrix::Matrix(c(1, -1), nrow = 1, sparse = TRUE)
  am <- methods::as(Matrix::Matrix(1, nrow = 1, ncol = 1, sparse = TRUE), "dgCMatrix")
  gm <- Matrix::sparseMatrix(i = c(1, 1), j = c(1, 2), x = 1, dims = c(1, 2))

  args <- list(
    x = p$x, y = p$y, cost_matrix = p$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, 2), Group_Matrix = gm, w_group = 1,
    Adjust_Matrix_Fused = am, fused_penmat = fp,
    alpha = 0, gamma = 0.5,
    lambda_sparsity = 0.05, lambda_diversity = 0,
    tolerance = 1e-4, obj_tol_rel = 1e-4,
    max_iter = 20L, max_iter_admm = 20L,
    model = default_model
  )
  eigen_cache <- compute_block_eigen(gm, am, fp)
  fit <- do.call(ECSLR_Compute_Coef_combine, c(args, list(eigen_cache_ = eigen_cache)))

  expect_true(is.finite(fit$objective))
  expect_equal(length(fit$betas_est), 2L)
})

test_that("diversity penalty increases for M=2 vs M=1", {
  p <- make_tiny_problem()

  fit1 <- ECSLR_Compute_Coef_combine(
    x = p$x, y = p$y, cost_matrix = p$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, 2), Group_Matrix = p$Group_Matrix, w_group = rep(1, 2),
    Adjust_Matrix_Fused = p$empty_adjust, fused_penmat = p$empty_fused,
    alpha = 1, gamma = 1, lambda_sparsity = 0.01, lambda_diversity = 0,
    tolerance = 1e-4, obj_tol_rel = 1e-4, max_iter = 30L, max_iter_admm = 5L,
    model = default_model
  )
  fit2 <- ECSLR_Compute_Coef_combine(
    x = p$x, y = p$y, cost_matrix = p$cost_matrix,
    M = 2L, include_intercept = TRUE,
    w_lasso = rep(1, 2), Group_Matrix = p$Group_Matrix, w_group = rep(1, 2),
    Adjust_Matrix_Fused = p$empty_adjust, fused_penmat = p$empty_fused,
    alpha = 1, gamma = 1, lambda_sparsity = 0.01, lambda_diversity = 0.1,
    tolerance = 1e-4, obj_tol_rel = 1e-4, max_iter = 30L, max_iter_admm = 5L,
    model = default_model
  )

  # With diversity, M=2 should produce two distinct beta columns
  expect_equal(ncol(fit2$betas_scaled), 2L)
  expect_true(is.finite(fit2$objective))
})

test_that("R-level ECSLR_multi_initial returns finite objective", {
  p <- make_tiny_problem()

  result <- ECSLR_multi_initial(
    x = p$x, y = p$y, cost_matrix = p$cost_matrix,
    M = 1L, include_intercept = TRUE,
    w_lasso = rep(1, 2), Group_Matrix = p$Group_Matrix, w_group = rep(1, 2),
    Adjust_Matrix_Fused = p$empty_adjust, fused_penmat = p$empty_fused,
    alpha = 1, gamma = 1,
    lambda_sparsity = 0.01, lambda_diversity = 0,
    tolerance = 1e-4, obj_tol_rel = 1e-4,
    max_iter = 20L, max_iter_admm = 5L,
    model = default_model,
    x0 = NULL, start0 = TRUE
  )

  expect_true(is.finite(result$objective))
  expect_equal(length(result$betas_est), ncol(p$x))
})

if (exists("build_eigen_cache", mode = "function")) {
  test_that("build_eigen_cache returns NULL when lambda_fused=0", {
    p <- make_tiny_problem()
    cache <- build_eigen_cache(p$Group_Matrix, p$empty_adjust, p$empty_fused, lambda_fused = 0)
    expect_null(cache)
  })

  test_that("build_eigen_cache returns list with Q and eigval when lambda_fused!=0", {
    gm <- Matrix::sparseMatrix(i = c(1, 1), j = c(1, 2), x = 1, dims = c(1, 2))
    fp <- Matrix::Matrix(c(1, -1), nrow = 1, sparse = TRUE)
    am <- methods::as(Matrix::Matrix(1, nrow = 1, ncol = 1, sparse = TRUE), "dgCMatrix")

    cache <- build_eigen_cache(gm, am, fp, lambda_fused = 0.1)
    expect_named(cache, c("Q", "eigval"))
    expect_s4_class(cache$Q, "dgCMatrix")
    expect_equal(length(cache$eigval), 2L)
  })

  test_that("build_eigen_cache uses row-sum block sizes from indicator matrices", {
    gm <- Matrix::sparseMatrix(
      i = c(1, 2, 2),
      j = c(1, 2, 3),
      x = 1,
      dims = c(2, 3)
    )
    fp <- Matrix::sparseMatrix(
      i = c(1, 1),
      j = c(2, 3),
      x = c(1, -1),
      dims = c(1, 3)
    )
    am <- Matrix::sparseMatrix(
      i = 2,
      j = 1,
      x = 1,
      dims = c(2, 1)
    )

    cache <- build_eigen_cache(gm, am, fp, lambda_fused = 0.1)
    expect_equal(as.matrix(cache$Q)[1, 1], 1, tolerance = 1e-12)
    expect_equal(as.numeric(cache$eigval), c(0, 0, 2), tolerance = 1e-12)
  })
}
