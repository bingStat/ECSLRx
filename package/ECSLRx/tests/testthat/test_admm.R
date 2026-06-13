# test_admm.R  --  Numerical correctness tests for admm_flasso (OPT-1)

library(testthat)
library(Matrix)

# Helper to dynamically get Q and eigval (OPT-3 fallback with diagonal safety check)
get_Q_eigval <- function(Group_Matrix, Adjust_Matrix_Fused, fused_penmat) {
  if (exists("compute_block_eigen", mode = "function")) {
    return(compute_block_eigen(Group_Matrix, Adjust_Matrix_Fused, fused_penmat))
  } else {
    A <- as.matrix(Matrix::t(fused_penmat) %*% fused_penmat)
    # Check if A is diagonal (off-diagonals are all zero) to avoid degenerate eigen rotation
    if (nrow(A) > 1 && all(A[row(A) != col(A)] == 0)) {
      Q <- methods::as(Matrix::Diagonal(nrow(A)), "dgCMatrix")
      eigval <- diag(A)
      return(list(Q = Q, eigval = eigval))
    }
    eig <- eigen(A)
    ord <- order(eig$values)
    eigval <- eig$values[ord]
    Q <- methods::as(eig$vectors[, ord], "dgCMatrix")
    return(list(Q = Q, eigval = eigval))
  }
}

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

# Shared ADMM fixture ----------------------------------------------------------
make_fused_setup <- function(p = 4) {
  # 1 group containing all p variables (within-group fusion setup)
  Group_Matrix <- methods::as(Matrix::Matrix(1, nrow = 1, ncol = p, sparse = TRUE), "dgCMatrix")
  # Fused penalty: consecutive differences
  D <- matrix(0, nrow = p - 1, ncol = p)
  for (i in seq_len(p - 1)) { D[i, i] <- 1; D[i, i + 1] <- -1 }
  fused_penmat <- methods::as(Matrix::Matrix(D, sparse = TRUE), "dgCMatrix")
  # Adjust_Matrix_Fused: map all p-1 fused pairs to the 1 group
  am_vals <- matrix(1, nrow = 1, ncol = p - 1)
  Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(am_vals, sparse = TRUE), "dgCMatrix")

  eigen_res <- get_Q_eigval(Group_Matrix, Adjust_Matrix_Fused, fused_penmat)
  Q    <- eigen_res$Q
  eigval <- eigen_res$eigval
  numfused <- as.numeric(Adjust_Matrix_Fused %*% rep(1, p - 1))
  df       <- as.numeric(Group_Matrix %*% rep(1, p))

  list(p = p, Group_Matrix = Group_Matrix,
       Adjust_Matrix_Fused = Adjust_Matrix_Fused,
       fused_penmat = fused_penmat,
       Q = Q, eigval = eigval,
       numfused = numfused, df = df)
}

test_that("update_betas with lambda_fused=0 returns vv unchanged (no ADMM called)", {
  s <- make_fused_setup(4)
  vv <- c(1.5, -0.5, 0.8, -1.2)

  out <- update_betas(
    vv = vv,
    betas_old = rep(0, s$p),
    Group_Matrix = s$Group_Matrix,
    Adjust_Matrix_Fused = s$Adjust_Matrix_Fused,
    fused_penmat = s$fused_penmat,
    Q = s$Q, eigval = s$eigval,
    numfused = s$numfused,
    df = s$df,
    max_iter_admm = 5L,
    lambda_lasso = 0,
    w_lasso = rep(1, s$p),
    lambda_fused = 0,          # <-- zero: admm_flasso is NOT called
    group_soft_penalty = rep(0, s$p),
    eta_m = rep(1, s$p),
    exist_grouplasso = FALSE,
    lambda_1w = rep(0, s$p),
    lambda_2 = rep(0, s$p),
    lambda_3 = rep(0, s$p)
  )

  expect_equal(as.numeric(out), vv, tolerance = 1e-14)
})

test_that("admm_flasso with very large lambda_3 strongly reduces consecutive differences", {
  s <- make_fused_setup(4)
  vv <- c(3.0, -3.0, 3.0, -3.0)  # alternating, max consecutive diffs
  lambda_3 <- rep(1000, nrow(s$Group_Matrix))  # huge fused penalty

  out <- call_admm_flasso(
    vv = vv,
    Group_Matrix = s$Group_Matrix,
    Adjust_Matrix_Fused = s$Adjust_Matrix_Fused,
    lambda_3 = lambda_3,
    fused_penmat = s$fused_penmat,
    Q = s$Q, eigval = s$eigval,
    max_iter_admm = 2000L,
    betas_old = rep(0, s$p),
    numfused = s$numfused,
    df = s$df,
    iter_count = 0L
  )

  # With huge lambda, consecutive differences should be much smaller than input
  total_diff_in  <- sum(abs(diff(vv)))
  total_diff_out <- sum(abs(diff(as.numeric(out))))
  expect_lt(total_diff_out, total_diff_in * 0.1)  # at least 90% reduction
})

test_that("admm_flasso decreases fused penalty relative to input", {
  s <- make_fused_setup(6)
  # vv has large differences between consecutive entries
  vv <- c(3, -3, 3, -3, 3, -3)
  lambda_3 <- rep(1.0, nrow(s$Group_Matrix))

  out <- call_admm_flasso(
    vv = vv,
    Group_Matrix = s$Group_Matrix,
    Adjust_Matrix_Fused = s$Adjust_Matrix_Fused,
    lambda_3 = lambda_3,
    fused_penmat = s$fused_penmat,
    Q = s$Q, eigval = s$eigval,
    max_iter_admm = 500L,
    betas_old = rep(0, s$p),
    numfused = s$numfused,
    df = s$df,
    iter_count = 0L
  )

  fused_penalty_in  <- sum(abs(diff(vv)))
  fused_penalty_out <- sum(abs(diff(as.numeric(out))))
  expect_lt(fused_penalty_out, fused_penalty_in)
})

test_that("admm_flasso output is close to soft-thresh when fused structure is identity", {
  # When fused_penmat is diagonal (each predictor fused only with itself),
  # the problem reduces to element-wise soft thresholding.
  p <- 4
  Group_Matrix <- methods::as(Matrix::Diagonal(p), "dgCMatrix")
  fused_penmat <- methods::as(Matrix::Diagonal(p), "dgCMatrix")
  Adjust_Matrix_Fused <- methods::as(Matrix::Diagonal(p), "dgCMatrix")

  eigen_res <- get_Q_eigval(Group_Matrix, Adjust_Matrix_Fused, fused_penmat)
  Q    <- eigen_res$Q
  eigval <- eigen_res$eigval
  numfused <- rep(1, p)
  df       <- rep(1, p)

  vv       <- c(2, -1.5, 0.3, 3)
  lam      <- 0.5
  lambda_3 <- rep(lam, p)

  out <- call_admm_flasso(
    vv = vv,
    Group_Matrix = Group_Matrix,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    lambda_3 = lambda_3,
    fused_penmat = fused_penmat,
    Q = Q, eigval = eigval,
    max_iter_admm = 1000L,
    betas_old = rep(0, p),
    numfused = numfused,
    df = df,
    iter_count = 0L
  )

  # Compare to direct soft-thresholding (admm should converge close to this)
  expected <- as.numeric(Soft(vv, rep(lam, p)))
  expect_equal(as.numeric(out), expected, tolerance = 1e-4)
})
