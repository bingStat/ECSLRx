library(testthat)
library(Matrix)

context("Cost and penalty calculations")

test_that("cost and penalty functions match hand calculations", {
  x <- matrix(c(0, 1, 1, 0), nrow = 2)
  y <- c(0, 1)
  cost_matrix <- matrix(c(
    0, 4, 2, 0,
    0, 3, 1, 0
  ), nrow = 2, byrow = TRUE)
  betas <- matrix(c(1, -2, 2, 1), nrow = 2)
  intercept <- matrix(c(0.1, -0.2), nrow = 1)
  Group_Matrix <- methods::as(Matrix::Diagonal(2), "dgCMatrix")
  fused_penmat <- Matrix::Matrix(c(1, -1), nrow = 1, sparse = TRUE)
  w <- c(1, 2)

  scores <- c(0.25, 0.75)
  expected_aec <- mean(c(0.25 * 4 + 0.75 * 0, 0.75 * 0 + 0.25 * 1))
  expect_equal(Compute_AEC(scores, y, cost_matrix), expected_aec)

  expect_equal(Compute_penalty_sparsity_Lasso(0L, betas, w, 0.5), 2.5)
  expect_equal(Compute_penalty_sparsity_groupLasso(0L, betas, Group_Matrix, w, 0.5), 2.5)
  expect_equal(Compute_penalty_fused(0L, betas, fused_penmat, 0.5), 1.5)
  expect_equal(Compute_penalty_diversity_group(0L, betas, Group_Matrix, w, 0.5), 2.5)

  obj1 <- Compute_objective_multi(
    0L, x, y, cost_matrix,
    0.5, 0.5, 0.5, 0.5,
    w, w, Group_Matrix, fused_penmat,
    betas, intercept
  )
  manual_scores <- sigmoid(as.numeric(x %*% betas[, 1] + intercept[1, 1]))
  manual_obj <- Compute_AEC(manual_scores, y, cost_matrix) + 2.5 + 2.5 + 1.5 + 2.5
  expect_equal(obj1, manual_obj, tolerance = 1e-12)
})

test_that("ensemble objective equals sum of component objectives except diversity is not double-counted", {
  x <- matrix(c(0, 1, 1, 0), nrow = 2)
  y <- c(0, 1)
  cost_matrix <- matrix(c(0, 4, 2, 0, 0, 3, 1, 0), nrow = 2, byrow = TRUE)
  betas <- matrix(c(1, -2, 2, 1), nrow = 2)
  intercept <- matrix(c(0.1, -0.2), nrow = 1)
  Group_Matrix <- methods::as(Matrix::Diagonal(2), "dgCMatrix")
  fused_penmat <- Matrix::Matrix(c(1, -1), nrow = 1, sparse = TRUE)
  w <- c(1, 2)

  ensemble_obj <- Compute_objective_ensemble_multi(
    2L, x, y, cost_matrix,
    0.5, 0.5, 0.5, 0.5,
    w, w, Group_Matrix, fused_penmat,
    betas, intercept
  )
  expected <- sum(vapply(0:1, function(m) {
    Compute_objective_multi(
      m, x, y, cost_matrix,
      0.5, 0.5, 0.5, 0.5,
      w, w, Group_Matrix, fused_penmat,
      betas, intercept
    ) - 0.5 * Compute_penalty_diversity_group(m, betas, Group_Matrix, w, 0.5)
  }, numeric(1)))

  expect_equal(ensemble_obj, expected, tolerance = 1e-12)
})
