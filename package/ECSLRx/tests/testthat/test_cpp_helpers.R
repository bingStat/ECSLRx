library(testthat)
library(Matrix)

context("C++ math and matrix helper functions")

test_that("matrix helpers match base R", {
  A <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  B <- matrix(c(2, 0, -1, 3, 1, 4), nrow = 3)

  expect_equal(Matrix_Mult(A, B, 1L), A %*% B)
  expect_equal(Matrix_crossprod(A, A, 1L), crossprod(A, A))
})

test_that("soft-thresholding and sigmoid are numerically correct", {
  z <- c(-3, -0.5, 0, 0.5, 3)
  gamma <- c(1, 1, 1, 0.25, 4)

  expect_equal(as.numeric(Soft(z, gamma)), c(-2, 0, 0, 0.25, 0))
  expect_equal(as.numeric(sigmoid(c(-Inf, 0, Inf))), c(0, 0.5, 1))
})

test_that("group norms and group soft-thresholding use group structure", {
  Group_Matrix <- Matrix::Matrix(
    c(1, 1, 0,
      0, 0, 1),
    nrow = 2,
    byrow = TRUE,
    sparse = TRUE
  )
  vv <- c(3, 4, 2)

  expect_equal(as.numeric(group_norm(vv, Group_Matrix)), c(5, 2))
  expect_equal(
    as.numeric(group_soft_thresh(vv, c(1, 3), Group_Matrix)),
    c(2.4, 3.2, 0),
    tolerance = 1e-12
  )
})
