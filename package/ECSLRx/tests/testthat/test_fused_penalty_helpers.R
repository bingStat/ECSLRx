library(testthat)
library(Matrix)

context("Fused penalty helper construction")

test_that("construct_fused_penalty drops none blocks cleanly", {
  fused_type <- list(a = "none", b = "none")
  fused_par <- list(a = 1L, b = 2L)

  out <- construct_fused_penalty(fused_type, fused_par)

  expect_s4_class(out$pen.mat, "Matrix")
  expect_s4_class(out$Adjust_Matrix_Fused, "Matrix")
  expect_equal(dim(out$pen.mat), c(0L, 3L))
  expect_equal(dim(out$Adjust_Matrix_Fused), c(2L, 0L))
  expect_equal(rownames(out$Adjust_Matrix_Fused), names(fused_type))
})

test_that("construct_fused_penalty builds a standard 1D fused block", {
  fused_type <- list(a = "fuse1d")
  fused_par <- list(a = 4L)

  out <- construct_fused_penalty(fused_type, fused_par)

  expect_equal(dim(out$pen.mat), c(3L, 4L))
  expect_equal(dim(out$Adjust_Matrix_Fused), c(1L, 3L))
  expect_equal(rownames(out$Adjust_Matrix_Fused), names(fused_type))
  expect_equal(
    as.matrix(out$pen.mat),
    rbind(
      c(-1, 1, 0, 0),
      c(0, -1, 1, 0),
      c(0, 0, -1, 1)
    )
  )
  expect_equal(unname(as.matrix(out$Adjust_Matrix_Fused)), matrix(1, nrow = 1, ncol = 3L))
})

test_that("construct_fused_penalty builds a non-degenerate 2D fused block", {
  fused_type <- list(a = "fuse2d")
  fused_par <- list(a = c(2L, 3L))

  out <- construct_fused_penalty(fused_type, fused_par)

  expect_equal(dim(out$pen.mat), c(7L, 6L))
  expect_equal(dim(out$Adjust_Matrix_Fused), c(1L, 7L))
  expect_equal(rownames(out$Adjust_Matrix_Fused), names(fused_type))
  expect_equal(
    as.matrix(out$pen.mat),
    rbind(
      c(-1, 1, 0, 0, 0, 0),
      c(0, 0, -1, 1, 0, 0),
      c(0, 0, 0, 0, -1, 1),
      c(-1, 0, 1, 0, 0, 0),
      c(0, -1, 0, 1, 0, 0),
      c(0, 0, -1, 0, 1, 0),
      c(0, 0, 0, -1, 0, 1)
    )
  )
  expect_equal(unname(as.matrix(out$Adjust_Matrix_Fused)), matrix(1, nrow = 1, ncol = 7L))
})

test_that("construct_fused_penalty handles mixed fuse regimes", {
  # This fixture mixes three fuse regimes:
  # - a: no fusion, so the block should collapse to a zero row and be dropped
  # - b: standard 1D fusion over 3 ordered levels
  # - c: a 2D fuse request with one degenerate dimension, so it falls back to 1D on 4 levels
  fused_type <- list(a = "none", b = "fuse1d", c = "fuse2d")
  fused_par <- list(a = 1L, b = 3L, c = c(1L, 4L))

  out <- construct_fused_penalty(fused_type, fused_par)

  expect_s4_class(out$pen.mat, "Matrix")
  expect_s4_class(out$Adjust_Matrix_Fused, "Matrix")
  expect_equal(dim(out$pen.mat), c(5L, 8L))
  expect_equal(dim(out$Adjust_Matrix_Fused), c(3L, 5L))
  expect_equal(rownames(out$Adjust_Matrix_Fused), names(fused_type))
  expect_equal(as.matrix(out$pen.mat[1:2, 2:4]), rbind(c(-1, 1, 0), c(0, -1, 1)))
  expect_equal(as.matrix(out$pen.mat[3:5, 5:8]), rbind(c(-1, 1, 0, 0), c(0, -1, 1, 0), c(0, 0, -1, 1)))
})

test_that("custom_getD2dSparse falls back to 1D when a dimension is degenerate", {
  expect_equal(as.matrix(custom_getD2dSparse(1L, 4L)), as.matrix(genlasso::getD1dSparse(4L)))
})

test_that("construct_fused_penalty rejects unknown fuse types", {
  expect_error(
    construct_fused_penalty(list(a = "mystery"), list(a = 1L)),
    "Unknown fused penalty type"
  )
})
