# Build block-diagonal fused-lasso penalty and adjustment matrices from feature metadata.
construct_fused_penalty <- function(fused_type, fused_par) {
  pen.mat.list <- lapply(seq_along(fused_type), function(j) {
    switch(fused_type[[j]],
      none = matrix(0, nrow = 1, ncol = fused_par[[j]]),
      fuse1d = genlasso::getD1dSparse(fused_par[[j]]),
      fuse2d = custom_getD2dSparse(fused_par[[j]][1], fused_par[[j]][2]),
      fuse4gragh = genlasso::getDgSparse(fused_par[[j]]),
      stop("Unknown fused penalty type: ", fused_type[[j]])
    )
  })

  index.fused.list <- lapply(pen.mat.list, function(mat) {
    if (nrow(mat) == 1 && all(mat == 0)) {
      0
    } else {
      matrix(1, nrow = 1, ncol = nrow(mat))
    }
  })

  pen.mat <- Matrix::bdiag(pen.mat.list)
  Adjust_Matrix_Fused <- Matrix::bdiag(index.fused.list)
  pen.mat <- pen.mat[Matrix::rowSums(abs(pen.mat)) != 0, , drop = FALSE]
  Adjust_Matrix_Fused <- Adjust_Matrix_Fused[, Matrix::colSums(Adjust_Matrix_Fused) != 0, drop = FALSE]
  rownames(Adjust_Matrix_Fused) <- names(fused_type)

  list(pen.mat = pen.mat, Adjust_Matrix_Fused = Adjust_Matrix_Fused)
}

# Return a 1D or 2D fusion difference matrix, falling back to 1D when a grid is degenerate.
custom_getD2dSparse <- function(a, b) {
  if (a == 1 || b == 1) {
    genlasso::getD1dSparse(a * b)
  } else {
    genlasso::getD2dSparse(a, b)
  }
}
