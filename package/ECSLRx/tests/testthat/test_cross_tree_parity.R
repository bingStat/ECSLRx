# test_cross_tree_parity.R -- Cross-tree behavior parity checks

library(testthat)
library(Matrix)

skip_on_cran()
skip_if_not_installed("callr")
skip_if_not_installed("pkgload")
skip_if_not_installed("caret")

locate_pkg_root <- function(start = getwd()) {
  path <- normalizePath(start, winslash = "/", mustWork = TRUE)

  repeat {
    desc <- file.path(path, "DESCRIPTION")
    if (file.exists(desc)) {
      lines <- readLines(desc, warn = FALSE)
      if (any(grepl("^Package:\\s*ECSLRx\\s*$", lines))) {
        return(path)
      }
    }

    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("Could not locate an ECSLRx package root from ", start)
    }
    path <- parent
  }
}

current_pkg_root <- locate_pkg_root()
other_pkg_root <- NA_character_
can_compare_trees <- TRUE

if (grepl("/package/ECSLRx-dev$", current_pkg_root)) {
  other_pkg_root <- sub("/package/ECSLRx-dev$", "/.worktrees/public/package/ECSLRx", current_pkg_root)
} else if (grepl("/\\.worktrees/public/package/ECSLRx$", current_pkg_root)) {
  other_pkg_root <- sub("/\\.worktrees/public/package/ECSLRx$", "/package/ECSLRx-dev", current_pkg_root)
} else {
  can_compare_trees <- FALSE
}

run_in_pkg <- function(pkg_root, expr_fun) {
  callr::r(
    function(pkg_root, expr_fun) {
      pkgload::load_all(pkg_root, quiet = TRUE, export_all = TRUE, attach = FALSE)
      expr_fun()
    },
    args = list(pkg_root = pkg_root, expr_fun = expr_fun),
    spinner = FALSE
  )
}

make_cache_fixture <- function() {
  group_matrix <- methods::as(Matrix::Matrix(
    c(1, 1, 0,
      0, 0, 1),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  adjust_matrix_fused <- methods::as(Matrix::Matrix(
    c(1, 0,
      0, 1),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  fused_penmat <- methods::as(Matrix::Matrix(
    c(1, -1, 0,
      0, 0, 1),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  list(
    group_matrix = group_matrix,
    adjust_matrix_fused = adjust_matrix_fused,
    fused_penmat = fused_penmat
  )
}

make_solver_fixture <- function() {
  x <- matrix(c(
    -1, -1,
    -0.5, 0,
     0.5, 0,
     1, 1
  ), ncol = 2, byrow = TRUE)
  y <- c(0, 0, 1, 1)
  cost_matrix <- matrix(
    c(0, 2, 1, 0,
      0, 2, 1, 0,
      0, 2, 1, 0,
      0, 2, 1, 0),
    nrow = 4, byrow = TRUE
  )
  group_matrix <- methods::as(Matrix::Diagonal(2), "dgCMatrix")
  empty_fused <- Matrix::Matrix(0, nrow = 0, ncol = 2, sparse = TRUE)
  empty_adjust <- Matrix::Matrix(0, nrow = 2, ncol = 0, sparse = TRUE)

  list(
    x = x,
    y = y,
    cost_matrix = cost_matrix,
    group_matrix = group_matrix,
    empty_fused = empty_fused,
    empty_adjust = empty_adjust
  )
}

make_cv_fixture <- function() {
  set.seed(42)
  n <- 20
  p <- 4
  x <- matrix(as.numeric(rnorm(n * p)), n, p)
  colnames(x) <- paste0("V", 1:p)
  y <- as.numeric(rbinom(n, 1, 0.4))
  cost_matrix <- matrix(0.0, n, 4)
  cost_matrix[, 2] <- as.numeric(runif(n, 2, 4))
  cost_matrix[, 3] <- as.numeric(runif(n, 1, 2))

  group_matrix <- methods::as(Matrix::Matrix(
    c(1.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 1.0),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  fused_penmat <- methods::as(Matrix::Matrix(
    c(1.0, -1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, -1.0),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  adjust_matrix_fused <- methods::as(Matrix::Matrix(
    c(1.0, 0.0,
      0.0, 1.0),
    nrow = 2, ncol = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")

  list(
    x = x,
    y = y,
    cost_matrix = cost_matrix,
    group_matrix = group_matrix,
    fused_penmat = fused_penmat,
    adjust_matrix_fused = adjust_matrix_fused
  )
}

test_that("build_eigen_cache matches across package trees", {
  skip_if_not(can_compare_trees, "Could not infer the sibling ECSLRx package tree.")

  fixture <- make_cache_fixture()

  current_res <- run_in_pkg(current_pkg_root, function() {
    pkg_fn <- getExportedValue("ECSLRx", "build_eigen_cache")
    res <- pkg_fn(
      fixture$group_matrix,
      fixture$adjust_matrix_fused,
      fixture$fused_penmat,
      lambda_fused = 0.1
    )
    list(Q = as.matrix(res$Q), eigval = as.numeric(res$eigval))
  })

  other_res <- run_in_pkg(other_pkg_root, function() {
    pkg_fn <- getExportedValue("ECSLRx", "build_eigen_cache")
    res <- pkg_fn(
      fixture$group_matrix,
      fixture$adjust_matrix_fused,
      fixture$fused_penmat,
      lambda_fused = 0.1
    )
    list(Q = as.matrix(res$Q), eigval = as.numeric(res$eigval))
  })

  expect_equal(current_res, other_res, tolerance = 1e-12)
})

test_that("ECSLR_Compute_Coef_combine matches across package trees", {
  skip_if_not(can_compare_trees, "Could not infer the sibling ECSLRx package tree.")

  fixture <- make_solver_fixture()

  current_res <- run_in_pkg(current_pkg_root, function() {
    pkg_fn <- getExportedValue("ECSLRx", "ECSLR_Compute_Coef_combine")
    fit <- do.call(pkg_fn, list(
      x = fixture$x,
      y = fixture$y,
      cost_matrix = fixture$cost_matrix,
      M = 1L,
      include_intercept = TRUE,
      w_lasso = rep(1, ncol(fixture$x)),
      Group_Matrix = fixture$group_matrix,
      w_group = rep(1, nrow(fixture$group_matrix)),
      Adjust_Matrix_Fused = fixture$empty_adjust,
      fused_penmat = fixture$empty_fused,
      alpha = 1,
      gamma = 1,
      lambda_sparsity = 0.01,
      lambda_diversity = 0,
      tolerance = 1e-4,
      obj_tol_rel = 1e-4,
      max_iter = 20L,
      max_iter_admm = 5L,
      model = "ECSLRx"
    ))
    list(
      betas_est = fit$betas_est,
      intercept_est = fit$intercept_est,
      objective = fit$objective
    )
  })

  other_res <- run_in_pkg(other_pkg_root, function() {
    pkg_fn <- getExportedValue("ECSLRx", "ECSLR_Compute_Coef_combine")
    fit <- do.call(pkg_fn, list(
      x = fixture$x,
      y = fixture$y,
      cost_matrix = fixture$cost_matrix,
      M = 1L,
      include_intercept = TRUE,
      w_lasso = rep(1, ncol(fixture$x)),
      Group_Matrix = fixture$group_matrix,
      w_group = rep(1, nrow(fixture$group_matrix)),
      Adjust_Matrix_Fused = fixture$empty_adjust,
      fused_penmat = fixture$empty_fused,
      alpha = 1,
      gamma = 1,
      lambda_sparsity = 0.01,
      lambda_diversity = 0,
      tolerance = 1e-4,
      obj_tol_rel = 1e-4,
      max_iter = 20L,
      max_iter_admm = 5L,
      model = "ECSLRx"
    ))
    list(
      betas_est = fit$betas_est,
      intercept_est = fit$intercept_est,
      objective = fit$objective
    )
  })

  expect_equal(current_res, other_res, tolerance = 1e-8)
})

test_that("alternating CV wrapper and direct function agree across package trees", {
  skip_if_not(can_compare_trees, "Could not infer the sibling ECSLRx package tree.")

  fixture <- make_cv_fixture()

  run_cv <- function(pkg_root) {
    run_in_pkg(pkg_root, function() {
      is_new_tree <- grepl("/\\.worktrees/public/package/ECSLRx$", normalizePath(pkg_root, winslash = "/", mustWork = TRUE))
      cv_fn <- if (is_new_tree) {
        getExportedValue("ECSLRx", "cv.ECSLRx")
      } else {
        getExportedValue("ECSLRx", "CV.ECSLRx.alternating")
      }

      if (is_new_tree) {
        out <- cv_fn(
          fixture$x,
          fixture$y,
          fixture$cost_matrix,
          w_lasso = rep(1, ncol(fixture$x)),
          Group_Matrix = fixture$group_matrix,
          w_group = rep(1, nrow(fixture$group_matrix)),
          Adjust_Matrix_Fused = fixture$adjust_matrix_fused,
          fused_penmat = fixture$fused_penmat,
          config = list(
            cv_method = "alternating",
            M = c(1L, 2L),
            include_intercept = TRUE,
            alpha = 0.5,
            gamma = 0.5,
            n_lambda_sparsity = 3L,
            n_lambda_diversity = 3L,
            n_folds = 2L,
            tolerance = 1e-3,
            obj_tol_rel = 1e-3,
            max_iter = 5L,
            max_iter_admm = 3L,
            model = "ECSLRx",
            CV_criterion = "cost",
            start0 = TRUE,
            n_threads = 1L
          )
        )
      } else {
        out <- cv_fn(
          x = fixture$x,
          y = fixture$y,
          cost_matrix = fixture$cost_matrix,
          Ms = c(1L, 2L),
          include_intercept = TRUE,
          w_lasso = rep(1, ncol(fixture$x)),
          Group_Matrix = fixture$group_matrix,
          w_group = rep(1, nrow(fixture$group_matrix)),
          Adjust_Matrix_Fused = fixture$adjust_matrix_fused,
          fused_penmat = fixture$fused_penmat,
          alpha = 0.5,
          gamma = 0.5,
          n_lambda_sparsity = 3L,
          n_lambda_diversity = 3L,
          n_folds = 2L,
          tolerance = 1e-3,
          obj_tol_rel = 1e-3,
          max_iter = 5L,
          max_iter_admm = 3L,
          model = "ECSLRx",
          CV_criterion = "cost",
          start0 = TRUE,
          n_threads = 1L
        )
      }

      list(
        M = out$M,
        alpha = out$alpha,
        gamma = out$gamma,
        lambda_sparsity_opt = out$lambda_sparsity_opt,
        lambda_diversity_opt = out$lambda_diversity_opt,
        cv_opt_new = out$cv_opt_new,
        Optimal_Index = out$Optimal_Index,
        betas_est = out$ECSLR_model_opt$betas_est,
        intercept_est = out$ECSLR_model_opt$intercept_est,
        objective = out$ECSLR_model_opt$objective,
        model = out$model
      )
    })
  }

  current_res <- run_cv(current_pkg_root)
  other_res <- run_cv(other_pkg_root)

  expect_equal(current_res, other_res, tolerance = 1e-10)
})
