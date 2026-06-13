library(testthat)
library(Matrix)

context("Computational Log Generation")

test_that("Generate computational log for regression comparison", {
  # Determine correct path for writing the log
  log_path <- "test_run_computational_log.txt"
  if (dir.exists("tests")) {
    log_path <- "tests/test_run_computational_log.txt"
  } else if (dir.exists("../tests")) {
    log_path <- "../tests/test_run_computational_log.txt"
  }

  log_conn <- file(log_path, open = "w")
  on.exit(close(log_conn), add = TRUE)

  write_line <- function(...) {
    cat(paste0(...), "\n", file = log_conn, sep = "")
  }

  # Format floating-point numbers consistently
  fmt <- function(val) {
    if (is.matrix(val) || is(val, "Matrix")) {
      val <- as.matrix(val)
      paste(apply(val, 1, function(row) paste(sprintf("%.7f", row), collapse = ", ")), collapse = " | ")
    } else {
      paste(sprintf("%.7f", val), collapse = ", ")
    }
  }

  write_line("=== ECSLR Computational Log ===")
  write_line("Timestamp: 2026-06-06")

  # 1. Dataset Setup
  set.seed(999)
  n <- 25
  p <- 6
  x <- matrix(as.numeric(rnorm(n * p)), n, p)
  colnames(x) <- paste0("V", 1:p)
  y <- as.numeric(rbinom(n, 1, 0.5))
  cost_matrix <- matrix(0.0, n, 4)
  cost_matrix[, 2] <- as.numeric(runif(n, 3, 6)) # FN cost
  cost_matrix[, 3] <- as.numeric(runif(n, 1, 3)) # FP cost

  Group_Matrix <- methods::as(Matrix::Matrix(
    c(1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    nrow = 2, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")
  
  fused_penmat <- methods::as(Matrix::Matrix(
    c(1.0, -1.0,  0.0,  0.0,  0.0,  0.0,
      0.0,  1.0, -1.0,  0.0,  0.0,  0.0,
      0.0,  0.0,  0.0,  1.0, -1.0,  0.0,
      0.0,  0.0,  0.0,  0.0,  1.0, -1.0),
    nrow = 4, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")
  
  Adjust_Matrix_Fused <- methods::as(Matrix::Matrix(
    c(1.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 1.0),
    nrow = 2, ncol = 4, byrow = TRUE, sparse = TRUE
  ), "dgCMatrix")
  
  w_lasso <- as.numeric(c(1.0, 1.2, 0.8, 1.0, 1.1, 0.9))
  w_group <- as.numeric(c(1.0, 1.5))

  write_line("Setup Dimensions:")
  write_line("  n = ", n, ", p = ", p)
  write_line("  Group_Matrix dim: ", paste(dim(Group_Matrix), collapse = "x"))
  write_line("  fused_penmat dim: ", paste(dim(fused_penmat), collapse = "x"))
  write_line("  Adjust_Matrix_Fused dim: ", paste(dim(Adjust_Matrix_Fused), collapse = "x"))
  write_line("  y sum: ", sum(y))
  write_line("  cost_matrix mean (cols 2, 3): ", fmt(colMeans(cost_matrix[, 2:3])))

  # 2. C++ Helpers
  write_line("\n--- C++ Helpers ---")
  C_mult <- Matrix_Mult(x[1:5, ], matrix(as.numeric(1:24), 6, 4), 1L)
  write_line("Matrix_Mult sample (row 1): ", fmt(C_mult[1, ]))
  
  C_cross <- Matrix_crossprod(x[1:10, 1:3], x[1:10, 1:3], 1L)
  write_line("Matrix_crossprod: ", fmt(C_cross))

  scores_sig <- sigmoid(c(-2.5, -0.5, 0, 0.5, 2.5))
  write_line("sigmoid: ", fmt(scores_sig))

  soft_res <- Soft(c(-1.5, -0.5, 0, 0.5, 1.5), c(0.8, 0.8, 0.8, 0.8, 0.8))
  write_line("Soft: ", fmt(soft_res))

  group_norm_res <- group_norm(c(2.0, -3.0, 1.0, 0.5, 1.2, -0.8), Group_Matrix)
  write_line("group_norm: ", fmt(group_norm_res))

  group_soft_res <- group_soft_thresh(c(2.0, -3.0, 1.0, 0.5, 1.2, -0.8), c(1.0, 1.2), Group_Matrix)
  write_line("group_soft_thresh: ", fmt(group_soft_res))

  # 3. Penalties & Objective
  write_line("\n--- Penalties and Objective Functions ---")
  betas <- matrix(c(0.5, -0.4, 0.8, -0.2, 0.1, -0.6,
                    -0.3, 0.7, -0.5, 0.4, 0.2, 0.1), nrow = p, ncol = 2)
  intercept <- matrix(c(0.15, -0.25), nrow = 1, ncol = 2)

  # Check AEC cost on mock scores
  mock_scores <- sigmoid(as.numeric(x %*% betas[, 1] + intercept[1, 1]))
  aec_val <- Compute_AEC(mock_scores, y, cost_matrix)
  write_line("Compute_AEC: ", fmt(aec_val))

  pen_lasso_m0 <- Compute_penalty_sparsity_Lasso(0L, betas, w_lasso, 0.25)
  pen_lasso_m1 <- Compute_penalty_sparsity_Lasso(1L, betas, w_lasso, 0.25)
  write_line("Compute_penalty_sparsity_Lasso (m=0): ", fmt(pen_lasso_m0))
  write_line("Compute_penalty_sparsity_Lasso (m=1): ", fmt(pen_lasso_m1))

  pen_group_m0 <- Compute_penalty_sparsity_groupLasso(0L, betas, Group_Matrix, w_group, 0.3)
  pen_group_m1 <- Compute_penalty_sparsity_groupLasso(1L, betas, Group_Matrix, w_group, 0.3)
  write_line("Compute_penalty_sparsity_groupLasso (m=0): ", fmt(pen_group_m0))
  write_line("Compute_penalty_sparsity_groupLasso (m=1): ", fmt(pen_group_m1))

  pen_fused_m0 <- Compute_penalty_fused(0L, betas, fused_penmat, 0.15)
  pen_fused_m1 <- Compute_penalty_fused(1L, betas, fused_penmat, 0.15)
  write_line("Compute_penalty_fused (m=0): ", fmt(pen_fused_m0))
  write_line("Compute_penalty_fused (m=1): ", fmt(pen_fused_m1))

  # Check diversity penalty (only if diversity group penalty exists or matches)
  pen_div_m0 <- Compute_penalty_diversity_group(0L, betas, Group_Matrix, w_group, 0.2)
  pen_div_m1 <- Compute_penalty_diversity_group(1L, betas, Group_Matrix, w_group, 0.2)
  write_line("Compute_penalty_diversity_group (m=0): ", fmt(pen_div_m0))
  write_line("Compute_penalty_diversity_group (m=1): ", fmt(pen_div_m1))

  obj_m0 <- Compute_objective_multi(
    0L, x, y, cost_matrix,
    0.25, 0.3, 0.15, 0.2,
    w_lasso, w_group, Group_Matrix, fused_penmat,
    betas, intercept
  )
  write_line("Compute_objective_multi (m=0): ", fmt(obj_m0))

  obj_ensemble <- Compute_objective_ensemble_multi(
    2L, x, y, cost_matrix,
    0.25, 0.3, 0.15, 0.2,
    w_lasso, w_group, Group_Matrix, fused_penmat,
    betas, intercept
  )
  write_line("Compute_objective_ensemble_multi: ", fmt(obj_ensemble))

  # 4. ADMM Updates Trajectory
  write_line("\n--- ADMM Updates Trajectory ---")
  vv <- c(0.8, -0.3, 0.5, -0.6, 0.2, 0.4)
  betas_old <- rep(0.0, p)
  A_mat <- as.matrix(Matrix::t(fused_penmat) %*% fused_penmat)
  eig <- eigen(A_mat)
  ord <- order(eig$values)
  eigval <- eig$values[ord]
  Q_mat <- methods::as(eig$vectors[, ord], "dgCMatrix")
  numfused <- as.numeric(Adjust_Matrix_Fused %*% rep(1.0, 4))
  df_vec <- as.numeric(Group_Matrix %*% rep(1.0, p))

  lambda_lasso <- 0.15
  lambda_fused <- 0.08
  group_soft_penalty <- c(0.12, 0.18)
  eta_m <- c(1.2, 1.2)

  betas_current <- betas_old
  for (step in 1:5) {
    lambda_1w <- rep(0.0, p)
    lambda_2 <- rep(0.0, 2)
    lambda_3 <- rep(0.0, 2)
    
    betas_current <- update_betas(
      vv = vv,
      betas_old = betas_current,
      Group_Matrix = Group_Matrix,
      Adjust_Matrix_Fused = Adjust_Matrix_Fused,
      fused_penmat = fused_penmat,
      Q = Q_mat,
      eigval = eigval,
      numfused = numfused,
      df = df_vec,
      max_iter_admm = 10L,
      lambda_lasso = lambda_lasso,
      w_lasso = w_lasso,
      lambda_fused = lambda_fused,
      group_soft_penalty = group_soft_penalty,
      eta_m = eta_m,
      exist_grouplasso = TRUE,
      lambda_1w = lambda_1w,
      lambda_2 = lambda_2,
      lambda_3 = lambda_3
    )
    write_line("  ADMM update step ", step, ": ", fmt(betas_current))
  }

  default_model <- if (exists("compute_block_eigen", mode = "function")) "ECSLRx" else "ECSLRmulti"
  eigen_cache <- build_eigen_cache(
    Group_Matrix, Adjust_Matrix_Fused, fused_penmat,
    lambda_fused = 0.08 * (1 - 0.6)
  )

  # 5. Core Solver
  write_line("\n--- Core Solver ECSLR_Compute_Coef_combine ---")
  fit_solver <- ECSLR_Compute_Coef_combine(
    x = x, y = y, cost_matrix = cost_matrix,
    M = 2L, include_intercept = TRUE,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix, w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused, fused_penmat = fused_penmat,
    alpha = 0.4, gamma = 0.6,
    lambda_sparsity = 0.08, lambda_diversity = 0.03,
    tolerance = 1e-4, obj_tol_rel = 1e-4, max_iter = 15L, max_iter_admm = 8L,
    model = default_model,
    eigen_cache_ = eigen_cache
  )

  write_line("Solver objective: ", fmt(fit_solver$objective))
  write_line("Solver intercept_est: ", fmt(fit_solver$intercept_est))
  write_line("Solver betas_est: ", fmt(fit_solver$betas_est))
  write_line("Solver betas_scaled col 1: ", fmt(fit_solver$betas_scaled[, 1]))
  write_line("Solver betas_scaled col 2: ", fmt(fit_solver$betas_scaled[, 2]))
  write_line("Solver intercept_scaled: ", fmt(fit_solver$intercept_scaled))

  # 6. High-level Grid Search
  write_line("\n--- High-level Grid Search CV.ECSLR.alternating ---")
  cv_func_name <- if (exists("CV.ECSLRx.alternating", mode = "function")) "CV.ECSLRx.alternating" else "CV.ECSLR.alternating"
  cv_func <- get(cv_func_name)
  cv_fit <- cv_func(
    x = x, y = y, cost_matrix = cost_matrix,
    Ms = c(1L, 2L), include_intercept = TRUE,
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix, w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused, fused_penmat = fused_penmat,
    alpha = 0.4, gamma = 0.6,
    n_lambda_sparsity = 3L, n_lambda_diversity = 3L, n_folds = 2L,
    tolerance = 1e-4, obj_tol_rel = 1e-4, max_iter = 8L, max_iter_admm = 5L,
    model = default_model, CV_criterion = "cost", start0 = TRUE, n_threads = 1L
  )

  write_line("CV optimal M: ", cv_fit$M)
  write_line("CV optimal lambda_sparsity_opt: ", fmt(cv_fit$lambda_sparsity_opt))
  write_line("CV optimal lambda_diversity_opt: ", fmt(cv_fit$lambda_diversity_opt))
  write_line("CV model objective: ", fmt(cv_fit$ECSLR_model_opt$objective))
  write_line("CV model betas_est: ", fmt(cv_fit$ECSLR_model_opt$betas_est))

  expect_true(file.exists(log_path))
})
