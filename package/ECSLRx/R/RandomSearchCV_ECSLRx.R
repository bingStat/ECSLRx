sample_ecslrx_param_grid <- function(param_grid, n_params) {
  dplyr::sample_n(param_grid, size = min(as.integer(n_params)[1L], nrow(param_grid)))
}

build_ecslrx_initial_state <- function(p, M, intercept_initial) {
  M <- as.integer(M)
  list(
    betas = matrix(0, nrow = p, ncol = M),
    intercept = matrix(intercept_initial, nrow = 1, ncol = M)
  )
}

run_ecslrx_sparsity_path <- function(lambda_sparsity_values,
                                     x_train, y_train, cost_train,
                                     x_validation, y_validation, cost_validation,
                                     p, M, alpha, gamma, lambda_diversity,
                                     intercept_initial,
                                     include_intercept,
                                     w_lasso,
                                     Group_Matrix, w_group,
                                     Adjust_Matrix_Fused, fused_penmat,
                                     tolerance, obj_tol_rel, max_iter, max_iter_admm,
                                     model, start0) {
  M <- as.integer(M)
  alpha <- as.numeric(alpha)
  gamma <- as.numeric(gamma)
  lambda_diversity <- as.numeric(lambda_diversity)

  x0 <- list(
    betas = matrix(0, nrow = p, ncol = M),
    intercept = matrix(intercept_initial, nrow = 1, ncol = M)
  )
  fold_cost_without_algorithm <- ECSLRx::cost_without_algorithm(cost_validation, y_validation)
  cv_rows <- vector("list", length(lambda_sparsity_values))
  ECSLR_model_path <- vector("list", length(lambda_sparsity_values))

  for (i_lambda in seq_along(lambda_sparsity_values)) {
    lambda_sparsity <- as.numeric(lambda_sparsity_values[i_lambda])

    timing_result <- system.time({
      ECSLR_model_path[[i_lambda]] <- ECSLRx::ECSLR_multi_initial(
        x_train, y_train, cost_train,
        M, include_intercept,
        w_lasso,
        Group_Matrix, w_group,
        Adjust_Matrix_Fused, fused_penmat,
        alpha, gamma,
        lambda_sparsity,
        lambda_diversity,
        tolerance, obj_tol_rel, max_iter, max_iter_admm,
        model, x0 = x0, start0 = start0
      )
    })

    x0 <- ECSLR_model_path[[i_lambda]]$coef_x_std
    aec_cost <- ECSLRx::Compute_aec_cost(
      ECSLR_model_path[[i_lambda]]$intercept_est,
      ECSLR_model_path[[i_lambda]]$betas_est,
      x_validation, y_validation,
      cost_validation, model
    )

    cv_rows[[i_lambda]] <- data.frame(
      alpha = alpha,
      gamma = gamma,
      M = M,
      lambda_sparsity = lambda_sparsity,
      lambda_diversity = lambda_diversity,
      insample_obj = ECSLR_model_path[[i_lambda]]$objective,
      AEC = aec_cost$AEC,
      cost = aec_cost$cost,
      cost_without_algorithm = fold_cost_without_algorithm,
      savings = 1 - aec_cost$cost / fold_cost_without_algorithm,
      elapsed_time = as.numeric(timing_result[3])
    )
  }

  cv_temp <- dplyr::bind_rows(cv_rows)
  print(cv_temp)
  list(path_results = cv_temp, ECSLR_model_path = ECSLR_model_path)
}

match_ecslrx_lambda_index <- function(lambda_sparsity_grid, lambda_sparsity_value) {
  exact_match <- which(lambda_sparsity_grid == lambda_sparsity_value)[1L]
  if (!is.na(exact_match)) {
    return(exact_match)
  }
  which.min(abs(lambda_sparsity_grid - lambda_sparsity_value))
}

resolve_penalty_args <- function(w_lasso, Group_Matrix, w_group,
                                 Adjust_Matrix_Fused, fused_penmat,
                                 model_parameters = NULL) {
  if (!is.null(model_parameters)) {
    w_lasso <- model_parameters$w_lasso
    Group_Matrix <- model_parameters$Group_Matrix
    w_group <- model_parameters$w_group
    Adjust_Matrix_Fused <- model_parameters$Adjust_Matrix_Fused
    fused_penmat <- model_parameters$fused_penmat
  }
  list(
    w_lasso = w_lasso,
    Group_Matrix = Group_Matrix,
    w_group = w_group,
    Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    fused_penmat = fused_penmat
  )
}

RandomSearchCV.ECSLRx <- function(x, y, cost_matrix,
                                  w_lasso = NULL,
                                  Group_Matrix, w_group,
                                  Adjust_Matrix_Fused, fused_penmat,
                                  include_intercept = TRUE,
                                  n_folds = 3L,
                                  ParamList,
                                  n_params = 50L,
                                  refit = TRUE,
                                  nested_sparsity_search = FALSE,
                                  model = "ECSLRx",
                                  CV_criterion = "cost",
                                  start0 = TRUE,
                                  n_threads = 4L,
                                  tolerance = 1e-5,
                                  obj_tol_rel = 1e-5,
                                  max_iter = 1e3,
                                  max_iter_admm = 1e3,
                                  seed = 1234L,
                                  logfile = NULL,
                                  model_parameters = NULL) {
  penalty <- resolve_penalty_args(
    w_lasso, Group_Matrix, w_group,
    Adjust_Matrix_Fused, fused_penmat,
    model_parameters = model_parameters
  )
  w_lasso <- penalty$w_lasso
  Group_Matrix <- penalty$Group_Matrix
  w_group <- penalty$w_group
  Adjust_Matrix_Fused <- penalty$Adjust_Matrix_Fused
  fused_penmat <- penalty$fused_penmat

  set.seed(seed)

  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  p <- ncol(x)
  if (is.null(w_lasso)) {
    w_lasso <- rep(1, p)
  }

  ParamList <- as.data.frame(ParamList)
  lambda_sparsity_grid <- unique(ParamList$lambda_sparsity)

  if (as.integer(n_params)[1L] == -1L) {
    if (nested_sparsity_search) {
      base_grid <- unique(ParamList[, setdiff(names(ParamList), "lambda_sparsity"), drop = FALSE])
      n_params <- nrow(base_grid) * length(lambda_sparsity_grid)
    } else {
      n_params <- nrow(ParamList)
    }
  }

  if (nested_sparsity_search) {
    base_grid <- unique(ParamList[, setdiff(names(ParamList), "lambda_sparsity"), drop = FALSE])
    ParamSet <- sample_ecslrx_param_grid(base_grid, ceiling(n_params / length(lambda_sparsity_grid)))
  } else {
    ParamSet <- sample_ecslrx_param_grid(ParamList, n_params)
  }
  n_params <- nrow(ParamSet)
  print(ParamSet)

  sample_ind <- seq_len(n)
  folds_index <- tryCatch({
    amount <- exp(x[, "logamount"]) - 1
    generateTrainIndex(y, amount, k = n_folds, times = 1L)
  }, error = function(e) {
    caret::createMultiFolds(as.factor(y), k = n_folds, times = 1L)
  })

  intercept_initial <- log(mean(y) / (1 - mean(y)))

  n_threads <- min(as.integer(n_threads), n_folds * n_params)
  n_threads <- max(1L, n_threads)

  cls <- parallel::makeCluster(n_threads)
  on.exit(if (exists("cls") && !is.null(cls)) parallel::stopCluster(cls), add = TRUE)
  doParallel::registerDoParallel(cls)

  cv_results_full <- foreach::foreach(
    fold = seq_len(n_folds),
    .packages = c("ECSLRx", "dplyr"),
    .export = c(
      "run_ecslrx_sparsity_path", "nested_sparsity_search",
      "lambda_sparsity_grid", "ParamSet", "sample_ind", "folds_index",
      "x", "y", "cost_matrix", "p", "intercept_initial",
      "include_intercept", "w_lasso", "Group_Matrix", "w_group",
      "Adjust_Matrix_Fused", "fused_penmat", "tolerance", "obj_tol_rel",
      "max_iter", "max_iter_admm", "model", "start0", "logfile"
    ),
    .combine = "rbind"
  ) %:%
    foreach::foreach(i_params = seq_len(n_params), .combine = "rbind") %dopar% {
      if (!is.null(logfile)) {
        worker_logfile <- sub("\\.txt$", paste0("_pid", Sys.getpid(), ".txt"), logfile)
        sink(worker_logfile, append = TRUE)
        on.exit(sink(), add = TRUE)
        cat("fold = ", fold, "i_params=", i_params, "\n")
      }

      train.index <- folds_index[[fold]]
      validation.index <- setdiff(sample_ind, train.index)

      param_row <- ParamSet[i_params, , drop = FALSE]
      lambda_sparsity_values <- if (nested_sparsity_search) {
        rev(lambda_sparsity_grid)
      } else {
        param_row$lambda_sparsity
      }

      res_path <- run_ecslrx_sparsity_path(
        lambda_sparsity_values = lambda_sparsity_values,
        x_train = x[train.index, , drop = FALSE],
        y_train = y[train.index],
        cost_train = cost_matrix[train.index, , drop = FALSE],
        x_validation = x[validation.index, , drop = FALSE],
        y_validation = y[validation.index],
        cost_validation = cost_matrix[validation.index, , drop = FALSE],
        p = p,
        M = param_row$M,
        alpha = param_row$alpha,
        gamma = param_row$gamma,
        lambda_diversity = param_row$lambda_diversity,
        intercept_initial = intercept_initial,
        include_intercept = include_intercept,
        w_lasso = w_lasso,
        Group_Matrix = Group_Matrix,
        w_group = w_group,
        Adjust_Matrix_Fused = Adjust_Matrix_Fused,
        fused_penmat = fused_penmat,
        tolerance = tolerance,
        obj_tol_rel = obj_tol_rel,
        max_iter = max_iter,
        max_iter_admm = max_iter_admm,
        model = model,
        start0 = start0
      )$path_results

      cbind(i_params = i_params, fold = fold, res_path)
    }

  parallel::stopCluster(cls)
  cls <- NULL

  ranking_column <- if (tolower(CV_criterion) == "aec") "cv_AEC" else "cv_cost"
  cv_results <- cv_results_full %>%
    dplyr::group_by(alpha, gamma, M, lambda_sparsity, lambda_diversity) %>%
    dplyr::summarise(
      cv_insample_obj = mean(insample_obj, na.rm = TRUE),
      cv_AEC = mean(AEC, na.rm = TRUE),
      cv_cost = mean(cost, na.rm = TRUE),
      cv_cost_without_algorithm = mean(cost_without_algorithm, na.rm = TRUE),
      cv_savings = mean(savings, na.rm = TRUE),
      mean_cv_elapsed_time = mean(elapsed_time, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(.data[[ranking_column]])

  print(cv_results, n = nrow(cv_results), width = Inf)
  opt_params <- cv_results[1, , drop = FALSE]

  output <- list(
    M = opt_params$M,
    alpha = opt_params$alpha,
    gamma = opt_params$gamma,
    lambda_diversity_opt = opt_params$lambda_diversity,
    lambda_sparsity_opt = opt_params$lambda_sparsity,
    cv_results_final = cv_results,
    model = model,
    cv_method = if (nested_sparsity_search) "nested_sparsity" else "random_search"
  )

  if (refit) {
    optimal_index <- match_ecslrx_lambda_index(lambda_sparsity_grid, opt_params$lambda_sparsity)
    sparsity_path_index <- seq(from = length(lambda_sparsity_grid), to = optimal_index)
    refit_path <- run_ecslrx_sparsity_path(
      lambda_sparsity_values = lambda_sparsity_grid[sparsity_path_index],
      x_train = x,
      y_train = y,
      cost_train = cost_matrix,
      x_validation = x,
      y_validation = y,
      cost_validation = cost_matrix,
      p = p,
      M = opt_params$M,
      alpha = opt_params$alpha,
      gamma = opt_params$gamma,
      lambda_diversity = opt_params$lambda_diversity,
      intercept_initial = intercept_initial,
      include_intercept = include_intercept,
      w_lasso = w_lasso,
      Group_Matrix = Group_Matrix,
      w_group = w_group,
      Adjust_Matrix_Fused = Adjust_Matrix_Fused,
      fused_penmat = fused_penmat,
      tolerance = tolerance,
      obj_tol_rel = obj_tol_rel,
      max_iter = max_iter,
      max_iter_admm = max_iter_admm,
      model = model,
      start0 = start0
    )
    ECSLR_model_full <- vector("list", length(lambda_sparsity_grid))
    ECSLR_model_full[sparsity_path_index] <- refit_path$ECSLR_model_path
    insample_results <- data.frame(
      M = as.integer(opt_params$M),
      alpha = as.numeric(opt_params$alpha),
      gamma = as.numeric(opt_params$gamma),
      lambda_sparsity = lambda_sparsity_grid,
      lambda_diversity_opt = as.numeric(opt_params$lambda_diversity),
      insample_obj = NA_real_,
      insample_AEC = NA_real_,
      insample_cost = NA_real_
    )
    insample_results$insample_obj[sparsity_path_index] <- refit_path$path_results$insample_obj
    insample_results$insample_AEC[sparsity_path_index] <- refit_path$path_results$AEC
    insample_results$insample_cost[sparsity_path_index] <- refit_path$path_results$cost
    cat("insample_results: \n")
    print(insample_results)
    output$Optimal_Index_Sparsity <- optimal_index
    output$ECSLR_model_opt <- ECSLR_model_full[[optimal_index]]
    output$ECSLR_model_full <- ECSLR_model_full
    output$insample_results <- insample_results
  }

  output
}

RandomSearchCV_ECSLRmulti <- RandomSearchCV.ECSLRx

RandomSearchCV_improved_ECSLRmulti <- function(...) {
  RandomSearchCV.ECSLRx(..., nested_sparsity_search = TRUE)
}
