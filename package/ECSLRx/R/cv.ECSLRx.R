cv.ECSLRx <- function(x, y, cost_matrix,
                      w_lasso = NULL,
                      Group_Matrix,
                      w_group,
                      Adjust_Matrix_Fused,
                      fused_penmat,
                      config = list(),
                      logfile = NULL) {
  default_config <- list(
    cv_method = "alternating",
    M = 10L,
    include_intercept = TRUE,
    alpha = 1 / 2,
    gamma = 2 / 3,
    n_lambda_sparsity = 20L,
    n_lambda_diversity = 20L,
    n_folds = 5L,
    tolerance = 1e-5,
    obj_tol_rel = 1e-5,
    max_iter = 1e3,
    max_iter_admm = 1e3,
    model = "ECSLRx",
    CV_criterion = "cost",
    start0 = TRUE,
    n_threads = 3L
  )
  config <- modifyList(default_config, config)
  if (is.null(w_lasso)) {
    w_lasso <- rep(1, ncol(x))
  }

  cv_method <- match.arg(config$cv_method, c("alternating", "random_search", "nested_sparsity"))
  if (!is.null(logfile)) {
    config$logfile <- logfile
  }

  switch(
    cv_method,
    alternating = {
      out <- CV.ECSLRx.alternating(
        x = x,
        y = y,
        cost_matrix = cost_matrix,
        Ms = config$M,
        include_intercept = config$include_intercept,
        w_lasso = w_lasso,
        Group_Matrix = Group_Matrix,
        w_group = w_group,
        Adjust_Matrix_Fused = Adjust_Matrix_Fused,
        fused_penmat = fused_penmat,
        alpha = config$alpha,
        gamma = config$gamma,
        n_lambda_sparsity = config$n_lambda_sparsity,
        n_lambda_diversity = config$n_lambda_diversity,
        n_folds = config$n_folds,
        tolerance = config$tolerance,
        obj_tol_rel = config$obj_tol_rel,
        max_iter = config$max_iter,
        max_iter_admm = config$max_iter_admm,
        model = config$model,
        CV_criterion = config$CV_criterion,
        start0 = config$start0,
        n_threads = config$n_threads
      )
      out$cv_method <- "alternating"
      out
    },
    random_search = {
      if (is.null(config$ParamList)) {
        stop("config$ParamList is required when cv_method = 'random_search'.")
      }
      RandomSearchCV.ECSLRx(
        x = x,
        y = y,
        cost_matrix = cost_matrix,
        w_lasso = w_lasso,
        Group_Matrix = Group_Matrix,
        w_group = w_group,
        Adjust_Matrix_Fused = Adjust_Matrix_Fused,
        fused_penmat = fused_penmat,
        include_intercept = config$include_intercept,
        n_folds = config$n_folds,
        ParamList = config$ParamList,
        n_params = if (is.null(config$n_params)) 50L else config$n_params,
        refit = if (is.null(config$refit)) TRUE else config$refit,
        nested_sparsity_search = FALSE,
        model = config$model,
        CV_criterion = config$CV_criterion,
        start0 = config$start0,
        n_threads = config$n_threads,
        tolerance = config$tolerance,
        obj_tol_rel = config$obj_tol_rel,
        max_iter = config$max_iter,
        max_iter_admm = config$max_iter_admm,
        seed = if (is.null(config$seed)) 1234L else config$seed,
        logfile = config$logfile
      )
    },
    nested_sparsity = {
      if (is.null(config$ParamList)) {
        stop("config$ParamList is required when cv_method = 'nested_sparsity'.")
      }
      RandomSearchCV.ECSLRx(
        x = x,
        y = y,
        cost_matrix = cost_matrix,
        w_lasso = w_lasso,
        Group_Matrix = Group_Matrix,
        w_group = w_group,
        Adjust_Matrix_Fused = Adjust_Matrix_Fused,
        fused_penmat = fused_penmat,
        include_intercept = config$include_intercept,
        n_folds = config$n_folds,
        ParamList = config$ParamList,
        n_params = if (is.null(config$n_params)) 50L else config$n_params,
        refit = if (is.null(config$refit)) TRUE else config$refit,
        nested_sparsity_search = TRUE,
        model = config$model,
        CV_criterion = config$CV_criterion,
        start0 = config$start0,
        n_threads = config$n_threads,
        tolerance = config$tolerance,
        obj_tol_rel = config$obj_tol_rel,
        max_iter = config$max_iter,
        max_iter_admm = config$max_iter_admm,
        seed = if (is.null(config$seed)) 1234L else config$seed,
        logfile = config$logfile
      )
    }
  )
}
