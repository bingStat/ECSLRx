# Fitting logic for paper comparator models: logit, cslogit, CSRF-cv, CSRP-cv.

# Fit standard logistic regression and evaluate cost-sensitive metrics.
fit_logit_model <- function(model, x.train, y.train, x.test, y.test, cost_matrix_train, cost_matrix_test, Group_Matrix) {
  train <- data.frame(x.train, class = y.train)

  timing_result <- system.time({
    lr <- glm(class ~ ., data = train, family = "binomial")
  })

  betas_est <- lr$coefficients[-1]
  betas_est[is.na(betas_est)] <- 0
  model.coef <- list(intercept_est = lr$coefficients[1], betas_est = betas_est, model = model)
  SPDI <- Calculate_SP_DI(betas_est, Group_Matrix)
  res_cm <- generate_metrics(model.coef, x.train, y.train, cost_matrix_train, x.test, y.test, cost_matrix_test)

  list(
    timing_result = timing_result,
    output = lr,
    model.coef = model.coef,
    SPDI = SPDI,
    res_cm = res_cm
  )
}

# Fit cost-sensitive sparse logistic regression with lambda CV via cslogit.
fit_cslogit_model <- function(model, x.train, y.train, x.test, y.test, cost_matrix_train, cost_matrix_test,
                              n_lambda_sparsity, n_folds, include_intercept, Group_Matrix) {
  library(cslogit)
  n <- nrow(x.train)
  p <- ncol(x.train)
  cost_matrix2 <- matrix(nrow = n, ncol = 2)
  cost_matrix2[, 1] <- ifelse(y.train == 1, cost_matrix_train[, 4], cost_matrix_train[, 1])
  cost_matrix2[, 2] <- ifelse(y.train == 1, cost_matrix_train[, 3], cost_matrix_train[, 2])

  timing_result <- system.time({
    mu_x <- apply(x.train, 2, mean)
    sd_x <- apply(x.train, 2, sd)
    x_std <- scale(x.train, center = mu_x, scale = sd_x)
    x_std[is.na(x_std)] <- 0
    train_std <- data.frame(x_std, class = y.train)

    lambda_path <- c(0, exp(seq(log(1e-05), log(1), length.out = n_lambda_sparsity)))

    cv_results <- tryCatch(
      cv.cslogit(
        formula = factor(class) ~ .,
        data = train_std,
        cost_matrix = cost_matrix2,
        nfolds = n_folds,
        lambda_path = lambda_path,
        seed = 2020,
        options = list(algorithm = "MMA")
      ),
      error = function(e) {
        cv.cslogit(
          formula = factor(class) ~ .,
          data = train_std,
          cost_matrix = cost_matrix2,
          nfolds = n_folds,
          lambda_path = lambda_path,
          seed = 2020,
          options = list(algorithm = "MMA", start = rep(0, p + 1), check_data = FALSE)
        )
      }
    )
    optimal_lambda <- cv_results$optimal_lambda

    cslogit.output <- tryCatch(
      cslogit(
        formula = factor(class) ~ .,
        data = train_std,
        cost_matrix = cost_matrix2,
        lambda = optimal_lambda,
        options = list(algorithm = "MMA")
      ),
      error = function(e) {
        cslogit(
          formula = factor(class) ~ .,
          data = train_std,
          cost_matrix = cost_matrix2,
          lambda = optimal_lambda,
          options = list(algorithm = "MMA", start = rep(0, p + 1), check_data = FALSE)
        )
      }
    )
  })

  betas <- cslogit.output$coefficients[-1]
  intercept <- cslogit.output$coefficients[1]
  betas_scaled <- betas / sd_x
  betas_scaled[is.na(betas_scaled) | is.infinite(betas_scaled)] <- 0
  intercept_scaled <- ifelse(include_intercept, 1, 0) * (intercept - mu_x %*% betas_scaled)

  model.coef <- list(
    intercept_est = as.numeric(intercept_scaled),
    betas_est = as.numeric(betas_scaled),
    lambda_sparsity = optimal_lambda,
    model = model
  )
  SPDI <- Calculate_SP_DI(betas_scaled, Group_Matrix)
  res_cm <- generate_metrics(model.coef, x.train, y.train, cost_matrix_train, x.test, y.test, cost_matrix_test)

  list(
    timing_result = timing_result,
    output = cslogit.output,
    model.coef = model.coef,
    SPDI = SPDI,
    res_cm = res_cm
  )
}

# Score hard class predictions from CSRF/CSRP without coefficient estimates.
calculate_metrics_rf <- function(pre.class.test, y.test, cost_matrix_test) {
  confusionM <- list()
  confusionM[["direct-classification"]] <- tryCatch(
    confusionMatrix(
      factor(pre.class.test, levels = c(0, 1)),
      factor(y.test, levels = c(0, 1)),
      positive = "1",
      mode = "prec_recall"
    ),
    error = function(e) list(overall = "NA", byClass = "NA")
  )

  pre.class.test <- as.numeric(as.character(pre.class.test))
  cost_without_algorithm.test <- cost_without_algorithm(cost_matrix_test, y.test)
  cost.test <- cost_with_algorithm(cost_matrix_test, as.numeric(y.test), as.numeric(pre.class.test))
  savings.test <- 1 - cost.test / cost_without_algorithm.test

  res <- data.frame(
    LogDeviance = NA,
    AUPRC = NA,
    AUROC = NA,
    MCC = tryCatch(cor(y.test, pre.class.test), error = function(e) NA),
    cost_without_algorithm = cost_without_algorithm.test,
    AEC = cost.test,
    ExpectedSavings = savings.test,
    MinCost = cost.test,
    opt.threshold.criteria = "direct-classification",
    Accuracy = as.numeric(confusionM[["direct-classification"]]$overall["Accuracy"]),
    Precision = as.numeric(confusionM[["direct-classification"]]$byClass["Precision"]),
    Recall = as.numeric(confusionM[["direct-classification"]]$byClass["Recall"]),
    F1 = as.numeric(confusionM[["direct-classification"]]$byClass["F1"]),
    AverageCost = cost.test,
    savings = savings.test
  )

  list(res = res, confusionM = confusionM)
}

# Fit CSRF or CSRP via Python (reticulate) with cross-validated hyperparameters.
fit_rf_csrf_model <- function(model, x.train, y.train, x.test, y.test, cost_matrix_train, cost_matrix_test,
                              M, n_folds, verbose, logfile) {
  library(reticulate)

  cost_mat_train <- cbind(cost_matrix_train[, 2], cost_matrix_train[, 3], cost_matrix_train[, 4], cost_matrix_train[, 1])
  cost_mat_test <- cbind(cost_matrix_test[, 2], cost_matrix_test[, 3], cost_matrix_test[, 4], cost_matrix_test[, 1])

  system_info <- Sys.info()[1]
  if (system_info == "Windows") {
    use_condaenv("py39", required = TRUE)
    py_run_string("import gc\ngc.collect()\n\nimport numpy as np\nimport sys\nimport joblib\nsys.modules['sklearn.externals.joblib'] = joblib")
    py_run_string("import six; sys.modules['sklearn.externals.six'] = six; sys.modules['sklearn.externals.six.moves'] = six.moves")
    py_run_string("import sklearn.ensemble; sys.modules['sklearn.ensemble.base'] = sklearn.ensemble._base")
  } else {
    use_python("/vsc-hard-mounts/leuven-data/356/vsc35603/miniconda3/envs/Renv/bin/", required = TRUE)
    py_run_string("import gc; gc.collect()")
  }

  py_run_string(sprintf("
import sys
f = open('%s', 'a', buffering=1)
sys.stdout = f
sys.stderr = f
sys.stdout.flush()
", logfile, logfile))

  py$x_train <- x.train
  py$y_train <- y.train
  py$cost_mat_train <- cost_mat_train
  py$x_test <- x.test
  py$y_test <- y.test
  py$cost_mat_test <- cost_mat_test
  py$verbose <- verbose
  py$modelname <- if (grepl("CSRF", model)) "CSRF" else "CSRP"
  py$n_folds <- n_folds
  py$n_iter <- 50L

  timing_result <- system.time({
    source_python("cvCSRF.py")
  })

  pre.class.test <- py$pre_class_test
  pre.prob.test <- py$pre_prob_test[, 2]
  pre.prob.train <- py$pre_prob_train[, 2]

  output <- list(cv_results_final = py$cv_results, M = py$best_params$n_estimators)
  model.coef <- py$best_params
  model.coef$model <- model
  SPDI <- list(SP = NA, SPfinal = NA, DI = NA, n_active_items = NA, Group_SPfinal = NA)

  res_cm <- calculate_metrics_rf(pre.class.test, y.test, cost_matrix_test)
  res.main <- generate_out_costmatrix(y.test, pre.prob.test, cost_matrix_test, criteria = c("ID-cost", "balance"), y.train, pre.prob.train)
  resCost.train <- generate_out_costmatrix(y.train, pre.prob.train, cost_matrix_train, criteria = "(Insample) ID-cost", y.train, pre.prob.train)
  res_cm$res <- rbind(res_cm$res, resCost.train$res, res.main$res)

  list(
    res_cm = res_cm,
    model = model,
    model.coef = model.coef,
    output = output,
    SPDI = SPDI,
    timing_result = timing_result
  )
}
