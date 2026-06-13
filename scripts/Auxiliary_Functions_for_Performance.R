# Performance evaluation for paper comparator models:
# cslogit, CSRF-cv, CSRP-cv, ECSLRx.

library(cslogit)
library(caret)
library(pROC)
library(PRROC)

source(file.path(if (exists("exp_dir")) exp_dir else ".", "fit_models.R"))

# Fit ECSLRx with cross-validated tuning and return metrics plus timing.
fit_ECSLRx <- function(x.train, y.train, x.test, y.test,
                       cost_matrix_train, cost_matrix_test,
                       model_parameters, config, logfile = NULL) {
  library(ECSLRx, lib.loc = package_lib)

  cat("alpha=", config$alpha, ", gamma=", config$gamma, "\n")
  cat("cv_method=", config$cv_method, "\n")

  timing_result <- system.time({
    output <- cv.ECSLRx(
      x.train, y.train, cost_matrix_train,
      w_lasso = NULL,
      Group_Matrix = model_parameters$Group_Matrix,
      w_group = model_parameters$w_group,
      Adjust_Matrix_Fused = model_parameters$Adjust_Matrix_Fused,
      fused_penmat = model_parameters$fused_penmat,
      config = config,
      logfile = logfile
    )
  })
  model.coef <- output$ECSLR_model_opt
  SPDI <- Calculate_SP_DI(model.coef$betas_scaled, model_parameters$Group_Matrix)
  res_cm <- generate_metrics(model.coef, x.train, y.train, cost_matrix_train,
                             x.test, y.test, cost_matrix_test)

  list(
    timing_result = timing_result,
    output = output,
    model.coef = model.coef,
    SPDI = SPDI,
    res_cm = res_cm
  )
}

# Fit a comparator model from a built config; dispatch by config$model.
fit_model <- function(model, model_parameters, config, x.train, y.train, x.test, y.test,
                      cost_matrix_train, cost_matrix_test,
                      logfile = NULL, verbose = TRUE) {
  switch(
    model,
    cslogit = fit_cslogit_model(model, x.train, y.train, x.test, y.test,
                                cost_matrix_train, cost_matrix_test,
                                config$n_lambda_sparsity, config$n_folds,
                                config$include_intercept, model_parameters$Group_Matrix),
    "CSRF-cv" = fit_rf_csrf_model("CSRF-cv", x.train, y.train, x.test, y.test,
                                  cost_matrix_train, cost_matrix_test,
                                  config$M, config$n_folds, verbose, logfile),
    "CSRP-cv" = fit_rf_csrf_model("CSRP-cv", x.train, y.train, x.test, y.test,
                                  cost_matrix_train, cost_matrix_test,
                                  config$M, config$n_folds, verbose, logfile),
    ECSLRx = fit_ECSLRx(x.train, y.train, x.test, y.test,
                        cost_matrix_train, cost_matrix_test,
                        model_parameters, config, logfile),
    stop("Unsupported model: ", model)
  )
}

# Fit one comparator model and return standardized performance output.
generate_performance3 <- function(model, x.train, y.train, x.test, y.test, cost_matrix_train, cost_matrix_test,
                                  model_parameters = list(), config, logfile = NULL, verbose = TRUE) {
  supported <- c("cslogit", "CSRF-cv", "CSRP-cv", "ECSLRx")
  if (!model %in% supported) {
    stop("Unknown model: ", model, ". Supported: ", paste(supported, collapse = ", "))
  }

  x.train <- as.matrix(x.train)
  x.test <- as.matrix(x.test)
  y.train <- as.numeric(as.character(y.train))
  y.test <- as.numeric(as.character(y.test))

  cat("model:", model, "\n")
  cat("inner CV n_folds=", config$n_folds, "\n")

  if (model == "ECSLRx") {
    p <- ncol(x.train)
    group_cols_n <- ncol(model_parameters$Group_Matrix)
    if (!identical(p, group_cols_n)) {
      stop(
        "ECSLRx feature/penalty mismatch: ncol(x.train)=", p,
        ", ncol(Group_Matrix)=", group_cols_n,
        ". Check application preprocessing and cache invalidation."
      )
    }
  }

  fit_res <- fit_model(model, model_parameters, config, x.train, y.train, x.test, y.test,
                       cost_matrix_train, cost_matrix_test, logfile, verbose)

  timing_result <- fit_res$timing_result
  elapsed_time <- as.numeric(timing_result[3])
  cat("time result: \n")
  print(timing_result)

  list(
    res = fit_res$res_cm$res,
    model = model,
    model.coef = fit_res$model.coef,
    output = fit_res$output,
    elapsed_time = elapsed_time,
    SPDI = fit_res$SPDI,
    model_parameters = model_parameters,
    config = config
  )
}

# Compute sparsity (SP), diversity (DI), and grouped sparsity summaries from coefficients.
Calculate_SP_DI <- function(betas, Group_Matrix = NULL) {
  if (is.null(dim(betas))) {
    betas <- matrix(betas, ncol = 1)
  }
  sp <- apply(betas == 0, 2, mean)
  SP <- mean(sp)
  betas_final <- apply(betas, 1, mean)
  SPfinal <- mean(betas_final == 0)

  if (!is.null(Group_Matrix)) {
    betas_norm2s <- Group_Matrix %*% (betas_final^2)
    n_active_items <- sum(betas_norm2s != 0)
    Group_SPfinal <- 1 - n_active_items / nrow(Group_Matrix)
  } else {
    n_active_items <- sum(betas_final != 0)
    Group_SPfinal <- SPfinal
  }

  di <- apply(betas == 0, 1, mean)
  DI <- mean(di[di != 1])
  list(SP = SP, SPfinal = SPfinal, DI = DI, n_active_items = n_active_items, Group_SPfinal = Group_SPfinal)
}

# Score train/test predictions with cost-sensitive and probabilistic metrics.
generate_metrics <- function(model.coef, x.train, y.train, cost_matrix_train, x.test, y.test, cost_matrix_test) {
  pre.prob.train <- as.numeric(sigmoid(model.coef$intercept_est + x.train %*% model.coef$betas_est))
  pre.prob.test <- as.numeric(sigmoid(model.coef$intercept_est + x.test %*% model.coef$betas_est))
  res.main <- generate_out_costmatrix(y.test, pre.prob.test, cost_matrix_test, criteria = c("ID-cost", "balance"), y.train, pre.prob.train)
  resCost.train <- generate_out_costmatrix(y.train, pre.prob.train, cost_matrix_train, criteria = "(Insample) ID-cost", y.train, pre.prob.train)

  res <- rbind(resCost.train$res, res.main$res)
  confusionM <- c(res.main$confusionM, resCost.train$confusionM)
  list(res = res, confusionM = confusionM)
}

# Evaluate predictions under one or more threshold criteria and cost matrices.
generate_out_costmatrix <- function(y.test, pre.prob.test, cost_matrix_test, criteria = "ID-cost",
                                    y.train = NA, pre.prob.train = NA) {
  LogDeviance <- Compute_LD(pre.prob.test, y.test)
  cost_without_algorithm.test <- cost_without_algorithm(cost_matrix_test, y.test)
  AEC.test <- Compute_AEC(pre.prob.test, y.test, cost_matrix_test)
  ExpectedSavings <- (cost_without_algorithm.test - AEC.test) / cost_without_algorithm.test

  AUPRC.test <- tryCatch({
    prcurve.test <- PRROC::pr.curve(scores.class0 = pre.prob.test, weights.class0 = y.test, curve = FALSE)
    prcurve.test$auc.integral
  }, error = function(e) NA)

  AUROC.test <- tryCatch({
    suppressMessages(pROC::auc(y.test, pre.prob.test))
  }, error = function(e) NA)

  MCC.test <- calculate_max_correlation(y.test, pre.prob.test)
  minCost.test <- calculate_min_cost(y.test, pre.prob.test, cost_matrix_test)

  metrics1 <- data.frame(
    LogDeviance = LogDeviance,
    AUPRC = AUPRC.test,
    AUROC = AUROC.test,
    MCC = MCC.test,
    cost_without_algorithm = cost_without_algorithm.test,
    AEC = AEC.test,
    ExpectedSavings = ExpectedSavings,
    MinCost = minCost.test
  )

  nc <- length(criteria)
  metrics2 <- data.frame(
    opt.threshold.criteria = rep(NA, nc),
    Accuracy = rep(NA, nc),
    Precision = rep(NA, nc),
    Recall = rep(NA, nc),
    F1 = rep(NA, nc),
    AverageCost = rep(NA, nc),
    savings = rep(NA, nc)
  )
  confusionM <- list()
  cc <- 1
  for (crit in criteria) {
    opt_threshold <- calculate_opt_threshold(pre.prob.train, y.train, cost_matrix_test, crit)
    pre.class.test <- ifelse(pre.prob.test > opt_threshold, 1, 0)

    confusionM[[crit]] <- tryCatch(
      confusionMatrix(
        factor(pre.class.test, levels = c(0, 1)),
        factor(y.test, levels = c(0, 1)),
        positive = "1",
        mode = "prec_recall"
      ),
      error = function(e) list(overall = "NA", byClass = "NA")
    )

    cost.test <- cost_with_algorithm(cost_matrix_test, as.numeric(y.test), as.numeric(pre.class.test))
    savings.test <- 1 - cost.test / cost_without_algorithm.test

    metrics2[cc, ] <- data.frame(
      opt.threshold.criteria = crit,
      Accuracy = as.numeric(confusionM[[crit]]$overall["Accuracy"]),
      Precision = as.numeric(confusionM[[crit]]$byClass["Precision"]),
      Recall = as.numeric(confusionM[[crit]]$byClass["Recall"]),
      F1 = as.numeric(confusionM[[crit]]$byClass["F1"]),
      AverageCost = cost.test,
      savings = savings.test
    )
    cc <- cc + 1
  }

  res <- cbind(do.call("rbind", replicate(nrow(metrics2), metrics1, simplify = FALSE)), metrics2)
  list(res = res, confusionM = confusionM)
}

# Maximize Pearson correlation between labels and thresholded predicted probabilities.
calculate_max_correlation <- function(true_labels, pred_probs) {
  sorted_probs <- sort(unique(pred_probs))
  if (length(sorted_probs) >= 2) {
    thresholds <- (head(sorted_probs, -1) + tail(sorted_probs, -1)) / 2
    correlations <- sapply(thresholds, function(th) {
      cor(true_labels, ifelse(pred_probs >= th, 1, 0))
    })
    max(correlations, na.rm = TRUE)
  } else {
    NA
  }
}

# Find the lowest achievable average misclassification cost over probability thresholds.
calculate_min_cost <- function(true_labels, pred_probs, cost_matrix) {
  sorted_probs <- c(0, sort(unique(pred_probs)), 1)
  thresholds <- (head(sorted_probs, -1) + tail(sorted_probs, -1)) / 2
  costs <- sapply(thresholds, function(th) {
    Compute_AEC(ifelse(pred_probs >= th, 1, 0), true_labels, cost_matrix)
  })
  min(costs, na.rm = TRUE)
}

# Choose a classification threshold for a named criterion (F1, ID-cost, balance, etc.).
calculate_opt_threshold <- function(pre.prob.train, y.train, cost_matrix_test, criterion = NA) {
  if (criterion == "F1") {
    tryCatch({
      prcurve <- PRROC::pr.curve(scores.class0 = pre.prob.train, weights.class0 = y.train, curve = TRUE)
      f1_scores <- 2 * (prcurve$curve[, 2] * prcurve$curve[, 1]) / (prcurve$curve[, 2] + prcurve$curve[, 1])
      prcurve$curve[, 3][which.max(f1_scores)]
    }, error = function(e) mean(pre.prob.train))
  } else if (grepl("ID-cost", criterion)) {
    instance_dependent_cost_threshold(cost_matrix_test)
  } else if (criterion == "balance") {
    0.5
  } else if (criterion == "imbalance") {
    mean(y.train)
  }
}

# Average misclassification cost for fixed class predictions.
cost_with_algorithm <- function(cost_matrix, labels, predictions) {
  Compute_AEC(predictions, labels, cost_matrix)
}

# Best constant-all-negative or all-positive baseline cost on the test set.
cost_without_algorithm <- function(cost_matrix, labels) {
  cost_neg <- Compute_AEC(rep(0, length(labels)), labels, cost_matrix)
  cost_pos <- Compute_AEC(rep(1, length(labels)), labels, cost_matrix)
  min(cost_neg, cost_pos)
}

# Mean log loss (logistic deviance) for predicted probabilities.
Compute_LD <- function(predictions, y) {
  -mean(y * predictions - log(1 + exp(predictions)))
}
