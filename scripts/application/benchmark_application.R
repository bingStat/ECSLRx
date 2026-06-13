# Application benchmark: six paper cases, comparator + ECSLRx models.
# Usage: Rscript benchmark_application.R <casenum> [k ...]
#   e.g. Rscript benchmark_application.R 1 1 2 3

script_dir <- normalizePath(getwd(), winslash = "/")
repo_root <- normalizePath(file.path(script_dir, "../.."), winslash = "/")
exp_dir <- file.path(repo_root, "scripts")
savedir <- file.path(repo_root, "runs")
datadir <- file.path(savedir, "data_application")
package_lib <- Sys.getenv("APPLICATION_PACKAGE_LIB", file.path(repo_root, "Rlib"))
if (!nzchar(package_lib) || !dir.exists(package_lib)) {
  stop("Repository package library not found: ", package_lib)
}

library(ECSLRx, lib.loc = package_lib)
library(foreach)

source(file.path(exp_dir, "Auxiliary_Functions_for_Performance.R"))

# Flatten a 3-D instance-dependent cost array to a 4-column matrix.
flatten_cost_matrix <- function(cm3) {
  cbind(cm3[, 1, 1], cm3[, 2, 1], cm3[, 1, 2], cm3[, 2, 2])
}

# Assemble one application benchmark result row (metrics + coefficients).
append_performance_row <- function(resAll, model, meta) {
  is_tree <- model %in% c("CSRF-cv", "CSRP-cv")
  is_ecslrx <- model == "ECSLRx"

  beta_est <- if (is_tree) {
    data.frame(matrix(NA, 1, meta$p.train))
  } else {
    data.frame(t(resAll$model.coef$betas_est))
  }
  colnames(beta_est) <- colnames(meta$x.train)

  cbind(
    data.frame(
      case = meta$case, N = meta$N, k = meta$k,
      J = meta$J, p = meta$p.train, n.train = meta$n.train, n.test = meta$n.test,
      p.minority.real = meta$p.minority.real,
      model = model, resAll$res, elapsed_time = resAll$elapsed_time
    ),
    M = if (is_tree || is_ecslrx) resAll$output$M else NA_integer_,
    lambda_sparsity = if (is_ecslrx) resAll$model.coef$lambda_sparsity else NA,
    lambda_diversity = if (is_ecslrx) resAll$model.coef$lambda_diversity else NA,
    SP = resAll$SPDI$SP,
    SPfinal = resAll$SPDI$SPfinal,
    Group_SPfinal = resAll$SPDI$Group_SPfinal,
    n_active_items = resAll$SPDI$n_active_items,
    DI = resAll$SPDI$DI,
    intercept_est = if (is_tree) NA else resAll$model.coef$intercept_est,
    beta_est
  )
}

# --- CLI ---
cli <- commandArgs(trailingOnly = TRUE)
casenum <- as.integer(if (length(cli) >= 1) cli[1] else 1)
k.all <- if (length(cli) >= 2) as.integer(cli[-1]) else 1L

datasets <- c("MTCC", "KTCC", "KVIC", "BMTL", "DCCC", "UGC")
case <- datasets[casenum]
cat("Case:", case, "\n")

cache_file <- Sys.getenv(
  "APPLICATION_CACHE_FILE",
  file.path(datadir, paste0(case, "_train_index_full_5_2folds.RData"))
)
if (!file.exists(cache_file)) {
  stop("Cached RData not found: ", cache_file, ". Run the preprocessing step first.")
}
load(cache_file)

tag <- Sys.getenv("APPLICATION_FILE_TAG", "")
if (!nzchar(tag)) {
  tag <- paste0("_", format(Sys.Date(), "%Y%m%d"))
} else if (!startsWith(tag, "_")) {
  tag <- paste0("_", tag)
}

config <- list(
  M = 10L,
  include_intercept = TRUE,
  alpha = 1 / 2,
  gamma = 2 / 3,
  cv_method = "alternating",
  CV_criterion = "cost",
  start0 = TRUE,
  n_lambda_sparsity = 20L,
  n_lambda_diversity = 20L,
  n_folds = 5L,
  tolerance = 1e-5,
  obj_tol_rel = 1e-5,
  max_iter = 1e3,
  max_iter_admm = 1e3,
  n_threads = as.integer(Sys.getenv("ECSLR_N_THREADS", Sys.getenv("SLURM_CPUS_PER_TASK", "8")))
)

models <- c("ECSLRx")
cat("models:", paste(models, collapse = ", "), "\n")

N <- length(k.all)
log_dir <- file.path(savedir, "logfile", case)
temp_dir <- file.path(savedir, "temp", case)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)

result_final <- foreach(
  k = k.all,
  .combine = rbind,
  .init = data.frame(),
  .packages = c("ECSLRx", "Matrix", "cslogit", "caret", "pROC", "PRROC")
) %do% {
  logfile <- file.path(log_dir, paste0("log_", case, tag, "_k=", k, ".txt"))
  sink(logfile, append = TRUE)
  on.exit(sink(NULL), add = TRUE)

  cat("=== k =", k, "===", format(Sys.time()), "\n")
  temp_filename <- file.path(temp_dir, paste0(case, tag, "_k=", k, ".csv"))

  if (file.exists(temp_filename)) {
    res2 <- read.csv(temp_filename, check.names = FALSE)
    if (ncol(res2) > 0 && grepl("^Unnamed", names(res2)[1])) res2 <- res2[, -1, drop = FALSE]
    models_calculate <- setdiff(models, unique(res2$model))
  } else {
    res2 <- data.frame()
    models_calculate <- models
  }
  cat("models_calculate:", paste(models_calculate, collapse = ", "), "\n")

  if (length(models_calculate) > 0) {
    train_index <- train_index_all[[k]]
    x.train <- covariates[train_index, , drop = FALSE]
    y.train <- labels[train_index]
    x.test <- covariates[-train_index, , drop = FALSE]
    y.test <- labels[-train_index]

    cost_matrix_train <- flatten_cost_matrix(cost_matrix_raw[train_index, , , drop = FALSE])
    cost_matrix_test <- flatten_cost_matrix(cost_matrix_raw[-train_index, , , drop = FALSE])

    meta <- list(
      case = case, N = N, k = k,
      J = nrow(model_parameters$Group_Matrix),
      p.train = ncol(x.train),
      n.train = length(y.train),
      n.test = length(y.test),
      p.minority.real = mean(as.numeric(as.character(y.train))),
      x.train = x.train, y.train = y.train,
      x.test = x.test, y.test = y.test,
      cost_matrix_train = cost_matrix_train,
      cost_matrix_test = cost_matrix_test
    )

    for (model in models_calculate) {
      resAll <- generate_performance3(
        model, x.train, y.train, x.test, y.test,
        cost_matrix_train, cost_matrix_test,
        model_parameters = model_parameters, config = config,
        logfile = logfile
      )
      res2 <- rbind(res2, append_performance_row(resAll, model, meta))
      write.csv(res2, temp_filename, row.names = FALSE)
    }
  }
  res2
}

cat("Done.", format(Sys.time()), "\n")
invisible(result_final)
