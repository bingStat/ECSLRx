# Simulation benchmark: synthesized grouped data, paper comparator + ECSLRx models.
# Usage (HPC): Rscript benchmark_simulation.R <I|II|III> <pmino> <rho> <rhoc> <B> <tag>
# Usage (local): Rscript benchmark_simulation.R   # defaults to scenario III

library(ECSLRx)
library(foreach)
library(doParallel)

repo_root <- normalizePath("../..", winslash = "/")
script_root <- file.path(repo_root, "scripts")
savedir <- file.path(repo_root, "runs")
source(file.path(script_root, "Auxiliary_Functions_for_Performance.R"))

mpi_workers <- as.numeric(Sys.getenv("ECSLR_MPI_WORKERS", Sys.getenv("SLURM_CPUS_PER_TASK", "8")))
if (is.na(mpi_workers) || mpi_workers <= 0) mpi_workers <- 8L

# Flatten a 3-D instance-dependent cost array to a 4-column matrix.
flatten_cost_matrix <- function(cm3) {
  cbind(cm3[, 1, 1], cm3[, 2, 1], cm3[, 1, 2], cm3[, 2, 2])
}

# Build train/test cost matrices with optional amount-dependent misclassification costs.
build_cost_matrices <- function(x.train, x.test, include.amount, fixed_cost = 10) {
  n.train <- nrow(x.train)
  n.test <- nrow(x.test)
  cost_matrix_train0 <- array(0, dim = c(n.train, 2, 2))
  cost_matrix_test0 <- array(0, dim = c(n.test, 2, 2))
  if (include.amount) {
    cost_matrix_train0[, 1, 2] <- exp(x.train$logamount) - 1
    cost_matrix_train0[, 2, 1] <- fixed_cost
    cost_matrix_train0[, 2, 2] <- fixed_cost
    cost_matrix_test0[, 1, 2] <- exp(x.test$logamount) - 1
    cost_matrix_test0[, 2, 1] <- fixed_cost
    cost_matrix_test0[, 2, 2] <- fixed_cost
  } else {
    cost_ratio <- 2
    cost_matrix_train0[, 1, 2] <- cost_ratio
    cost_matrix_train0[, 2, 1] <- 1
    cost_matrix_test0[, 1, 2] <- cost_ratio
    cost_matrix_test0[, 2, 1] <- 1
  }
  list(
    train = flatten_cost_matrix(cost_matrix_train0),
    test = flatten_cost_matrix(cost_matrix_test0)
  )
}

# Assemble one simulation benchmark result row (metrics, beta recovery, coefficients).
append_simulation_row <- function(resAll, model, meta) {
  is_tree <- model %in% c("CSRF-cv", "CSRP-cv")
  is_ecslrx <- model == "ECSLRx"
  real_betas <- meta$real_betas

  if (is_tree) {
    beta_est <- data.frame(matrix(NA, 1, meta$p.train))
    intercept_est <- NA
    MSE_betas <- sensitivity_betas <- specificity_betas <- precision_betas <- NA
  } else {
    est <- resAll$model.coef$betas_est
    beta_est <- data.frame(t(est))
    intercept_est <- resAll$model.coef$intercept_est
    MSE_betas <- mean((est - real_betas)^2)
    sensitivity_betas <- sum(est != 0 & real_betas != 0) / sum(real_betas != 0)
    specificity_betas <- sum(est == 0 & real_betas == 0) / sum(real_betas == 0)
    precision_betas <- sum(est != 0 & real_betas != 0) / sum(est != 0)
  }
  colnames(beta_est) <- colnames(meta$x.train)

  cbind(
    data.frame(
      N = meta$N, k = meta$k, J = meta$J, p = meta$p.train,
      n.train = meta$n.train, n.test = meta$n.test,
      p.minority.real = meta$p.minority.real,
      scenario = meta$scenario, p.minority = meta$p.mino,
      rho = meta$rho, rho.cross = meta$rho.cross, Bayes.risk = meta$Bayes.risk,
      model = model, resAll$res,
      M = if (is_tree || is_ecslrx) resAll$output$M else NA_integer_,
      lambda_sparsity = if (is_ecslrx) resAll$model.coef$lambda_sparsity else NA,
      lambda_diversity = if (is_ecslrx) resAll$model.coef$lambda_diversity else NA,
      SP = resAll$SPDI$SP, SPfinal = resAll$SPDI$SPfinal,
      Group_SPfinal = resAll$SPDI$Group_SPfinal,
      n_active_items = resAll$SPDI$n_active_items, DI = resAll$SPDI$DI,
      MSE_betas = MSE_betas, sensitivity_betas = sensitivity_betas,
      specificity_betas = specificity_betas, precision_betas = precision_betas,
      elapsed_time = resAll$elapsed_time, intercept_est = intercept_est
    ),
    beta_est
  )
}

# --- scenario parameters ---
cli <- commandArgs(trailingOnly = TRUE)
source(file.path(script_root, "simulation", "Data_Simulation_Group.R"))
if (length(cli) >= 6) {
  scenario <- cli[1]
  p.mino <- as.numeric(cli[2])
  rho <- as.numeric(cli[3])
  rho.cross <- as.numeric(cli[4])
  Bayes.risk <- as.numeric(cli[5])
  tag <- cli[6]
} else {
  scenario <- "III"
  p.mino <- 0.2
  rho <- 0.5
  rho.cross <- 0.2
  Bayes.risk <- 0.15
  tag <- paste0("_", format(Sys.Date(), "%Y%m%d"))
}
spec <- resolve_scenario(scenario)
num.categorical <- spec$num.categorical
num.continuous <- spec$num.continuous
n.all <- spec$n.all
group.size <- spec$group.size
include.amount <- TRUE
fixed_cost <- 10
set.seed(1234)

N <- 50L
k.all <- seq_len(N)
cluster_name <- if (.Platform$OS.type == "windows") "wice" else Sys.getenv("SLURM_CLUSTER_NAME")
job_partition <- if (.Platform$OS.type == "windows") "batch" else Sys.getenv("SLURM_JOB_PARTITION")

case <- paste0(
  "simu_s", scenario, "_N", N,
  "_ca", num.categorical, "_co", num.continuous, "_n", n.all,
  "_pmino", p.mino, "_rho", rho, "_rhoc", rho.cross,
  "_B", Bayes.risk, "_g", group.size, "_", cluster_name, "_", job_partition
)
cat("scenario:", scenario,
    " ca/co/n/g =", num.categorical, num.continuous, n.all, group.size, "\n")
case_data <- sub(paste0("_", cluster_name, "_", job_partition),
                 paste0("_genius_", job_partition), case)
cat("case:", case, "\n")

log_dir <- file.path(savedir, "logfile", case)
temp_dir <- file.path(savedir, "temp", case)
data_dir <- file.path(savedir, "data_simulation")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

file_path <- file.path(data_dir, paste0("data_", case_data, ".RData"))
if (!file.exists(file_path)) {
  cat("Generating simulation data:", file_path, "\n")
  simu.parameters <- generate_param(
    scenario = scenario, p.mino = p.mino, rho = rho,
    rho.cross = rho.cross, Bayes.risk = Bayes.risk
  )
  all.data <- lapply(seq_len(N), function(i) simulate_data(simu.parameters, n.all = n.all))
  train_index_all <- lapply(all.data, function(data) {
    caret::createMultiFolds(as.factor(data$y), k = 2, times = 1)[[1]]
  })
  save(all.data, train_index_all, simu.parameters, file = file_path)
}
load(file_path)

real_betas <- simu.parameters$betas
df <- simu.parameters$df
J <- length(df)
w_lasso <- rep(1, sum(df))
I_group <- rep(0, J)
I_group[df != 1] <- 1
model_parameters <- list(
  w_lasso = w_lasso,
  w_group = as.matrix(I_group * sqrt(df)),
  Group_Matrix = simu.parameters$Group_Matrix,
  fused_penmat = simu.parameters$pen.mat,
  Adjust_Matrix_Fused = simu.parameters$Adjust_Matrix_Fused
)

config <- list(
  M = 10L, include_intercept = TRUE,
  alpha = 1 / 2, gamma = 2 / 3, cv_method = "alternating", CV_criterion = "cost",
  start0 = TRUE, n_lambda_sparsity = 20L, n_lambda_diversity = 20L, n_folds = 5L,
  tolerance = 1e-5, obj_tol_rel = 1e-5, max_iter = 1e3, max_iter_admm = 1e3,
  n_threads = 3L
)
models <- c("logit", "cslogit", "CSRF-cv", "CSRP-cv", "ECSLRx")
final_filename <- file.path(savedir, "result", paste0(case, tag, "_N=", N, ".csv"))

if (.Platform$OS.type == "windows") {
  cl <- makeCluster(N)
  registerDoParallel(cl)
  on.exit(stopCluster(cl), add = TRUE)
} else {
  library(doMPI)
  cl <- startMPIcluster(maxcores = mpi_workers)
  registerDoMPI(cl)
  on.exit({ closeCluster(cl); mpi.quit() }, add = TRUE)
}

result_final <- foreach(
  k = k.all, .combine = rbind, .init = data.frame(), .verbose = TRUE,
  .export = c(
    "config", "real_betas", "case", "tag", "savedir", "N", "exp_dir",
    "log_dir", "temp_dir", "models", "all.data", "train_index_all", "model_parameters",
    "scenario", "p.mino", "rho", "rho.cross", "Bayes.risk", "J",
    "include.amount", "fixed_cost", "append_simulation_row", "build_cost_matrices"
  ),
  .packages = c("ECSLRx", "Matrix", "cslogit", "caret", "pROC", "PRROC")
) %dopar% {
  logfile <- file.path(log_dir, paste0("log_simulation_", case, tag, "_N=", N, "_k=", k, ".txt"))
  sink(logfile, append = TRUE)
  on.exit(sink(NULL), add = TRUE)
  cat("=== k =", k, "===", format(Sys.time()), "\n")

  temp_filename <- file.path(temp_dir, paste0(case, tag, "_N=", N, "_k=", k, ".csv"))
  if (file.exists(temp_filename)) {
    res2 <- tryCatch({
      dat <- read.csv(temp_filename, check.names = FALSE)
      if (ncol(dat) > 0 && grepl("^Unnamed", names(dat)[1])) dat <- dat[, -1, drop = FALSE]
      dat
    }, error = function(e) data.frame())
    models_calculate <- setdiff(models, unique(res2$model))
  } else {
    res2 <- data.frame()
    models_calculate <- models
  }

  if (length(models_calculate) > 0) {
    source(file.path(script_root, "Auxiliary_Functions_for_Performance.R"))
    data <- all.data[[k]]
    train_index <- train_index_all[[k]]
    p <- ncol(data) - 1
    x.train <- data[train_index, 1:p, drop = FALSE]
    x.test <- data[-train_index, 1:p, drop = FALSE]
    y.train <- data[train_index, "y"]
    y.test <- data[-train_index, "y"]
    cm <- build_cost_matrices(x.train, x.test, include.amount, fixed_cost)

    meta <- list(
      N = N, k = k, J = J, p.train = p,
      n.train = length(y.train), n.test = length(y.test),
      p.minority.real = sum(y.train) / length(y.train),
      scenario = scenario, p.mino = p.mino, rho = rho,
      rho.cross = rho.cross, Bayes.risk = Bayes.risk,
      real_betas = real_betas, x.train = x.train
    )

    for (model in models_calculate) {
      resAll <- generate_performance3(
        model, x.train, y.train, x.test, y.test,
        cm$train, cm$test,
        model_parameters = model_parameters, config = config, logfile = logfile
      )
      res2 <- rbind(res2, append_simulation_row(resAll, model, meta))
      write.csv(res2, temp_filename, row.names = FALSE)
    }
  }
  res2
}

write.csv(result_final, final_filename, row.names = FALSE)
cat("Done.", format(Sys.time()), "\n")
