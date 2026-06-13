# Grouped simulation data generation for paper scenarios I, II, III.
library(MASS)
library(dplyr)
library(mclust)
library(ECSLRx)

# Generate simulation parameters for paper scenarios I–III with fixed DGP hyperparameters.
generate_param <- function(scenario, p.mino, rho, rho.cross = NA, Bayes.risk = 0.15) {
  scenario <- toupper(as.character(scenario))
  spec <- switch(
    scenario,
    I = list(scenario = "I", num.categorical = 25L, num.continuous = 25L, n.all = 1000L, group.size = 10L),
    II = list(scenario = "II", num.categorical = 50L, num.continuous = 50L, n.all = 1000L, group.size = 10L),
    III = list(scenario = "III", num.categorical = 50L, num.continuous = 50L, n.all = 1000L, group.size = 2L),
    stop("Unknown scenario: ", scenario, ". Use I, II, or III.")
  )

  if (scenario %in% c("I", "II")) {
    fused.index <- 1:5
    zeta.continuous <- 0.4
    zeta.categorical <- 0.4
  } else {
    fused.index <- 1:2
    zeta.continuous <- 0.1
    zeta.categorical <- 0.1
  }

  gmm.1 <- list(
    modelName = "V",
    parameters = list(
      pro = c(0.333, 0.667), mean = c(0.667, 4),
      variance = list(modelName = "V", d = 1, G = 2, sigmasq = c(0.003, 3.2), scale = c(0.003, 3.2))
    )
  )
  gmm.0 <- list(
    modelName = "V",
    parameters = list(
      pro = c(0.1, 0.4, 0.2, 0.3), mean = c(0.667, 2, 4, 5),
      variance = list(modelName = "V", d = 1, G = 4, sigmasq = c(0.002, 0.8, 0.7, 1.5), scale = c(0.002, 0.8, 0.7, 1.5))
    )
  )

  generate_param_full(
    scenario = spec$scenario,
    num.categorical = spec$num.categorical,
    num.continuous = spec$num.continuous,
    group.size = spec$group.size,
    levels = 4L,
    zeta.categorical = zeta.categorical,
    zeta.continuous = zeta.continuous,
    fused.index = fused.index,
    include.amount = TRUE,
    gmm.1 = gmm.1,
    gmm.0 = gmm.0,
    inter.degree = 1L,
    p.mino = p.mino,
    rho = rho,
    rho.cross = rho.cross,
    Bayes.risk = Bayes.risk
  )
}

# Draw coefficients, correlation structure, and penalty matrices for one simulation design.
generate_param_full <- function(scenario, num.categorical, num.continuous, group.size,
                                levels = 4L, zeta.categorical = 0.4, zeta.continuous = 0.4,
                                fused.index = NULL, include.amount = TRUE,
                                gmm.1 = NULL, gmm.0 = NULL, inter.degree = 1L,
                                p.mino, rho, rho.cross = 0, Bayes.risk = 0.25) {
  num.Xpredictors <- num.categorical + num.continuous
  p.active.ca <- num.categorical * zeta.categorical
  p.active.co <- num.continuous * zeta.continuous
  p.active <- p.active.ca + p.active.co
  nblock <- p.active / group.size
  size.ca <- p.active.ca / nblock
  size.co <- p.active.co / nblock

  if (nblock != floor(nblock) || size.ca != floor(size.ca) || size.co != floor(size.co)) {
    stop("Block sizes must be integers; check scenario dimensions and group.size.")
  }

  Sigma <- matrix(rho.cross, nrow = num.Xpredictors, ncol = num.Xpredictors)
  for (b in 0:(nblock - 1)) {
    block <- c()
    if (size.ca > 0) block <- c(block, (b * size.ca + 1):(b * size.ca + size.ca))
    if (size.co > 0) block <- c(block, (num.categorical + b * size.co + 1):(num.categorical + b * size.co + size.co))
    Sigma[block, block] <- rho
  }
  inactive.index <- c()
  if (p.active.ca < num.categorical) inactive.index <- c(inactive.index, (p.active.ca + 1):num.categorical)
  if (p.active.co < num.continuous) inactive.index <- c(inactive.index, (num.categorical + p.active.co + 1):num.Xpredictors)
  if (length(inactive.index) > 0) Sigma[inactive.index, inactive.index] <- rho
  diag(Sigma) <- 1

  df <- c(rep(levels - 1, num.categorical), rep(1, num.continuous))
  if (include.amount) df <- c(df, 1)
  num.predictors <- length(df)

  fused.active <- rep(0, num.predictors)
  fused.active[fused.index] <- 1
  fused_type <- as.list(rep("none", num.predictors))
  fused_type[fused.index] <- "fuse1d"
  fused_par <- as.list(df)

  fus <- construct_fused_penalty(fused_type, fused_par)
  pen.mat <- fus$pen.mat
  Adjust_Matrix_Fused <- fus$Adjust_Matrix_Fused

  J <- length(df)
  p <- sum(df)
  active.maineffect <- integer(num.predictors)
  active.maineffect[seq_len(p.active.ca)] <- 1
  active.maineffect[(num.categorical + 1):(num.categorical + p.active.co)] <- 1
  active.maineffect[num.predictors] <- 1
  active <- active.maineffect

  betas <- numeric(p)
  cc <- 1
  for (j in seq_len(J)) {
    if (j <= length(active) && active[j] == 1) {
      if (grepl("fuse", fused_type[[j]])) {
        beta_g <- rep(rnorm(1), df[j])
      } else {
        beta_tilde <- rnorm(df[j] + 1)
        beta_g <- beta_tilde - mean(beta_tilde)
      }
      betas[cc:(cc + df[j] - 1)] <- beta_g[seq_len(df[j])]
    }
    cc <- cc + df[j]
  }

  nn <- 4000
  X_interaction <- simulate_X(nn, num.categorical, Sigma, levels, inter.degree, include.amount, num.continuous, gmm.1, gmm.0, p.mino)
  names(betas) <- colnames(X_interaction)[seq_len(p)]
  if (include.amount) {
    betas["logamount"] <- betas["logamount"] / var(X_interaction[, "logamount"])
  }

  rn <- 100
  bayes.risk <- numeric(rn)
  ra_candidates <- seq(-10, 10, length.out = rn)
  for (cyc in seq_len(100)) {
    for (r in seq_len(rn)) {
      betas_temp <- betas * ra_candidates[r]
      prob <- sigmoid(as.matrix(X_interaction[, seq_len(p)]) %*% betas_temp)
      if (include.amount) {
        index.select <- c(
          sample(which(prob > 0.5 & X_interaction$amount_label == 1), nn * p.mino, replace = TRUE),
          sample(which(prob <= 0.5 & X_interaction$amount_label == 0), nn * (1 - p.mino), replace = TRUE)
        )
      } else {
        index.select <- c(
          sample(which(prob > 0.5), nn * p.mino, replace = TRUE),
          sample(which(prob <= 0.5), nn * (1 - p.mino), replace = TRUE)
        )
      }
      prob.select <- prob[index.select]
      bayes.risk[r] <- mean(pmin(prob.select, 1 - prob.select))
    }
    diff <- bayes.risk - Bayes.risk
    closest_indices <- order(abs(diff))
    min.index <- closest_indices[1]
    risk_adjustment <- ra_candidates[min.index]
    if (abs(diff[min.index]) <= 1e-4) break
    for (k in 2:rn) {
      if (diff[min.index] * diff[closest_indices[k]] <= 0) {
        ra_candidates <- seq(ra_candidates[closest_indices[1]], ra_candidates[closest_indices[k]], length.out = rn)
        break
      }
    }
  }
  betas_rescaled <- betas * risk_adjustment

  Group_Matrix <- matrix(0, nrow = J, ncol = p)
  cc <- 1
  for (j in seq_len(J)) {
    Group_Matrix[j, cc:(cc + df[j] - 1)] <- 1
    cc <- cc + df[j]
  }

  list(
    scenario = scenario,
    p.mino = p.mino, rho = rho, rho.cross = rho.cross, Bayes.risk = Bayes.risk,
    betas = betas_rescaled, df = df, p = p,
    Group_Matrix = Matrix::Matrix(Group_Matrix, sparse = TRUE),
    pen.mat = pen.mat, Adjust_Matrix_Fused = Adjust_Matrix_Fused,
    num.categorical = num.categorical, num.continuous = num.continuous,
    Sigma = Sigma, levels = levels, inter.degree = inter.degree,
    gmm.1 = gmm.1, gmm.0 = gmm.0
  )
}

# Sample one replicated dataset with target class balance and Bayes risk.
simulate_data <- function(simu.parameters, n.all = 400) {
  num.categorical <- simu.parameters$num.categorical
  num.continuous <- simu.parameters$num.continuous
  Sigma <- simu.parameters$Sigma
  levels <- simu.parameters$levels
  inter.degree <- simu.parameters$inter.degree
  p.mino <- simu.parameters$p.mino
  betas <- simu.parameters$betas
  p <- simu.parameters$p
  gmm.1 <- simu.parameters$gmm.1
  gmm.0 <- simu.parameters$gmm.0
  include.amount <- TRUE

  n1 <- n.all * p.mino
  n0 <- n.all * (1 - p.mino)
  times <- 5
  while (TRUE) {
    X_interaction <- simulate_X(
      n.all * times, num.categorical, Sigma, levels, inter.degree,
      include.amount, num.continuous, gmm.1, gmm.0, p.mino
    )
    prob <- sigmoid(as.matrix(X_interaction[, seq_len(p)]) %*% betas)
    y <- ifelse(prob > 0.5, 1, 0)
    data <- data.frame(X_interaction, y, prob)
    data.1 <- dplyr::filter(data, amount_label == 1 & y == 1)
    data.0 <- dplyr::filter(data, amount_label == 0 & y == 0)
    if (nrow(data.1) >= n1 && nrow(data.0) >= n0) {
      data.select <- rbind(
        data.1[sample(seq_len(nrow(data.1)), n1), , drop = FALSE],
        data.0[sample(seq_len(nrow(data.0)), n0), , drop = FALSE]
      )
      cat("Bayes risk:", mean(pmin(data.select$prob, 1 - data.select$prob)), "\n")
      break
    }
    times <- times * 2
    cat("Increase sampling times to", times, "\n")
  }
  dplyr::select(data.select, -c(prob, amount_label))
}

# Simulate covariates (with optional log-amount) and expand interactions via model.matrix.
simulate_X <- function(n, num.categorical, Sigma, levels, inter.degree = 1, include.amount,
                       num.continuous, gmm.1, gmm.0, p.mino) {
  num.Xpredictors <- num.categorical + num.continuous
  mn_data <- as.data.frame(mvrnorm(n, mu = rep(0, num.Xpredictors), Sigma = Sigma))
  cutvalue <- qnorm(seq(0, 1, length.out = levels + 1))
  X <- mn_data %>%
    mutate(across(all_of(seq_len(num.categorical)), ~ cut(., breaks = cutvalue, include.lowest = TRUE, labels = FALSE))) %>%
    mutate(across(all_of(seq_len(num.categorical)), as.factor))

  n1 <- as.integer(n * p.mino)
  n0 <- n - n1
  if (include.amount) {
    fixed_cost <- 10
    truncation1 <- log(fixed_cost + 1)
    truncation2 <- log(5000 + 1)
    l1 <- sim(modelName = gmm.1$modelName, parameters = gmm.1$parameters, n = 5 * n1)[, 2]
    logamount.1 <- sample(l1[l1 > truncation1 & l1 < truncation2], n1)
    l0 <- sim(modelName = gmm.0$modelName, parameters = gmm.0$parameters, n = 5 * n0)[, 2]
    logamount.0 <- sample(l0[l0 > truncation1 & l0 < truncation2], n0)
    X <- data.frame(X, logamount = c(logamount.1, logamount.0))
  }

  formula <- if (inter.degree == 1) as.formula("~ .") else as.formula(paste("~ .^", inter.degree, sep = ""))
  X_interaction <- model.matrix(formula, data = X)[, -1, drop = FALSE]
  if (include.amount) {
    X_interaction <- cbind(X_interaction, amount_label = c(rep(1, n1), rep(0, n0)))
  }
  as.data.frame(X_interaction)
}
