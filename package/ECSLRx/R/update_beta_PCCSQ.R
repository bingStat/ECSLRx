"%**%" <- function(A, B){
  return(Matrix_Mult(A, B))
}

# OPT-3: eigen_cache is a pre-computed list(Q, eigval) from compute_block_eigen().
# Pass it through to ECSLR_Compute_Coef_combine to avoid redundant block-eigen
# calls in CV loops.
ECSLR_x_initial <- function(x, y, cost_matrix,
                                M, include_intercept,
                                w_lasso,
                                Group_Matrix, w_group,
                                Adjust_Matrix_Fused, fused_penmat,
                                alpha, gamma,
                                lambda_sparsity,
                                lambda_diversity,
                                tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                x0 = NULL, start0 = TRUE,
                                eigen_cache = NULL){  # OPT-3 new param
  p <- ncol(x)
  lambda_fused <- lambda_sparsity * (1 - gamma)
  if (is.null(eigen_cache)) {
    eigen_cache <- ECSLRx::build_eigen_cache(
      Group_Matrix, Adjust_Matrix_Fused, fused_penmat, lambda_fused
    )
  }

  # start from given start value
  if(is.null(x0)){
    ECSLR_model_start_previous <- list(objective=NA)
  }else{
    ECSLR_model_start_previous <- ECSLR_Compute_Coef_combine(x, y, cost_matrix,
                                                             M, include_intercept,
                                                             w_lasso,
                                                             Group_Matrix, w_group,
                                                             Adjust_Matrix_Fused, fused_penmat,
                                                             alpha, gamma,
                                                             lambda_sparsity,
                                                             lambda_diversity,
                                                             tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                                             x0_=x0,
                                                             eigen_cache_=eigen_cache)  # OPT-3
    ECSLR_model_start_previous$start <- "previous"
    cat("start_previous:",ECSLR_model_start_previous$objective,"\n\n")
  }

  # Safety: if x0=NULL and start0=FALSE, there would be no valid runs at all.
  # Force start0=TRUE in this degenerate case.
  if(is.null(x0) && !start0){
    warning("start0=FALSE but x0 is NULL -- forcing start0=TRUE to avoid empty model list")
    start0 <- TRUE
  }

  if(start0){
    # start from given 0
    ECSLR_model_start_0 <- ECSLR_Compute_Coef_combine(x, y, cost_matrix,
                                                      M, include_intercept,
                                                      w_lasso,
                                                      Group_Matrix, w_group,
                                                      Adjust_Matrix_Fused, fused_penmat,
                                                      alpha, gamma,
                                                      lambda_sparsity,
                                                      lambda_diversity,
                                                      tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                                      x0_=NULL,
                                                      eigen_cache_=eigen_cache)  # OPT-3
    ECSLR_model_start_0$start <- "0"
    cat("start_0:",ECSLR_model_start_0$objective,"\n\n")
  }else{
    ECSLR_model_start_0 <- list(objective=NA)
  }

  models <- list(ECSLR_model_start_previous
                 ,ECSLR_model_start_0)

  valid_models <- Filter(function(m) !is.null(m$objective) && !is.na(m$objective), models)
  if(length(valid_models) == 0) stop("No valid model found in ECSLR_x_initial")
  ECSLR_opt <- valid_models[[which.min(sapply(valid_models, function(x) x$objective))]]
  cat("optimal start: ")
  print(ECSLR_opt$start)

  return(ECSLR_opt)
}


# Backward-compatibility alias: keeps aux_code and any external caller working
# without modification after the rename.
ECSLR_multi_initial <- ECSLR_x_initial

instance_dependent_cost_threshold <- function(cost_matrix){
  (cost_matrix[, 2] - cost_matrix[, 1]) /(cost_matrix[, 2] - cost_matrix[, 1] +
                                                  cost_matrix[, 3] - cost_matrix[, 4])
}

Compute_coef_cost <- function(y,cost_matrix){
  a <- y * (cost_matrix[,4] - cost_matrix[,3]) + (1-y) * (cost_matrix[,2]-cost_matrix[,1])
  b <- y * cost_matrix[,3] + (1-y)*cost_matrix[,1]
  return(list(a=a,b=b))
}

Compute_aec_cost <- function(intercept1, betas1, x, y, cost_matrix, model){
  scores <- sigmoid(intercept1 + Matrix_Mult(as.matrix(x), betas1))
  opt_threshold <- instance_dependent_cost_threshold(cost_matrix)
  pre.class <- ifelse(scores>opt_threshold,1,0)

  cost <- Compute_AEC(pre.class, y, cost_matrix=cost_matrix)
  AEC <- Compute_AEC(scores, y, cost_matrix=cost_matrix)
  return(list(cost=cost,AEC=AEC))
}

cost_without_algorithm <- function(cost_matrix, labels){
  cost_neg <- Compute_AEC(rep(0, length(labels)), labels, cost_matrix)
  cost_pos <- Compute_AEC(rep(1, length(labels)), labels, cost_matrix)
  return(min(cost_neg, cost_pos))
}

# OPT-3: Helper to build eigen_cache once for a given structural setup.
# Call this once before any CV loop when lambda_fused != 0.
build_eigen_cache <- function(Group_Matrix, Adjust_Matrix_Fused, fused_penmat, lambda_fused) {
  if (lambda_fused == 0 || nrow(fused_penmat) == 0) return(NULL)
  compute_block_eigen(Group_Matrix, Adjust_Matrix_Fused, fused_penmat)
}
