# library(foreach)
# library(doParallel)
# library(caret)

# source("./R/update_beta_PCCSQ.R")


CV.ECSLRx.alternating <- function(x, y, cost_matrix = NULL,
                                 Ms = 10, include_intercept = TRUE,
                                 w_lasso = NULL,
                                 Group_Matrix, w_group,
                                 Adjust_Matrix_Fused, fused_penmat,
                                 alpha = 0, gamma = 0.2,
                                 n_lambda_sparsity=20L, n_lambda_diversity=20L, n_folds = 3L,
                                 tolerance = 1e-5, obj_tol_rel = 1e-5, max_iter = 1e3, max_iter_admm = 1e3, model = "ECSLR", CV_criterion = "cost", start0 = TRUE, n_threads=1L){

  print(paste0("n_lambda_sparsity=",n_lambda_sparsity,", n_lambda_diversity=",n_lambda_diversity))
  print(paste0("start0=",start0))

  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  p <- ncol(x)
  if (is.null(w_lasso)) {
    w_lasso <- rep(1, p)
  }
  Group_Matrix <- Matrix::Matrix(Group_Matrix, sparse = TRUE)
  Adjust_Matrix_Fused <- Matrix::Matrix(Adjust_Matrix_Fused, sparse = TRUE)
  fused_penmat <- Matrix::Matrix(fused_penmat, sparse = TRUE)
  eigen_cache <- ECSLRx::build_eigen_cache(
    Group_Matrix, Adjust_Matrix_Fused, fused_penmat,
    lambda_fused = ifelse(any(gamma != 1), 1, 0)
  )

  # Creating indices for the folds of the data
  sample_ind <- seq(1, n)
  set.seed(1234)

  folds_index <- tryCatch({
    amount <- exp(x[,"logamount"])-1
    ECSLRx::generateTrainIndex(y, amount, k=n_folds, times = 1)
  }, error = function(e){
    caret::createMultiFolds(as.factor(y), k=n_folds, times = 1)
  })

  cv_results_final <- data.frame() # save all the results during alternating grid search

  eps <- ifelse(p > n, 1e-2, 1e-4) #if p is too large, we should encourage sparsity and diversity
  cat("eps=",eps,"\n")
  CV_ITERATIONS_TOLERANCE <- 1e-5
  CV_ITERATIONS_MAX <- 10

  mu_x <- apply(x, 2, mean)
  sd_x <- apply(x, 2, sd)
  x_std <- scale(x, center = mu_x, scale = sd_x)
  # if a predictor always equal to a constant， then sd=0，and x_std would be NaN
  # this kind of predictors contribute no information, so set them to 0
  x_std[is.na(x_std)] <- 0

  ######## Warm start ############
  intercept_initial <- log(mean(y)/(1-mean(y)))
  x0_M1 <- list(
    betas = matrix(0, nrow = p, ncol = 1),
    intercept = matrix(intercept_initial, nrow = 1, ncol = 1)
  )

  cat("initial value from 0 \n")
  cat("head(x0_M1$betas)",head(x0_M1$betas),"\n")
  cat("x0_M1$intercept",x0_M1$intercept,"\n")

  ### Cross validation for M, alpha, and gamma
  cv_Mag <- data.frame()

  for(alpha1 in alpha){
    for(gamma1 in gamma){
      ########Warm start############
      temp <- ECSLRx::Compute_Lambda_Sparsity_Grid(x_std, y, cost_matrix,
                                           include_intercept,
                                           w_lasso,
                                           Group_Matrix, w_group,
                                           Adjust_Matrix_Fused, fused_penmat,
                                           alpha1, gamma1,
                                           tolerance, obj_tol_rel, max_iter, max_iter_admm,
                                           eps, n_lambda_sparsity, model,
                                           eigen_cache = eigen_cache)
      lambda_sparsity_opt <- temp$lambda_sparsity_max
      lambda_sparsity_grid <- temp$lambda_sparsity_grid


      # Initial iteration with no diversity
      lambda_diversity_opt <- 0
      sparsity_search <- TRUE

      cv_opt_new <- Inf
      ##### Initial sparsity search ########
      x0 <- x0_M1
      cv_opt_old <- cv_opt_new
      # lambda_sparsity_opt_old <- lambda_sparsity_opt
      # lambda_diversity_opt_old <- lambda_diversity_opt


      temp <- ECSLRx::Compute_CV_Grid_alternating(sample_ind, folds_index, sparsity_search,
                                          n_lambda_sparsity, n_lambda_diversity, n_folds,
                                          x, y,
                                          M=1L, include_intercept,
                                          w_lasso,
                                          Group_Matrix, w_group,
                                          Adjust_Matrix_Fused, fused_penmat,
                                          alpha1, gamma1,
                                          lambda_sparsity_grid,
                                          lambda_sparsity_opt,
                                          lambda_diversity_opt=0,
                                          tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                          eps, cost_matrix, x0, CV_criterion, start0 = start0, n_threads = n_threads,
                                          cv_iteration = 0L,
                                          eigen_cache = eigen_cache)

      cv_opt_new <- temp$cv_opt_new
      lambda_sparsity_opt <- temp$lambda_sparsity_opt
      lambda_diversity_opt <- temp$lambda_diversity_opt
      cv_results_final <- rbind(cv_results_final,temp$cv_results)

      # Print iteration data to console
      cat("Search:",ifelse(sparsity_search,"sparsity","diversity"),"\n")
      cat("M:", 1L, ", alpha:", alpha1, ", gamma:", gamma1, "\n")
      cat("Iteration: ", 0, "(initial sparsity) \n")
      cat("cv_opt_old: ", cv_opt_old, "\n")
      cat("cv_opt_new: ", cv_opt_new, "\n")
      cat("lambda_sparsity_opt: ", lambda_sparsity_opt, "\n")
      cat("lambda_diversity_opt: ", lambda_diversity_opt, "\n\n")
      ################## initial sparsity search ######################################

      lambda_sparsity_opt_initial <- lambda_sparsity_opt
      lambda_diversity_opt_initial <- lambda_diversity_opt
      cv_opt_initial <- cv_opt_new

      ### Cross validation for M
      for(M in Ms){
        lambda_sparsity_opt <- lambda_sparsity_opt_initial
        lambda_diversity_opt <- lambda_diversity_opt_initial
        cv_opt_new <- cv_opt_initial

        # if M = 1, it is not necessary to do the next step and lambda_diversity_opt=0 as before.
        if(M > 1){
          sparsity_search <- FALSE
          x0_M <- list(
            betas = matrix(0, nrow = p, ncol = M),
            intercept = matrix(intercept_initial, nrow = 1, ncol = M)
          )
          x0 <- x0_M
          # Compute the solutions until the optimal is no longer a significant improvement
          for(cv_iterations in 0:CV_ITERATIONS_MAX){

            # Variables to store the old penalty parameters
            cv_opt_old <- cv_opt_new
            lambda_sparsity_opt_old <- lambda_sparsity_opt
            lambda_diversity_opt_old <- lambda_diversity_opt
            temp <- ECSLRx::Compute_CV_Grid_alternating(sample_ind, folds_index, sparsity_search,
                                                n_lambda_sparsity, n_lambda_diversity, n_folds,
                                                x, y,
                                                M, include_intercept,
                                                w_lasso,
                                                Group_Matrix, w_group,
                                                Adjust_Matrix_Fused, fused_penmat,
                                                alpha1, gamma1,
                                                lambda_sparsity_grid,
                                                lambda_sparsity_opt,
                                                lambda_diversity_opt,
                                                tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                                eps, cost_matrix, x0, CV_criterion, start0=start0, n_threads = n_threads,
                                                cv_iteration = cv_iterations,
                                                eigen_cache = eigen_cache)

            cv_opt_new <- temp$cv_opt_new
            lambda_sparsity_opt <- temp$lambda_sparsity_opt
            lambda_diversity_opt <- temp$lambda_diversity_opt
            cv_results_final <- rbind(cv_results_final,temp$cv_results)

            # Print iteration data to console
            cat("Search:",ifelse(sparsity_search,"sparsity","diversity"),"\n")
            cat("M:", M, ", alpha:", alpha1, ", gamma:", gamma1, "\n")
            if(cv_iterations==0){
              cat("Iteration: ", 0, "(initial diversity) \n")
            }else{
              cat("Iteration: ", cv_iterations, "\n")
            }
            cat("cv_opt_old: ", cv_opt_old, "\n")
            cat("cv_opt_new: ", cv_opt_new, "\n")
            cat("lambda_sparsity_opt: ", lambda_sparsity_opt, "\n")
            cat("lambda_diversity_opt: ", lambda_diversity_opt, "\n\n")
            if(cv_iterations==0){
              cat("Complete initializing lambda_sparsity and lambda_diversity, continue to optimize", "\n\n")
            }

            if(cv_iterations >= 1){#>=1,make sure we have do sparsity search and diversity search at least once, do not compare at the first diversity search

              # Conditions for breaking out of search for optimal penalty parameters;
              # The old result is better, can break and keep the old result
              if (cv_opt_new > cv_opt_old){
                lambda_sparsity_opt <- lambda_sparsity_opt_old
                lambda_diversity_opt <- lambda_diversity_opt_old
                cv_opt_new <- cv_opt_old
                print("The old result is better, stop optimize and keep the old result")
                break
              }

              # If it comes here, it means the new cv_opt_new is preferred and the new result is better.
              if( (sparsity_search && (lambda_sparsity_opt == lambda_sparsity_opt_old)) |
                  (!sparsity_search && (lambda_diversity_opt == lambda_diversity_opt_old)) ) {
                print("lambda_sparsity_opt=lambda_sparsity_opt_old or lambda_diversity_opt=lambda_diversity_opt_old, no need to search further")
                break
              }

              if(abs(cv_opt_new - cv_opt_old) <= CV_ITERATIONS_TOLERANCE){
                print("abs(cv_opt_new - cv_opt_old) <= CV_ITERATIONS_TOLERANCE, the algorithm converge.")
                break
              }
            }

            # New search over the penalty parameters
            sparsity_search <- !sparsity_search

          } # End of loop over the CV iterations
        }

        ### Cross validation for M
        cv_Mag <- rbind(cv_Mag,
                        data.frame(M = M, alpha = alpha1, gamma = gamma1,
                                   lambda_sparsity_opt = lambda_sparsity_opt,
                                   lambda_diversity_opt = lambda_diversity_opt,
                                   cv_opt_new = cv_opt_new))
        print(cv_Mag)

      }
    }
  }

  cat("cv_results_final: \n")
  print(cv_results_final)

  cat("cv_Mag: \n")
  print(cv_Mag)


  index.opt <- which.min(cv_Mag$cv_opt_new)
  M_opt <- cv_Mag$M[index.opt]
  alpha_opt <- cv_Mag$alpha[index.opt]
  gamma_opt <- cv_Mag$gamma[index.opt]
  lambda_sparsity_opt <- cv_Mag$lambda_sparsity_opt[index.opt]
  lambda_diversity_opt <- cv_Mag$lambda_diversity_opt[index.opt]

  opt_sparsity_rows <- cv_results_final[
    cv_results_final$search_direction == "sparsity" &
      cv_results_final$alpha == alpha_opt &
      cv_results_final$gamma == gamma_opt,
    ,
    drop = FALSE
  ]
  lambda_sparsity_grid <- sort(unique(opt_sparsity_rows$lambda_sparsity))
  n_lambda_sparsity <- length(lambda_sparsity_grid)

  cat("optimal parameters:\n")
  print(cv_Mag[index.opt,])

  # The optimal lambda_sparsity_opt and lambda_diversity_opt has been calculated, the next step is calculate the coefficient for the optimal lambda,
  # The problem is if we just calculate the coefficient for lambda_sparsity_opt, the result is not stable,
  # so calculate from lambda_sparsity_max, and iterated use the result as the initial coefficient to calculate the next one.
  # We compute the solutions for a decreasing sequence among the candidates for lambda_sparsity, similar to the grid search for the LASSO model described in Friedman(2001).

  ## output a series of result for different lambda_sparsity
  ECSLR_model_full <- list()
  cv_errors_sparsity <- rep(NA, n_lambda_sparsity)
  cv_cost_sparsity <- rep(NA, n_lambda_sparsity)
  insample_obj_sparsity <- rep(NA, n_lambda_sparsity)

  Optimal_Index <- which(lambda_sparsity_grid==lambda_sparsity_opt)
  # Optimal_Index <- which(signif(lambda_sparsity_grid, digits = 5)==signif(lambda_sparsity_opt, digits = 5))
  x0_M <- list(
    betas = matrix(0, nrow = p, ncol = M_opt),
    intercept = matrix(intercept_initial, nrow = 1, ncol = M_opt)
  )
  x0 <- x0_M

  # Computing the parameters for the full data
  for(sparsity_ind in n_lambda_sparsity:Optimal_Index){

    ECSLR_model_full[[sparsity_ind]] <- ECSLRx::ECSLR_multi_initial(x = x, y = y, cost_matrix=cost_matrix,
                                                            M = M_opt,
                                                            include_intercept,
                                                            w_lasso,
                                                            Group_Matrix, w_group,
                                                            Adjust_Matrix_Fused, fused_penmat,
                                                            alpha = alpha_opt, gamma = gamma_opt,
                                                            lambda_sparsity = lambda_sparsity_grid[sparsity_ind],
                                                            lambda_diversity = lambda_diversity_opt,
                                                            tolerance, obj_tol_rel, max_iter, max_iter_admm,
                                                            model, x0=x0, start0 = start0,
                                                            eigen_cache = eigen_cache)

    x0 <- ECSLR_model_full[[sparsity_ind]]$coef_x_std

    insample_obj_sparsity[sparsity_ind] <- ECSLR_model_full[[sparsity_ind]]$objective

    aec_cost <- ECSLRx::Compute_aec_cost(ECSLR_model_full[[sparsity_ind]]$intercept_est, ECSLR_model_full[[sparsity_ind]]$betas_est,
                                 x, y,
                                 cost_matrix, model)

    cv_errors_sparsity[sparsity_ind] <- aec_cost$AEC
    cv_cost_sparsity[sparsity_ind] <- aec_cost$cost
  }

  insample_results <- data.frame(M=M_opt,
                                 alpha=alpha_opt,
                                 gamma=gamma_opt,
                                 lambda_sparsity=lambda_sparsity_grid,
                                 lambda_diversity_opt = lambda_diversity_opt,
                                 insample_obj = insample_obj_sparsity,
                                 insample_AEC = cv_errors_sparsity,
                                 insample_cost = cv_cost_sparsity)
  cat("insample_results: \n")
  print(insample_results)

  ECSLR_model_opt <- ECSLR_model_full[[Optimal_Index]]

  output <- list(M=M_opt,
                 alpha=alpha_opt,
                 gamma=gamma_opt,
                 lambda_sparsity_grid=lambda_sparsity_grid,
                 lambda_sparsity_opt=lambda_sparsity_opt,
                 lambda_diversity_opt=lambda_diversity_opt,
                 cv_results_final = cv_results_final,
                 cv_Mag = cv_Mag,
                 Optimal_Index = Optimal_Index,
                 ECSLR_model_opt = ECSLR_model_opt,
                 ECSLR_model_full = ECSLR_model_full,
                 model=model)
  return(output)
}


Compute_CV_Grid_alternating <- function(sample_ind, folds_index, sparsity_search,
                                        n_lambda_sparsity, n_lambda_diversity, n_folds,
                                        x, y,
                                        M, include_intercept,
                                        w_lasso,
                                        Group_Matrix, w_group,
                                        Adjust_Matrix_Fused, fused_penmat,
                                        alpha, gamma,
                                        lambda_sparsity_grid,
                                        lambda_sparsity_opt,
                                        lambda_diversity_opt,
                                        tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                        eps, cost_matrix, x0, CV_criterion, start0, n_threads,
                                        cv_iteration = NA_integer_,
                                        eigen_cache){
  # n_folds <- 1
  p <- ncol(x)
  x00 <- x0
  search_direction <- ifelse(sparsity_search, "sparsity", "diversity")

  if(sparsity_search){ # Search for optimal sparsity parameter

    n_lambda_sparsity <- length(lambda_sparsity_grid)

    ####### parallel calculation ###
    cls <- makeCluster(n_threads)
    registerDoParallel(cls)
    cv_results_list <- foreach(fold = 1:n_folds, .packages = c("Rcpp","ECSLRx","Matrix")) %dopar% {

      train.index <- folds_index[[fold]]
      validation.index <- setdiff(sample_ind, train.index)
      fold_cost_without_algorithm <- cost_without_algorithm(cost_matrix[validation.index,,drop=FALSE],
                                                            y[validation.index])

      x_train <- x[train.index,]
      x_validation <- x[validation.index,]

      cv_errors_sparsity <- rep(0, n_lambda_sparsity)
      cv_cost_sparsity <- rep(0, n_lambda_sparsity)
      insample_obj_sparsity <- rep(0, n_lambda_sparsity)
      x0 <- x00
      for(sparsity_ind in n_lambda_sparsity:1){
        # If lambda_diversity=0, it is not necessary to set M>1.
        # G_2 <- ifelse(sparsity_search && lambda_diversity_opt==0,1,M)
        lambda_sparsity <- lambda_sparsity_grid[sparsity_ind]
        lambda_diversity <- lambda_diversity_opt
        #####################start from two, and choose the better one
        ECSLR_model_fold <- ECSLRx::ECSLR_multi_initial(x_train, y[train.index], cost_matrix[train.index,],
                                                M, include_intercept,
                                                w_lasso,
                                                Group_Matrix, w_group,
                                                Adjust_Matrix_Fused, fused_penmat,
                                                alpha, gamma,
                                                lambda_sparsity,
                                                lambda_diversity,
                                                tolerance, obj_tol_rel, max_iter, max_iter_admm, model, x0=x0, start0 = start0,
                                                eigen_cache = eigen_cache)
        insample_obj_sparsity[sparsity_ind] <- ECSLR_model_fold$objective
        #####################

        x0 <- ECSLR_model_fold$coef_x_std

        aec_cost <- ECSLRx::Compute_aec_cost(ECSLR_model_fold$intercept_est, ECSLR_model_fold$betas_est,
                                     x_validation, y[validation.index],
                                     cost_matrix[validation.index,], model)

        cv_errors_sparsity[sparsity_ind] <- aec_cost$AEC
        cv_cost_sparsity[sparsity_ind] <- aec_cost$cost

      }

      data.frame(insample_obj_sparsity = insample_obj_sparsity,
                 cv_errors_sparsity = cv_errors_sparsity,
                 cv_cost_sparsity = cv_cost_sparsity,
                 fold_cost_without_algorithm = fold_cost_without_algorithm,
                 cv_savings_sparsity = 1 - cv_cost_sparsity / fold_cost_without_algorithm)
    }
    stopCluster(cls)
    cv_cost_sparsity_mat <- sapply(cv_results_list, function(df) df$cv_cost_sparsity)
    cv_savings_sparsity_mat <- sapply(cv_results_list, function(df) df$cv_savings_sparsity)
    insample_obj_sparsity_mat <- sapply(cv_results_list, function(df) df$insample_obj_sparsity)
    cv_errors_sparsity_mat <- sapply(cv_results_list, function(df) df$cv_errors_sparsity)
    cv_cost_without_algorithm <- mean(sapply(cv_results_list,
                                             function(df) df$fold_cost_without_algorithm[1]),
                                      na.rm = TRUE)

    cv_errors_sparsity_mean <- apply(cv_errors_sparsity_mat, 1, mean)
    cv_cost_sparsity_mean <- apply(cv_cost_sparsity_mat, 1, mean)
    cv_savings_sparsity_mean <- apply(cv_savings_sparsity_mat, 1, mean)
    insample_obj_sparsity_mean <- apply(insample_obj_sparsity_mat, 1, mean)

    cv_results <- data.frame(M=M,
                             alpha = alpha, gamma = gamma,
                             lambda_sparsity = lambda_sparsity_grid,
                             lambda_diversity = lambda_diversity_opt,
                             insample_obj = insample_obj_sparsity_mean,
                             cv_errors = cv_errors_sparsity_mean,
                             cv_cost = cv_cost_sparsity_mean,
                             cv_cost_without_algorithm = cv_cost_without_algorithm,
                             cv_savings = cv_savings_sparsity_mean,
                             search_direction = search_direction,
                             iteration_id = cv_iteration)

    print(cv_results)

    if(CV_criterion=="cost"){
      cv_opt_new <- min(cv_cost_sparsity_mean, na.rm = TRUE)
      # index_sparsity_opt <- which(cv_cost_sparsity_mean == cv_opt_new)[1]
      index_sparsity_opt <- max(which(cv_cost_sparsity_mean == cv_opt_new)) #if lambda_1, lambda_2 all give the minimum cost, then choose large one.
    }else if(CV_criterion=="AEC"){
      cv_opt_new <- min(cv_errors_sparsity_mean, na.rm = TRUE)
      # index_sparsity_opt <- which(cv_errors_sparsity_mean == cv_opt_new)[1]
      index_sparsity_opt <- max(which(cv_errors_sparsity_mean == cv_opt_new))
    }
    lambda_sparsity_opt <- lambda_sparsity_grid[index_sparsity_opt]

  } else {# diversity search

    temp <- ECSLRx::Compute_Lambda_Diversity_Grid(x, y, cost_matrix,
                                          M, include_intercept,
                                          w_lasso,
                                          Group_Matrix, w_group,
                                          Adjust_Matrix_Fused, fused_penmat,
                                          alpha, gamma,
                                          lambda_sparsity_opt, n_lambda_diversity,
                                          tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                          eps,
                                          eigen_cache = eigen_cache)
    lambda_diversity_max <- temp$lambda_diversity_max
    lambda_diversity_grid <- temp$lambda_diversity_grid
    n_lambda_diversity <- length(lambda_diversity_grid)

    # library(doSNOW)
    # cld <- makeCluster(n_threads, outfile="log.txt")
    # registerDoSNOW(cld)
    cld <- makeCluster(n_threads)
    registerDoParallel(cld)
    cv_results_list <- foreach(fold = 1:n_folds, .packages =  c("Rcpp", "ECSLRx", "Matrix")) %dopar% {

      train.index <- folds_index[[fold]]
      validation.index <- setdiff(sample_ind, train.index)
      fold_cost_without_algorithm <- cost_without_algorithm(cost_matrix[validation.index,,drop=FALSE],
                                                            y[validation.index])

      x_train <- x[train.index,]
      x_validation <- x[validation.index,]

      cv_errors_diversity <- rep(0, n_lambda_diversity)
      cv_cost_diversity <- rep(0, n_lambda_diversity)
      insample_obj_diversity <- rep(0, n_lambda_diversity)
      x0 <- x00
      for(diversity_ind in n_lambda_diversity:1){

        lambda_diversity <- lambda_diversity_grid[diversity_ind]
        lambda_sparsity <- lambda_sparsity_opt

        ECSLR_model_fold <- ECSLRx::ECSLR_multi_initial(x_train, y[train.index], cost_matrix[train.index,],
                                                M, include_intercept,
                                                w_lasso,
                                                Group_Matrix, w_group,
                                                Adjust_Matrix_Fused, fused_penmat,
                                                alpha, gamma,
                                                lambda_sparsity,
                                                lambda_diversity,
                                                tolerance, obj_tol_rel, max_iter, max_iter_admm, model, x0=x0, start0 = start0,
                                                eigen_cache = eigen_cache)
        insample_obj_diversity[diversity_ind] <- ECSLR_model_fold$objective

        x0 <- ECSLR_model_fold$coef_x_std

        aec_cost <- ECSLRx::Compute_aec_cost(ECSLR_model_fold$intercept_est, ECSLR_model_fold$betas_est,
                                     x_validation, y[validation.index],
                                     cost_matrix[validation.index,], model)

        cv_errors_diversity[diversity_ind] <- aec_cost$AEC
        cv_cost_diversity[diversity_ind] <- aec_cost$cost

      }
      data.frame(insample_obj_diversity = insample_obj_diversity,
                 cv_errors_diversity = cv_errors_diversity,
                 cv_cost_diversity = cv_cost_diversity,
                 fold_cost_without_algorithm = fold_cost_without_algorithm,
                 cv_savings_diversity = 1 - cv_cost_diversity / fold_cost_without_algorithm)
    }
    stopCluster(cld)

    cv_cost_diversity_mat <- sapply(cv_results_list, function(df) df$cv_cost_diversity)
    cv_savings_diversity_mat <- sapply(cv_results_list, function(df) df$cv_savings_diversity)
    insample_obj_diversity_mat <- sapply(cv_results_list, function(df) df$insample_obj_diversity)
    cv_errors_diversity_mat <- sapply(cv_results_list, function(df) df$cv_errors_diversity)
    cv_cost_without_algorithm <- mean(sapply(cv_results_list,
                                             function(df) df$fold_cost_without_algorithm[1]),
                                      na.rm = TRUE)

    cv_errors_diversity_mean <- apply(cv_errors_diversity_mat, 1, mean)
    cv_cost_diversity_mean <- apply(cv_cost_diversity_mat, 1, mean)
    cv_savings_diversity_mean <- apply(cv_savings_diversity_mat, 1, mean)
    insample_obj_diversity_mean <- apply(insample_obj_diversity_mat, 1, mean)

    cv_results <- data.frame(M=M,
                             alpha = alpha, gamma = gamma,
                             lambda_sparsity = lambda_sparsity_opt,
                             lambda_diversity = lambda_diversity_grid,
                             insample_obj = insample_obj_diversity_mean,
                             cv_errors = cv_errors_diversity_mean,
                             cv_cost = cv_cost_diversity_mean,
                             cv_cost_without_algorithm = cv_cost_without_algorithm,
                             cv_savings = cv_savings_diversity_mean,
                             search_direction = search_direction,
                             iteration_id = cv_iteration)
    print(cv_results)

    if(CV_criterion=="cost"){
      cv_opt_new <- min(cv_cost_diversity_mean, na.rm = TRUE)
      # index_diversity_opt <- which(cv_cost_diversity_mean == cv_opt_new)[1]
      index_diversity_opt <- max(which(cv_cost_diversity_mean == cv_opt_new))
    }else if(CV_criterion=="AEC"){
      cv_opt_new <- min(cv_errors_diversity_mean, na.rm = TRUE)
      # index_diversity_opt <- which(cv_errors_diversity_mean == cv_opt_new)[1]
      index_diversity_opt <- max(which(cv_errors_diversity_mean == cv_opt_new))
    }
    lambda_diversity_opt <- lambda_diversity_grid[index_diversity_opt]
  }

  return(list(lambda_sparsity_opt = lambda_sparsity_opt,
              lambda_diversity_opt = lambda_diversity_opt,
              cv_opt_new = cv_opt_new,
              cv_results = cv_results))
}

amount2category <- function(labels, amount){
  # Amount category, used for stratification:
  print(quantile(amount[labels == 1], probs = c(1/3, 2/3)))

  value1 <- quantile(amount[labels == 1], probs = 2/3)
  value2 <- quantile(amount[labels == 1], probs = 1/3)

  amount_category <- rep("high", length(amount))
  amount_category[amount < value1] <- "middle"
  amount_category[amount < value2] <- "low"
  amount_category <- factor(amount_category, levels = c("low", "middle", "high"))

  # print(table(amount_category))
  # print(prop.table(table(amount_category)))
  print(table(Class = labels, amount_category = amount_category))

  data.frame(Class = labels,
             amount_category = amount_category)
}

generateTrainIndex <- function(labels, amount, k, times){ # 2 * 5-folds

  indt <- ECSLRx::amount2category(labels, amount)

  folds <- caret::createMultiFolds(as.factor(paste0(labels,"-",indt$amount_category)), k=k, times = times)

  for(kk in 1:(k*times)){
    print(table(indt[folds[[kk]],]))
  }
  return(folds)
}


# Method to set lambda to new value and return current lambda
Compute_Lambda_Sparsity_Grid <- function(x_std, y, cost_matrix = NULL,
                                         include_intercept,
                                         w_lasso,
                                         Group_Matrix, w_group,
                                         Adjust_Matrix_Fused, fused_penmat,
                                         alpha, gamma,
                                         tolerance, obj_tol_rel, max_iter, max_iter_admm,
                                         eps, n_lambda_sparsity, model = "blank",
                                         eigen_cache){
  n <- nrow(x_std)
  # Maximum lambda_sparsity that kills all variables
  if(grepl("ECSLR", model)){
    xb <- 1/(4 * n) * abs(Matrix_crossprod(x_std, y * (cost_matrix[,4]-cost_matrix[,3]) + (1-y) *(cost_matrix[,2]-cost_matrix[,1])))

    lambdas_max_lasso <- 1/(alpha * gamma) * ECSLRx::max.noinf(xb / w_lasso) # if w_lasso_j=0, it mean that beta_j do not have lasso penalty, we can ignore its lambda_lasso_max. which means that we should ignore
    lambdas_max_group <- 1/((1-alpha) * gamma) * ECSLRx::max.noinf(sqrt(Group_Matrix %*% (xb^2)) / w_group)
    lambdas_max_fused <- 0
    # 1/(1-gamma) * ECSLRx::max.noinf(xb / apply(abs(fused_penmat),2,sum))

    lambda_sparsity_max <- ECSLRx::max.noinf(c(lambdas_max_lasso, lambdas_max_group, lambdas_max_fused))
  }

  n_lambda_sparsity.pre <- n_lambda_sparsity
  lambda_sparsity_grid <- exp(seq(log(eps*lambda_sparsity_max), log(lambda_sparsity_max), length.out=n_lambda_sparsity.pre))
  cat("lambda_sparsity_grid.pre:",lambda_sparsity_grid,"\n")
  # lambda_sparsity_grid <- c(0,exp(seq(log(eps*lambda_sparsity_max), log(lambda_sparsity_max), length.out=n_lambda_sparsity)))

  for(sparsity_ind in (n_lambda_sparsity.pre-1):1){
    lambda_sparsity <- lambda_sparsity_grid[sparsity_ind]
    coef1 <- ECSLRx::ECSLR_Compute_Coef_combine(x_std, y, cost_matrix,
                                        M=1, include_intercept,
                                        w_lasso,
                                        Group_Matrix, w_group,
                                        Adjust_Matrix_Fused, fused_penmat,
                                        alpha, gamma,
                                        lambda_sparsity=lambda_sparsity, lambda_diversity=0,
                                        tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                        eigen_cache_=eigen_cache)
    if(ECSLRx::Check_Zeros_Beta(coef1$betas_scaled)){
      lambda_sparsity_max <- lambda_sparsity_grid[sparsity_ind]
    }else{
      break
    }
  }
  print(paste0("finally lambda_sparsity_max=",lambda_sparsity_max))
  lambda_sparsity_grid <- c(exp(seq(log(eps*lambda_sparsity_max), log(lambda_sparsity_max), length.out=n_lambda_sparsity)))

  cat("lambda_sparsity_grid: \n")
  cat(lambda_sparsity_grid,"\n")

  return(list(lambda_sparsity_max=lambda_sparsity_max, lambda_sparsity_grid=lambda_sparsity_grid))
}

# calculate the maximum value (ignore the infinite value).
# all of the input is positive
max.noinf <- function(vec){
  s <- vec[is.finite(vec)]
  max(s,0)
}

# Check whether betas all equal to 0 (tolerance-based)
Check_Zeros_Beta_bak <- function(betas) {
  Zeros <- FALSE
  if(sum(abs(betas) > 1e-7)==0){
    Zeros <- TRUE
  }
  return(Zeros)
}

# Check whether betas all equal to 0 (exact-zero legacy strategy)
Check_Zeros_Beta <- function(betas) {
  Zeros <- FALSE
  if(sum(abs(betas) > 0)==0){
    Zeros <- TRUE
  }
  return(Zeros)
}

#Function to returns a vector with ones corresponding to the betas that have interactions.
Check_Interactions <- function(betas) {
  k <- dim(betas)[3]
  checks <- rep(0, k)
  all_ones <- rep(1, k)
  for(ii in 1:k) {
    checks[ii] <- ECSLRx::Check_Interactions_Beta(betas[,,ii])
  }
  return(checks == all_ones)
}

Check_Interactions_Beta <- function(beta){
  p <- nrow(beta)
  interactions <- FALSE
  for (i in 1:p) {
    temp <- abs(beta[i, ])
    num <- sum(temp > 0) #sum(temp > 1e-8)
    if (num > 1) {
      interactions <- TRUE
      break
    }
  }
  return(interactions)
}


Get_Lambda_Diversity_Max <- function(x, y, cost_matrix,
                                     M, include_intercept,
                                     w_lasso,
                                     Group_Matrix, w_group,
                                     Adjust_Matrix_Fused, fused_penmat,
                                     alpha, gamma,
                                     lambda_sparsity_opt,
                                     tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                     eigen_cache){

  n <- nrow(x)
  p <- ncol(x)

  # Maximum lambda_diversity that kills all interactions
  betas_scaled <- array(0, dim=c(p, M))

  # Initial guess for the diversity penalty
  lambda_diversity_max <- M

  # Split model to determine the maximum diversity parameter
  coef1 <- ECSLRx::ECSLR_Compute_Coef_combine(x, y, cost_matrix,
                                      M, include_intercept,
                                      w_lasso,
                                      Group_Matrix, w_group,
                                      Adjust_Matrix_Fused, fused_penmat,
                                      alpha, gamma,
                                         lambda_sparsity=lambda_sparsity_opt, lambda_diversity=lambda_diversity_max,
                                         tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                         eigen_cache_=eigen_cache)
  betas_scaled <- coef1$betas_scaled
  GRID_INTERACTION_MAX_COUNTER <- 5
  for(i in 1:GRID_INTERACTION_MAX_COUNTER){
    # If interactions remain, increase lambda_diversity_max by scaling it by a constant factor of two
    if(ECSLRx::Check_Interactions_Beta(betas_scaled)){
      lambda_diversity_max <- lambda_diversity_max * 2
      coef1 <- ECSLRx::ECSLR_Compute_Coef_combine(x, y, cost_matrix,
                                          M, include_intercept,
                                          w_lasso,
                                          Group_Matrix, w_group,
                                          Adjust_Matrix_Fused, fused_penmat,
                                          alpha, gamma,
                                             lambda_sparsity=lambda_sparsity_opt, lambda_diversity=lambda_diversity_max,
                                             tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                             eigen_cache_=eigen_cache)
      betas_scaled <- coef1$betas_scaled
      # print(betas_scaled)
    }else{
      break
    }
  }
  # If we could not kill all the interactions
  if(ECSLRx::Check_Interactions_Beta(betas_scaled)){
    warning("Failure to find lambda_diversity that kills all interactions.")
  }

  # Return the diversity penalty parameter
  return(list(lambda_diversity_max=lambda_diversity_max,betas_scaled=betas_scaled))
}

# Method to set lambda to new value and return current lambda
Compute_Lambda_Diversity_Grid <- function(x, y, cost_matrix,
                                          M, include_intercept,
                                          w_lasso,
                                          Group_Matrix, w_group,
                                          Adjust_Matrix_Fused, fused_penmat,
                                          alpha, gamma,
                                          lambda_sparsity_opt, n_lambda_diversity,
                                          tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                          eps,
                                          eigen_cache){

  Lambda_Diversity_temp <- ECSLRx::Get_Lambda_Diversity_Max(x, y, cost_matrix,
                                                     M, include_intercept,
                                                     w_lasso,
                                                     Group_Matrix, w_group,
                                                     Adjust_Matrix_Fused, fused_penmat,
                                                     alpha, gamma,
                                                     lambda_sparsity_opt,
                                                     tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                                     eigen_cache = eigen_cache)
  lambda_diversity_max <- Lambda_Diversity_temp$lambda_diversity_max
  betas_scaled <- Lambda_Diversity_temp$betas_scaled
  p <- ncol(x)
  # for preliminary grid
  eps.pre <- eps
  n_lambda_diversity.pre <- n_lambda_diversity #20
  lambda_diversity_grid.pre <- exp(seq(log(eps.pre * lambda_diversity_max), log(lambda_diversity_max), length = n_lambda_diversity.pre))
  cat("lambda_diversity_grid.pre: \n")
  cat(lambda_diversity_grid.pre,"\n \n")

  # If we could not kill all the interactions, keep lambda_diversity_max
  if(ECSLRx::Check_Interactions_Beta(beta = betas_scaled)){
    cat("Failure to find lambda_diversity that kills all interactions. \n")
  } else {

    # Find smallest lambda_diversity in the grid such that there are no interactions
    for(diversity_ind in ((n_lambda_diversity.pre-1):1)){
      lambda_diversity <- lambda_diversity_grid.pre[diversity_ind]
      coef1 <- ECSLRx::ECSLR_Compute_Coef_combine(x, y, cost_matrix,
                                          M, include_intercept,
                                          w_lasso,
                                          Group_Matrix, w_group,
                                          Adjust_Matrix_Fused, fused_penmat,
                                          alpha, gamma,
                                             lambda_sparsity=lambda_sparsity_opt, lambda_diversity=lambda_diversity,
                                             tolerance, obj_tol_rel, max_iter, max_iter_admm, model,
                                             eigen_cache_=eigen_cache)

      if(ECSLRx::Check_Interactions_Beta(coef1$betas_scaled)){
        break
      }else{
        lambda_diversity_max <- lambda_diversity_grid.pre[diversity_ind]
      }
    }
  }

  lambda_diversity_grid <- c(exp(seq(log(eps * lambda_diversity_max), log(lambda_diversity_max), length = n_lambda_diversity)))
  cat("lambda_diversity_grid: \n")
  cat(lambda_diversity_grid,"\n \n")

  return(list(lambda_diversity_max=lambda_diversity_max,lambda_diversity_grid=lambda_diversity_grid))
}

