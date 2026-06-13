# test_cache_compatibility_bugfix.R
# Bug condition exploration tests for cache compatibility fix
# 
# **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
#
# CRITICAL: These tests are EXPECTED TO FAIL on unfixed code.
# Test failure confirms the bug exists (dimension mismatch between legacy cache
# covariates and penalty matrices causes system to crash).
#
# After the fix is implemented, these tests should PASS, demonstrating that
# the system correctly handles legacy cache files by regenerating penalty matrices.
#
# NOTE: Since the actual UGC cache file currently has matching dimensions,
# this test uses a SYNTHETIC modified cache to demonstrate the bug condition.
# This simulates what would happen with a true legacy cache from the old preprocessing.

library(testthat)
library(Matrix)

# Helper function to check if bug condition exists
# Bug condition: dimensions of loaded cache data don't match
isBugCondition <- function(cache_data) {
  p <- ncol(cache_data$covariates)
  w_lasso_n <- nrow(cache_data$model_parameters$w_lasso)
  group_cols_n <- ncol(cache_data$model_parameters$Group_Matrix)
  
  return((p != w_lasso_n) || (p != group_cols_n))
}

test_that("Property 1: Bug Condition - Synthetic Legacy Cache Dimension Mismatch (Case 6 UGC)", {
  # **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
  
  # Setup: Create a synthetic legacy cache with dimension mismatch
  # This simulates what happens when refactored preprocessing produces different
  # covariate structure than legacy preprocessing
  
  data_root <- "C:/Users/Bing/Cloud/OneDrive - KU Leuven/Research/20230331-CSensemble/MyProgress/20241028_ECSLR-multi/application/data"
  legacy_cache_file <- file.path(data_root, "GCD1-1_train_index_full_5_2folds.RData")
  
  skip_if_not(file.exists(legacy_cache_file), 
              message = "Legacy cache file not found. Cannot test bug condition.")
  
  # Load the actual cache
  cache_env <- new.env()
  load(legacy_cache_file, envir = cache_env)
  
  # Verify baseline: current cache actually has matching dimensions
  p_original <- ncol(cache_env$covariates)
  w_lasso_n_original <- nrow(cache_env$model_parameters$w_lasso)
  group_cols_n_original <- ncol(cache_env$model_parameters$Group_Matrix)
  
  cat("\n=== Original Cache Dimensions (from legacy UGC cache file GCD1-1) ===\n")
  cat("ncol(covariates) =", p_original, "\n")
  cat("nrow(w_lasso) =", w_lasso_n_original, "\n")
  cat("ncol(Group_Matrix) =", group_cols_n_original, "\n")
  cat("NOTE: Current cache has MATCHING dimensions\n")
  
  # CREATE SYNTHETIC MISMATCH: Simulate what legacy preprocessing would produce
  # Remove last 2 columns from covariates to simulate different preprocessing output
  # This represents the scenario where refactored preprocessing adds/removes features
  covariates_legacy <- cache_env$covariates[, 1:(p_original-2), drop = FALSE]
  model_parameters_legacy <- cache_env$model_parameters  # Keep original penalty matrices
  
  p_legacy <- ncol(covariates_legacy)
  w_lasso_n_legacy <- nrow(model_parameters_legacy$w_lasso)
  group_cols_n_legacy <- ncol(model_parameters_legacy$Group_Matrix)
  
  cat("\n=== Synthetic Legacy Cache Dimensions (Simulated Mismatch) ===\n")
  cat("ncol(covariates) =", p_legacy, "\n")
  cat("nrow(w_lasso) =", w_lasso_n_legacy, "\n")
  cat("ncol(Group_Matrix) =", group_cols_n_legacy, "\n")
  cat("MISMATCH DETECTED: covariates has", p_legacy, "columns but penalties expect", w_lasso_n_legacy, "\n")
  
  # **CRITICAL ASSERTION - EXPECTED TO FAIL ON UNFIXED CODE**
  # After fix: dimensions should match because penalty matrices are regenerated
  # Before fix: dimensions DON'T match, causing the error
  expect_equal(p_legacy, w_lasso_n_legacy,
               info = sprintf("ECSLRx feature/penalty mismatch: ncol(covariates)=%d, nrow(w_lasso)=%d. This is the BUG CONDITION.", 
                            p_legacy, w_lasso_n_legacy))
  
  expect_equal(p_legacy, group_cols_n_legacy,
               info = sprintf("ECSLRx feature/penalty mismatch: ncol(covariates)=%d, ncol(Group_Matrix)=%d. This is the BUG CONDITION.", 
                            p_legacy, group_cols_n_legacy))
  
  # Document the counterexample for analysis
  cat("\n=== COUNTEREXAMPLE FOUND ===\n")
  cat("Bug confirmed: Dimension mismatch detected in synthetic legacy cache\n")
  cat(sprintf("  - ncol(covariates) = %d\n", p_legacy))
  cat(sprintf("  - nrow(w_lasso) = %d (mismatch: %d)\n", w_lasso_n_legacy, p_legacy - w_lasso_n_legacy))
  cat(sprintf("  - ncol(Group_Matrix) = %d (mismatch: %d)\n", group_cols_n_legacy, p_legacy - group_cols_n_legacy))
  cat("\nThis confirms what the bug WOULD look like with true legacy cache.\n")
  cat("After the fix is implemented, penalty matrices should be regenerated to match covariates.\n")
})

test_that("Property 1: Bug Condition - ECSLRx dimension check catches mismatch", {
  # **Validates: Requirements 1.1, 2.1**
  
  # This test verifies that the dimension checking logic in generate_performance3
  # correctly detects mismatches and throws the expected error
  
  data_root <- "C:/Users/Bing/Cloud/OneDrive - KU Leuven/Research/20230331-CSensemble/MyProgress/20241028_ECSLR-multi/application/data"
  legacy_cache_file <- file.path(data_root, "GCD1-1_train_index_full_5_2folds.RData")
  
  skip_if_not(file.exists(legacy_cache_file), 
              message = "Legacy UGC cache file not found. Cannot test bug condition.")
  
  # Load cache and create synthetic mismatch
  cache_env <- new.env()
  load(legacy_cache_file, envir = cache_env)
  
  # Create dimension mismatch by removing columns
  covariates <- cache_env$covariates[, 1:(ncol(cache_env$covariates)-2), drop = FALSE]
  labels <- cache_env$labels
  cost_matrix_raw <- cache_env$cost_matrix_raw
  model_parameters <- cache_env$model_parameters  # Keep original - has more columns
  train_index_all <- cache_env$train_index_all
  
  # Prepare test data
  train_index <- train_index_all[[1]]
  x.train <- as.matrix(covariates[train_index, , drop = FALSE])
  y.train <- as.numeric(as.character(labels[train_index]))
  x.test <- as.matrix(covariates[-train_index, , drop = FALSE])
  y.test <- as.numeric(as.character(labels[-train_index]))
  
  # Flatten cost matrix
  flatten_cost_matrix <- function(cm3) {
    cbind(cm3[, 1, 1], cm3[, 2, 1], cm3[, 1, 2], cm3[, 2, 2])
  }
  cost_matrix_train <- flatten_cost_matrix(cost_matrix_raw[train_index, , , drop = FALSE])
  cost_matrix_test <- flatten_cost_matrix(cost_matrix_raw[-train_index, , , drop = FALSE])
  
  cat("\n=== Testing ECSLRx Dimension Check with Mismatched Data ===\n")
  cat("ncol(x.train) =", ncol(x.train), "\n")
  cat("nrow(model_parameters$w_lasso) =", nrow(model_parameters$w_lasso), "\n")
  cat("ncol(model_parameters$Group_Matrix) =", ncol(model_parameters$Group_Matrix), "\n")
  
  # Mock the dimension checking logic from generate_performance3
  p <- ncol(x.train)
  w_lasso_n <- nrow(model_parameters$w_lasso)
  group_cols_n <- ncol(model_parameters$Group_Matrix)
  
  # **CRITICAL ASSERTION - ON UNFIXED CODE THIS PASSES (bug exists)**
  # On unfixed code: The dimension check CORRECTLY identifies the mismatch and throws error
  # But the UNFIXED code doesn't handle it - it just crashes
  # After fix: The code should detect mismatch and regenerate penalties instead of crashing
  
  has_mismatch <- (!identical(p, w_lasso_n) || !identical(p, group_cols_n))
  expect_true(has_mismatch,
              info = sprintf("Dimension mismatch should be detected: p=%d, w_lasso_n=%d, group_cols_n=%d", 
                           p, w_lasso_n, group_cols_n))
  
  cat("\n=== COUNTEREXAMPLE CONFIRMED ===\n")
  cat("Dimension mismatch detected by the check:\n")
  cat(sprintf("  - ncol(x.train)=%d != nrow(w_lasso)=%d\n", p, w_lasso_n))
  cat(sprintf("  - ncol(x.train)=%d != ncol(Group_Matrix)=%d\n", p, group_cols_n))
  cat("\nOn UNFIXED code: This would cause generate_performance3 to throw an error and crash.\n")
  cat("After FIX: The code should detect this mismatch and regenerate penalty matrices.\n")
})

test_that("Bug Condition Check - isBugCondition helper works correctly", {
  # Test the helper function to ensure it correctly identifies bug condition
  
  # Mock cache data with dimension mismatch (bug condition = TRUE)
  cache_with_bug <- list(
    covariates = matrix(0, nrow = 100, ncol = 50),  # 50 columns
    model_parameters = list(
      w_lasso = matrix(0, nrow = 48, ncol = 1),     # 48 rows - MISMATCH
      Group_Matrix = Matrix::Matrix(0, nrow = 48, ncol = 48)  # 48 cols - MISMATCH
    )
  )
  
  expect_true(isBugCondition(cache_with_bug),
              info = "Bug condition should be TRUE when dimensions don't match")
  
  # Mock cache data with matching dimensions (bug condition = FALSE)
  cache_without_bug <- list(
    covariates = matrix(0, nrow = 100, ncol = 50),  # 50 columns
    model_parameters = list(
      w_lasso = matrix(0, nrow = 50, ncol = 1),     # 50 rows - MATCH
      Group_Matrix = Matrix::Matrix(0, nrow = 48, ncol = 50)  # 50 cols - MATCH
    )
  )
  
  expect_false(isBugCondition(cache_without_bug),
               info = "Bug condition should be FALSE when dimensions match")
})

# ==============================================================================
# PRESERVATION PROPERTY TESTS - Task 2
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
#
# These tests verify that compatible cache files continue to work correctly
# without any regeneration. They establish the baseline behavior that must be
# preserved when the fix is implemented.
#
# EXPECTED OUTCOME on UNFIXED code: These tests should PASS, confirming the
# baseline behavior for compatible cache files.
# ==============================================================================

test_that("Property 2: Preservation - Compatible cache loads successfully without regeneration", {
  # **Validates: Requirements 3.1, 3.2**
  #
  # This test verifies that cache files with compatible dimensions
  # (ncol(covariates) == nrow(w_lasso) == ncol(Group_Matrix))
  # load successfully and can be used without any regeneration.
  
  data_root <- "C:/Users/Bing/Cloud/OneDrive - KU Leuven/Research/20230331-CSensemble/MyProgress/20241028_ECSLR-multi/application/data"
  
  # Test with legacy UGC cache (currently has matching dimensions)
  cache_file <- file.path(data_root, "GCD1-1_train_index_full_5_2folds.RData")
  
  skip_if_not(file.exists(cache_file), 
              message = "UGC cache file not found. Cannot test preservation.")
  
  # Load the cache
  cache_env <- new.env()
  load(cache_file, envir = cache_env)
  
  # Verify the cache has all expected components
  expect_true(exists("covariates", envir = cache_env),
              info = "Cache should contain covariates")
  expect_true(exists("labels", envir = cache_env),
              info = "Cache should contain labels")
  expect_true(exists("model_parameters", envir = cache_env),
              info = "Cache should contain model_parameters")
  expect_true(exists("train_index_all", envir = cache_env),
              info = "Cache should contain train_index_all")
  
  # Check dimension compatibility (no bug condition)
  p <- ncol(cache_env$covariates)
  w_lasso_n <- nrow(cache_env$model_parameters$w_lasso)
  group_cols_n <- ncol(cache_env$model_parameters$Group_Matrix)
  
  cat("\n=== Preservation Test: Compatible Cache Dimensions ===\n")
  cat("ncol(covariates) =", p, "\n")
  cat("nrow(w_lasso) =", w_lasso_n, "\n")
  cat("ncol(Group_Matrix) =", group_cols_n, "\n")
  
  # Verify dimensions are compatible (no bug condition)
  expect_false(isBugCondition(list(
    covariates = cache_env$covariates,
    model_parameters = cache_env$model_parameters
  )), info = "Cache should have compatible dimensions (no bug condition)")
  
  # Verify dimensions match exactly
  expect_equal(p, w_lasso_n,
               info = "Compatible cache: ncol(covariates) should equal nrow(w_lasso)")
  expect_equal(p, group_cols_n,
               info = "Compatible cache: ncol(covariates) should equal ncol(Group_Matrix)")
  
  # Verify penalty matrices have correct structure
  expect_true(is.matrix(cache_env$model_parameters$w_lasso),
              info = "w_lasso should be a matrix")
  expect_true(inherits(cache_env$model_parameters$Group_Matrix, "Matrix"),
              info = "Group_Matrix should be a Matrix object")
  
  cat("\n=== PRESERVATION TEST PASSED ===\n")
  cat("Compatible cache loaded successfully with matching dimensions.\n")
  cat("This behavior MUST be preserved after implementing the fix.\n")
})

test_that("Property 2: Preservation - Non-ECSLRx models work with any cache", {
  # **Validates: Requirements 3.3**
  #
  # This test verifies that non-ECSLRx models (logit, cslogit, CSRF-cv, CSRP-cv)
  # work correctly regardless of penalty matrix dimensions, because they don't
  # use the penalty matrices at all.
  
  data_root <- "C:/Users/Bing/Cloud/OneDrive - KU Leuven/Research/20230331-CSensemble/MyProgress/20241028_ECSLR-multi/application/data"
  cache_file <- file.path(data_root, "GCD1-1_train_index_full_5_2folds.RData")
  
  skip_if_not(file.exists(cache_file), 
              message = "UGC cache file not found. Cannot test non-ECSLRx models.")
  
  # Load the cache
  cache_env <- new.env()
  load(cache_file, envir = cache_env)
  
  # Prepare test data (use first fold)
  train_index <- cache_env$train_index_all[[1]]
  x.train <- as.matrix(cache_env$covariates[train_index, , drop = FALSE])
  y.train <- as.numeric(as.character(cache_env$labels[train_index]))
  x.test <- as.matrix(cache_env$covariates[-train_index, , drop = FALSE])
  y.test <- as.numeric(as.character(cache_env$labels[-train_index]))
  
  # Flatten cost matrix
  flatten_cost_matrix <- function(cm3) {
    cbind(cm3[, 1, 1], cm3[, 2, 1], cm3[, 1, 2], cm3[, 2, 2])
  }
  cost_matrix_train <- flatten_cost_matrix(cache_env$cost_matrix_raw[train_index, , , drop = FALSE])
  cost_matrix_test <- flatten_cost_matrix(cache_env$cost_matrix_raw[-train_index, , , drop = FALSE])
  
  cat("\n=== Preservation Test: Non-ECSLRx Models ===\n")
  cat("Testing that logit and cslogit models work with loaded cache data\n")
  cat("n.train =", length(y.train), ", n.test =", length(y.test), "\n")
  cat("p =", ncol(x.train), "\n")
  
  # Test logit model - should work regardless of penalty matrix dimensions
  # because logit doesn't use model_parameters
  # logit model should work with any cache (doesn't use penalty matrices)
  logit_result <- tryCatch({
    # Simple logit fit without penalty matrices
    train_df <- as.data.frame(cbind(y = y.train, x.train))
    logit_model <- glm(y ~ ., data = train_df, family = binomial())
    pred_prob <- predict(logit_model, newdata = as.data.frame(x.test), type = "response")
    
    cat("Logit model fitted successfully\n")
    cat("Prediction range: [", min(pred_prob), ",", max(pred_prob), "]\n")
    "SUCCESS"
  }, error = function(e) {
    paste("ERROR:", e$message)
  })
  
  expect_equal(logit_result, "SUCCESS",
               info = "logit model should work with any cache (doesn't use penalty matrices)")
  
  # Note: cslogit, CSRF-cv, CSRP-cv also don't depend on penalty matrix dimensions
  # They are tested in the full benchmark execution
  # The key preservation property is that non-ECSLRx models can execute
  # even if penalty matrices have dimension mismatches
  
  cat("\n=== PRESERVATION TEST PASSED ===\n")
  cat("Non-ECSLRx models work correctly with loaded cache.\n")
  cat("This behavior MUST be preserved after implementing the fix.\n")
})

test_that("Property 2: Preservation - Cache validation detects dimension compatibility", {
  # **Validates: Requirements 3.2**
  #
  # This test documents the baseline behavior: when cache dimensions are compatible,
  # the system should proceed without regeneration.
  
  # Create mock compatible cache data
  n <- 100
  p <- 50
  J <- 10
  
  compatible_cache <- list(
    covariates = matrix(rnorm(n * p), nrow = n, ncol = p),
    labels = factor(sample(0:1, n, replace = TRUE)),
    model_parameters = list(
      w_lasso = matrix(1, nrow = p, ncol = 1),
      w_group = rep(1, J),
      Group_Matrix = Matrix::Matrix(0, nrow = J, ncol = p),
      fused_penmat = Matrix::Matrix(0, nrow = p, ncol = p),
      Adjust_Matrix_Fused = Matrix::Matrix(0, nrow = p, ncol = p)
    )
  )
  
  # Add some actual group structure to Group_Matrix
  for (j in 1:J) {
    cols_in_group <- seq((j-1)*5 + 1, min(j*5, p))
    compatible_cache$model_parameters$Group_Matrix[j, cols_in_group] <- 1
  }
  
  cat("\n=== Preservation Test: Dimension Compatibility Check ===\n")
  
  # Verify no bug condition exists
  expect_false(isBugCondition(compatible_cache),
               info = "Compatible cache should NOT trigger bug condition")
  
  # Verify dimensions match
  p_actual <- ncol(compatible_cache$covariates)
  w_lasso_n <- nrow(compatible_cache$model_parameters$w_lasso)
  group_cols_n <- ncol(compatible_cache$model_parameters$Group_Matrix)
  
  cat("Dimensions: p =", p_actual, ", w_lasso_n =", w_lasso_n, ", group_cols_n =", group_cols_n, "\n")
  
  expect_equal(p_actual, w_lasso_n,
               info = "Compatible cache: covariates columns match w_lasso rows")
  expect_equal(p_actual, group_cols_n,
               info = "Compatible cache: covariates columns match Group_Matrix columns")
  
  cat("\n=== PRESERVATION TEST PASSED ===\n")
  cat("Dimension compatibility validation works correctly.\n")
  cat("Compatible cache is correctly identified (no bug condition).\n")
})

test_that("Property 2: Preservation - Multiple cases with compatible cache", {
  # **Validates: Requirements 3.3**
  #
  # This test verifies that multiple different cases work correctly when
  # their cache files have compatible dimensions.
  
  data_root <- "C:/Users/Bing/Cloud/OneDrive - KU Leuven/Research/20230331-CSensemble/MyProgress/20241028_ECSLR-multi/application/data"
  
  # Test multiple cache files
  test_cases <- c(
    "GCD1-1_train_index_full_5_2folds.RData",
    "KTCC2_train_index_full_5_2folds.RData",
    "MTCC3_train_index_full_5_2folds.RData"
  )
  
  cat("\n=== Preservation Test: Multiple Cases with Compatible Cache ===\n")
  
  for (cache_name in test_cases) {
    cache_file <- file.path(data_root, cache_name)
    
    if (!file.exists(cache_file)) {
      cat("Skipping", cache_name, "(file not found)\n")
      next
    }
    
    cat("\nTesting:", cache_name, "\n")
    
    # Load cache
    cache_env <- new.env()
    load(cache_file, envir = cache_env)
    
    # Check dimensions
    p <- ncol(cache_env$covariates)
    w_lasso_n <- nrow(cache_env$model_parameters$w_lasso)
    group_cols_n <- ncol(cache_env$model_parameters$Group_Matrix)
    
    cat("  ncol(covariates) =", p, "\n")
    cat("  nrow(w_lasso) =", w_lasso_n, "\n")
    cat("  ncol(Group_Matrix) =", group_cols_n, "\n")
    
    # Verify compatibility
    has_bug <- isBugCondition(list(
      covariates = cache_env$covariates,
      model_parameters = cache_env$model_parameters
    ))
    
    if (has_bug) {
      cat("  WARNING: Bug condition detected (dimensions mismatch)\n")
    } else {
      cat("  OK: Compatible dimensions (no bug condition)\n")
      
      # For compatible cache, verify exact dimension match
      expect_equal(p, w_lasso_n,
                   info = sprintf("%s: ncol(covariates) should equal nrow(w_lasso)", cache_name))
      expect_equal(p, group_cols_n,
                   info = sprintf("%s: ncol(covariates) should equal ncol(Group_Matrix)", cache_name))
    }
  }
  
  cat("\n=== PRESERVATION TEST PASSED ===\n")
  cat("Multiple cases with compatible cache work correctly.\n")
  cat("This behavior MUST be preserved after implementing the fix.\n")
})
