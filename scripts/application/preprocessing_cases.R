# Final application-case preprocessing (public reproduction branch).
# Each function hardcodes the parameter settings used in the paper experiments.

library(dplyr)
library(tidyr)
library(stringr)

# Allow the caller to override the raw-data root; keep the legacy relative path as fallback.
data_root <- if (exists("data_root", inherits = TRUE)) data_root else "../../Data"

# Dispatch to the paper-case preprocessor by case name.
preprocess_case <- function(case) {
  switch(case,
    "KTCC" = preprocess_KTCC(),
    "UGC" = preprocess_UGC(),
    "BMTL" = preprocess_BMTL(),
    "MTCC" = preprocess_MTCC(),
    "KVIC" = preprocess_KVIC(),
    "DCCC" = preprocess_DCCC(),
    stop("Unknown application case: ", case)
  )
}

# KTCC — Kaggle Telco Customer Churn
preprocess_KTCC <- function() {
  # --- Load raw data ---
  telco_churn <- read.csv(file.path(data_root, "Kaggle Telco Customer Churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

  # --- Clean rows and harmonize service-level categories ---
  data <- telco_churn %>%
    dplyr::select(-customerID) %>%
    filter(!(rowSums(. == "" | . == " " | is.na(.)) > 0)) %>%
    mutate(
      across(c(MultipleLines), ~ ifelse(. == "No phone service", "No", .)),
      across(c(OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies),
             ~ ifelse(. == "No internet service", "No", .))
    )

  # --- Factorize categoricals; derive labels and monthly charge amounts ---
  categorical_variables <- c(
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
  )

  raw_cova <- data %>%
    mutate(across(all_of(categorical_variables), ~ factor(.x, levels = sort(unique(.x))))) %>%
    mutate(Contract = factor(Contract, levels = c("Month-to-month", "One year", "Two year"))) %>%
    dplyr::select(-Churn)

  labels <- as.integer(data$Churn == "Yes")
  amounts <- raw_cova$MonthlyCharges
  raw_cova <- dplyr::select(raw_cova, -MonthlyCharges, -TotalCharges)

  # --- Build design matrix, penalties, and default churn cost matrix ---
  .preprocess_finalize(
    raw_cova = raw_cova,
    labels = labels,
    fused_variable = "tenure",
    amounts = amounts
  )
}

# UGC — UCI German Credit
preprocess_UGC <- function() {
  # --- Load UCI German Credit data ---
  german_colnames <- c(
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "class"
  )

  german <- read.table(
    file.path(data_root, "UCI German Credit", "german.data"),
    sep = " ", col.names = german_colnames, stringsAsFactors = FALSE
  )

  labels <- as.integer(german$class == 2)
  credit_amount <- german$credit_amount

  # --- Bin duration and factorize categoricals (age kept numeric) ---
  raw_cova <- german %>%
    mutate(
      duration = cut(
        duration,
        breaks = c(0, 12, 24, 36, 48, Inf),
        labels = c("1-12", "13-24", "25-36", "37-48", "49+"),
        right = TRUE
      ),
      across(
        c(checking_status, credit_history, purpose, savings, employment, installment_rate,
          personal_status, other_debtors, residence_since, property, other_plans, housing,
          existing_credits, job, num_dependents, telephone, foreign_worker, duration),
        ~ factor(.x, levels = sort(unique(.x)))
      ) # age kept numeric
    ) %>%
    dplyr::select(-class, -credit_amount)

  # --- Bahnsen et al. instance-dependent cost matrix ---
  pi_1 <- mean(labels)
  cl <- credit_amount
  cl_avg <- mean(cl)
  n_samples <- length(labels)
  cost_matrix <- array(0, dim = c(n_samples, 2, 2))
  cost_matrix[, 1, 1] <- 0.0
  cost_matrix[, 1, 2] <- v_calculate_cost_fn(cl, lgd)
  cost_matrix[, 2, 1] <- v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
  cost_matrix[, 2, 2] <- 0.0

  # --- Finalize with duration fused penalty ---
  result <- .preprocess_finalize(
    raw_cova = raw_cova,
    labels = labels,
    fused_variable = "duration",
    amounts = credit_amount,
    cost_matrix = cost_matrix
  )
  result
}

# BMTL — Belgian Motor Third-party Liability (CASdatasets)
preprocess_BMTL <- function() {
  # --- Load Belgian motor third-party liability data ---
  library(CASdatasets)
  data(beMTPL16)

  labels <- beMTPL16$signal
  categorical_variables <- c("policy_year", "vehicle_brand", "claim_time", "driving_training_label")

  # --- Derive binned claim_time from hour; factorize covariates ---
  raw_cova <- beMTPL16 %>%
    mutate(across(all_of(categorical_variables), ~ factor(.x, levels = sort(unique(.x))))) %>%
    mutate(
      claim_hour = as.numeric(str_sub(as.character(claim_time), 1, 2)),
      logamount = log(claim_value + 1),
      logcatalog_value = log(catalog_value - min(catalog_value) + 1),
      claim_time = cut(
        claim_hour,
        breaks = c(-1, 3, 6, 9, 12, 15, 18, 21, 24),
        labels = c("00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"),
        right = TRUE
      ),
      claim_time = factor(
        claim_time,
        levels = c("00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24")
      )
    ) %>%
    dplyr::select(-insurance_contract, -signal, -vehicle_model, -claim_hour)

  # --- Expand to model matrix and build claim-based cost matrix ---
  covariates <- as.data.frame(model.matrix(~ ., raw_cova)[, -1, drop = FALSE])
  claim_value <- covariates$claim_value
  fixed_cost <- 350
  n_samples <- length(labels)
  cost_matrix <- array(0, dim = c(n_samples, 2, 2))
  cost_matrix[, 1, 1] <- 0.0
  cost_matrix[, 1, 2] <- claim_value
  cost_matrix[, 2, 1] <- fixed_cost
  cost_matrix[, 2, 2] <- fixed_cost

  # --- Penalty structure: fuse claim_time; group lasso on multi-level factors ---
  df <- vapply(raw_cova, function(x) {
    if (is.factor(x) || is.character(x)) length(unique(x)) - 1 else 1
  }, numeric(1))
  J <- length(df)
  fused_variable <- "claim_time"
  fused_type <- as.list(rep("none", J))
  names(fused_type) <- names(df)
  fused_type[fused_variable] <- "fuse1d"
  fus <- construct_fused_penalty(fused_type, as.list(df))
  Group_Matrix <- construct_Group_matrix(df)
  weight <- generate_weight(df, J, fused_variable, Group_Matrix, colnames(covariates))

  # --- Drop raw scales after the derived log features have entered the design matrix ---
  covariates <- covariates %>%
    dplyr::select(-claim_value, -catalog_value)

  list(
    covariates = covariates,
    labels = labels,
    cost_matrix = cost_matrix,
    df = df,
    w_lasso = weight$w_lasso,
    w_group = weight$w_group,
    Group_Matrix = Group_Matrix,
    fused_penmat = fus[[1]],
    Adjust_Matrix_Fused = fus[[2]]
  )
}

# MTCC — Maven Telecom Customer Churn
preprocess_MTCC <- function() {
  # --- Load Maven churn data and ZIP-level population ---
  maven_telco_customer_churn <- read.csv(file.path(data_root, "Maven Telecom Customer Churn", "telecom_customer_churn.csv"))
  population <- read.csv(file.path(data_root, "Maven Telecom Customer Churn", "telecom_zipcode_population.csv"))

  # --- Join population; impute service-dependent NA; filter valid customers ---
  data <- maven_telco_customer_churn %>%
    left_join(population, by = "Zip.Code") %>%
    mutate(
      Avg.Monthly.Long.Distance.Charges = if_else(
        Phone.Service == "No" & is.na(Avg.Monthly.Long.Distance.Charges), 0, Avg.Monthly.Long.Distance.Charges
      ),
      Multiple.Lines = if_else(Phone.Service == "No" & Multiple.Lines == "", "No", Multiple.Lines),
      Avg.Monthly.GB.Download = if_else(
        Internet.Service == "No" & is.na(Avg.Monthly.GB.Download), 0, Avg.Monthly.GB.Download
      ),
      across(
        c(Internet.Type, Online.Security, Online.Backup, Device.Protection.Plan,
          Premium.Tech.Support, Streaming.TV, Streaming.Movies, Streaming.Music, Unlimited.Data),
        ~ if_else(Internet.Service == "No" & . == "", "No", .)
      )
    ) %>%
    filter(Customer.Status %in% c("Stayed", "Churned") & Monthly.Charge > 0) %>%
    mutate(y = ifelse(Customer.Status == "Stayed", 0, 1)) %>%
    mutate(Zip = substr(Zip.Code, 1, 3)) %>%
    mutate(
      Tenure.in.Months = cut(
        Tenure.in.Months,
        breaks = c(0, 3, 12, 36, 60, Inf),
        labels = c("1-3", "4-12", "13-36", "37-60", "61+"),
        right = TRUE
      )
    ) %>%
    dplyr::select(-Customer.ID, -City, -Latitude, -Longitude, -Internet.Service,
                  -Total.Revenue, -Customer.Status, -Churn.Category, -Churn.Reason, -Zip.Code)

  assertthat::assert_that(sum(is.na(data)) == 0)

  # --- Factorize categoricals; extract labels and monthly charges ---
  categorical_variables <- c(
    "Gender", "Married", "Offer", "Phone.Service", "Multiple.Lines", "Internet.Type",
    "Online.Security", "Online.Backup", "Device.Protection.Plan", "Premium.Tech.Support",
    "Streaming.TV", "Streaming.Movies", "Streaming.Music", "Unlimited.Data", "Contract",
    "Paperless.Billing", "Payment.Method", "Zip", "Tenure.in.Months"
  )

  labels <- as.integer(data$y)
  amounts <- data$Monthly.Charge
  raw_cova <- data %>%
    mutate(
      logamount = log(Monthly.Charge + 1),
      across(
        c(Total.Charges, Total.Refunds, Total.Extra.Data.Charges, Total.Long.Distance.Charges,
          Avg.Monthly.Long.Distance.Charges, Avg.Monthly.GB.Download, Population),
        ~ log(. + 1), .names = "log{.col}"
      )
    ) %>%
    mutate(across(all_of(categorical_variables), ~ factor(.x))) %>%
    mutate(across(Zip, ~ factor(.x, levels = sort(unique(.x))))) %>%
    dplyr::select(-y)

  # --- Expand to model matrix; build fused penalty (1D tenure + graph Zip) ---
  covariates <- as.data.frame(model.matrix(~ ., raw_cova)[, -1, drop = FALSE])
  full_names <- colnames(covariates)
  df <- vapply(raw_cova, function(x) {
    if (is.factor(x) || is.character(x)) length(unique(x)) - 1 else 1
  }, numeric(1))
  J <- length(df)
  fused_variable <- c("Zip", "Tenure.in.Months")
  fused_type <- as.list(rep("none", J))
  names(fused_type) <- names(df)
  fused_type[fused_variable] <- "fuse1d"
  fused_type[["Zip"]] <- "fuse4gragh"
  fused_par <- as.list(df)
  zips <- colnames(covariates) %>% grep("Zip", ., value = TRUE) %>% sub("Zip", "", .)
  fused_par[["Zip"]] <- getMTCCgragh(zips)

  fus <- construct_fused_penalty(fused_type, fused_par)
  Group_Matrix <- construct_Group_matrix(df)
  weight <- generate_weight(df, J, fused_variable, Group_Matrix, full_names)

  # --- Churn cost matrix scaled by monthly charge ---
  n_samples <- length(labels)
  cost_matrix <- array(0, dim = c(n_samples, 2, 2))
  cost_matrix[, 1, 1] <- 0
  cost_matrix[, 1, 2] <- 12 * amounts
  cost_matrix[, 2, 1] <- 2 * amounts
  cost_matrix[, 2, 2] <- 0

  # --- Drop raw scales after the derived log features have entered the design matrix ---
  covariates <- covariates %>%
    dplyr::select(
      -Monthly.Charge, -Total.Charges, -Total.Refunds, -Total.Extra.Data.Charges,
      -Total.Long.Distance.Charges, -Avg.Monthly.Long.Distance.Charges, -Avg.Monthly.GB.Download,
      -Population
    )

  list(
    covariates = covariates,
    labels = labels,
    cost_matrix = cost_matrix,
    df = df,
    w_lasso = weight$w_lasso,
    w_group = weight$w_group,
    Group_Matrix = Group_Matrix,
    fused_penmat = fus[[1]],
    Adjust_Matrix_Fused = fus[[2]]
  )
}

# KVIC — Kaggle Vehicle Insurance Claim Fraud Detection
preprocess_KVIC <- function() {
  # --- Load vehicle insurance fraud data ---
  data <- read.csv(file.path(data_root, "Kaggle Vehicle Insurance Claim Fraud Detection", "fraud_oracle.csv"))
  labels <- data$FraudFound_P

  # --- Drop date/policy ID columns; enforce ordered factor levels ---
  remove_variables <- c(
    "Month", "WeekOfMonth", "DayOfWeek", "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed",
    "PolicyNumber", "PolicyType", "Age", "FraudFound_P"
  )

  raw_cova <- data %>%
    dplyr::select(-all_of(remove_variables)) %>%
    mutate(across(everything(), ~ factor(.x, levels = sort(unique(.x))))) %>%
    mutate(
      VehiclePrice = factor(VehiclePrice, levels = c(
        "less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000",
        "60000 to 69000", "more than 69000"
      )),
      Days_Policy_Accident = factor(Days_Policy_Accident, levels = c(
        "none", "1 to 7", "8 to 15", "15 to 30", "more than 30"
      )),
      Days_Policy_Claim = factor(Days_Policy_Claim, levels = c("none", "8 to 15", "15 to 30", "more than 30")),
      PastNumberOfClaims = factor(PastNumberOfClaims, levels = c("none", "1", "2 to 4", "more than 4")),
      AgeOfVehicle = factor(AgeOfVehicle, levels = c(
        "new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"
      )),
      NumberOfSuppliments = factor(NumberOfSuppliments, levels = c("none", "1 to 2", "3 to 5", "more than 5")),
      AddressChange_Claim = factor(AddressChange_Claim, levels = c(
        "no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"
      )),
      NumberOfCars = factor(NumberOfCars, levels = c("1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"))
    )

  # --- Model matrix and DriverRating fused penalty ---
  covariates <- as.data.frame(model.matrix(~ ., raw_cova)[, -1, drop = FALSE])
  df <- vapply(raw_cova, function(x) {
    if (is.factor(x) || is.character(x)) length(unique(x)) - 1 else 1
  }, numeric(1))
  J <- length(df)
  fused_variable <- "DriverRating"
  fused_type <- as.list(rep("none", J))
  names(fused_type) <- names(df)
  fused_type[fused_variable] <- "fuse1d"
  fus <- construct_fused_penalty(fused_type, as.list(df))
  Group_Matrix <- construct_Group_matrix(df)
  weight <- generate_weight(df, J, fused_variable, Group_Matrix, colnames(covariates))

  # --- Fixed fraud-detection cost matrix (claim amount = 3500) ---
  fixed_cost <- 350
  cost_matrix <- array(0, dim = c(nrow(covariates), 2, 2))
  cost_matrix[, 1, 1] <- 0.0
  cost_matrix[, 1, 2] <- 3500
  cost_matrix[, 2, 1] <- fixed_cost
  cost_matrix[, 2, 2] <- fixed_cost

  list(
    covariates = covariates,
    labels = labels,
    cost_matrix = cost_matrix,
    df = df,
    w_lasso = weight$w_lasso,
    w_group = weight$w_group,
    Group_Matrix = Group_Matrix,
    fused_penmat = fus[[1]],
    Adjust_Matrix_Fused = fus[[2]]
  )
}

# DCCC — UCI Default of Credit Card Clients
preprocess_DCCC <- function() {
  # --- Load UCI credit-card default data ---
  library(readxl)
  dccc <- read_excel(file.path(data_root, "UCI Default of Credit Card Clients", "default of credit card clients.xls"), skip = 1)
  dccc <- as.data.frame(dccc)

  # --- Factorize PAY status and demographic variables ---
  categorical_variables <- c(
    "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
  )
  labels <- dccc$`default payment next month`

  raw_cova <- dccc %>%
    mutate(across(all_of(categorical_variables), ~ factor(.x, levels = sort(unique(.x))))) %>%
    mutate(
      logamount = log(LIMIT_BAL + 1),
      across(all_of(amt_var), ~ log(. - min(., na.rm = TRUE) + 1), .names = "log{.col}")
    ) %>%
    dplyr::select(-ID, -`default payment next month`)

  # --- Model matrix; fuse all PAY_* status variables ---
  covariates <- as.data.frame(model.matrix(~ ., raw_cova)[, -1, drop = FALSE])
  full_names <- colnames(covariates)
  df <- vapply(raw_cova, function(x) {
    if (is.factor(x) || is.character(x)) length(unique(x)) - 1 else 1
  }, numeric(1))
  J <- length(df)
  fused_variable <- c("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
  fused_type <- as.list(rep("none", J))
  names(fused_type) <- names(df)
  fused_type[fused_variable] <- "fuse1d"
  fus <- construct_fused_penalty(fused_type, as.list(df))
  Group_Matrix <- construct_Group_matrix(df)
  weight <- generate_weight(df, J, fused_variable, Group_Matrix, full_names)

  # --- Bahnsen cost matrix on credit limit (LIMIT_BAL) ---
  credit_lines <- covariates$LIMIT_BAL
  pi_1 <- mean(labels)
  n_samples <- length(credit_lines)
  cost_matrix <- array(0, dim = c(n_samples, 2, 2))
  cost_matrix[, 1, 1] <- 0.0
  cost_matrix[, 1, 2] <- v_calculate_cost_fn(credit_lines, lgd)
  cost_matrix[, 2, 1] <- v_calculate_cost_fp(credit_lines, int_r, n_term, int_cf, pi_1, lgd, mean(credit_lines))
  cost_matrix[, 2, 2] <- 0.0

  # --- Drop raw scales after the derived log features have entered the design matrix ---
  covariates <- covariates %>%
    dplyr::select(-LIMIT_BAL, -all_of(amt_var))

  list(
    covariates = covariates,
    labels = labels,
    cost_matrix = cost_matrix,
    df = df,
    w_lasso = weight$w_lasso,
    w_group = weight$w_group,
    Group_Matrix = Group_Matrix,
    fused_penmat = fus[[1]],
    Adjust_Matrix_Fused = fus[[2]]
  )
}



# Bahnsen et al. (2014a) cost helpers for credit-scoring cases (UGC, DCCC).
int_r <- 0.0479 / 12
n_term <- 24
int_cf <- 0.0294 / 12
lgd <- 0.75
cl_max <- 25000

# Monthly annuity payment for a loan of principal cl_i at rate int_ over n_term months.
calculate_a <- function(cl_i, int_, n_term) {
  cl_i * ((int_ * (1 + int_) ^ n_term) / ((1 + int_) ^ n_term - 1))
}

# Present value of annuity payments a at discount rate int_ over n_term months.
calculate_pv <- function(a, int_, n_term) {
  a / int_ * (1 - 1 / (1 + int_) ^ n_term)
}

# Loss-given-default cost for a false negative on credit limit cl_i.
calculate_cost_fn <- function(cl_i, lgd) {
  cl_i * lgd
}

# False-positive misclassification cost under Bahnsen et al. (2014a) credit-scoring economics.
calculate_cost_fp <- function(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg) {
  a <- calculate_a(cl_i, int_r, n_term)
  pv <- calculate_pv(a, int_cf, n_term)
  r <- pv - cl_i
  r_avg <- calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
  max(0, r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd))
}

v_calculate_cost_fp <- Vectorize(calculate_cost_fp)
v_calculate_cost_fn <- Vectorize(calculate_cost_fn)

APPLICATION_CASES <- c(
  "MTCC",
  "KTCC",
  "KVIC",
  "BMTL",
  "DCCC",
  "UGC"
)

# Build design matrix, penalty matrices, weights, and cost matrix for one application case.
.preprocess_finalize <- function(raw_cova, labels, fused_variable, amounts = NULL,
                                 cost_matrix = NULL, log_cols = character(0),
                                 drop_cols = character(0), zip_graph = NULL) {
  if (!is.null(amounts)) {
    raw_cova$logamount <- log(amounts + 1)
  }

  # --- Expand factors to dummy columns ---
  covariates <- as.data.frame(model.matrix(~ ., raw_cova)[, -1, drop = FALSE])
  full_names <- colnames(covariates)

  # --- Group sizes per raw predictor ---
  df <- vapply(raw_cova, function(x) {
    if (is.factor(x) || is.character(x)) length(unique(x)) - 1 else 1
  }, numeric(1))

  # --- Fused-lasso structure (1D and optional graph-based Zip) ---
  J <- length(df)
  fused_type <- as.list(rep("none", J))
  names(fused_type) <- names(df)
  fused_type[fused_variable] <- "fuse1d"
  fused_par <- as.list(df)
  if (!is.null(zip_graph)) {
    fused_type[["Zip"]] <- "fuse4gragh"
    fused_par[["Zip"]] <- zip_graph
  }

  fus <- construct_fused_penalty(fused_type, fused_par)
  Group_Matrix <- construct_Group_matrix(df)
  weight <- generate_weight(df, J, fused_variable, Group_Matrix, full_names)

  # --- Default churn-style cost matrix when none is supplied ---
  if (is.null(cost_matrix)) {
    n_samples <- length(labels)
    cost_matrix <- array(0, dim = c(n_samples, 2, 2))
    cost_matrix[, 1, 1] <- 0
    cost_matrix[, 1, 2] <- 12 * amounts
    cost_matrix[, 2, 1] <- 2 * amounts
    cost_matrix[, 2, 2] <- 0
  }

  if (length(log_cols) > 0) {
    covariates <- covariates %>%
      mutate(across(all_of(log_cols), ~ log(. + 1), .names = "log{.col}")) %>%
      dplyr::select(-all_of(log_cols))
  }

  if (length(drop_cols) > 0) {
    covariates <- dplyr::select(covariates, -all_of(drop_cols))
  }

  list(
    covariates = covariates,
    labels = labels,
    cost_matrix = cost_matrix,
    df = df,
    w_lasso = weight$w_lasso,
    w_group = weight$w_group,
    Group_Matrix = Group_Matrix,
    fused_penmat = fus[[1]],
    Adjust_Matrix_Fused = fus[[2]]
  )
}


# Build a spatial adjacency graph for California ZIP codes at a given prefix length.
getMTCCgragh <- function(zips){
  library(igraph)
  library(tigris)
  library(sf)
  options(tigris_use_cache = TRUE)

  # --- Fetch California ZCTA geometries ---
  zips_ca <- zctas(cb = TRUE, starts_with = "9", year = 2020)

  # --- Group to requested ZIP prefix length ---
  zipdigit <- nchar(zips[1])
  zipdigit0 <- nchar(zips_ca$ZCTA5CE20[1])

  if(zipdigit < zipdigit0){
    zips_grouped <- zips_ca %>%
      mutate(zip_prefix = substr(ZCTA5CE20, 1, zipdigit)) %>%
      filter(zip_prefix %in% zips) %>%
      group_by(zip_prefix) %>%
      summarise(geometry = st_union(geometry)) %>%
      arrange(zip_prefix)
  }else{
    zips_grouped <- zips_ca %>%
      mutate(zip_prefix = ZCTA5CE20) %>%
      filter(zip_prefix %in% zips) %>%
      arrange(zip_prefix)
  }

  # --- Spatial adjacency -> edge list (include self-loops) ---
  neighbors_matrix <- st_touches(zips_grouped, retain_unique = TRUE)

  edge_list <- lapply(1:length(neighbors_matrix), function(i) {
    neighbors <- neighbors_matrix[[i]]
    data.frame(from = rep(i, length(neighbors)+1), to = c(i,neighbors))
  })

  edges <- do.call(rbind, edge_list)

  # --- Build undirected igraph and remove duplicate edges ---
  g <- graph_from_data_frame(edges, directed = FALSE)
  V(g)$name <- as.numeric(V(g)$name)
  g_clean <- igraph::simplify(g, remove.loops = TRUE, remove.multiple = TRUE)

  return(g_clean)
}

# Map each predictor group to its expanded one-hot or continuous columns.
construct_Group_matrix <- function(df){

  J <- length(df)
  p <- sum(df)

  # One-hot block per predictor group
  Group_Matrix <- matrix(0, nrow = J, ncol = p)
  cc <- 1
  for (j in 1:J) {
    Group_Matrix[j, cc:(cc+df[j]-1)] <- 1
    cc <- cc + df[j]
  }

  rownames(Group_Matrix) <- names(df)

  return(Matrix::Matrix(Group_Matrix,sparse = TRUE))

}

# Construct group-lasso and lasso penalty weights from feature metadata.
generate_weight <- function(df, J, fused_variable, Group_Matrix, full_names){

  raw_names <- names(df)

  # Group-lasso weights: active on multi-level categoricals only
  glasso_variables <- names(which(df != 1))
  I_group <- rep(0, J)
  names(I_group) <- raw_names
  I_group[glasso_variables] <- 1
  nu_group <- sqrt(df)
  w_group <- as.matrix(I_group * nu_group)

  # Lasso weights: zero on fused variables, one elsewhere (mapped to columns)
  lasso_variables <- setdiff(raw_names, fused_variable)

  I_lasso <- rep(0, J)
  names(I_lasso) <- raw_names
  I_lasso[lasso_variables] <- 1
  w_lasso <- as.matrix(t(Group_Matrix) %*% as.matrix(I_lasso))
  row.names(w_lasso) <- full_names

  return(
    list(
      w_lasso  = w_lasso,
      w_group = w_group
    )
  )
}
