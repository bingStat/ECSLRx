# ECSLRx

Ensemble cost-sensitive logistic regression with elastic net, group lasso, fused lasso, and diversity penalties, solved via PCCSQA-SUM.

## Installation

```r
install.packages("devtools")
devtools::install_github("bingStat/ECSLRx")
# or from a local clone:
devtools::install(".", dependencies = TRUE)
```

Requirements: R (>= 3.5.0), Rcpp, RcppEigen, foreach, doParallel, Matrix, caret.

Optional: `pROC`, `PRROC`, `testthat`, `microbenchmark`.

## Quick start

```r
library(ECSLRx)
testthat::test_check("ECSLRx")
```

## License

GPL-3
