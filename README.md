# ECSLRx

Ensemble cost-sensitive logistic regression with elastic net, group lasso, fused lasso, and diversity penalties, solved via PCCSQA-SUM.

## Installation

```r
install.packages("devtools")
devtools::install_github("bingStat/ECSLRx", subdir = "package/ECSLRx", dependencies = TRUE)
```


## Quick start

```r
library(ECSLRx)
devtools::test("package/ECSLRx")
```

## Reproducing experiments

This repository exposes the package plus the paper reproduction scripts in the flattened `scripts/` + `runs/` layout.

### Simulation

```bash
Rscript scripts/simulation/benchmark_simulation.R III 0.2 0.5 0.2 0.15 _20260607
bash scripts/simulation/loop.sh
```

### Application (real-data cases)

Six paper cases (`casenum` 1–6): MTCC, KTCC, KVIC, BMTL, DCCC, UGC.
<!--
## Data Availability

The datasets analysed during the current study are publicly available.

| Dataset | URL |
|---|---|
| Maven Telecom Customer Churn | https://www.mavenanalytics.io/data-playground |
| Kaggle Telco Customer Churn | https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data |
| Kaggle Vehicle Insurance Claim | https://www.kaggle.com/datasets/gowthamdd/vehicle-insurance-claim-analysis |
| Belgian Motor Third-part Liability | https://dutangc.github.io/CASdatasets/reference/beMTPL16.html |
| UCI Default of Credit Card Clients | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| UCI German Credit | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data | -->

```bash
Rscript scripts/application/benchmark_application.R <casenum> [k ...]
bash scripts/application/app.sh 4 1-10
```

Models: logit, cslogit, CSRF-cv, CSRP-cv, ECSLRx.

> **Note:** Experiment scripts expect local data paths and may require adaptation for your environment.

## Repository layout

```
package/ECSLRx/                  R package source (R/, src/, tests/)
scripts/                         Paper reproduction scripts
  Auxiliary_Functions_for_Performance.R
  fit_models.R
  application/benchmark_application.R
  application/preprocessing_cases.R
  application/app.sh
  simulation/benchmark_simulation.R
  simulation/Data_Simulation_Group.R
  simulation/loop.sh
```

## Citation

If you use this package, please cite the paper: `ECSLRx`.

## License

See [LICENSE](LICENSE).
