# Bayesian Logistic Regression via Metropolis-Hastings

From-scratch multi-chain **Random Walk Metropolis-Hastings (RWMH)** sampler written in pure **R** to estimate full posterior distributions and credible intervals for binary diabetes prediction on the **Pima Indians Diabetes Dataset** (`N = 768`, 8 clinical predictors).

Completed for *On the Theory and Practice of Monte Carlo Simulations* (Prof. Dr. Katja Ickstadt & Zeyu Ding, Winter Semester 2025/26, TU Dortmund University) by **Raj Pawar** and Harsh Rana.

![Marginal posterior densities vs. Frequentist MLE across all 9 parameters](https://raw.githubusercontent.com/r-a-j/Project_Bayes_Logistic/main/Report/images/posterior_density.png)

---

## Overview

Logistic regression does not have a conjugate prior for a multivariate Gaussian prior on its regression coefficients, so the posterior normalizing constant cannot be computed in closed form. Instead of calling Stan or JAGS as a black box, `Code/main_analysis.R` implements the log-posterior and Metropolis-Hastings transition kernel directly in R, verifies multi-chain convergence via `coda`, and validates posterior means and 95% credible intervals against Frequentist Maximum Likelihood (`glm`).

### Key Implementation Details

- **Model:** Binary logistic regression on 8 z-score standardized predictors plus an intercept (`p = 9`), with a weakly informative Gaussian prior $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, 100\mathbf{I})$.
- **Custom RWMH Sampler:** Symmetric isotropic multivariate normal proposal (`step_size = 0.003`, tuned for ~30 to 50% acceptance), evaluated in log-space to avoid numerical underflow.
- **Multi-Chain Setup:** 3 independent chains initialized from overdispersed $\mathcal{N}(0, 2)$ draws, run for 50,000 iterations each with 10,000 burn-in discarded (**120,000 pooled posterior draws**).
- **Reproducibility:** Portable paths via `here::here()`, fixed RNG seeds (`2026` for the 80/20 split, `2025` for MCMC), and locked package versions in `renv.lock`.

---

## Convergence Diagnostics

Multi-chain convergence was checked with `coda` across the 3 post-burn-in chains:

- **Gelman-Rubin Statistic ($\hat{R}$):** Strictly $\le 1.01$ across all 9 parameters as well as the multivariate PSRF (well below the 1.05 threshold).
- **Effective Sample Size (ESS):** Above 2,000 independent draws for every coefficient, ranging from **2,054** (`BMI`) to **3,113** (`Intercept`).

![Pooled 120,000-iteration MCMC trace plots](https://raw.githubusercontent.com/r-a-j/Project_Bayes_Logistic/main/Report/images/trace_plots.png)

---

## Bayesian vs. Frequentist Parameter Estimates

Pooled posterior means and 95% credible intervals across 120,000 post-burn-in draws compared against `glm(..., family = binomial)` on the 615-patient training split (`Report/images/comparison_table.csv`):

| Parameter | Freq. MLE | MLE 95% CI | Bayes Mean | Bayes 95% CI | Signal (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glucose** | 1.020 | [0.773, 1.283] | **1.042** | [0.785, 1.308] | Strong positive driver |
| **BMI** | 0.794 | [0.541, 1.062] | **0.813** | [0.556, 1.079] | Strong positive driver |
| **Pregnancies** | 0.379 | [0.148, 0.616] | **0.384** | [0.153, 0.617] | Positive effect (> 0) |
| **DiabetesPedigreeFunction** | 0.325 | [0.107, 0.548] | **0.334** | [0.107, 0.561] | Positive effect (> 0) |
| Age | 0.206 | [-0.030, 0.442] | 0.212 | [-0.026, 0.451] | Mild positive (crosses 0) |
| BloodPressure | -0.191 | [-0.409, 0.025] | -0.198 | [-0.418, 0.019] | Uncertain (crosses 0) |
| Insulin | -0.090 | [-0.319, 0.141] | -0.091 | [-0.324, 0.143] | Centered near 0 (374 raw zeros) |
| SkinThickness | -0.018 | [-0.258, 0.225] | -0.017 | [-0.264, 0.232] | Centered near 0 (227 raw zeros) |
| Intercept | -0.853 | [-1.070, -0.646] | -0.863 | [-1.074, -0.654] | Baseline log-odds |

All 9 posterior means match Frequentist MLE within `0.022` standardized log-odds units, with the red dashed MLE lines landing directly on the modes of the posterior density curves.

---

## Held-Out Predictive Performance (`N = 153`)

Posterior-mean predictions scored on the 20% test split (`153` patients, `0.5` probability threshold, `Report/images/accuracy_metrics.txt`):

| Predicted \ Actual | Healthy (`0`) | Diabetic (`1`) |
| :--- | :---: | :---: |
| **Predicted `0`** | 86 (TN) | 20 (FN) |
| **Predicted `1`** | 14 (FP) | 33 (TP) |

- **Overall Accuracy:** `77.78%` (`95% CI [70.36%, 84.09%]`, vs. `65.36%` No Information Rate, `p = 0.00059`)
- **Balanced Accuracy:** `74.13%` (`Kappa = 0.4958`)
- **Diabetes Recall / Sensitivity (`Outcome = 1`):** `62.26%` (`33 / 53`)
- **Healthy Specificity (`Outcome = 0`):** `86.00%` (`86 / 100`)
- **Diabetes Precision (`Outcome = 1`):** `70.21%` (`33 / 47`)

*(Note: `caret::confusionMatrix()` prints `Sensitivity: 0.8600` because it defaults to `'Positive' Class : 0`; the table above reports metrics explicitly with `Outcome = 1` as the clinical positive class.)*

---

## Repository Structure

```text
.
├── Code/
│   └── main_analysis.R                # End-to-end RWMH sampler, coda diagnostics, plots, and test scoring
├── Data/
│   └── diabetes.csv                   # Pima Indians Diabetes dataset (768 rows, 8 predictors + Outcome)
├── Report/
│   └── images/
│       ├── posterior_density.pdf/png  # 3x3 marginal posterior densities overlaid with Frequentist MLE
│       ├── trace_plots.pdf/png        # 3x3 pooled post-burn-in MCMC trace plots with LOWESS curves
│       ├── comparison_table.csv       # Full MLE vs. Bayesian mean & 95% interval table
│       └── accuracy_metrics.txt       # Held-out confusion matrix and summary statistics
└── renv.lock                          # Reproducible R package snapshot
```

## How to Run

1. Clone the repository and restore the `renv` environment (or install `tidyverse`, `MASS`, `coda`, `caret`, `mvtnorm`, `knitr`, `here`):
   ```bash
   git clone https://github.com/r-a-j/Project_Bayes_Logistic.git
   cd Project_Bayes_Logistic
   ```
2. Run the full 3-chain analysis from R or the terminal:
   ```bash
   Rscript Code/main_analysis.R
   ```
   Outputs (`trace_plots.pdf`, `posterior_density.pdf`, `comparison_table.csv`, and `accuracy_metrics.txt`) are written to `Report/images/`.

---

## Limitations & Next Steps

- **Zero-encoded missing values:** The raw Pima dataset encodes missing physiological measurements as zeros (`374` in `Insulin`, `227` in `SkinThickness`, `35` in `BloodPressure`, `11` in `BMI`, `5` in `Glucose`). Modeling them as latent missing values during MCMC would reduce coefficient attenuation.
- **Proposal efficiency:** Isotropic RWMH delivers ~2% ESS per raw draw in 9 dimensions; switching to an adaptive covariance proposal or Hamiltonian Monte Carlo (HMC/NUTS) scales much better to correlated high-dimensional posteriors.
- **Screening cutoff & posterior predictive draws:** Lowering the classification threshold below `0.5` and averaging predictions across posterior draws (instead of plug-in posterior means) would improve diabetes recall and uncertainty calibration.
