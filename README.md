# IFRS 9 Lifetime ECL Engine

An end-to-end credit risk modelling app built with Streamlit, demonstrating a full IFRS 9 Expected Credit Loss (ECL) framework on LendingClub data.

## Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ifrs9-ecl.streamlit.app)

## Overview

This app walks through every layer of the IFRS 9 ECL calculation — from raw variable selection through to portfolio-level provisions — using a publicly available LendingClub dataset (~2.2M loans, 2007–2018).

| Component | Method | Key metric |
|---|---|---|
| PD | Logistic regression + WoE scorecard | Gini 0.40 (OOT) |
| LGD | Portfolio constant (beats Beta regression) | MAE 0.036 |
| EAD | Amortisation schedule (outperforms OLS) | R² –0.02 (OLS fails) |
| ECL | Σ PD_t × LGD × EAD_t | Stage 1 / Stage 2 split |

## Pages

| # | Page | Description |
|---|---|---|
| 🏠 | Home | Portfolio ECL dashboard — Stage 1 vs Stage 2 headline numbers |
| 1 | Theoretical Framework & Variable Selection | 5C framework, WoE binning, IV screening |
| 2 | PD Model & Scorecard | Logistic regression, ROC, score distribution, PD simulator |
| 3 | Model Monitoring | PSI, score stability, out-of-time validation |
| 4 | EAD Approach | Amortisation-based EAD vs OLS benchmark |
| 5 | Default Timing & ECL Simulator | Timing curves, individual loan ECL calculator |
| 6 | LGD Estimation | Beta regression vs portfolio constant head-to-head |
| 7 | Portfolio ECL | Timeline, ECL by grade and term, stress testing |

## Quick Start

```bash
# Clone
git clone https://github.com/AnNguyen37/ifrs9-ecl-app.git
cd ifrs9-ecl-app

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run Home.py
```

## Key Technical Decisions

- **PD model target**: `good_bad = 1` (good) / `0` (bad). `result.predict()` returns P(good), so `PD_default = 1 − P(good)` throughout.
- **Stage 1 ECL**: sum of monthly ECL for months 1–12
- **Stage 2 ECL**: sum of monthly ECL for all months (lifetime)
- **Timing weights**: `w_t = defaults_in_month_t / total_defaults_in_segment` — no lifelines dependency
- **EAD**: beginning-of-period outstanding balance using `EAD_t = P × [(1+r)^n − (1+r)^(t−1)] / [(1+r)^n − 1]`

## Data

Source: [LendingClub Loan Data 2007–2018 Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (public dataset, not included in this repo).

Pre-processed model artefacts are included in `data/`.

## Requirements

```
streamlit==1.37.0
pandas==2.2.2
numpy==1.26.4
plotly==5.22.0
scikit-learn==1.5.1
statsmodels==0.14.2
```
