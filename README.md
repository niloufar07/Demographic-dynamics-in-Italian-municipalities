
# Italian Municipalities Population & Geohazard Analysis Pipeline
To access inputs, please visit: The data and codes are openly available for researchers and policymakers at Figshare:https://figshare.com/articles/dataset/Population_Trend_in_Italy/32647404, https://doi.org/10.6084/m9.figshare.32647404
This repository contains a machine learning pipeline utilizing **PyCaret** to analyze and model population dynamics and geohazard risk indexes across 7,800+ Italian municipalities (*comuni*). The solution processes complex geospatial, remote sensing (EGMS/INGV landslide data), and historical census datasets to evaluate and predict demographic and structural trends.

## 📌 Project Overview
The pipeline integrates geographic, structural, and population metrics spanning over two decades (2001–2024) to engineer custom socio-demographic indicators. Using automated machine learning (AutoML), it trains and benchmarks several advanced regressors to discover underlying drivers of regional population shifts, accessibility changes, and land vulnerability.

---

## 🛠️ Tech Stack & Requirements

The codebase is engineered to run in a Python 3.11 environment. Key packages and dependencies include:

*   **Python**: `^3.11`
*   **AutoML Framework**: `pycaret[regression] == 3.3.2`
*   **Data Analysis & Processing**: `pandas < 2.2.0`, `numpy >= 1.21, < 1.27`, `scipy <= 1.11.4`
*   **Machine Learning Underlayers**: `scikit-learn > 1.4.0`, `lightgbm >= 3.0.0`
*   **Data Visualization**: `matplotlib < 3.8.0`, `plotly >= 5.14.0`, `yellowbrick >= 1.4`
*   **File Handling**: `openpyxl` (for Excel parsing)

---

## 💾 Dataset Architecture & Merging Pipeline

The pipeline digests three distinct data matrices located within the remote sensing repository structure:

1.  **Feature Matrix (`input_INGV_GRINS_Landslide_EGMS.xlsx`)**: Remote sensing telemetry, landslide risk classifications, and raw geomorphological variables indexed by `PROCOM` (Italian national municipality identification code).
2.  **Target Framework 2024 (`P2_2024_it_Comuni_percentage score.xlsx`)**: Recent multi-criteria index percentage scores and official 2024 municipal population counts (`Popolazione censita al 31 dicembre - Totale`).
3.  **Historical Base Dataset (`RBD-Dataset-2001processed.xlsx`)**: Historic demographic baselines containing the target feature `POP_2001` mapped by `PROCOM`.

### Data Harmonization Flow:
*   An **inner merge** joins the telemetry array with the 2024 framework via structural keys (`PROCOM` $\leftrightarrow$ `Codice comune`).
*   A secondary **inner merge** hooks the 2001 historical matrix to finalize a unified shape of **(7,858 rows $	imes$ 568 columns)**.

---

## ⚙️ Feature Engineering & Preprocessing

```python
# Target definition formulation
merged_df['target'] = ((merged_df['Popolazione censita al 31...'] - merged_df['POP_2001']) * 100) / merged_df['POP_2001']
```

Before passing data into the AutoML engine, the framework executes strict deterministic transformations:
*   **Numeric Type Normalization**: Ingests unstructured object representations and converts standard European decimal formats (commas `,`) to standard floats (`.`).
*   **Target Variable Formula**: Computes percentage population growth/shrinkage metrics over a 23-year timeframe.
*   **Infinite/Null Value Sanitation**: Replaces arithmetic boundaries ($\pm\infty$) generated during zero-division with `NaN` thresholds, subsequently dropping incomplete matrix cells.
*   **Advanced Feature Masking**: Drops **140+ uninformative, administrative, or highly correlated parameters** (e.g., structural attributes, system coordinates, specific macro-classes, spatial coordinates `lat`/`long`, specific age quantiles, regional metadata strings) to mitigate over-fitting.
*   **Categorical Encodings**: Auto-checks high-cardinality values, running standard `One-Hot Encoding` transformations for nominal columns with less than 20 unique items.
*   **Imputation Strategies**: Missing variables are handled dynamically by executing a dataset median imputation fallback.

---

## 🤖 Model Evaluation & Performance Metrics

Using PyCaret's zero-knowledge pipeline initialization, models are evaluated via K-Fold Cross Validation sorted directly against the $R^2$ coefficient metric.

### Comparative Results Summary Table

| Model | MAE | MSE | RMSE | $R^2$ | RMSLE | MAPE | TT (Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Light Gradient Boosting Machine (`lightgbm`)** | **5.9251** | **76.9219** | **8.6976** | **0.8123** | **0.6820** | **1.5288** | **2.4480** |
| Random Forest Regressor (`rf`) | 6.2392 | 84.4834 | 9.1248 | 0.7936 | 0.7047 | 1.5396 | 71.8850 |
| Decision Tree Regressor (`dt`) | 9.3613 | 174.2733 | 13.1730 | 0.5687 | 0.8833 | 2.4371 | 0.9150 |
| Dummy Regressor (`dummy`) | 15.1344 | 405.8020 | 20.1019 | -0.0014 | 1.8267 | 1.2286 | 0.0680 |
after removing the feature SOC_04 the new R*2 from the best model is 0,7701
### Key Takeaways
*   **Top Performer**: The `LightGBM` framework yields superior performance with an **$R^2$ score of 0.8123**, outperforming ensemble models and classical algorithms.
*   **Computational Efficiency**: `LightGBM` finalized training execution in a swift **2.45 seconds**, whereas the standard `Random Forest Regressor` required **71.88 seconds** to resolve similar metric optimization boundaries.

---

##  Quick Start Guide

### 1. Environment Setup
Ensure your local system contains Python 3.11, then pull down dependencies via pip:
```bash
pip install -U pycaret
```

### 2. Implementation Pipeline Blueprint
```python
import pandas as pd
import numpy as np
from pycaret.regression import *

# 1. Load data streams
df_features = pd.read_excel("input_INGV_GRINS_Landslide_EGMS.xlsx")
df_target_2024 = pd.read_excel("P2_2024_it_Comuni_percentage score.xlsx")
df_target_2001 = pd.read_excel("RBD-Dataset-2001processed.xlsx")

# 2. Execute target building, feature pruning, and one-hot alignments
# [Refer to code logic for complete exclude_cols array details]

# 3. Setup and compare ML models
regression_setup = setup(data=X, target=y, session_id=123, verbose=False, preprocess=False)
best_model = compare_models(sort='R2')
```

---
*Developed by NILOOFAR KHEIRKHAHAN as part of the Remote Sensing research initiative mapping landslide and demographic vulnerabilities across Italian municipalities BARI UNIVERSITY ALDO MORO*



with open('LICENSE', 'w', encoding='utf-8') as f:
    f.write(license_text)
print("License file created successfully.")
