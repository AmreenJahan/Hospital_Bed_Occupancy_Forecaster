# Hospital Bed Occupancy Forecaster

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)

End-to-end **healthcare analytics and machine learning** project that forecasts hospital bed occupancy by department, detects overflow risk, and supports data-driven capacity planning.

---

## Project Overview

Hospitals operate under constant pressure to balance admissions, discharges, staffing, and finite bed capacity. This project delivers a complete data science pipeline—from raw operational records to an interactive dashboard—that:

1. Cleans and deduplicates department-level occupancy data
2. Engineers **time-aware, leakage-free** features (lags and rolling windows per department)
3. Trains and compares **Random Forest**, **Gradient Boosting**, and **XGBoost** regressors
4. Generates **7-day occupancy forecasts** (rate, occupied beds, available beds)
5. Triggers **overflow alerts** (GREEN / YELLOW / ORANGE / RED) with operational recommendations

**Target variable:** `Occupancy_Rate` (%)

| Dataset | Records | Date Range |
|---------|---------|------------|
| Raw | 10,000 | 2021–2025 |
| Cleaned (deduplicated) | 6,943 | 2021–2025 |
| Train / Test | 5,554 / 1,389 | Chronological split |

---

## Features

| Module | Capability |
|--------|------------|
| **Data Pipeline** | Loading, validation, deduplication, quality reports |
| **EDA** | Occupancy, admissions, seasonal, department, correlation analysis |
| **Operational Insights** | Peak/low periods, holiday effects, staffing impact |
| **Feature Engineering** | Department-grouped lags (1/7/30 day), rolling means/std |
| **ML Models** | RF, GBR, XGBoost with `GridSearchCV` + `TimeSeriesSplit` |
| **Forecasting** | 7-day horizon per department |
| **Overflow System** | Risk classification + actionable recommendations |
| **Dashboard** | 5-page Streamlit app with Plotly charts |

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Language | Python 3.8+ |
| Data | Pandas, NumPy, SciPy |
| ML | scikit-learn, XGBoost, joblib |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Notebooks | Jupyter (optional) |

---

## Project Structure

```
Hospital_Bed_Occupancy_Forecaster/
├── main.py                      # Day 1 pipeline (EDA → features → split)
├── run_all.py                   # Full end-to-end execution
├── train_models.py              # Model training entry point
├── model_evaluation.py          # Metrics, plots, feature importance
├── forecasting.py               # 7-day forecast (generate_forecast / save_forecast)
├── overflow_alert_system.py     # Risk detection & recommendations
├── config.py                    # Paths, constants, leakage column list
│
├── scripts/                     # Modular pipeline scripts
│   ├── load_data.py
│   ├── data_cleaning.py
│   ├── exploratory_data_analysis.py
│   ├── operational_insights.py
│   ├── feature_engineering.py
│   ├── dataset_preparation.py
│   ├── train_models.py
│   └── model_evaluation.py
│
├── data/
│   ├── hospital_bed_occupancy_10000.csv   # Raw dataset
│   ├── cleaned_hospital_bed_occupancy.csv
│   ├── engineered_hospital_bed_occupancy.csv
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
│
├── models/                      # Trained models, encoders, split metadata
├── saved_models/
│   └── best_model.pkl           # Production model (Random Forest)
│
├── reports/                     # Reports, CSV outputs, figures
│   ├── figures/                 # EDA & model visualizations (PNG)
│   ├── forecast_7_day.csv
│   ├── overflow_alerts.csv
│   └── *.md / *.txt
│
└── streamlit_app/
    └── app.py                   # Interactive dashboard
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AmreenJahan/Hospital_Bed_Occupancy_Forecaster.git
cd Hospital_Bed_Occupancy_Forecaster
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Windows tip** — if you see Unicode console errors:

```bash
set PYTHONIOENCODING=utf-8
```

---

## How to Run

### Full pipeline (recommended first run)

```bash
python run_all.py
```

### Step-by-step

```bash
python main.py                  # Data cleaning, EDA, features, train/test split
python train_models.py          # Train RF, GBR, XGBoost; save best model
python model_evaluation.py      # MAE, RMSE, R², residual & importance plots
python forecasting.py           # Export reports/forecast_7_day.csv
python overflow_alert_system.py # Export reports/overflow_alerts.csv
```

### Run the dashboard

```bash
streamlit run streamlit_app/app.py
```

Open **http://localhost:8501** in your browser.

**Dashboard pages:**
1. Executive Overview — KPIs and occupancy trends
2. EDA Dashboard — Department, seasonal, correlation charts
3. Forecasting — 7-day predictions with CSV download
4. Overflow Monitoring — Risk levels and recommendations
5. Model Performance — Comparison table, metrics, feature importance

---

## Model Results

Chronological train/test split (test period starts **2024-12-30**).

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **RandomForest** | **9.14** | **11.30** | **0.553** |
| GradientBoosting | 9.14 | 11.27 | 0.556 |
| XGBoost | 9.15 | 11.30 | 0.554 |

Best model: **RandomForestRegressor** → `saved_models/best_model.pkl`

---

## Sample Screenshots

> Visual outputs are generated under `reports/figures/` when you run the pipeline.

| Analysis | File |
|----------|------|
| Occupancy distribution | `reports/figures/B_occupancy_analysis.png` |
| Department comparison | `reports/figures/D_department_analysis.png` |
| Seasonal patterns | `reports/figures/E_seasonal_analysis.png` |
| Correlation heatmap | `reports/figures/H_correlation_heatmap.png` |
| Peak occupancy periods | `reports/figures/I_peak_occupancy_periods.png` |
| Actual vs Predicted | `reports/figures/actual_vs_predicted.png` |
| Feature importance | `reports/figures/feature_importance.png` |
| Residual analysis | `reports/figures/residual_analysis.png` |

**Streamlit dashboard** — run `streamlit run streamlit_app/app.py` for the live interactive UI.

---

## Overflow Risk Levels

| Level | Occupancy | Recommended Action |
|-------|-----------|-------------------|
| GREEN | < 70% | Standard monitoring |
| YELLOW | 70–85% | Increase staffing, review elective admissions |
| ORANGE | 85–95% | Prepare overflow units, delay non-critical admissions |
| RED | > 95% | Open temporary wards, activate incident command |

---

## Key Outputs

| Output | Path |
|--------|------|
| Train / test data | `data/processed/train_data.csv`, `test_data.csv` |
| Model comparison | `reports/model_comparison_table.csv` |
| Evaluation report | `reports/model_evaluation_report.md` |
| 7-day forecast | `reports/forecast_7_day.csv` |
| Overflow alerts | `reports/overflow_alerts.csv` |
| Full documentation | `reports/documentation_report.md` |
| Submission checklist | `reports/SUBMISSION_CHECKLIST.md` |

---

## Future Improvements

- [ ] **Real-time API** — FastAPI endpoint for live occupancy scoring
- [ ] **Probabilistic forecasts** — Prediction intervals (quantile regression)
- [ ] **Hospital-wide aggregation** — Roll up department forecasts to facility level
- [ ] **External features** — Weather, epidemiological signals, regional events
- [ ] **Deep learning baselines** — LSTM / Temporal Fusion Transformer comparison
- [ ] **MLOps** — Scheduled retraining, model versioning (MLflow / DVC)
- [ ] **CI/CD** — Automated pipeline tests on push
- [ ] **Multi-hospital support** — Generalize across facility networks

---

## Documentation

- [Full Project Documentation](reports/documentation_report.md)
- [Capacity Planning Report](reports/capacity_planning_report.md)
- [Model Evaluation Report](reports/model_evaluation_report.md)
- [Submission Checklist](reports/SUBMISSION_CHECKLIST.md)

---

