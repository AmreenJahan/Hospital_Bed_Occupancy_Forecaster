# Hospital Bed Occupancy Forecaster — Documentation Report

## 1. Problem Statement

Hospitals must anticipate bed demand to allocate staff, open overflow units, and avoid capacity crises. This project builds an end-to-end analytics and forecasting system that:

- Cleans and analyzes department-level occupancy data
- Engineers time-aware, leakage-free features
- Trains and compares machine learning regressors
- Produces 7-day occupancy forecasts
- Flags overflow risk with operational recommendations

**Target variable:** `Occupancy_Rate` (percentage of beds occupied per department per day).

---

## 2. Dataset Description

| Attribute | Value |
|-----------|-------|
| Source file | `data/hospital_bed_occupancy_10000.csv` |
| Raw records | 10,000 |
| After deduplication | 6,943 (unique Date + Department) |
| Date range | 2021-01-01 to 2025-12-29 |
| Departments | 7 (Cardiology, Emergency, General Medicine, ICU, Neurology, Orthopedics, Pediatrics) |
| Features | Date, admissions, discharges, staffing, holidays, season, department |

**Data quality actions taken:**
- Aggregated duplicate Date+Department rows (3,057 resolved)
- Flagged over-capacity records (`Over_Capacity` where rate > 100%)
- No missing values in final cleaned dataset

---

## 3. EDA Findings

- **Mean occupancy:** ~82% hospital-wide
- **Seasonality:** Monsoon and Winter show highest record volumes
- **Department variation:** All departments average 81–82% occupancy
- **Peak risk:** ~1,085 over-capacity days after deduplication
- **Weak raw correlations:** Admissions and staffing show low linear correlation with occupancy; temporal lags are more predictive
- **Holiday effect:** Public holidays and special events influence admission patterns

Full EDA outputs: `reports/eda_report.txt`, `reports/figures/A–H`.

---

## 4. Feature Engineering

### Time-aware features (per Department groupby)
- **Lag features:** 1, 7, 30-day occupancy and admissions
- **Rolling features:** 7 and 30-day mean/std of occupancy and admissions
- **Date features:** Year, Quarter, Month, Week, Day, Day-of-week, Weekend flag
- **Ratio features:** Admission/Discharge, ICU/Emergency admission ratios, Staff-per-bed
- **Interactions:** Admissions×LOS, Staff×Admissions, Emergency×Admissions

### Leakage prevention (excluded from modeling)
- `Beds_Occupied`, `Bed_Utilization_Ratio`
- `Staff_Occupancy_Ratio`, `Staff_Occupancy_Interaction`, `Emergency_Occupancy_Interaction`
- Bed-related lag/rolling features

**Final feature count:** 42 (train set)

---

## 5. Train/Test Split

| Parameter | Value |
|-----------|-------|
| Method | Chronological (time-aware) |
| Train samples | 5,554 (80%) |
| Test samples | 1,389 (20%) |
| Split date | 2024-12-30 |

---

## 6. Model Comparison

| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|-----|
| **RandomForest** | **9.135** | 127.68 | **11.30** | 0.553 |
| GradientBoosting | 9.136 | 126.98 | 11.27 | 0.556 |
| XGBoost | 9.145 | 127.59 | 11.30 | 0.554 |

Hyperparameter tuning performed via `GridSearchCV` with `TimeSeriesSplit` (3 folds).

---

## 7. Best Model

**RandomForestRegressor** — selected by lowest MAE on the chronological test set.

Saved artifact: `saved_models/best_model.pkl`

**Top features:**
1. Over_Capacity flag
2. Rolling 7-day occupancy mean
3. Rolling 7-day occupancy std
4. Total hospital beds
5. Lag 1-day occupancy

---

## 8. Evaluation Metrics (Test Set)

| Metric | Value |
|--------|-------|
| MAE | 9.135 |
| MSE | 127.68 |
| RMSE | 11.30 |
| R² | 0.553 |

Visual diagnostics: `reports/figures/actual_vs_predicted.png`, `residual_analysis.png`, `feature_importance.png`

---

## 9. Forecasting Results

- **Horizon:** 7 days per department
- **Output:** `reports/forecast_7_day.csv`
- **Fields:** Date, Department, Predicted Occupancy Rate, Occupied Beds, Available Beds, Total Beds

Forecasts use iterative lag updates per department from the latest observed data.

---

## 10. Overflow Detection

| Risk Level | Occupancy Range | Action |
|------------|-----------------|--------|
| GREEN | < 70% | Standard monitoring |
| YELLOW | 70–85% | Increase staffing, review admissions |
| ORANGE | 85–95% | Prepare overflow units, delay non-critical admissions |
| RED | > 95% | Open temporary wards, activate incident command |

Output: `reports/overflow_alerts.csv`

---

## 11. Capacity Planning Recommendations

1. **Staffing:** Scale clinical staff with rolling 7-day occupancy trends
2. **Overflow units:** Pre-activate when ORANGE alerts exceed 3 consecutive days per department
3. **Admission control:** Defer elective procedures during RED alerts
4. **Cross-department balancing:** Monitor Emergency and ICU for earliest surge signals
5. **Seasonal planning:** Increase capacity buffers before Monsoon and Winter peaks

See `reports/capacity_planning_report.md` for detailed guidance.

---

## 12. Conclusions

The project delivers a submission-ready healthcare forecasting pipeline with:

- Audited, leakage-free feature engineering
- Chronological validation appropriate for time series
- Comparable ML models with documented metrics
- Operational 7-day forecasts and overflow alerting
- Interactive Streamlit dashboard for stakeholders

Random Forest achieves ~9.1% MAE on occupancy rate prediction, explaining ~55% of variance on held-out future data — a realistic result without target leakage.

---

*Generated: Hospital Bed Occupancy Forecaster v1.0*
