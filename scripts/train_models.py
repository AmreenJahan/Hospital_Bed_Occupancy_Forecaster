"""
Train and compare forecasting models for hospital bed occupancy.
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BEST_MODEL_PATH, MODELS_DIR, REPORTS_DIR, SAVED_MODELS_DIR,
    TEST_DATA_PATH, TRAIN_DATA_PATH,
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_train_test_data():
    """Load prepared train and test datasets."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_train = train_df.drop(columns=['Target'])
    y_train = train_df['Target']
    X_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train RandomForest, GradientBoosting, and XGBoost with tuning."""
    models = {}

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    rf_search = GridSearchCV(
        rf, rf_grid, cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    models['RandomForest'] = rf_search.best_estimator_

    gbr = GradientBoostingRegressor(random_state=42)
    gbr_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
    }
    gbr_search = GridSearchCV(
        gbr, gbr_grid, cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    gbr_search.fit(X_train, y_train)
    models['GradientBoosting'] = gbr_search.best_estimator_

    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    xgb_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
    }
    xgb_search = GridSearchCV(
        xgb, xgb_grid, cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    models['XGBoost'] = xgb_search.best_estimator_

    tuning_results = {
        'RandomForest': rf_search.best_params_,
        'GradientBoosting': gbr_search.best_params_,
        'XGBoost': xgb_search.best_params_,
    }

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics['Model'] = name
        results.append(metrics)
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower()}_model.pkl'))

    comparison = pd.DataFrame(results)[['Model', 'MAE', 'MSE', 'RMSE', 'R2']]
    comparison = comparison.sort_values('MAE')
    return models, comparison, tuning_results


def select_best_model(comparison, models):
    """Select best model by lowest MAE."""
    best_name = comparison.iloc[0]['Model']
    best_model = models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(list(comparison.columns), os.path.join(MODELS_DIR, 'feature_columns_meta.pkl'))
    return best_name, best_model


def main():
    print("=" * 80)
    print("MODEL TRAINING - HOSPITAL BED OCCUPANCY FORECASTER")
    print("=" * 80)

    X_train, X_test, y_train, y_test = load_train_test_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    feature_names = list(X_train.columns)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    models, comparison, tuning_results = train_baseline_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best_model(comparison, models)

    comparison_path = os.path.join(REPORTS_DIR, 'model_comparison_table.csv')
    comparison.to_csv(comparison_path, index=False)

    summary = {
        'best_model': best_name,
        'best_metrics': comparison.iloc[0].to_dict(),
        'tuning_params': tuning_results,
        'feature_count': len(feature_names),
    }
    with open(os.path.join(REPORTS_DIR, 'model_training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved to: {BEST_MODEL_PATH}")
    return best_model, comparison


if __name__ == '__main__':
    main()
