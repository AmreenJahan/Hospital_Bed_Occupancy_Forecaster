"""
Model evaluation: metrics, feature importance, residuals, actual vs predicted.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BEST_MODEL_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR,
    TEST_DATA_PATH, TRAIN_DATA_PATH,
)

sns.set_style('whitegrid')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data_and_model():
    """Load test data and best model."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']
    model = joblib.load(BEST_MODEL_PATH)
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    X_test = X_test[feature_names]
    return model, X_test, y_test


def compute_metrics(y_true, y_pred):
    """Return evaluation metrics dict."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path):
    """Scatter plot of actual vs predicted occupancy."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors='k', linewidth=0.3, c='#2E86AB')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Occupancy Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted Occupancy Rate (%)', fontsize=12)
    ax.set_title(f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    """Residual distribution and residual vs predicted plots."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=40, color='#A23B72', edgecolor='black', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_title('Residual Distribution', fontweight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted)')

    axes[1].scatter(y_pred, residuals, alpha=0.4, c='#F18F01', edgecolors='k', linewidth=0.2)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs Predicted', fontweight='bold')
    axes[1].set_xlabel('Predicted Occupancy Rate (%)')
    axes[1].set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Bar chart of top feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color='#3A7D44',
        edgecolor='black',
    )
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in idx],
        'Importance': importances[idx],
    })
    importance_df.to_csv(os.path.join(REPORTS_DIR, 'feature_importance.csv'), index=False)
    return importance_df


def generate_evaluation_report(metrics, model_name, importance_df=None):
    """Write model evaluation report."""
    lines = [
        "=" * 80,
        "MODEL EVALUATION REPORT - HOSPITAL BED OCCUPANCY FORECASTER",
        "=" * 80,
        f"Best Model: {model_name}",
        "",
        "EVALUATION METRICS (Test Set)",
        "-" * 80,
        f"MAE:  {metrics['MAE']:.4f}",
        f"MSE:  {metrics['MSE']:.4f}",
        f"RMSE: {metrics['RMSE']:.4f}",
        f"R2:   {metrics['R2']:.4f}",
        "",
    ]
    if importance_df is not None:
        lines.extend(["TOP FEATURES", "-" * 80])
        for _, row in importance_df.head(10).iterrows():
            lines.append(f"  {row['Feature']}: {row['Importance']:.4f}")

    lines.extend(["", "=" * 80])
    report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ Report saved: {report_path}")
    return report_path


def main():
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    import json
    summary_path = os.path.join(REPORTS_DIR, 'model_training_summary.json')
    model_name = 'Best Model'
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            model_name = json.load(f).get('best_model', model_name)

    model, X_test, y_test = load_data_and_model()
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    feature_names = list(X_test.columns)

    print(f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    plot_actual_vs_predicted(
        y_test, y_pred, model_name,
        os.path.join(FIGURES_DIR, 'actual_vs_predicted.png'),
    )
    plot_residuals(y_test, y_pred, os.path.join(FIGURES_DIR, 'residual_analysis.png'))
    importance_df = plot_feature_importance(
        model, feature_names, os.path.join(FIGURES_DIR, 'feature_importance.png'),
    )
    generate_evaluation_report(metrics, model_name, importance_df)

    eval_df = pd.DataFrame([{**metrics, 'Model': model_name}])
    eval_df.to_csv(os.path.join(REPORTS_DIR, 'model_evaluation_metrics.csv'), index=False)
    print("✓ Evaluation complete")


if __name__ == '__main__':
    main()
