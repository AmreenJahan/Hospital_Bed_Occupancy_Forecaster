"""
Hospital Bed Occupancy Forecaster - Dataset Preparation Script
=============================================================
Chronological train/test split with leakage-free features.
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import LEAKAGE_COLUMNS, TARGET_COLUMN, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import LEAKAGE_COLUMNS, TARGET_COLUMN, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def handle_categorical_variables(df):
    """Encode categorical variables; preserve Date for chronological split."""
    print("\n" + "=" * 80)
    print("HANDLING CATEGORICAL VARIABLES")
    print("=" * 80)

    df_encoded = df.copy()
    label_encoders = {}

    if 'Date' in df_encoded.columns:
        df_encoded['Date'] = pd.to_datetime(df_encoded['Date'])

    categorical_cols = [
        c for c in df_encoded.select_dtypes(include=['object']).columns
        if c != 'Date'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        df_encoded = df_encoded.drop(columns=[col])
        print(f"✓ Encoded {col}")

    bool_cols = df_encoded.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded, label_encoders


def remove_unnecessary_columns(df):
    """Remove non-feature columns and leakage columns."""
    print("\n" + "=" * 80)
    print("REMOVING UNNECESSARY / LEAKAGE COLUMNS")
    print("=" * 80)

    df_clean = df.copy()
    columns_to_remove = []

    for col in LEAKAGE_COLUMNS:
        if col in df_clean.columns:
            columns_to_remove.append(col)

    for col in df_clean.columns:
        if df_clean[col].nunique() <= 1:
            columns_to_remove.append(col)

    columns_to_remove = list(set(columns_to_remove))
    if columns_to_remove:
        df_clean = df_clean.drop(columns=columns_to_remove)
        print(f"✓ Removed: {columns_to_remove}")

    return df_clean


def create_feature_matrix_and_target(df, target_column=TARGET_COLUMN):
    """Create X and y; keep Date column in X for chronological split."""
    print("\n" + "=" * 80)
    print("CREATING FEATURE MATRIX AND TARGET VARIABLE")
    print("=" * 80)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].copy()
    y = df[target_column].copy()

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    return X, y


def perform_chronological_split(X, y, test_size=0.2):
    """Chronological 80/20 split sorted by Date."""
    print("\n" + "=" * 80)
    print("PERFORMING CHRONOLOGICAL TRAIN-TEST SPLIT")
    print("=" * 80)

    if 'Date' not in X.columns:
        raise ValueError("Date column required for chronological split")

    combined = X.copy()
    combined['Target'] = y.values
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.sort_values('Date').reset_index(drop=True)

    split_idx = int(len(combined) * (1 - test_size))
    train_df = combined.iloc[:split_idx]
    test_df = combined.iloc[split_idx:]

    split_date = test_df['Date'].min()

    y_train = train_df['Target']
    y_test = test_df['Target']
    X_train = train_df.drop(columns=['Target'])
    X_test = test_df.drop(columns=['Target'])

    # Remove Date from features after split
    if 'Date' in X_train.columns:
        X_train = X_train.drop(columns=['Date'])
        X_test = X_test.drop(columns=['Date'])

    print(f"Split date (test starts): {split_date.date()}")
    print(f"Training: {len(X_train):,} ({(1-test_size)*100:.0f}%)")
    print(f"Testing:  {len(X_test):,} ({test_size*100:.0f}%)")
    print(f"Train target mean: {y_train.mean():.4f}")
    print(f"Test target mean:  {y_test.mean():.4f}")

    split_info = {
        'split_type': 'chronological',
        'test_size': test_size,
        'split_date': str(split_date.date()),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }
    return X_train, X_test, y_train, y_test, split_info


def save_processed_datasets(X_train, X_test, y_train, y_test, output_dir=None):
    if output_dir is None:
        output_dir = PROCESSED_DIR
    """Save train/test CSV files."""
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATASETS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    train_data = X_train.copy()
    train_data['Target'] = y_train.values
    test_data = X_test.copy()
    test_data['Target'] = y_test.values

    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"✓ {train_path} ({train_data.shape})")
    print(f"✓ {test_path} ({test_data.shape})")
    return train_path, test_path


def generate_dataset_preparation_report(df_original, X_train, X_test, y_train, y_test, split_info):
    """Generate dataset preparation report."""
    report_lines = [
        "=" * 80,
        "HOSPITAL BED OCCUPANCY FORECASTER - DATASET PREPARATION REPORT",
        "=" * 80,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Original rows: {df_original.shape[0]:,}",
        f"Features: {X_train.shape[1]}",
        f"Training samples: {len(X_train):,}",
        f"Testing samples: {len(X_test):,}",
        "",
        "SPLIT METHOD: Chronological (time-aware)",
        f"Split date: {split_info['split_date']}",
        f"Train mean occupancy: {y_train.mean():.4f}",
        f"Test mean occupancy: {y_test.mean():.4f}",
        "",
        "LEAKAGE COLUMNS EXCLUDED:",
    ]
    for col in LEAKAGE_COLUMNS:
        report_lines.append(f"  - {col}")

    report_lines.extend([
        "",
        "OUTPUT FILES:",
        "  - data/processed/train_data.csv",
        "  - data/processed/test_data.csv",
        "  - models/label_encoders.pkl",
        "  - models/split_info.json",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    report_path = os.path.join(REPORTS_DIR, 'dataset_preparation_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)


def main():
    """Execute dataset preparation pipeline."""
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - DATASET PREPARATION")
    print("=" * 80)

    from config import ENGINEERED_DATA_PATH
    df = pd.read_csv(ENGINEERED_DATA_PATH)
    df_original = df.copy()

    df, label_encoders = handle_categorical_variables(df)
    df = remove_unnecessary_columns(df)
    X, y = create_feature_matrix_and_target(df)
    X_train, X_test, y_train, y_test, split_info = perform_chronological_split(X, y)

    save_processed_datasets(X_train, X_test, y_train, y_test)
    generate_dataset_preparation_report(df_original, X_train, X_test, y_train, y_test, split_info)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    with open(os.path.join(MODELS_DIR, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)

    print("\n✓ Dataset preparation completed")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
