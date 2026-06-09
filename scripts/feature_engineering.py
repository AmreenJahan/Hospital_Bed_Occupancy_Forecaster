"""
Hospital Bed Occupancy Forecaster - Feature Engineering Script
==============================================================
Department-aware lag/rolling features without target leakage.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Columns that leak target information and must not be used as features
LEAKAGE_COLUMNS = [
    'Beds_Occupied',
    'Bed_Utilization_Ratio',
    'Staff_Occupancy_Ratio',
    'Staff_Occupancy_Interaction',
    'Emergency_Occupancy_Interaction',
    'Lag_1_Day_Beds',
    'Lag_7_Day_Beds',
    'Rolling_7_Day_Beds_Mean',
    'Rolling_30_Day_Beds_Mean',
]


def _sort_by_department_date(df):
    """Sort dataframe by department then date for time-aware features."""
    df_fe = df.copy()
    if 'Date' in df_fe.columns:
        df_fe['Date'] = pd.to_datetime(df_fe['Date'])
    if 'Department' in df_fe.columns:
        return df_fe.sort_values(['Department', 'Date']).reset_index(drop=True)
    return df_fe.sort_values('Date').reset_index(drop=True)


def create_date_features(df):
    """Create date-based features from the Date column."""
    print("\n" + "=" * 80)
    print("CREATING DATE FEATURES")
    print("=" * 80)

    df_fe = df.copy()
    if 'Date' not in df_fe.columns:
        print("⚠️  Date column not found in dataset")
        return df_fe

    df_fe['Date'] = pd.to_datetime(df_fe['Date'])
    df_fe['Year'] = df_fe['Date'].dt.year
    df_fe['Quarter'] = df_fe['Date'].dt.quarter
    df_fe['Month_Num'] = df_fe['Date'].dt.month
    df_fe['Week_Number'] = df_fe['Date'].dt.isocalendar().week.astype(int)
    df_fe['Day'] = df_fe['Date'].dt.day
    df_fe['Day_of_Week_Num'] = df_fe['Date'].dt.dayofweek
    df_fe['Day_of_Year'] = df_fe['Date'].dt.dayofyear
    df_fe['Is_Weekend'] = (df_fe['Day_of_Week_Num'] >= 5).astype(int)

    print(f"\n✓ Date features created (range: {df_fe['Date'].min()} to {df_fe['Date'].max()})")
    return df_fe


def create_lag_features(df):
    """Create lag features per Department using groupby."""
    print("\n" + "=" * 80)
    print("CREATING LAG FEATURES (PER DEPARTMENT)")
    print("=" * 80)

    df_fe = _sort_by_department_date(df)
    group = df_fe.groupby('Department') if 'Department' in df_fe.columns else None

    if 'Occupancy_Rate' in df_fe.columns:
        if group is not None:
            df_fe['Lag_1_Day_Occupancy'] = group['Occupancy_Rate'].shift(1)
            df_fe['Lag_7_Day_Occupancy'] = group['Occupancy_Rate'].shift(7)
            df_fe['Lag_30_Day_Occupancy'] = group['Occupancy_Rate'].shift(30)
        else:
            df_fe['Lag_1_Day_Occupancy'] = df_fe['Occupancy_Rate'].shift(1)
            df_fe['Lag_7_Day_Occupancy'] = df_fe['Occupancy_Rate'].shift(7)
            df_fe['Lag_30_Day_Occupancy'] = df_fe['Occupancy_Rate'].shift(30)
        print("✓ Occupancy lag features (1, 7, 30 day) per department")

    if 'Daily_Admissions' in df_fe.columns:
        if group is not None:
            df_fe['Lag_1_Day_Admissions'] = group['Daily_Admissions'].shift(1)
            df_fe['Lag_7_Day_Admissions'] = group['Daily_Admissions'].shift(7)
            df_fe['Lag_30_Day_Admissions'] = group['Daily_Admissions'].shift(30)
        else:
            df_fe['Lag_1_Day_Admissions'] = df_fe['Daily_Admissions'].shift(1)
            df_fe['Lag_7_Day_Admissions'] = df_fe['Daily_Admissions'].shift(7)
            df_fe['Lag_30_Day_Admissions'] = df_fe['Daily_Admissions'].shift(30)
        print("✓ Admissions lag features per department")

    return df_fe


def create_rolling_features(df):
    """Create rolling window features per Department using groupby."""
    print("\n" + "=" * 80)
    print("CREATING ROLLING FEATURES (PER DEPARTMENT)")
    print("=" * 80)

    df_fe = _sort_by_department_date(df)
    group = df_fe.groupby('Department') if 'Department' in df_fe.columns else None

    if 'Occupancy_Rate' in df_fe.columns:
        if group is not None:
            df_fe['Rolling_7_Day_Occupancy_Mean'] = (
                group['Occupancy_Rate'].transform(lambda s: s.rolling(7, min_periods=1).mean())
            )
            df_fe['Rolling_7_Day_Occupancy_Std'] = (
                group['Occupancy_Rate'].transform(lambda s: s.rolling(7, min_periods=1).std())
            )
            df_fe['Rolling_30_Day_Occupancy_Mean'] = (
                group['Occupancy_Rate'].transform(lambda s: s.rolling(30, min_periods=1).mean())
            )
            df_fe['Rolling_30_Day_Occupancy_Std'] = (
                group['Occupancy_Rate'].transform(lambda s: s.rolling(30, min_periods=1).std())
            )
        else:
            df_fe['Rolling_7_Day_Occupancy_Mean'] = df_fe['Occupancy_Rate'].rolling(7, min_periods=1).mean()
            df_fe['Rolling_7_Day_Occupancy_Std'] = df_fe['Occupancy_Rate'].rolling(7, min_periods=1).std()
            df_fe['Rolling_30_Day_Occupancy_Mean'] = df_fe['Occupancy_Rate'].rolling(30, min_periods=1).mean()
            df_fe['Rolling_30_Day_Occupancy_Std'] = df_fe['Occupancy_Rate'].rolling(30, min_periods=1).std()
        print("✓ Occupancy rolling features per department")

    if 'Daily_Admissions' in df_fe.columns:
        if group is not None:
            df_fe['Rolling_7_Day_Admissions_Mean'] = (
                group['Daily_Admissions'].transform(lambda s: s.rolling(7, min_periods=1).mean())
            )
            df_fe['Rolling_7_Day_Admissions_Std'] = (
                group['Daily_Admissions'].transform(lambda s: s.rolling(7, min_periods=1).std())
            )
            df_fe['Rolling_30_Day_Admissions_Mean'] = (
                group['Daily_Admissions'].transform(lambda s: s.rolling(30, min_periods=1).mean())
            )
        else:
            df_fe['Rolling_7_Day_Admissions_Mean'] = df_fe['Daily_Admissions'].rolling(7, min_periods=1).mean()
            df_fe['Rolling_7_Day_Admissions_Std'] = df_fe['Daily_Admissions'].rolling(7, min_periods=1).std()
            df_fe['Rolling_30_Day_Admissions_Mean'] = df_fe['Daily_Admissions'].rolling(30, min_periods=1).mean()
        print("✓ Admissions rolling features per department")

    return df_fe


def create_ratio_features(df):
    """Create ratio-based features without target leakage."""
    print("\n" + "=" * 80)
    print("CREATING RATIO FEATURES (NO LEAKAGE)")
    print("=" * 80)

    df_fe = df.copy()

    if 'Daily_Admissions' in df_fe.columns and 'Daily_Discharges' in df_fe.columns:
        df_fe['Admission_Discharge_Ratio'] = np.where(
            df_fe['Daily_Discharges'] != 0,
            df_fe['Daily_Admissions'] / df_fe['Daily_Discharges'],
            df_fe['Daily_Admissions'],
        )
        print("✓ Admission_Discharge_Ratio")

    if 'ICU_Admissions' in df_fe.columns and 'Daily_Admissions' in df_fe.columns:
        df_fe['ICU_Admission_Ratio'] = np.where(
            df_fe['Daily_Admissions'] != 0,
            df_fe['ICU_Admissions'] / df_fe['Daily_Admissions'],
            0,
        )
        print("✓ ICU_Admission_Ratio")

    if 'Emergency_Admissions' in df_fe.columns and 'Daily_Admissions' in df_fe.columns:
        df_fe['Emergency_Admission_Ratio'] = np.where(
            df_fe['Daily_Admissions'] != 0,
            df_fe['Emergency_Admissions'] / df_fe['Daily_Admissions'],
            0,
        )
        print("✓ Emergency_Admission_Ratio")

    if 'Staff_Availability' in df_fe.columns and 'Total_Hospital_Beds' in df_fe.columns:
        df_fe['Staff_Per_Bed_Ratio'] = df_fe['Staff_Availability'] / (df_fe['Total_Hospital_Beds'] + 1)
        print("✓ Staff_Per_Bed_Ratio")

    return df_fe


def create_interaction_features(df):
    """Create interaction features without using Occupancy_Rate."""
    print("\n" + "=" * 80)
    print("CREATING INTERACTION FEATURES (NO LEAKAGE)")
    print("=" * 80)

    df_fe = df.copy()

    if 'Daily_Admissions' in df_fe.columns and 'Avg_Length_of_Stay' in df_fe.columns:
        df_fe['Admissions_LOS_Interaction'] = df_fe['Daily_Admissions'] * df_fe['Avg_Length_of_Stay']
        print("✓ Admissions_LOS_Interaction")

    if 'Staff_Availability' in df_fe.columns and 'Daily_Admissions' in df_fe.columns:
        df_fe['Staff_Admissions_Interaction'] = df_fe['Staff_Availability'] * df_fe['Daily_Admissions']
        print("✓ Staff_Admissions_Interaction")

    if 'Emergency_Admissions' in df_fe.columns and 'Daily_Admissions' in df_fe.columns:
        df_fe['Emergency_Admissions_Interaction'] = df_fe['Emergency_Admissions'] * df_fe['Daily_Admissions']
        print("✓ Emergency_Admissions_Interaction")

    return df_fe


def remove_leakage_columns(df):
    """Drop columns that leak target information."""
    existing = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if existing:
        print(f"\n✓ Removing leakage columns: {existing}")
        df = df.drop(columns=existing)
    return df


def handle_missing_values_in_features(df):
    """Handle missing values in engineered features."""
    print("\n" + "=" * 80)
    print("HANDLING MISSING VALUES IN ENGINEERED FEATURES")
    print("=" * 80)

    df_fe = df.copy()
    missing_before = df_fe.isnull().sum().sum()
    print(f"\nMissing values before handling: {missing_before:,}")

    lag_columns = [col for col in df_fe.columns if 'Lag' in col]
    rolling_columns = [col for col in df_fe.columns if 'Rolling' in col]

    for col in lag_columns + rolling_columns:
        if col in df_fe.columns:
            df_fe[col] = df_fe[col].ffill().bfill().fillna(0)

    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_fe[col].isnull().sum() > 0:
            df_fe[col] = df_fe[col].fillna(df_fe[col].median())

    missing_after = df_fe.isnull().sum().sum()
    print(f"Missing values after handling: {missing_after:,}")
    return df_fe


def generate_feature_engineering_report(df_original, df_engineered, report_path='../reports/feature_engineering_report.txt'):
    """Generate feature engineering report."""
    print("\n" + "=" * 80)
    print("GENERATING FEATURE ENGINEERING REPORT")
    print("=" * 80)

    new_features = sorted(set(df_engineered.columns) - set(df_original.columns))
    report_lines = [
        "=" * 80,
        "HOSPITAL BED OCCUPANCY FORECASTER - FEATURE ENGINEERING REPORT",
        "=" * 80,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. DATASET COMPARISON",
        "-" * 80,
        f"Original rows: {df_original.shape[0]:,}, columns: {df_original.shape[1]}",
        f"Engineered rows: {df_engineered.shape[0]:,}, columns: {df_engineered.shape[1]}",
        f"New features: {len(new_features)}",
        "",
        "2. TIME-AWARE FEATURE ENGINEERING",
        "-" * 80,
        "✓ Lag features computed per Department (groupby)",
        "✓ Rolling features computed per Department (groupby)",
        "✓ Chronological ordering preserved within each department",
        "",
        "3. LEAKAGE PREVENTION",
        "-" * 80,
        "Removed or excluded leakage features:",
    ]
    for col in LEAKAGE_COLUMNS:
        report_lines.append(f"  - {col}")

    report_lines.extend([
        "",
        "4. NEW FEATURES",
        "-" * 80,
    ])
    for feat in new_features:
        report_lines.append(f"  - {feat}")

    report_lines.extend([
        "",
        "5. READY FOR MODELING",
        "-" * 80,
        "Engineered dataset is ready for chronological train/test split.",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n✓ Feature engineering report saved to: {report_path}")


def main():
    """Execute feature engineering pipeline."""
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - FEATURE ENGINEERING")
    print("=" * 80)

    try:
        df = pd.read_csv('../data/cleaned_hospital_bed_occupancy.csv')
        print("\n✓ Loaded cleaned dataset")
    except FileNotFoundError:
        df = pd.read_csv('../data/hospital_bed_occupancy_10000.csv')
        print("\n✓ Loaded raw dataset")

    df_original = df.copy()
    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_ratio_features(df)
    df = create_interaction_features(df)
    df = remove_leakage_columns(df)
    df = handle_missing_values_in_features(df)

    print(f"\nEngineered dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    generate_feature_engineering_report(df_original, df)

    engineered_data_path = '../data/engineered_hospital_bed_occupancy.csv'
    df.to_csv(engineered_data_path, index=False)
    print(f"\n✓ Engineered dataset saved to: {engineered_data_path}")
    return df


if __name__ == "__main__":
    main()
