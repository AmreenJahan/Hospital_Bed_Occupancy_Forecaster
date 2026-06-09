"""
Hospital Bed Occupancy Forecaster - Data Cleaning Script
========================================================
This script performs comprehensive data cleaning including:
- Missing value handling
- Duplicate removal
- Invalid value detection and correction
- Data consistency checks
- Data quality report generation

Author: Data Science Intern
Date: June 2026
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check
        
    Returns:
    --------
    dict : Dictionary with missing value information
    """
    print("\n" + "=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    missing_info = {}
    
    # Overall missing statistics
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100
    
    missing_info['total_cells'] = total_cells
    missing_info['total_missing'] = total_missing
    missing_info['missing_percentage'] = missing_percentage
    
    print(f"\n📊 Overall Missing Statistics:")
    print(f"   - Total cells: {total_cells:,}")
    print(f"   - Missing cells: {total_missing:,}")
    print(f"   - Missing percentage: {missing_percentage:.2f}%")
    
    # Column-wise missing statistics
    print(f"\n📋 Column-wise Missing Values:")
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            col_percentage = (count / df.shape[0]) * 100
            print(f"   - {col}: {count:,} ({col_percentage:.2f}%)")
            missing_info[col] = {
                'count': count,
                'percentage': col_percentage
            }
    else:
        print("   ✓ No missing values found in any column!")
    
    return missing_info


def check_duplicates(df):
    """
    Check for duplicate rows in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check
        
    Returns:
    --------
    dict : Dictionary with duplicate information
    """
    print("\n" + "=" * 80)
    print("DUPLICATE ROWS ANALYSIS")
    print("=" * 80)
    
    duplicate_info = {}
    
    # Check for exact duplicates
    exact_duplicates = df.duplicated().sum()
    duplicate_percentage = (exact_duplicates / df.shape[0]) * 100
    
    duplicate_info['exact_duplicates'] = exact_duplicates
    duplicate_info['duplicate_percentage'] = duplicate_percentage
    
    print(f"\n📊 Duplicate Statistics:")
    print(f"   - Exact duplicate rows: {exact_duplicates:,}")
    print(f"   - Duplicate percentage: {duplicate_percentage:.2f}%")
    
    if exact_duplicates > 0:
        print(f"\n   ⚠️  Found {exact_duplicates:,} duplicate rows")
        print("   These will be removed during cleaning")
    else:
        print("   ✓ No duplicate rows found!")
    
    # Check for key column duplicates (if Date column exists)
    if 'Date' in df.columns:
        date_duplicates = df['Date'].duplicated().sum()
        print(f"\n📅 Date Column Duplicates:")
        print(f"   - Duplicate dates: {date_duplicates:,}")
        duplicate_info['date_duplicates'] = date_duplicates
    
    return duplicate_info


def check_invalid_values(df):
    """
    Check for invalid or inconsistent values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check
        
    Returns:
    --------
    dict : Dictionary with invalid value information
    """
    print("\n" + "=" * 80)
    print("INVALID VALUES ANALYSIS")
    print("=" * 80)
    
    invalid_info = {}
    
    # Check for negative values in columns that shouldn't have them
    print(f"\n🔍 Negative Values Check:")
    columns_to_check = [
        'Total_Hospital_Beds', 'Beds_Occupied', 'Occupancy_Rate',
        'Daily_Admissions', 'Daily_Discharges', 'Emergency_Admissions',
        'ICU_Admissions', 'Staff_Availability', 'Avg_Length_of_Stay'
    ]
    
    for col in columns_to_check:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   - {col}: {negative_count:,} negative values ⚠️")
                invalid_info[f'{col}_negative'] = negative_count
            else:
                print(f"   - {col}: ✓ No negative values")
    
    # Check for values outside expected ranges
    print(f"\n🔍 Range Validation:")
    
    # Occupancy rate should be between 0 and 1 (or 0 and 100)
    if 'Occupancy_Rate' in df.columns:
        max_occupancy = df['Occupancy_Rate'].max()
        min_occupancy = df['Occupancy_Rate'].min()
        
        if max_occupancy > 1.0:  # Assuming it's a percentage
            if max_occupancy > 100:
                print(f"   - Occupancy_Rate: Max value {max_occupancy:.2f} exceeds 100% ⚠️")
                invalid_info['occupancy_exceeds_100'] = True
        else:
            if max_occupancy > 1.0:
                print(f"   - Occupancy_Rate: Max value {max_occupancy:.2f} exceeds 1.0 ⚠️")
                invalid_info['occupancy_exceeds_1'] = True
        
        print(f"   - Occupancy_Rate: Range [{min_occupancy:.2f}, {max_occupancy:.2f}]")
    
    # Beds occupied should not exceed total beds
    if 'Beds_Occupied' in df.columns and 'Total_Hospital_Beds' in df.columns:
        overflow = (df['Beds_Occupied'] > df['Total_Hospital_Beds']).sum()
        if overflow > 0:
            print(f"   - Beds_Occupied > Total_Hospital_Beds: {overflow:,} cases ⚠️")
            invalid_info['beds_overflow'] = overflow
        else:
            print(f"   - Beds_Occupied <= Total_Hospital_Beds: ✓ Valid")
    
    # Check for infinite values
    print(f"\n🔍 Infinite Values Check:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"   - {col}: {inf_count:,} infinite values ⚠️")
            invalid_info[f'{col}_infinite'] = inf_count
    
    return invalid_info


def check_data_consistency(df):
    """
    Check for data consistency across related columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check
        
    Returns:
    --------
    dict : Dictionary with consistency information
    """
    print("\n" + "=" * 80)
    print("DATA CONSISTENCY CHECK")
    print("=" * 80)
    
    consistency_info = {}
    
    # Check if Occupancy_Rate matches calculated rate
    if all(col in df.columns for col in ['Beds_Occupied', 'Total_Hospital_Beds', 'Occupancy_Rate']):
        calculated_rate = (df['Beds_Occupied'] / df['Total_Hospital_Beds']) * 100
        # Allow small tolerance for floating point comparison
        tolerance = 0.1
        mismatch = ((calculated_rate - df['Occupancy_Rate']).abs() > tolerance).sum()
        
        print(f"\n📊 Occupancy Rate Consistency:")
        print(f"   - Mismatch between calculated and reported rate: {mismatch:,} cases")
        
        if mismatch > 0:
            print(f"   ⚠️  Some occupancy rates don't match (Beds_Occupied/Total_Hospital_Beds)*100")
            consistency_info['occupancy_rate_mismatch'] = mismatch
        else:
            print(f"   ✓ Occupancy rates are consistent")
    
    # Check day of week consistency with date
    if 'Date' in df.columns and 'Day_of_Week' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            actual_day_of_week = df['Date'].dt.day_name()
            mismatch = (actual_day_of_week != df['Day_of_Week']).sum()
            
            print(f"\n📅 Day of Week Consistency:")
            print(f"   - Mismatch between date and Day_of_Week: {mismatch:,} cases")
            
            if mismatch > 0:
                consistency_info['day_of_week_mismatch'] = mismatch
            else:
                print(f"   ✓ Day of week is consistent with date")
        except Exception as e:
            print(f"   ⚠️  Could not validate day of week: {e}")
    
    # Check month consistency with date
    if 'Date' in df.columns and 'Month' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            actual_month = df['Date'].dt.month
            mismatch = (actual_month != df['Month']).sum()
            
            print(f"\n📅 Month Consistency:")
            print(f"   - Mismatch between date and Month: {mismatch:,} cases")
            
            if mismatch > 0:
                consistency_info['month_mismatch'] = mismatch
            else:
                print(f"   ✓ Month is consistent with date")
        except Exception as e:
            print(f"   ⚠️  Could not validate month: {e}")
    
    return consistency_info


def clean_data(df):
    """
    Perform data cleaning operations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to clean
        
    Returns:
    --------
    pandas.DataFrame : Cleaned dataframe
    dict : Dictionary with cleaning operations performed
    """
    print("\n" + "=" * 80)
    print("DATA CLEANING OPERATIONS")
    print("=" * 80)
    
    cleaning_log = {}
    original_shape = df.shape
    df_cleaned = df.copy()
    
    # 1. Remove duplicate rows
    print(f"\n🧹 Removing duplicate rows...")
    duplicates_before = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = duplicates_before - df_cleaned.duplicated().sum()
    print(f"   - Removed {duplicates_removed:,} duplicate rows")
    cleaning_log['duplicates_removed'] = duplicates_removed
    
    # 2. Handle missing values
    print(f"\n🧹 Handling missing values...")
    missing_before = df_cleaned.isnull().sum().sum()
    
    # For numerical columns, fill with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            print(f"   - Filled {col} missing values with median: {median_value:.2f}")
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_value, inplace=True)
            print(f"   - Filled {col} missing values with mode: {mode_value}")
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"   - Total missing values handled: {missing_before - missing_after:,}")
    cleaning_log['missing_values_filled'] = missing_before - missing_after
    
    # 3. Handle invalid values
    print(f"\n🧹 Handling invalid values...")
    
    # Replace negative values with 0 for columns that shouldn't have negatives
    columns_to_fix = [
        'Total_Hospital_Beds', 'Beds_Occupied', 'Occupancy_Rate',
        'Daily_Admissions', 'Daily_Discharges', 'Emergency_Admissions',
        'ICU_Admissions', 'Staff_Availability', 'Avg_Length_of_Stay'
    ]
    
    for col in columns_to_fix:
        if col in df_cleaned.columns:
            negative_count = (df_cleaned[col] < 0).sum()
            if negative_count > 0:
                df_cleaned.loc[df_cleaned[col] < 0, col] = 0
                print(f"   - Replaced {negative_count:,} negative values in {col} with 0")
                cleaning_log[f'{col}_negatives_fixed'] = negative_count
    
    # 4. Handle infinite values
    print(f"\n🧹 Handling infinite values...")
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        inf_count = np.isinf(df_cleaned[col]).sum()
        if inf_count > 0:
            df_cleaned.loc[np.isinf(df_cleaned[col]), col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan).median()
            print(f"   - Replaced {inf_count:,} infinite values in {col}")
            cleaning_log[f'{col}_infinite_fixed'] = inf_count
    
    # 5. Ensure date is datetime
    if 'Date' in df_cleaned.columns:
        print(f"\n🧹 Converting Date column to datetime...")
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
        print(f"   ✓ Date column converted to datetime")

    # 6. Deduplicate Date + Department (keep mean for numeric, first for categorical)
    if 'Date' in df_cleaned.columns and 'Department' in df_cleaned.columns:
        print(f"\n🧹 Deduplicating Date + Department records...")
        dup_before = df_cleaned.duplicated(subset=['Date', 'Department']).sum()
        bool_cols = [c for c in df_cleaned.columns if df_cleaned[c].dtype == bool]
        cat_cols = [c for c in df_cleaned.select_dtypes(include=['object']).columns if c not in ['Date', 'Department']]
        num_cols = [c for c in df_cleaned.select_dtypes(include=[np.number]).columns]

        agg_dict = {c: 'mean' for c in num_cols}
        for c in cat_cols:
            agg_dict[c] = 'first'
        for c in bool_cols:
            agg_dict[c] = 'max'

        df_cleaned = (
            df_cleaned.groupby(['Date', 'Department'], as_index=False)
            .agg(agg_dict)
            .sort_values(['Department', 'Date'])
            .reset_index(drop=True)
        )
        cleaning_log['date_department_deduped'] = dup_before
        print(f"   - Resolved {dup_before:,} duplicate Date+Department rows via aggregation")
        print(f"   - Shape after dedup: {df_cleaned.shape[0]:,} rows")

    # 7. Flag over-capacity records (Occupancy_Rate > 100%)
    if 'Occupancy_Rate' in df_cleaned.columns:
        over_capacity = (df_cleaned['Occupancy_Rate'] > 100).sum()
        df_cleaned['Over_Capacity'] = (df_cleaned['Occupancy_Rate'] > 100).astype(int)
        cleaning_log['over_capacity_flagged'] = over_capacity
        print(f"\n🧹 Over-capacity flag added: {over_capacity:,} records exceed 100% occupancy")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"   Original shape: {original_shape[0]:,} rows × {original_shape[1]} columns")
    print(f"   Cleaned shape: {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
    print(f"   Rows removed: {original_shape[0] - df_cleaned.shape[0]:,}")
    print(f"   Missing values filled: {cleaning_log.get('missing_values_filled', 0):,}")
    print(f"   Duplicates removed: {cleaning_log.get('duplicates_removed', 0):,}")
    
    return df_cleaned, cleaning_log


def generate_data_quality_report(df, df_cleaned, missing_info, duplicate_info, invalid_info, consistency_info, cleaning_log):
    """
    Generate a comprehensive data quality report.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    df_cleaned : pandas.DataFrame
        Cleaned dataframe
    missing_info : dict
        Missing value information
    duplicate_info : dict
        Duplicate information
    invalid_info : dict
        Invalid value information
    consistency_info : dict
        Consistency information
    cleaning_log : dict
        Cleaning operations log
    """
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HOSPITAL BED OCCUPANCY FORECASTER - DATA QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset Overview
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Original Dataset:")
    report_lines.append(f"  - Rows: {df.shape[0]:,}")
    report_lines.append(f"  - Columns: {df.shape[1]}")
    report_lines.append(f"  - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report_lines.append("")
    report_lines.append(f"Cleaned Dataset:")
    report_lines.append(f"  - Rows: {df_cleaned.shape[0]:,}")
    report_lines.append(f"  - Columns: {df_cleaned.shape[1]}")
    report_lines.append(f"  - Memory Usage: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report_lines.append("")
    
    # Missing Values
    report_lines.append("2. MISSING VALUES ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total missing cells: {missing_info['total_missing']:,}")
    report_lines.append(f"Missing percentage: {missing_info['missing_percentage']:.2f}%")
    if len(missing_info) > 2:  # More than just total_cells and total_missing
        report_lines.append("")
        report_lines.append("Missing values by column:")
        for col, info in missing_info.items():
            if col not in ['total_cells', 'total_missing', 'missing_percentage']:
                report_lines.append(f"  - {col}: {info['count']:,} ({info['percentage']:.2f}%)")
    report_lines.append("")
    
    # Duplicates
    report_lines.append("3. DUPLICATE ROWS ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Exact duplicate rows: {duplicate_info['exact_duplicates']:,}")
    report_lines.append(f"Duplicate percentage: {duplicate_info['duplicate_percentage']:.2f}%")
    if 'date_duplicates' in duplicate_info:
        report_lines.append(f"Duplicate dates: {duplicate_info['date_duplicates']:,}")
    report_lines.append("")
    
    # Invalid Values
    report_lines.append("4. INVALID VALUES ANALYSIS")
    report_lines.append("-" * 80)
    if invalid_info:
        for issue, count in invalid_info.items():
            if isinstance(count, bool):
                report_lines.append(f"  - {issue}: Detected")
            else:
                report_lines.append(f"  - {issue}: {count:,} cases")
    else:
        report_lines.append("  No invalid values detected")
    report_lines.append("")
    
    # Data Consistency
    report_lines.append("5. DATA CONSISTENCY CHECK")
    report_lines.append("-" * 80)
    if consistency_info:
        for issue, count in consistency_info.items():
            report_lines.append(f"  - {issue}: {count:,} cases")
    else:
        report_lines.append("  All consistency checks passed")
    report_lines.append("")
    
    # Cleaning Operations
    report_lines.append("6. CLEANING OPERATIONS PERFORMED")
    report_lines.append("-" * 80)
    for operation, value in cleaning_log.items():
        if isinstance(value, int):
            report_lines.append(f"  - {operation}: {value:,}")
        else:
            report_lines.append(f"  - {operation}: {value}")
    report_lines.append("")
    
    # Final Assessment
    report_lines.append("7. FINAL DATA QUALITY ASSESSMENT")
    report_lines.append("-" * 80)
    final_missing = df_cleaned.isnull().sum().sum()
    final_duplicates = df_cleaned.duplicated().sum()
    
    if final_missing == 0 and final_duplicates == 0:
        report_lines.append("✓ DATA QUALITY: EXCELLENT")
        report_lines.append("  - No missing values")
        report_lines.append("  - No duplicate rows")
        report_lines.append("  - Ready for analysis and modeling")
    elif final_missing == 0:
        report_lines.append("✓ DATA QUALITY: GOOD")
        report_lines.append("  - No missing values")
        report_lines.append("  - No duplicate rows")
        report_lines.append("  - Ready for analysis and modeling")
    else:
        report_lines.append("⚠ DATA QUALITY: NEEDS ATTENTION")
        report_lines.append(f"  - Remaining missing values: {final_missing:,}")
        report_lines.append(f"  - Remaining duplicate rows: {final_duplicates:,}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report to file
    try:
        from config import REPORTS_DIR, CLEANED_DATA_PATH, RAW_DATA_PATH
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import REPORTS_DIR, CLEANED_DATA_PATH, RAW_DATA_PATH
    report_path = os.path.join(REPORTS_DIR, 'data_quality_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Data quality report saved to: {report_path}")
    
    return report_text


def main():
    """
    Main function to execute data cleaning pipeline.
    """
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - DATA CLEANING")
    print("=" * 80)
    
    # Load data (assuming load_data.py has been run)
    try:
        from load_data import load_data
        df = load_data('../data/hospital_bed_occupancy_10000.csv')
    except:
        # If load_data import fails, load directly
        print("\nLoading data directly...")
        df = pd.read_csv('../data/hospital_bed_occupancy_10000.csv')
    
    # Perform data quality checks
    missing_info = check_missing_values(df)
    duplicate_info = check_duplicates(df)
    invalid_info = check_invalid_values(df)
    consistency_info = check_data_consistency(df)
    
    # Clean the data
    df_cleaned, cleaning_log = clean_data(df)
    
    # Generate data quality report
    generate_data_quality_report(
        df, df_cleaned, missing_info, duplicate_info,
        invalid_info, consistency_info, cleaning_log
    )
    
    print("\n" + "=" * 80)
    print("DATA CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Save cleaned data
    cleaned_data_path = '../data/cleaned_hospital_bed_occupancy.csv'
    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"\n✓ Cleaned dataset saved to: {cleaned_data_path}")
    
    return df_cleaned


if __name__ == "__main__":
    df_cleaned = main()
