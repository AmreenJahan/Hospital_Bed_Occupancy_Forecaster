"""
Hospital Bed Occupancy Forecaster - Data Loading Script
========================================================
This script loads the hospital bed occupancy dataset and performs initial validation.
It displays dataset information including shape, columns, data types, and summary statistics.

Author: Data Science Intern
Date: June 2026
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(file_path):
    """
    Load the hospital bed occupancy dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    print("=" * 80)
    print("LOADING HOSPITAL BED OCCUPANCY DATASET")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # Load the dataset
    print(f"\nLoading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    print("✓ Dataset loaded successfully!")
    
    return df


def display_dataset_info(df):
    """
    Display comprehensive information about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    """
    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    
    # Display shape
    print(f"\n📊 Dataset Shape:")
    print(f"   - Number of Rows: {df.shape[0]:,}")
    print(f"   - Number of Columns: {df.shape[1]}")
    
    # Display column names
    print(f"\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Display data types
    print(f"\n🔢 Data Types:")
    print(df.dtypes)
    
    # Display memory usage
    print(f"\n💾 Memory Usage:")
    print(f"   Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def display_summary_statistics(df):
    """
    Display summary statistics for numerical and categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Numerical columns statistics
    print("\n📈 Numerical Columns Statistics:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe().round(2))
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\n📝 Categorical Columns Statistics:")
        for col in categorical_cols:
            print(f"\n   {col}:")
            print(f"   - Unique values: {df[col].nunique()}")
            print(f"   - Most frequent: {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}")


def display_sample_data(df, n=5):
    """
    Display sample data from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to display
    n : int
        Number of rows to display (default: 5)
    """
    print("\n" + "=" * 80)
    print(f"SAMPLE DATA (First {n} Rows)")
    print("=" * 80)
    print(df.head(n).to_string())


def validate_dataset_quality(df):
    """
    Perform basic quality validation on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to validate
        
    Returns:
    --------
    dict : Dictionary containing validation results
    """
    print("\n" + "=" * 80)
    print("DATASET QUALITY VALIDATION")
    print("=" * 80)
    
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    validation_results['total_missing'] = total_missing
    validation_results['missing_percentage'] = (total_missing / (df.shape[0] * df.shape[1])) * 100
    
    print(f"\n🔍 Missing Values:")
    print(f"   - Total missing values: {total_missing:,}")
    print(f"   - Percentage: {validation_results['missing_percentage']:.2f}%")
    
    if total_missing > 0:
        print("\n   Missing values by column:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"   - {col}: {count:,} ({(count/df.shape[0])*100:.2f}%)")
    else:
        print("   ✓ No missing values found!")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    validation_results['duplicates'] = duplicates
    validation_results['duplicate_percentage'] = (duplicates / df.shape[0]) * 100
    
    print(f"\n🔍 Duplicate Rows:")
    print(f"   - Total duplicates: {duplicates:,}")
    print(f"   - Percentage: {validation_results['duplicate_percentage']:.2f}%")
    
    if duplicates == 0:
        print("   ✓ No duplicate rows found!")
    
    # Check for negative values in numerical columns (where inappropriate)
    print(f"\n🔍 Negative Values Check:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['Day_of_Week', 'Month']:  # These can have negative values in some contexts
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   - {col}: {negative_count:,} negative values")
            else:
                print(f"   - {col}: ✓ No negative values")
    
    # Check date column if exists
    if 'Date' in df.columns:
        print(f"\n🔍 Date Column Validation:")
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"   - Total days: {(df['Date'].max() - df['Date'].min()).days}")
            print("   ✓ Date column is valid")
        except Exception as e:
            print(f"   ✗ Error parsing dates: {e}")
    
    return validation_results


def main():
    """
    Main function to execute data loading and validation.
    """
    # Define file path
    file_path = '../data/hospital_bed_occupancy_10000.csv'
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Display dataset information
        display_dataset_info(df)
        
        # Display summary statistics
        display_summary_statistics(df)
        
        # Display sample data
        display_sample_data(df, n=5)
        
        # Validate dataset quality
        validation_results = validate_dataset_quality(df)
        
        # Save validation results
        print("\n" + "=" * 80)
        print("DATA LOADING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n✓ Dataset loaded and validated")
        print(f"✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"✓ Missing values: {validation_results['total_missing']:,}")
        print(f"✓ Duplicate rows: {validation_results['duplicates']:,}")
        
        # Return the dataframe for further processing
        return df
        
    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")
        raise


if __name__ == "__main__":
    # Execute main function
    df = main()
