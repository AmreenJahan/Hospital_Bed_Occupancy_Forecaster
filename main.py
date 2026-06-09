"""
Hospital Bed Occupancy Forecaster - Main Execution Script
========================================================
This script executes the complete Day 1 pipeline:
1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Operational Insights
5. Feature Engineering
6. Dataset Preparation

Author: Data Science Intern
Date: June 2026
"""

import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))


def print_section_header(title):
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Title of the section
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_step_summary(step_number, step_name, status):
    """
    Print a step summary.
    
    Parameters:
    -----------
    step_number : int
        Step number
    step_name : str
        Name of the step
    status : str
        Status of the step (COMPLETED, FAILED, etc.)
    """
    print(f"\n{'='*80}")
    print(f"STEP {step_number}: {step_name} - {status}")
    print(f"{'='*80}")


def main():
    """
    Main function to execute the complete Day 1 pipeline.
    """
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - DAY 1 PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will execute the complete Day 1 pipeline:")
    print("  1. Data Loading")
    print("  2. Data Cleaning")
    print("  3. Exploratory Data Analysis (EDA)")
    print("  4. Operational Insights")
    print("  5. Feature Engineering")
    print("  6. Dataset Preparation")
    print("\n" + "=" * 80)
    
    # Track execution status
    execution_log = []
    
    # STEP 1: Data Loading
    print_step_summary(1, "DATA LOADING", "IN PROGRESS")
    try:
        from scripts.load_data import load_data, display_dataset_info, display_summary_statistics, display_sample_data, validate_dataset_quality
        
        file_path = 'data/hospital_bed_occupancy_10000.csv'
        df = load_data(file_path)
        display_dataset_info(df)
        display_summary_statistics(df)
        display_sample_data(df, n=5)
        validation_results = validate_dataset_quality(df)
        
        print_step_summary(1, "DATA LOADING", "COMPLETED ✓")
        execution_log.append(("Data Loading", "SUCCESS", f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns"))
    except Exception as e:
        print_step_summary(1, "DATA LOADING", f"FAILED ✗ - {str(e)}")
        execution_log.append(("Data Loading", "FAILED", str(e)))
        return
    
    # STEP 2: Data Cleaning
    print_step_summary(2, "DATA CLEANING", "IN PROGRESS")
    try:
        from scripts.data_cleaning import (
            check_missing_values, check_duplicates, check_invalid_values,
            check_data_consistency, clean_data, generate_data_quality_report
        )
        
        missing_info = check_missing_values(df)
        duplicate_info = check_duplicates(df)
        invalid_info = check_invalid_values(df)
        consistency_info = check_data_consistency(df)
        from config import CLEANED_DATA_PATH
        df_cleaned, cleaning_log = clean_data(df)
        generate_data_quality_report(
            df, df_cleaned, missing_info, duplicate_info,
            invalid_info, consistency_info, cleaning_log
        )
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
        
        print_step_summary(2, "DATA CLEANING", "COMPLETED ✓")
        execution_log.append(("Data Cleaning", "SUCCESS", f"Cleaned {df_cleaned.shape[0]:,} rows"))
    except Exception as e:
        print_step_summary(2, "DATA CLEANING", f"FAILED ✗ - {str(e)}")
        execution_log.append(("Data Cleaning", "FAILED", str(e)))
        return
    
    # STEP 3: Exploratory Data Analysis
    print_step_summary(3, "EXPLORATORY DATA ANALYSIS (EDA)", "IN PROGRESS")
    try:
        from scripts.exploratory_data_analysis import (
            dataset_overview, occupancy_analysis, admissions_analysis,
            department_analysis, seasonal_analysis, time_based_analysis,
            hospital_operations_analysis, correlation_analysis, generate_eda_report
        )
        
        dataset_overview(df_cleaned)
        occupancy_analysis(df_cleaned)
        admissions_analysis(df_cleaned)
        department_analysis(df_cleaned)
        seasonal_analysis(df_cleaned)
        time_based_analysis(df_cleaned)
        hospital_operations_analysis(df_cleaned)
        correlation_analysis(df_cleaned)
        generate_eda_report(df_cleaned)
        
        print_step_summary(3, "EXPLORATORY DATA ANALYSIS (EDA)", "COMPLETED ✓")
        execution_log.append(("EDA", "SUCCESS", "Generated 8 visualizations and report"))
    except Exception as e:
        print_step_summary(3, "EXPLORATORY DATA ANALYSIS (EDA)", f"FAILED ✗ - {str(e)}")
        execution_log.append(("EDA", "FAILED", str(e)))
        return
    
    # STEP 4: Operational Insights
    print_step_summary(4, "OPERATIONAL INSIGHTS", "IN PROGRESS")
    try:
        from scripts.operational_insights import (
            identify_peak_occupancy_periods, identify_low_occupancy_periods,
            identify_busiest_departments, identify_highest_admission_seasons,
            analyze_holiday_effects, analyze_staff_availability_effects,
            generate_operational_insights_report
        )
        
        peak_periods = identify_peak_occupancy_periods(df_cleaned)
        low_periods = identify_low_occupancy_periods(df_cleaned)
        dept_metrics = identify_busiest_departments(df_cleaned)
        seasonal_metrics = identify_highest_admission_seasons(df_cleaned)
        holiday_comparison = analyze_holiday_effects(df_cleaned)
        staff_analysis = analyze_staff_availability_effects(df_cleaned)
        generate_operational_insights_report(
            df_cleaned, peak_periods, low_periods, dept_metrics,
            seasonal_metrics, holiday_comparison, staff_analysis
        )
        
        print_step_summary(4, "OPERATIONAL INSIGHTS", "COMPLETED ✓")
        execution_log.append(("Operational Insights", "SUCCESS", "Generated 6 visualizations and report"))
    except Exception as e:
        print_step_summary(4, "OPERATIONAL INSIGHTS", f"FAILED ✗ - {str(e)}")
        execution_log.append(("Operational Insights", "FAILED", str(e)))
        return
    
    # STEP 5: Feature Engineering
    print_step_summary(5, "FEATURE ENGINEERING", "IN PROGRESS")
    try:
        from scripts.feature_engineering import (
            create_date_features, create_lag_features, create_rolling_features,
            create_ratio_features, create_interaction_features,
            remove_leakage_columns, handle_missing_values_in_features,
            generate_feature_engineering_report
        )
        from config import ENGINEERED_DATA_PATH, REPORTS_DIR
        
        df_fe = create_date_features(df_cleaned)
        df_fe = create_lag_features(df_fe)
        df_fe = create_rolling_features(df_fe)
        df_fe = create_ratio_features(df_fe)
        df_fe = create_interaction_features(df_fe)
        df_fe = remove_leakage_columns(df_fe)
        df_fe = handle_missing_values_in_features(df_fe)
        report_path = os.path.join(REPORTS_DIR, 'feature_engineering_report.txt')
        generate_feature_engineering_report(df_cleaned, df_fe, report_path=report_path)
        df_fe.to_csv(ENGINEERED_DATA_PATH, index=False)
        
        print_step_summary(5, "FEATURE ENGINEERING", "COMPLETED ✓")
        execution_log.append(("Feature Engineering", "SUCCESS", f"Created {df_fe.shape[1] - df_cleaned.shape[1]} new features"))
    except Exception as e:
        print_step_summary(5, "FEATURE ENGINEERING", f"FAILED ✗ - {str(e)}")
        execution_log.append(("Feature Engineering", "FAILED", str(e)))
        return
    
    # STEP 6: Dataset Preparation
    print_step_summary(6, "DATASET PREPARATION", "IN PROGRESS")
    try:
        from scripts.dataset_preparation import (
            handle_categorical_variables, remove_unnecessary_columns,
            create_feature_matrix_and_target, perform_chronological_split,
            save_processed_datasets, generate_dataset_preparation_report
        )
        import joblib
        import json
        from config import MODELS_DIR
        
        df_prep, label_encoders = handle_categorical_variables(df_fe)
        df_prep = remove_unnecessary_columns(df_prep)
        X, y = create_feature_matrix_and_target(df_prep, target_column='Occupancy_Rate')
        X_train, X_test, y_train, y_test, split_info = perform_chronological_split(X, y, test_size=0.2)
        save_processed_datasets(X_train, X_test, y_train, y_test)
        generate_dataset_preparation_report(df_fe, X_train, X_test, y_train, y_test, split_info)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
        with open(os.path.join(MODELS_DIR, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2)
        
        print_step_summary(6, "DATASET PREPARATION", "COMPLETED ✓")
        execution_log.append(("Dataset Preparation", "SUCCESS", "Created train/test splits"))
    except Exception as e:
        print_step_summary(6, "DATASET PREPARATION", f"FAILED ✗ - {str(e)}")
        execution_log.append(("Dataset Preparation", "FAILED", str(e)))
        return
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("DAY 1 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 Execution Log:")
    for i, (step, status, details) in enumerate(execution_log, 1):
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"   {i}. {step}: {status} {status_symbol}")
        print(f"      Details: {details}")
    
    print("\n📁 Deliverables Created:")
    print("   Data Files:")
    print("     - data/cleaned_hospital_bed_occupancy.csv")
    print("     - data/engineered_hospital_bed_occupancy.csv")
    print("     - data/processed/train_data.csv")
    print("     - data/processed/test_data.csv")
    
    print("\n   Reports:")
    print("     - reports/data_quality_report.txt")
    print("     - reports/eda_report.txt")
    print("     - reports/operational_insights_report.txt")
    print("     - reports/feature_engineering_report.txt")
    print("     - reports/dataset_preparation_report.txt")
    
    print("\n   Visualizations:")
    print("     - reports/figures/A_dataset_overview.png")
    print("     - reports/figures/B_occupancy_analysis.png")
    print("     - reports/figures/C_admissions_analysis.png")
    print("     - reports/figures/D_department_analysis.png")
    print("     - reports/figures/E_seasonal_analysis.png")
    print("     - reports/figures/F_time_based_analysis.png")
    print("     - reports/figures/G_hospital_operations_analysis.png")
    print("     - reports/figures/H_correlation_heatmap.png")
    print("     - reports/figures/I_peak_occupancy_periods.png")
    print("     - reports/figures/J_low_occupancy_periods.png")
    print("     - reports/figures/K_busiest_departments.png")
    print("     - reports/figures/L_highest_admission_seasons.png")
    print("     - reports/figures/M_holiday_effects.png")
    print("     - reports/figures/N_staff_availability_effects.png")
    print("     - reports/figures/correlation_matrix.csv")
    
    print("\n" + "=" * 80)
    print("🎉 DAY 1 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n✓ All Day 1 tasks completed successfully")
    print("✓ Dataset is ready for machine learning modeling (Day 2)")
    print("\nNext Steps:")
    print("  1. Review all generated reports and visualizations")
    print("  2. Analyze the insights and findings")
    print("  3. Proceed to Day 2: Machine Learning Model Development")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
