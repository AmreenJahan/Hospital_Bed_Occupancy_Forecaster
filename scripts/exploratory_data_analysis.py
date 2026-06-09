"""
Hospital Bed Occupancy Forecaster - Exploratory Data Analysis (EDA) Script
===========================================================================
This script performs comprehensive exploratory data analysis including:
- Dataset Overview
- Occupancy Analysis
- Admissions Analysis
- Department Analysis
- Seasonal Analysis
- Time-Based Analysis
- Hospital Operations Analysis
- Correlation Analysis

All visualizations are saved to reports/figures/

Author: Data Science Intern
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

try:
    from config import FIGURES_DIR, REPORTS_DIR, CLEANED_DATA_PATH, RAW_DATA_PATH
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import FIGURES_DIR, REPORTS_DIR, CLEANED_DATA_PATH, RAW_DATA_PATH

os.makedirs(FIGURES_DIR, exist_ok=True)


def dataset_overview(df):
    """
    A. Dataset Overview Analysis
    """
    print("\n" + "=" * 80)
    print("A. DATASET OVERVIEW")
    print("=" * 80)
    
    # Basic information
    print(f"\n📊 Dataset Overview:")
    print(f"   - Number of rows: {df.shape[0]:,}")
    print(f"   - Number of columns: {df.shape[1]}")
    
    # Feature types
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n🔢 Numerical Features ({len(numerical_features)}):")
    for i, feat in enumerate(numerical_features, 1):
        print(f"   {i:2d}. {feat}")
    
    print(f"\n📝 Categorical Features ({len(categorical_features)}):")
    for i, feat in enumerate(categorical_features, 1):
        print(f"   {i:2d}. {feat}")
    
    # Visualize feature distribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Numerical features count
    axes[0].barh(numerical_features, [1]*len(numerical_features), color='skyblue')
    axes[0].set_title('Numerical Features', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].set_xlim(0, 2)
    axes[0].set_yticks(range(len(numerical_features)))
    axes[0].set_yticklabels(numerical_features)
    
    # Categorical features count
    axes[1].barh(categorical_features, [1]*len(categorical_features), color='lightcoral')
    axes[1].set_title('Categorical Features', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].set_xlim(0, 2)
    axes[1].set_yticks(range(len(categorical_features)))
    axes[1].set_yticklabels(categorical_features)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/A_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Dataset overview visualization saved to: {FIGURES_DIR}/A_dataset_overview.png")


def occupancy_analysis(df):
    """
    B. Occupancy Analysis
    """
    print("\n" + "=" * 80)
    print("B. OCCUPANCY ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Occupancy Rate Distribution
    axes[0].hist(df['Occupancy_Rate'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['Occupancy_Rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Occupancy_Rate"].mean():.2f}%')
    axes[0].axvline(df['Occupancy_Rate'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Occupancy_Rate"].median():.2f}%')
    axes[0].set_xlabel('Occupancy Rate (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Occupancy Rate Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Beds Occupied Distribution
    axes[1].hist(df['Beds_Occupied'], bins=50, color='darkorange', edgecolor='black', alpha=0.7)
    axes[1].axvline(df['Beds_Occupied'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Beds_Occupied"].mean():.0f}')
    axes[1].axvline(df['Beds_Occupied'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Beds_Occupied"].median():.0f}')
    axes[1].set_xlabel('Beds Occupied', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Beds Occupied Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Total Beds Distribution
    axes[2].hist(df['Total_Hospital_Beds'], bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[2].axvline(df['Total_Hospital_Beds'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Total_Hospital_Beds"].mean():.0f}')
    axes[2].axvline(df['Total_Hospital_Beds'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Total_Hospital_Beds"].median():.0f}')
    axes[2].set_xlabel('Total Hospital Beds', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Total Hospital Beds Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/B_occupancy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Occupancy analysis visualization saved to: {FIGURES_DIR}/B_occupancy_analysis.png")
    
    # Print statistics
    print(f"\n📊 Occupancy Statistics:")
    print(f"   - Occupancy Rate: Mean={df['Occupancy_Rate'].mean():.2f}%, Median={df['Occupancy_Rate'].median():.2f}%, Std={df['Occupancy_Rate'].std():.2f}%")
    print(f"   - Beds Occupied: Mean={df['Beds_Occupied'].mean():.0f}, Median={df['Beds_Occupied'].median():.0f}, Std={df['Beds_Occupied'].std():.0f}")
    print(f"   - Total Beds: Mean={df['Total_Hospital_Beds'].mean():.0f}, Median={df['Total_Hospital_Beds'].median():.0f}, Std={df['Total_Hospital_Beds'].std():.0f}")


def admissions_analysis(df):
    """
    C. Admissions Analysis
    """
    print("\n" + "=" * 80)
    print("C. ADMISSIONS ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. Daily Admissions Distribution
    axes[0].hist(df['Daily_Admissions'], bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['Daily_Admissions'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Daily_Admissions"].mean():.1f}')
    axes[0].set_xlabel('Daily Admissions', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Daily Admissions Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Daily Discharges Distribution
    axes[1].hist(df['Daily_Discharges'], bins=50, color='crimson', edgecolor='black', alpha=0.7)
    axes[1].axvline(df['Daily_Discharges'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Daily_Discharges"].mean():.1f}')
    axes[1].set_xlabel('Daily Discharges', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Daily Discharges Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Emergency Admissions Analysis
    axes[2].hist(df['Emergency_Admissions'], bins=50, color='darkred', edgecolor='black', alpha=0.7)
    axes[2].axvline(df['Emergency_Admissions'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Emergency_Admissions"].mean():.1f}')
    axes[2].set_xlabel('Emergency Admissions', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Emergency Admissions Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. ICU Admissions Analysis
    axes[3].hist(df['ICU_Admissions'], bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[3].axvline(df['ICU_Admissions'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["ICU_Admissions"].mean():.1f}')
    axes[3].set_xlabel('ICU Admissions', fontsize=12)
    axes[3].set_ylabel('Frequency', fontsize=12)
    axes[3].set_title('ICU Admissions Distribution', fontsize=14, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/C_admissions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Admissions analysis visualization saved to: {FIGURES_DIR}/C_admissions_analysis.png")
    
    # Print statistics
    print(f"\n📊 Admissions Statistics:")
    print(f"   - Daily Admissions: Mean={df['Daily_Admissions'].mean():.1f}, Median={df['Daily_Admissions'].median():.1f}")
    print(f"   - Daily Discharges: Mean={df['Daily_Discharges'].mean():.1f}, Median={df['Daily_Discharges'].median():.1f}")
    print(f"   - Emergency Admissions: Mean={df['Emergency_Admissions'].mean():.1f}, Median={df['Emergency_Admissions'].median():.1f}")
    print(f"   - ICU Admissions: Mean={df['ICU_Admissions'].mean():.1f}, Median={df['ICU_Admissions'].median():.1f}")


def department_analysis(df):
    """
    D. Department Analysis
    """
    print("\n" + "=" * 80)
    print("D. DEPARTMENT ANALYSIS")
    print("=" * 80)
    
    if 'Department' not in df.columns:
        print("⚠️  Department column not found in dataset")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Department-wise Occupancy Rate
    dept_occupancy = df.groupby('Department')['Occupancy_Rate'].mean().sort_values(ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(dept_occupancy)))
    dept_occupancy.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_xlabel('Department', fontsize=12)
    axes[0].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
    axes[0].set_title('Department-wise Average Occupancy Rate', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Department-wise Admissions
    dept_admissions = df.groupby('Department')['Daily_Admissions'].mean().sort_values(ascending=False)
    colors = plt.cm.plasma(np.linspace(0, 1, len(dept_admissions)))
    dept_admissions.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
    axes[1].set_xlabel('Department', fontsize=12)
    axes[1].set_ylabel('Average Daily Admissions', fontsize=12)
    axes[1].set_title('Department-wise Average Daily Admissions', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Department-wise Bed Usage
    dept_beds = df.groupby('Department')['Beds_Occupied'].mean().sort_values(ascending=False)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(dept_beds)))
    dept_beds.plot(kind='bar', ax=axes[2], color=colors, edgecolor='black')
    axes[2].set_xlabel('Department', fontsize=12)
    axes[2].set_ylabel('Average Beds Occupied', fontsize=12)
    axes[2].set_title('Department-wise Average Bed Usage', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/D_department_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Department analysis visualization saved to: {FIGURES_DIR}/D_department_analysis.png")


def seasonal_analysis(df):
    """
    E. Seasonal Analysis
    """
    print("\n" + "=" * 80)
    print("E. SEASONAL ANALYSIS")
    print("=" * 80)
    
    if 'Season' not in df.columns:
        print("⚠️  Season column not found in dataset")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Occupancy Rate by Season
    season_occupancy = df.groupby('Season')['Occupancy_Rate'].mean()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_occupancy = season_occupancy.reindex(season_order)
    colors = ['lightgreen', 'orange', 'brown', 'lightblue']
    season_occupancy.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_xlabel('Season', fontsize=12)
    axes[0].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
    axes[0].set_title('Occupancy Rate by Season', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Admissions by Season
    season_admissions = df.groupby('Season')['Daily_Admissions'].mean()
    season_admissions = season_admissions.reindex(season_order)
    season_admissions.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
    axes[1].set_xlabel('Season', fontsize=12)
    axes[1].set_ylabel('Average Daily Admissions', fontsize=12)
    axes[1].set_title('Admissions by Season', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Discharges by Season
    season_discharges = df.groupby('Season')['Daily_Discharges'].mean()
    season_discharges = season_discharges.reindex(season_order)
    season_discharges.plot(kind='bar', ax=axes[2], color=colors, edgecolor='black')
    axes[2].set_xlabel('Season', fontsize=12)
    axes[2].set_ylabel('Average Daily Discharges', fontsize=12)
    axes[2].set_title('Discharges by Season', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E_seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Seasonal analysis visualization saved to: {FIGURES_DIR}/E_seasonal_analysis.png")


def time_based_analysis(df):
    """
    F. Time-Based Analysis
    """
    print("\n" + "=" * 80)
    print("F. TIME-BASED ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. Occupancy by Month
    if 'Month' in df.columns:
        month_occupancy = df.groupby('Month')['Occupancy_Rate'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_occupancy.index = month_names
        month_occupancy.plot(kind='line', ax=axes[0], marker='o', color='steelblue', linewidth=2, markersize=8)
        axes[0].set_xlabel('Month', fontsize=12)
        axes[0].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
        axes[0].set_title('Occupancy Rate by Month', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
    
    # 2. Occupancy by Day of Week
    if 'Day_of_Week' in df.columns:
        day_occupancy = df.groupby('Day_of_Week')['Occupancy_Rate'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_occupancy.index = day_names
        day_occupancy.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
        axes[1].set_title('Occupancy Rate by Day of Week', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Monthly Admission Trends
    if 'Month' in df.columns:
        month_admissions = df.groupby('Month')['Daily_Admissions'].mean()
        month_admissions.index = month_names
        month_admissions.plot(kind='line', ax=axes[2], marker='o', color='darkgreen', linewidth=2, markersize=8)
        axes[2].set_xlabel('Month', fontsize=12)
        axes[2].set_ylabel('Average Daily Admissions', fontsize=12)
        axes[2].set_title('Monthly Admission Trends', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
    
    # 4. Monthly Discharge Trends
    if 'Month' in df.columns:
        month_discharges = df.groupby('Month')['Daily_Discharges'].mean()
        month_discharges.index = month_names
        month_discharges.plot(kind='line', ax=axes[3], marker='o', color='purple', linewidth=2, markersize=8)
        axes[3].set_xlabel('Month', fontsize=12)
        axes[3].set_ylabel('Average Daily Discharges', fontsize=12)
        axes[3].set_title('Monthly Discharge Trends', fontsize=14, fontweight='bold')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F_time_based_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Time-based analysis visualization saved to: {FIGURES_DIR}/F_time_based_analysis.png")


def hospital_operations_analysis(df):
    """
    G. Hospital Operations Analysis
    """
    print("\n" + "=" * 80)
    print("G. HOSPITAL OPERATIONS ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. Staff Availability vs Occupancy
    if 'Staff_Availability' in df.columns:
        axes[0].scatter(df['Staff_Availability'], df['Occupancy_Rate'], alpha=0.5, color='steelblue', s=20)
        axes[0].set_xlabel('Staff Availability', fontsize=12)
        axes[0].set_ylabel('Occupancy Rate (%)', fontsize=12)
        axes[0].set_title('Staff Availability vs Occupancy Rate', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Staff_Availability'], df['Occupancy_Rate'], 1)
        p = np.poly1d(z)
        axes[0].plot(df['Staff_Availability'], p(df['Staff_Availability']), "r--", alpha=0.8, linewidth=2)
    
    # 2. Average Length of Stay vs Occupancy
    if 'Avg_Length_of_Stay' in df.columns:
        axes[1].scatter(df['Avg_Length_of_Stay'], df['Occupancy_Rate'], alpha=0.5, color='darkgreen', s=20)
        axes[1].set_xlabel('Average Length of Stay', fontsize=12)
        axes[1].set_ylabel('Occupancy Rate (%)', fontsize=12)
        axes[1].set_title('Average Length of Stay vs Occupancy Rate', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Avg_Length_of_Stay'], df['Occupancy_Rate'], 1)
        p = np.poly1d(z)
        axes[1].plot(df['Avg_Length_of_Stay'], p(df['Avg_Length_of_Stay']), "r--", alpha=0.8, linewidth=2)
    
    # 3. Holiday Impact Analysis
    if 'Public_Holiday' in df.columns:
        holiday_occupancy = df.groupby('Public_Holiday')['Occupancy_Rate'].mean()
        holiday_labels = ['Non-Holiday', 'Holiday']
        holiday_occupancy.index = holiday_labels
        colors = ['lightblue', 'red']
        holiday_occupancy.plot(kind='bar', ax=axes[2], color=colors, edgecolor='black')
        axes[2].set_xlabel('Public Holiday', fontsize=12)
        axes[2].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
        axes[2].set_title('Holiday Impact on Occupancy Rate', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=0)
        axes[2].grid(True, alpha=0.3, axis='y')
    
    # 4. Special Event Impact Analysis
    if 'Special_Event' in df.columns:
        event_occupancy = df.groupby('Special_Event')['Occupancy_Rate'].mean()
        event_labels = ['No Event', 'Special Event']
        event_occupancy.index = event_labels
        colors = ['lightgray', 'orange']
        event_occupancy.plot(kind='bar', ax=axes[3], color=colors, edgecolor='black')
        axes[3].set_xlabel('Special Event', fontsize=12)
        axes[3].set_ylabel('Average Occupancy Rate (%)', fontsize=12)
        axes[3].set_title('Special Event Impact on Occupancy Rate', fontsize=14, fontweight='bold')
        axes[3].tick_params(axis='x', rotation=0)
        axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/G_hospital_operations_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Hospital operations analysis visualization saved to: {FIGURES_DIR}/G_hospital_operations_analysis.png")


def correlation_analysis(df):
    """
    H. Correlation Analysis
    """
    print("\n" + "=" * 80)
    print("H. CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot_kws={'size': 8}, ax=ax)
    
    ax.set_title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{FIGURES_DIR}/H_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Correlation heatmap saved to: {FIGURES_DIR}/H_correlation_heatmap.png")
    
    # Identify important relationships
    print(f"\n📊 Important Correlations with Occupancy Rate:")
    if 'Occupancy_Rate' in correlation_matrix.columns:
        occupancy_corr = correlation_matrix['Occupancy_Rate'].sort_values(ascending=False)
        for feature, corr_value in occupancy_corr.items():
            if feature != 'Occupancy_Rate' and abs(corr_value) > 0.1:
                print(f"   - {feature}: {corr_value:.3f}")
    
    # Save correlation matrix to CSV
    correlation_matrix.to_csv(f'{FIGURES_DIR}/correlation_matrix.csv')
    print(f"\n✓ Correlation matrix saved to: {FIGURES_DIR}/correlation_matrix.csv")


def generate_eda_report(df):
    """
    Generate comprehensive EDA report
    """
    print("\n" + "=" * 80)
    print("GENERATING EDA REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HOSPITAL BED OCCUPANCY FORECASTER - EXPLORATORY DATA ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset Overview
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Records: {df.shape[0]:,}")
    report_lines.append(f"Total Features: {df.shape[1]}")
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    report_lines.append(f"Numerical Features: {len(numerical_features)}")
    report_lines.append(f"Categorical Features: {len(categorical_features)}")
    report_lines.append("")
    
    # Key Statistics
    report_lines.append("2. KEY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Occupancy Rate:")
    report_lines.append(f"  - Mean: {df['Occupancy_Rate'].mean():.2f}%")
    report_lines.append(f"  - Median: {df['Occupancy_Rate'].median():.2f}%")
    report_lines.append(f"  - Std Dev: {df['Occupancy_Rate'].std():.2f}%")
    report_lines.append(f"  - Min: {df['Occupancy_Rate'].min():.2f}%")
    report_lines.append(f"  - Max: {df['Occupancy_Rate'].max():.2f}%")
    report_lines.append("")
    
    report_lines.append(f"Beds Occupied:")
    report_lines.append(f"  - Mean: {df['Beds_Occupied'].mean():.0f}")
    report_lines.append(f"  - Median: {df['Beds_Occupied'].median():.0f}")
    report_lines.append(f"  - Std Dev: {df['Beds_Occupied'].std():.0f}")
    report_lines.append("")
    
    report_lines.append(f"Daily Admissions:")
    report_lines.append(f"  - Mean: {df['Daily_Admissions'].mean():.1f}")
    report_lines.append(f"  - Median: {df['Daily_Admissions'].median():.1f}")
    report_lines.append("")
    
    # Department Analysis
    if 'Department' in df.columns:
        report_lines.append("3. DEPARTMENT ANALYSIS")
        report_lines.append("-" * 80)
        dept_occupancy = df.groupby('Department')['Occupancy_Rate'].mean().sort_values(ascending=False)
        for dept, occ in dept_occupancy.head(5).items():
            report_lines.append(f"  - {dept}: {occ:.2f}%")
        report_lines.append("")
    
    # Seasonal Analysis
    if 'Season' in df.columns:
        report_lines.append("4. SEASONAL ANALYSIS")
        report_lines.append("-" * 80)
        season_occupancy = df.groupby('Season')['Occupancy_Rate'].mean()
        for season, occ in season_occupancy.items():
            report_lines.append(f"  - {season}: {occ:.2f}%")
        report_lines.append("")
    
    # Visualizations Generated
    report_lines.append("5. VISUALIZATIONS GENERATED")
    report_lines.append("-" * 80)
    report_lines.append("  - A_dataset_overview.png")
    report_lines.append("  - B_occupancy_analysis.png")
    report_lines.append("  - C_admissions_analysis.png")
    report_lines.append("  - D_department_analysis.png")
    report_lines.append("  - E_seasonal_analysis.png")
    report_lines.append("  - F_time_based_analysis.png")
    report_lines.append("  - G_hospital_operations_analysis.png")
    report_lines.append("  - H_correlation_heatmap.png")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF EDA REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(REPORTS_DIR, 'eda_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ EDA report saved to: {report_path}")


def main():
    """
    Main function to execute complete EDA pipeline
    """
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load cleaned data
    try:
        df = pd.read_csv('../data/cleaned_hospital_bed_occupancy.csv')
        print("\n✓ Loaded cleaned dataset")
    except FileNotFoundError:
        # If cleaned data doesn't exist, load raw data
        df = pd.read_csv('../data/hospital_bed_occupancy_10000.csv')
        print("\n✓ Loaded raw dataset (cleaned data not found)")
    
    # Convert Date column if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Execute all EDA analyses
    dataset_overview(df)
    occupancy_analysis(df)
    admissions_analysis(df)
    department_analysis(df)
    seasonal_analysis(df)
    time_based_analysis(df)
    hospital_operations_analysis(df)
    correlation_analysis(df)
    
    # Generate comprehensive EDA report
    generate_eda_report(df)
    
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n✓ All visualizations saved to: {FIGURES_DIR}")
    print("✓ EDA report generated")
    
    return df


if __name__ == "__main__":
    df = main()
