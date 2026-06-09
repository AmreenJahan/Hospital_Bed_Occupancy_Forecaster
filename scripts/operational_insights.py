"""
Hospital Bed Occupancy Forecaster - Operational Insights Script
==============================================================
This script generates operational insights including:
- Peak Occupancy Periods
- Low Occupancy Periods
- Busiest Departments
- Highest Admission Seasons
- Holiday Effects
- Staff Availability Effects

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


def identify_peak_occupancy_periods(df):
    """
    Identify peak occupancy periods in the dataset.
    """
    print("\n" + "=" * 80)
    print("PEAK OCCUPANCY PERIODS ANALYSIS")
    print("=" * 80)
    
    # Define high occupancy threshold (top 20%)
    high_occupancy_threshold = df['Occupancy_Rate'].quantile(0.80)
    
    # Filter high occupancy periods
    high_occupancy_periods = df[df['Occupancy_Rate'] >= high_occupancy_threshold].copy()
    
    print(f"\n📊 Peak Occupancy Threshold: {high_occupancy_threshold:.2f}%")
    print(f"   - Number of high occupancy days: {len(high_occupancy_periods):,}")
    print(f"   - Percentage of total days: {(len(high_occupancy_periods)/len(df))*100:.1f}%")
    
    # Analyze by month
    if 'Month' in df.columns:
        monthly_peak = high_occupancy_periods.groupby('Month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_peak.index = month_names
        
        print(f"\n📅 Peak Occupancy by Month:")
        for month, count in monthly_peak.sort_values(ascending=False).items():
            print(f"   - {month}: {count} high occupancy days")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_peak.plot(kind='bar', color='darkred', edgecolor='black', ax=ax)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of High Occupancy Days', fontsize=12)
        ax.set_title('Peak Occupancy Periods by Month', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/I_peak_occupancy_periods.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Peak occupancy visualization saved to: {FIGURES_DIR}/I_peak_occupancy_periods.png")
    
    return high_occupancy_periods


def identify_low_occupancy_periods(df):
    """
    Identify low occupancy periods in the dataset.
    """
    print("\n" + "=" * 80)
    print("LOW OCCUPANCY PERIODS ANALYSIS")
    print("=" * 80)
    
    # Define low occupancy threshold (bottom 20%)
    low_occupancy_threshold = df['Occupancy_Rate'].quantile(0.20)
    
    # Filter low occupancy periods
    low_occupancy_periods = df[df['Occupancy_Rate'] <= low_occupancy_threshold].copy()
    
    print(f"\n📊 Low Occupancy Threshold: {low_occupancy_threshold:.2f}%")
    print(f"   - Number of low occupancy days: {len(low_occupancy_periods):,}")
    print(f"   - Percentage of total days: {(len(low_occupancy_periods)/len(df))*100:.1f}%")
    
    # Analyze by month
    if 'Month' in df.columns:
        monthly_low = low_occupancy_periods.groupby('Month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_low.index = month_names
        
        print(f"\n📅 Low Occupancy by Month:")
        for month, count in monthly_low.sort_values(ascending=False).items():
            print(f"   - {month}: {count} low occupancy days")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_low.plot(kind='bar', color='darkgreen', edgecolor='black', ax=ax)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Low Occupancy Days', fontsize=12)
        ax.set_title('Low Occupancy Periods by Month', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/J_low_occupancy_periods.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Low occupancy visualization saved to: {FIGURES_DIR}/J_low_occupancy_periods.png")
    
    return low_occupancy_periods


def identify_busiest_departments(df):
    """
    Identify busiest departments based on multiple metrics.
    """
    print("\n" + "=" * 80)
    print("BUSIEST DEPARTMENTS ANALYSIS")
    print("=" * 80)
    
    if 'Department' not in df.columns:
        print("⚠️  Department column not found in dataset")
        return None
    
    # Calculate department metrics
    dept_metrics = df.groupby('Department').agg({
        'Occupancy_Rate': ['mean', 'max', 'std'],
        'Daily_Admissions': ['mean', 'sum'],
        'Beds_Occupied': ['mean', 'max']
    }).round(2)
    
    dept_metrics.columns = ['Avg_Occupancy', 'Max_Occupancy', 'Std_Occupancy',
                            'Avg_Admissions', 'Total_Admissions',
                            'Avg_Beds', 'Max_Beds']
    
    # Sort by average occupancy
    dept_metrics = dept_metrics.sort_values('Avg_Occupancy', ascending=False)
    
    print(f"\n📊 Department Performance Metrics:")
    print(dept_metrics.to_string())
    
    # Identify top 3 busiest departments
    top_3_depts = dept_metrics.head(3)
    print(f"\n🏆 Top 3 Busiest Departments (by Average Occupancy):")
    for i, (dept, row) in enumerate(top_3_depts.iterrows(), 1):
        print(f"   {i}. {dept}: {row['Avg_Occupancy']:.2f}% avg occupancy")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average Occupancy by Department
    dept_metrics['Avg_Occupancy'].plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Average Occupancy Rate by Department', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Occupancy Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Total Admissions by Department
    dept_metrics['Total_Admissions'].plot(kind='bar', ax=axes[0, 1], color='darkorange', edgecolor='black')
    axes[0, 1].set_title('Total Admissions by Department', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Total Admissions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Average Beds Occupied by Department
    dept_metrics['Avg_Beds'].plot(kind='bar', ax=axes[1, 0], color='forestgreen', edgecolor='black')
    axes[1, 0].set_title('Average Beds Occupied by Department', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Beds Occupied')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Occupancy Variability by Department
    dept_metrics['Std_Occupancy'].plot(kind='bar', ax=axes[1, 1], color='purple', edgecolor='black')
    axes[1, 1].set_title('Occupancy Variability (Std Dev) by Department', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/K_busiest_departments.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Busiest departments visualization saved to: {FIGURES_DIR}/K_busiest_departments.png")
    
    return dept_metrics


def identify_highest_admission_seasons(df):
    """
    Identify seasons with highest admissions.
    """
    print("\n" + "=" * 80)
    print("HIGHEST ADMISSION SEASONS ANALYSIS")
    print("=" * 80)
    
    if 'Season' not in df.columns:
        print("⚠️  Season column not found in dataset")
        return None
    
    # Calculate seasonal metrics
    seasonal_metrics = df.groupby('Season').agg({
        'Daily_Admissions': ['mean', 'sum', 'max'],
        'Emergency_Admissions': ['mean', 'sum'],
        'ICU_Admissions': ['mean', 'sum'],
        'Occupancy_Rate': 'mean'
    }).round(2)
    
    seasonal_metrics.columns = ['Avg_Admissions', 'Total_Admissions', 'Max_Admissions',
                                'Avg_Emergency', 'Total_Emergency',
                                'Avg_ICU', 'Total_ICU',
                                'Avg_Occupancy']
    
    # Reorder seasons
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_metrics = seasonal_metrics.reindex(season_order)
    
    print(f"\n📊 Seasonal Admission Metrics:")
    print(seasonal_metrics.to_string())
    
    # Identify peak admission season
    peak_season = seasonal_metrics['Avg_Admissions'].idxmax()
    print(f"\n🌟 Peak Admission Season: {peak_season}")
    print(f"   - Average daily admissions: {seasonal_metrics.loc[peak_season, 'Avg_Admissions']:.2f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average Admissions by Season
    colors = ['lightgreen', 'orange', 'brown', 'lightblue']
    seasonal_metrics['Avg_Admissions'].plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
    axes[0, 0].set_title('Average Daily Admissions by Season', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Daily Admissions')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Total Admissions by Season
    seasonal_metrics['Total_Admissions'].plot(kind='bar', ax=axes[0, 1], color=colors, edgecolor='black')
    axes[0, 1].set_title('Total Admissions by Season', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Total Admissions')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Emergency Admissions by Season
    seasonal_metrics['Avg_Emergency'].plot(kind='bar', ax=axes[1, 0], color=colors, edgecolor='black')
    axes[1, 0].set_title('Average Emergency Admissions by Season', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Emergency Admissions')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. ICU Admissions by Season
    seasonal_metrics['Avg_ICU'].plot(kind='bar', ax=axes[1, 1], color=colors, edgecolor='black')
    axes[1, 1].set_title('Average ICU Admissions by Season', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Average ICU Admissions')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/L_highest_admission_seasons.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Seasonal admissions visualization saved to: {FIGURES_DIR}/L_highest_admission_seasons.png")
    
    return seasonal_metrics


def analyze_holiday_effects(df):
    """
    Analyze the impact of holidays on hospital operations.
    """
    print("\n" + "=" * 80)
    print("HOLIDAY EFFECTS ANALYSIS")
    print("=" * 80)
    
    if 'Public_Holiday' not in df.columns:
        print("⚠️  Public_Holiday column not found in dataset")
        return None
    
    # Compare holiday vs non-holiday periods
    holiday_comparison = df.groupby('Public_Holiday').agg({
        'Occupancy_Rate': ['mean', 'median', 'std'],
        'Daily_Admissions': ['mean', 'median'],
        'Emergency_Admissions': ['mean', 'median'],
        'Daily_Discharges': ['mean', 'median']
    }).round(2)
    
    holiday_comparison.columns = ['Avg_Occupancy', 'Median_Occupancy', 'Std_Occupancy',
                                   'Avg_Admissions', 'Median_Admissions',
                                   'Avg_Emergency', 'Median_Emergency',
                                   'Avg_Discharges', 'Median_Discharges']
    
    holiday_comparison.index = ['Non-Holiday', 'Holiday']
    
    print(f"\n📊 Holiday vs Non-Holiday Comparison:")
    print(holiday_comparison.to_string())
    
    # Calculate percentage differences
    if len(holiday_comparison) == 2:
        holiday_metrics = holiday_comparison.loc['Holiday']
        non_holiday_metrics = holiday_comparison.loc['Non-Holiday']
        
        print(f"\n📈 Holiday Impact Analysis:")
        for col in holiday_metrics.index:
            if non_holiday_metrics[col] != 0:
                pct_change = ((holiday_metrics[col] - non_holiday_metrics[col]) / non_holiday_metrics[col]) * 100
                direction = "↑" if pct_change > 0 else "↓"
                print(f"   - {col}: {direction} {abs(pct_change):.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Occupancy Rate
    holiday_comparison['Avg_Occupancy'].plot(kind='bar', ax=axes[0, 0], color=['lightblue', 'red'], edgecolor='black')
    axes[0, 0].set_title('Average Occupancy Rate: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Occupancy Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Daily Admissions
    holiday_comparison['Avg_Admissions'].plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'red'], edgecolor='black')
    axes[0, 1].set_title('Average Daily Admissions: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Daily Admissions')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Emergency Admissions
    holiday_comparison['Avg_Emergency'].plot(kind='bar', ax=axes[1, 0], color=['lightblue', 'red'], edgecolor='black')
    axes[1, 0].set_title('Average Emergency Admissions: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Emergency Admissions')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Daily Discharges
    holiday_comparison['Avg_Discharges'].plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'red'], edgecolor='black')
    axes[1, 1].set_title('Average Daily Discharges: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Average Daily Discharges')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/M_holiday_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Holiday effects visualization saved to: {FIGURES_DIR}/M_holiday_effects.png")
    
    return holiday_comparison


def analyze_staff_availability_effects(df):
    """
    Analyze the impact of staff availability on hospital operations.
    """
    print("\n" + "=" * 80)
    print("STAFF AVAILABILITY EFFECTS ANALYSIS")
    print("=" * 80)
    
    if 'Staff_Availability' not in df.columns:
        print("⚠️  Staff_Availability column not found in dataset")
        return None
    
    # Categorize staff availability into Low, Medium, High
    df_staff = df.copy()
    df_staff['Staff_Category'] = pd.cut(
        df_staff['Staff_Availability'],
        bins=[0, df_staff['Staff_Availability'].quantile(0.33), 
              df_staff['Staff_Availability'].quantile(0.67), df_staff['Staff_Availability'].max()],
        labels=['Low', 'Medium', 'High']
    )
    
    # Analyze by staff category
    staff_analysis = df_staff.groupby('Staff_Category').agg({
        'Occupancy_Rate': ['mean', 'median', 'std'],
        'Daily_Admissions': ['mean', 'median'],
        'Emergency_Admissions': ['mean'],
        'Avg_Length_of_Stay': ['mean']
    }).round(2)
    
    staff_analysis.columns = ['Avg_Occupancy', 'Median_Occupancy', 'Std_Occupancy',
                             'Avg_Admissions', 'Median_Admissions',
                             'Avg_Emergency', 'Avg_LOS']
    
    print(f"\n📊 Staff Availability Analysis:")
    print(staff_analysis.to_string())
    
    # Calculate correlation
    correlation = df['Staff_Availability'].corr(df['Occupancy_Rate'])
    print(f"\n🔗 Correlation between Staff Availability and Occupancy Rate: {correlation:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Occupancy Rate by Staff Category
    staff_analysis['Avg_Occupancy'].plot(kind='bar', ax=axes[0, 0], color=['red', 'orange', 'green'], edgecolor='black')
    axes[0, 0].set_title('Average Occupancy Rate by Staff Availability', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Occupancy Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Admissions by Staff Category
    staff_analysis['Avg_Admissions'].plot(kind='bar', ax=axes[0, 1], color=['red', 'orange', 'green'], edgecolor='black')
    axes[0, 1].set_title('Average Daily Admissions by Staff Availability', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Daily Admissions')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Emergency Admissions by Staff Category
    staff_analysis['Avg_Emergency'].plot(kind='bar', ax=axes[1, 0], color=['red', 'orange', 'green'], edgecolor='black')
    axes[1, 0].set_title('Average Emergency Admissions by Staff Availability', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Emergency Admissions')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter plot: Staff Availability vs Occupancy
    axes[1, 1].scatter(df['Staff_Availability'], df['Occupancy_Rate'], alpha=0.5, color='steelblue', s=20)
    axes[1, 1].set_xlabel('Staff Availability', fontsize=12)
    axes[1, 1].set_ylabel('Occupancy Rate (%)', fontsize=12)
    axes[1, 1].set_title('Staff Availability vs Occupancy Rate', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['Staff_Availability'], df['Occupancy_Rate'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df['Staff_Availability'], p(df['Staff_Availability']), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/N_staff_availability_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Staff availability effects visualization saved to: {FIGURES_DIR}/N_staff_availability_effects.png")
    
    return staff_analysis


def generate_operational_insights_report(df, peak_periods, low_periods, dept_metrics, 
                                         seasonal_metrics, holiday_comparison, staff_analysis):
    """
    Generate comprehensive operational insights report.
    """
    print("\n" + "=" * 80)
    print("GENERATING OPERATIONAL INSIGHTS REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HOSPITAL BED OCCUPANCY FORECASTER - OPERATIONAL INSIGHTS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Peak Occupancy Periods
    report_lines.append("1. PEAK OCCUPANCY PERIODS")
    report_lines.append("-" * 80)
    high_threshold = df['Occupancy_Rate'].quantile(0.80)
    report_lines.append(f"Peak Occupancy Threshold: {high_threshold:.2f}%")
    report_lines.append(f"Number of high occupancy days: {len(peak_periods):,}")
    if 'Month' in df.columns:
        monthly_peak = peak_periods.groupby('Month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_peak.index = month_names
        peak_month = monthly_peak.idxmax()
        report_lines.append(f"Peak month: {peak_month} ({monthly_peak[peak_month]} days)")
    report_lines.append("")
    
    # Low Occupancy Periods
    report_lines.append("2. LOW OCCUPANCY PERIODS")
    report_lines.append("-" * 80)
    low_threshold = df['Occupancy_Rate'].quantile(0.20)
    report_lines.append(f"Low Occupancy Threshold: {low_threshold:.2f}%")
    report_lines.append(f"Number of low occupancy days: {len(low_periods):,}")
    if 'Month' in df.columns:
        monthly_low = low_periods.groupby('Month').size()
        monthly_low.index = month_names
        low_month = monthly_low.idxmax()
        report_lines.append(f"Lowest month: {low_month} ({monthly_low[low_month]} days)")
    report_lines.append("")
    
    # Busiest Departments
    if dept_metrics is not None:
        report_lines.append("3. BUSIEST DEPARTMENTS")
        report_lines.append("-" * 80)
        top_3 = dept_metrics.head(3)
        for i, (dept, row) in enumerate(top_3.iterrows(), 1):
            report_lines.append(f"{i}. {dept}:")
            report_lines.append(f"   - Average Occupancy: {row['Avg_Occupancy']:.2f}%")
            report_lines.append(f"   - Total Admissions: {row['Total_Admissions']:,.0f}")
            report_lines.append(f"   - Average Beds: {row['Avg_Beds']:.0f}")
        report_lines.append("")
    
    # Highest Admission Seasons
    if seasonal_metrics is not None:
        report_lines.append("4. HIGHEST ADMISSION SEASONS")
        report_lines.append("-" * 80)
        peak_season = seasonal_metrics['Avg_Admissions'].idxmax()
        report_lines.append(f"Peak Admission Season: {peak_season}")
        report_lines.append(f"Average Daily Admissions: {seasonal_metrics.loc[peak_season, 'Avg_Admissions']:.2f}")
        report_lines.append(f"Total Admissions: {seasonal_metrics.loc[peak_season, 'Total_Admissions']:,.0f}")
        report_lines.append("")
    
    # Holiday Effects
    if holiday_comparison is not None:
        report_lines.append("5. HOLIDAY EFFECTS")
        report_lines.append("-" * 80)
        holiday_occ = holiday_comparison.loc['Holiday', 'Avg_Occupancy']
        non_holiday_occ = holiday_comparison.loc['Non-Holiday', 'Avg_Occupancy']
        pct_change = ((holiday_occ - non_holiday_occ) / non_holiday_occ) * 100
        direction = "increase" if pct_change > 0 else "decrease"
        report_lines.append(f"Holiday occupancy {direction}: {abs(pct_change):.1f}%")
        report_lines.append(f"Holiday average occupancy: {holiday_occ:.2f}%")
        report_lines.append(f"Non-holiday average occupancy: {non_holiday_occ:.2f}%")
        report_lines.append("")
    
    # Staff Availability Effects
    if staff_analysis is not None:
        report_lines.append("6. STAFF AVAILABILITY EFFECTS")
        report_lines.append("-" * 80)
        high_staff_occ = staff_analysis.loc['High', 'Avg_Occupancy']
        low_staff_occ = staff_analysis.loc['Low', 'Avg_Occupancy']
        report_lines.append(f"High staff availability occupancy: {high_staff_occ:.2f}%")
        report_lines.append(f"Low staff availability occupancy: {low_staff_occ:.2f}%")
        correlation = df['Staff_Availability'].corr(df['Occupancy_Rate'])
        report_lines.append(f"Correlation with occupancy: {correlation:.3f}")
        report_lines.append("")
    
    # Key Recommendations
    report_lines.append("7. KEY OPERATIONAL RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("• Monitor occupancy closely during peak months identified")
    report_lines.append("• Consider resource reallocation during low occupancy periods")
    report_lines.append("• Focus staffing strategies on busiest departments")
    report_lines.append("• Prepare for increased admissions during peak seasons")
    report_lines.append("• Implement special protocols for holiday periods")
    report_lines.append("• Optimize staff scheduling based on availability patterns")
    report_lines.append("")
    
    # Visualizations Generated
    report_lines.append("8. VISUALIZATIONS GENERATED")
    report_lines.append("-" * 80)
    report_lines.append("  - I_peak_occupancy_periods.png")
    report_lines.append("  - J_low_occupancy_periods.png")
    report_lines.append("  - K_busiest_departments.png")
    report_lines.append("  - L_highest_admission_seasons.png")
    report_lines.append("  - M_holiday_effects.png")
    report_lines.append("  - N_staff_availability_effects.png")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF OPERATIONAL INSIGHTS REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(REPORTS_DIR, 'operational_insights_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Operational insights report saved to: {report_path}")


def main():
    """
    Main function to execute operational insights pipeline.
    """
    print("=" * 80)
    print("HOSPITAL BED OCCUPANCY FORECASTER - OPERATIONAL INSIGHTS")
    print("=" * 80)
    
    # Load cleaned data
    try:
        df = pd.read_csv('../data/cleaned_hospital_bed_occupancy.csv')
        print("\n✓ Loaded cleaned dataset")
    except FileNotFoundError:
        df = pd.read_csv('../data/hospital_bed_occupancy_10000.csv')
        print("\n✓ Loaded raw dataset (cleaned data not found)")
    
    # Convert Date column if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Execute all operational insights analyses
    peak_periods = identify_peak_occupancy_periods(df)
    low_periods = identify_low_occupancy_periods(df)
    dept_metrics = identify_busiest_departments(df)
    seasonal_metrics = identify_highest_admission_seasons(df)
    holiday_comparison = analyze_holiday_effects(df)
    staff_analysis = analyze_staff_availability_effects(df)
    
    # Generate comprehensive report
    generate_operational_insights_report(
        df, peak_periods, low_periods, dept_metrics,
        seasonal_metrics, holiday_comparison, staff_analysis
    )
    
    print("\n" + "=" * 80)
    print("OPERATIONAL INSIGHTS ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n✓ All visualizations saved to: {FIGURES_DIR}")
    print("✓ Operational insights report generated")
    
    return df


if __name__ == "__main__":
    df = main()
