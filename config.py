"""Project-wide configuration and paths."""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'hospital_bed_occupancy_10000.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'cleaned_hospital_bed_occupancy.csv')
ENGINEERED_DATA_PATH = os.path.join(DATA_DIR, 'engineered_hospital_bed_occupancy.csv')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'test_data.csv')

BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'best_model.pkl')
FORECAST_OUTPUT_PATH = os.path.join(REPORTS_DIR, 'forecast_7_day.csv')
OVERFLOW_ALERTS_PATH = os.path.join(REPORTS_DIR, 'overflow_alerts.csv')

TARGET_COLUMN = 'Occupancy_Rate'

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

OVERFLOW_THRESHOLDS = {
    'GREEN': (0, 70),
    'YELLOW': (70, 85),
    'ORANGE': (85, 95),
    'RED': (95, float('inf')),
}
