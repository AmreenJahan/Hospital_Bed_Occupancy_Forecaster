"""
7-day hospital bed occupancy forecasting system.
"""

import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from config import (
    BEST_MODEL_PATH, ENGINEERED_DATA_PATH, FORECAST_OUTPUT_PATH,
    MODELS_DIR, REPORTS_DIR,
)

os.makedirs(REPORTS_DIR, exist_ok=True)


def load_model_and_features():
    """Load trained model, feature names, and label encoders."""
    model = joblib.load(BEST_MODEL_PATH)
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
    return model, feature_names, encoders


def _encode_row(row, encoders):
    """Apply label encoders to categorical fields."""
    for col, le in encoders.items():
        val = str(row.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        row[f'{col}_Encoded'] = int(le.transform([val])[0])
    return row


def _prepare_base_data():
    """Load latest engineered data per department."""
    df = pd.read_csv(ENGINEERED_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['Department', 'Date'])


def generate_forecast(days=7, departments=None):
    """
    Generate multi-day occupancy forecasts per department.

    Uses the latest row per department and iteratively updates lag features.
    """
    model, feature_names, encoders = load_model_and_features()
    df = _prepare_base_data()

    if departments is None:
        departments = sorted(df['Department'].unique())

    forecasts = []
    for dept in departments:
        dept_df = df[df['Department'] == dept].copy()
        if dept_df.empty:
            continue

        latest = dept_df.iloc[-1].copy()
        total_beds = int(latest.get('Total_Hospital_Beds', 300))
        occupancy_history = dept_df['Occupancy_Rate'].tail(30).tolist()

        for day_offset in range(1, days + 1):
            forecast_date = latest['Date'] + timedelta(days=day_offset)
            row = latest.copy()
            row['Date'] = forecast_date
            row['Year'] = forecast_date.year
            row['Quarter'] = forecast_date.quarter
            row['Month_Num'] = forecast_date.month
            row['Month'] = forecast_date.month
            row['Week_Number'] = forecast_date.isocalendar().week
            row['Day'] = forecast_date.day
            row['Day_of_Week_Num'] = forecast_date.dayofweek
            row['Day_of_Week'] = forecast_date.day_name()
            row['Day_of_Year'] = forecast_date.timetuple().tm_yday
            row['Is_Weekend'] = int(forecast_date.dayofweek >= 5)

            if occupancy_history:
                row['Lag_1_Day_Occupancy'] = occupancy_history[-1]
            if len(occupancy_history) >= 7:
                row['Lag_7_Day_Occupancy'] = occupancy_history[-7]
            if len(occupancy_history) >= 30:
                row['Lag_30_Day_Occupancy'] = occupancy_history[-30]

            recent7 = occupancy_history[-7:] if occupancy_history else [80.0]
            recent30 = occupancy_history[-30:] if occupancy_history else recent7
            row['Rolling_7_Day_Occupancy_Mean'] = float(np.mean(recent7))
            row['Rolling_7_Day_Occupancy_Std'] = float(np.std(recent7)) if len(recent7) > 1 else 0.0
            row['Rolling_30_Day_Occupancy_Mean'] = float(np.mean(recent30))
            row['Rolling_30_Day_Occupancy_Std'] = float(np.std(recent30)) if len(recent30) > 1 else 0.0

            row = _encode_row(row, encoders)

            feature_row = {col: row.get(col, 0) for col in feature_names}
            X = pd.DataFrame([feature_row])[feature_names]
            predicted_rate = float(model.predict(X)[0])
            predicted_rate = max(0.0, predicted_rate)

            occupied_beds = int(round(total_beds * predicted_rate / 100))
            available_beds = max(0, total_beds - occupied_beds)

            forecasts.append({
                'Date': forecast_date.strftime('%Y-%m-%d'),
                'Department': dept,
                'Forecast_Day': day_offset,
                'Predicted_Occupancy_Rate': round(predicted_rate, 2),
                'Predicted_Occupied_Beds': occupied_beds,
                'Predicted_Available_Beds': available_beds,
                'Total_Hospital_Beds': total_beds,
            })
            occupancy_history.append(predicted_rate)

    return pd.DataFrame(forecasts)


def save_forecast(forecast_df, output_path=None):
    """Save forecast DataFrame to CSV."""
    if output_path is None:
        output_path = FORECAST_OUTPUT_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"Forecast saved: {output_path}")
    return output_path


def main():
    print("=" * 80)
    print("7-DAY OCCUPANCY FORECAST")
    print("=" * 80)
    forecast_df = generate_forecast(days=7)
    save_forecast(forecast_df)
    print(forecast_df.head(10).to_string(index=False))
    return forecast_df


if __name__ == '__main__':
    main()
