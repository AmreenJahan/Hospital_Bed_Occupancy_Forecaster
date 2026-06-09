"""
Hospital overflow risk detection and recommendation system.
"""

import os

import pandas as pd

from config import FORECAST_OUTPUT_PATH, OVERFLOW_ALERTS_PATH, OVERFLOW_THRESHOLDS, REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


def detect_overflow_risk(occupancy_rate):
    """
    Classify occupancy into risk levels.

    GREEN:  < 70%
    YELLOW: 70-85%
    ORANGE: 85-95%
    RED:    > 95%
    """
    rate = float(occupancy_rate)
    if rate < 70:
        return 'GREEN'
    if rate < 85:
        return 'YELLOW'
    if rate < 95:
        return 'ORANGE'
    return 'RED'


def generate_recommendations(risk_level):
    """Return operational recommendations based on risk level."""
    base = {
        'GREEN': [
            'Maintain standard staffing levels.',
            'Continue routine capacity monitoring.',
        ],
        'YELLOW': [
            'Increase staffing in high-traffic units.',
            'Review elective admission schedules.',
            'Monitor discharge planning efficiency.',
        ],
        'ORANGE': [
            'Increase staffing across affected departments.',
            'Prepare overflow units for activation.',
            'Delay non-critical admissions where clinically safe.',
            'Escalate to bed management team.',
        ],
        'RED': [
            'Increase staffing immediately across all units.',
            'Open temporary wards and overflow units.',
            'Delay non-critical admissions.',
            'Prepare overflow units and divert protocols.',
            'Activate hospital incident command procedures.',
        ],
    }
    return base.get(risk_level, base['GREEN'])


def run_overflow_alerts(forecast_path=None, output_path=None):
    """Process forecast CSV and export overflow alerts."""
    if forecast_path is None:
        forecast_path = FORECAST_OUTPUT_PATH
    if output_path is None:
        output_path = OVERFLOW_ALERTS_PATH

    df = pd.read_csv(forecast_path)
    rate_col = 'Predicted_Occupancy_Rate' if 'Predicted_Occupancy_Rate' in df.columns else 'Occupancy_Rate'

    alerts = []
    for _, row in df.iterrows():
        risk = detect_overflow_risk(row[rate_col])
        recs = generate_recommendations(risk)
        alerts.append({
            'Date': row.get('Date'),
            'Department': row.get('Department'),
            'Occupancy_Rate': row[rate_col],
            'Risk_Level': risk,
            'Recommendations': ' | '.join(recs),
        })

    alert_df = pd.DataFrame(alerts)
    alert_df.to_csv(output_path, index=False)
    print(f"✓ Overflow alerts saved: {output_path}")
    return alert_df


def main():
    print("=" * 80)
    print("OVERFLOW ALERT SYSTEM")
    print("=" * 80)
    alert_df = run_overflow_alerts()
    print(alert_df.groupby('Risk_Level').size().to_string())
    return alert_df


if __name__ == '__main__':
    main()
