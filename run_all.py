"""
Execute the complete Hospital Bed Occupancy Forecaster pipeline.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def main():
    print("\n" + "=" * 80)
    print("FULL PIPELINE EXECUTION")
    print("=" * 80)

    # Day 1
    from main import main as run_day1
    run_day1()

    # Day 2 - Modeling
    from scripts.train_models import main as train_models
    train_models()

    from scripts.model_evaluation import main as evaluate_models
    evaluate_models()

    # Forecasting & Alerts
    from forecasting import main as run_forecast
    run_forecast()

    from overflow_alert_system import main as run_alerts
    run_alerts()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - Project is submission-ready")
    print("=" * 80)
    print("Streamlit: streamlit run streamlit_app/app.py")


if __name__ == '__main__':
    main()
