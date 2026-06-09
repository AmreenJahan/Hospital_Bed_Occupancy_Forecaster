## Folder Structure

```
Hospital_Bed_Occupancy_Forecaster/
├── config.py
├── main.py
├── run_all.py
├── train_models.py
├── model_evaluation.py
├── forecasting.py
├── overflow_alert_system.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── hospital_bed_occupancy_10000.csv
│   ├── cleaned_hospital_bed_occupancy.csv
│   ├── engineered_hospital_bed_occupancy.csv
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
├── scripts/
│   ├── load_data.py
│   ├── data_cleaning.py
│   ├── exploratory_data_analysis.py
│   ├── operational_insights.py
│   ├── feature_engineering.py
│   ├── dataset_preparation.py
│   ├── train_models.py
│   └── model_evaluation.py
├── models/
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   ├── split_info.json
│   └── *_model.pkl
├── saved_models/
│   └── best_model.pkl
├── reports/
│   ├── *.txt, *.md, *.csv
│   └── figures/
├── streamlit_app/
│   └── app.py
└── notebooks/ (optional)
```

## Installation

```bash
cd Hospital_Bed_Occupancy_Forecaster
pip install -r requirements.txt
```

## Run Instructions

```bash
python run_all.py

# Or step-by-step:
python main.py
python train_models.py
python model_evaluation.py
python forecasting.py
python overflow_alert_system.py

# Dashboard
streamlit run streamlit_app/app.py
```

## Notes

- Set `PYTHONIOENCODING=utf-8` on Windows if Unicode console errors occur
- Best model: RandomForest (MAE 9.135, R² 0.553)
- Chronological split date: 2024-12-30
