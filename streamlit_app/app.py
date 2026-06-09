"""
Hospital Bed Occupancy Forecaster - Streamlit Dashboard
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    CLEANED_DATA_PATH, FIGURES_DIR, FORECAST_OUTPUT_PATH,
    OVERFLOW_ALERTS_PATH, REPORTS_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH,
)
from overflow_alert_system import detect_overflow_risk, generate_recommendations

st.set_page_config(
    page_title='Hospital Bed Occupancy Forecaster',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a5276; }
    .kpi-card {
        background: linear-gradient(135deg, #1a5276 0%, #2e86ab 100%);
        padding: 1.2rem; border-radius: 10px; color: white; text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: bold; }
    .kpi-label { font-size: 0.9rem; opacity: 0.9; }
    .risk-green { color: #27ae60; font-weight: bold; }
    .risk-yellow { color: #f39c12; font-weight: bold; }
    .risk-orange { color: #e67e22; font-weight: bold; }
    .risk-red { color: #c0392b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def kpi_card(label, value, suffix=''):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}{suffix}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def page_executive_overview():
    st.markdown('<p class="main-header">Executive Overview</p>', unsafe_allow_html=True)
    df = load_csv(CLEANED_DATA_PATH)
    forecast = load_csv(FORECAST_OUTPUT_PATH)

    if df is None:
        st.warning('Run the pipeline first to generate data.')
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card('Avg Occupancy', f"{df['Occupancy_Rate'].mean():.1f}", '%')
    with c2:
        kpi_card('Departments', df['Department'].nunique(), '')
    with c3:
        kpi_card('Records', f"{len(df):,}", '')
    with c4:
        high_risk = (df['Occupancy_Rate'] > 95).sum() if 'Occupancy_Rate' in df.columns else 0
        kpi_card('High Risk Days', high_risk, '')

    st.subheader('Occupancy Trend')
    daily = df.groupby('Date')['Occupancy_Rate'].mean().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    fig = px.line(daily, x='Date', y='Occupancy_Rate', title='Hospital-Wide Average Occupancy')
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)

    if forecast is not None:
        st.subheader('7-Day Forecast Summary')
        st.dataframe(forecast.head(14), use_container_width=True)


def page_eda():
    st.markdown('<p class="main-header">EDA Dashboard</p>', unsafe_allow_html=True)
    df = load_csv(CLEANED_DATA_PATH)
    if df is None:
        st.warning('Cleaned data not found.')
        return

    tab1, tab2, tab3 = st.tabs(['By Department', 'Seasonal', 'Correlations'])
    with tab1:
        dept_avg = df.groupby('Department')['Occupancy_Rate'].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(dept_avg, x='Occupancy_Rate', y='Department', orientation='h', color='Occupancy_Rate',
                     color_continuous_scale='Blues', title='Average Occupancy by Department')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'Season' in df.columns:
            season_avg = df.groupby('Season')['Occupancy_Rate'].mean().reset_index()
            fig = px.bar(season_avg, x='Season', y='Occupancy_Rate', title='Occupancy by Season')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num = df.select_dtypes(include='number')
        if 'Occupancy_Rate' in num.columns:
            corr = num.corr()['Occupancy_Rate'].drop('Occupancy_Rate').sort_values()
            fig = px.bar(x=corr.values, y=corr.index, orientation='h', title='Feature Correlations with Occupancy')
            st.plotly_chart(fig, use_container_width=True)

    fig_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')] if os.path.exists(FIGURES_DIR) else []
    if fig_files:
        st.subheader('Generated EDA Figures')
        selected = st.selectbox('Select figure', sorted(fig_files))
        st.image(os.path.join(FIGURES_DIR, selected), use_container_width=True)


def page_forecasting():
    st.markdown('<p class="main-header">Forecasting Dashboard</p>', unsafe_allow_html=True)
    forecast = load_csv(FORECAST_OUTPUT_PATH)
    if forecast is None:
        st.warning('Run forecasting.py to generate forecasts.')
        return

    dept = st.selectbox('Department', ['All'] + sorted(forecast['Department'].unique()))
    data = forecast if dept == 'All' else forecast[forecast['Department'] == dept]

    fig = go.Figure()
    for d in data['Department'].unique():
        sub = data[data['Department'] == d]
        fig.add_trace(go.Scatter(
            x=sub['Date'], y=sub['Predicted_Occupancy_Rate'],
            mode='lines+markers', name=d,
        ))
    fig.update_layout(title='7-Day Occupancy Forecast', template='plotly_white', height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(data, use_container_width=True)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button('Download Forecast CSV', csv, 'forecast_7_day.csv', 'text/csv')


def page_overflow():
    st.markdown('<p class="main-header">Overflow Monitoring</p>', unsafe_allow_html=True)
    alerts = load_csv(OVERFLOW_ALERTS_PATH)
    forecast = load_csv(FORECAST_OUTPUT_PATH)

    if alerts is None and forecast is not None:
        from overflow_alert_system import run_overflow_alerts
        alerts = run_overflow_alerts()

    if alerts is None:
        st.warning('No alert data available.')
        return

    risk_counts = alerts['Risk_Level'].value_counts()
    cols = st.columns(4)
    colors = {'GREEN': '🟢', 'YELLOW': '🟡', 'ORANGE': '🟠', 'RED': '🔴'}
    for i, level in enumerate(['GREEN', 'YELLOW', 'ORANGE', 'RED']):
        with cols[i]:
            count = risk_counts.get(level, 0)
            st.metric(f'{colors[level]} {level}', count)

    st.subheader('Alert Details')
    st.dataframe(alerts, use_container_width=True)

    selected_risk = st.selectbox('View recommendations for', ['RED', 'ORANGE', 'YELLOW', 'GREEN'])
    for rec in generate_recommendations(selected_risk):
        st.info(rec)


def page_model_performance():
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)

    comparison = load_csv(os.path.join(REPORTS_DIR, 'model_comparison_table.csv'))
    metrics = load_csv(os.path.join(REPORTS_DIR, 'model_evaluation_metrics.csv'))
    importance = load_csv(os.path.join(REPORTS_DIR, 'feature_importance.csv'))

    if comparison is not None:
        st.subheader('Model Comparison')
        st.dataframe(comparison, use_container_width=True)
        fig = px.bar(comparison, x='Model', y='MAE', color='Model', title='MAE by Model (lower is better)')
        st.plotly_chart(fig, use_container_width=True)

    if metrics is not None:
        st.subheader('Best Model Metrics')
        m = metrics.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('MAE', f"{m['MAE']:.3f}")
        c2.metric('RMSE', f"{m['RMSE']:.3f}")
        c3.metric('R²', f"{m['R2']:.4f}")
        c4.metric('MSE', f"{m['MSE']:.3f}")

    if importance is not None:
        st.subheader('Feature Importance')
        fig = px.bar(importance.head(15), x='Importance', y='Feature', orientation='h',
                     title='Top 15 Features')
        st.plotly_chart(fig, use_container_width=True)

    avp = os.path.join(FIGURES_DIR, 'actual_vs_predicted.png')
    if os.path.exists(avp):
        st.image(avp, caption='Actual vs Predicted', use_container_width=True)


def main():
    st.sidebar.title('🏥 Navigation')
    page = st.sidebar.radio(
        'Go to',
        ['Executive Overview', 'EDA Dashboard', 'Forecasting', 'Overflow Monitoring', 'Model Performance'],
    )
    st.sidebar.markdown('---')
    st.sidebar.caption('Hospital Bed Occupancy Forecaster v1.0')

    pages = {
        'Executive Overview': page_executive_overview,
        'EDA Dashboard': page_eda,
        'Forecasting': page_forecasting,
        'Overflow Monitoring': page_overflow,
        'Model Performance': page_model_performance,
    }
    pages[page]()


if __name__ == '__main__':
    main()
