# Capacity Planning Report

## Executive Summary

Based on 6,943 department-day records (2021–2025) and 7-day ML forecasts, this report provides actionable capacity planning guidance for hospital bed management.

---

## Current State

| Metric | Value |
|--------|-------|
| Average occupancy | 81.96% |
| Over-capacity days flagged | 1,085 (15.6%) |
| Departments monitored | 7 |
| Forecast horizon | 7 days |

---

## Risk Distribution (7-Day Forecast Alerts)

Recent forecast alerts indicate elevated monitoring needs:
- **YELLOW (70–85%):** Majority of forecast days — proactive staffing adjustments recommended
- **ORANGE (85–95%):** Surge preparation required
- **RED (>95%):** Immediate capacity escalation

---

## Department-Level Recommendations

### Cardiology, Emergency, ICU
- Maintain surge staffing pools during forecast YELLOW periods
- Pre-position overflow beds when 3+ consecutive ORANGE days forecast

### General Medicine, Neurology, Orthopedics, Pediatrics
- Optimize discharge planning during high rolling 7-day occupancy
- Coordinate elective surgery scheduling with forecast dips

---

## Operational Playbook

### GREEN (< 70%)
- Standard staffing and routine monitoring

### YELLOW (70–85%)
- Increase staffing in high-traffic units
- Review elective admission schedules
- Monitor discharge planning efficiency

### ORANGE (85–95%)
- Increase staffing across affected departments
- Prepare overflow units for activation
- Delay non-critical admissions where clinically safe
- Escalate to bed management team

### RED (> 95%)
- Increase staffing immediately across all units
- Open temporary wards and overflow units
- Delay non-critical admissions
- Prepare overflow units and divert protocols
- Activate hospital incident command procedures

---

## Seasonal Capacity Planning

| Season | Planning Action |
|--------|-----------------|
| Monsoon | Highest volume — add 10% buffer capacity |
| Winter | Second peak — extend ICU surge protocols |
| Summer | Lower volume — schedule maintenance and training |

---

## Data-Driven Monitoring KPIs

1. Rolling 7-day occupancy mean (primary early warning)
2. Lag-1 day occupancy (short-term momentum)
3. Over-capacity flag frequency per department
4. Forecasted available beds (7-day horizon)

---

## Conclusion

Integrate `overflow_alert_system.py` outputs into daily bed management meetings. Use Streamlit dashboard Overflow Monitoring page for real-time risk visualization and forecast CSV exports for ERP integration.
