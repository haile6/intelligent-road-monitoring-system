# RoadEye™ System Inspection & Mechanism-Based Architecture Assessment Report

**System Name:** RoadEye™ Intelligent Road Monitoring and Predictive Pavement Maintenance System  
**Corridor:** Route A2 · Addis Ababa – Debre Berhan Highway (Lat: `9.68°N`, Lon: `39.53°E`)  
**Target:** Mechanism-Based Pavement Deterioration Prediction Architecture  
**Date:** November 2026  
**Authority Reference:** Ethiopian Roads Administration (ERA) Pavement Design & Maintenance Directives / World Bank HDM-4 Standards  

---

## 1. Executive Summary

This inspection report evaluates the existing RoadEye codebase to guide the transition from a fixed $4.20\times$ seasonal deterioration multiplier to a **scientifically defensible, data-driven, mechanism-based pavement deterioration prediction architecture**.

The existing RoadEye platform possesses a production-grade Flask WSGI backend, SQLite longitudinal database, YOLOv8 computer vision damage detector, 3D Shape-from-Shading (SfS) volumetric depth estimator, automated ANPR vehicle tracker, and an Open-Meteo historical weather client. However, several modules previously relied on heuristic multipliers (such as an empirical/calibrated $4.20\times$ Kiremt multiplier) in primary prediction loops. This report audits existing assets, identifies field data gaps, establishes feature engineering feasibility, and details the exact code modules to be replaced.

---

## 2. Existing Components Available for Direct Reuse

The following components are robust, validated, and will be preserved and reused:

| Component | File Path | Reuse Scope & Capability |
|---|---|---|
| **Weather Integration Engine** | `data/weather_client.py` | Open-Meteo historical weather client for Debre Berhan (`9.68`, `39.53`). Fetches precipitation ($mm/day$), temperature ($^\circ C$), relative humidity ($%$), and ERA-Interim soil moisture. Features local SQLite caching, deduplication, and offline fallback. |
| **Corridor Segmentation & Pavement Intelligence** | `risk/pavement_intelligence.py` | ASTM D6433 Pavement Condition Index (PCI) deduction calculation, 100m dynamic corridor segmentation (`KP-01` to `KP-21`), defect spatial mapping, and structural condition indexing. |
| **Volumetric & 3D Depth Estimation** | `risk/volumetric_estimator.py` | Monocular Shape-from-Shading (SfS) depth estimation ($\lambda = 0.08$) and double numerical Simpson integration for cavity volume ($dm^3$/Liters). |
| **Traffic & Fleet Telemetry** | `risk/fleet_crowdsourcing.py` & `database/db_manager.py` | ANPR vehicle tracking, traffic volume counting, commercial truck classification (Sino-Truck freight proxy), and speed telemetry. |
| **Database Management** | `database/db_manager.py` | SQLite schema management for 101 corridor incidents, 2,415 plate readings, 378 multi-month unified observations, and contractor warranty ledger. |
| **Core ML Algorithms (NumPy Native)** | `risk/ml_deterioration_predictor.py` | CART Decision Trees, Random Forest Regressor (35 trees), Ridge L2 analytical solver, and chronological train/validation/test splitting logic. |
| **Web Server & UI Templates** | `web/app.py`, `web/templates/` | Flask WSGI routing, REST API infrastructure, and responsive Glassmorphism dashboard. |

---

## 3. Data Audit: Measured vs. Estimated vs. Missing

To maintain scientific integrity, every variable in RoadEye is explicitly classified by its empirical measurement method:

### A. Features Calculable from Existing Data
- **Moisture Exposure (Atmospheric):**
  - Daily precipitation ($mm/day$) from Open-Meteo Debre Berhan station.
  - Cumulative 7-day, 30-day, and 60-day rainfall ($mm$).
  - Number of rainy days ($\ge 1.0 mm$) in 30-day window.
  - Consecutive wet days count.
- **Heavy Traffic Exposure (Proxy):**
  - Average Daily Traffic (ADT / VPD).
  - Heavy commercial truck count (Sino-trucks, tippers, buses).
  - Commercial truck ratio ($\%$).
- **Fatigue & Surface Distress:**
  - Pothole count per 100m segment.
  - Longitudinal and transverse crack count.
  - Crack density ($m/m^2$ or $\%$ carriage area).
- **Pothole Dimensions (Camera-Estimated):**
  - Average and maximum surface length/width ($cm$).
  - Monocular 3D SfS depth profile ($cm$).
  - Cavity volume ($dm^3$ / Liters).

### B. Missing Data Requiring Direct Field Measurements
The following variables represent physical mechanisms that are **not directly measured by camera or weather APIs** and require physical field instruments:
1. **Pavement & Subgrade Pore-Water Pressure:** Direct pore water pressure ($kPa$) requires installed vibrating wire piezometers in the sub-base. In the current system, rainfall and ERA-Interim soil moisture serve as an atmospheric surface proxy, not a direct pore pressure sensor.
2. **Dynamic Axle Loadings (True ESALs):** True Equivalent Single Axle Loads (80 kN standard axle) require dynamic Weigh-in-Motion (WIM) piezoelectric road sensors. Camera ANPR provides vehicle classification counts, which act as a **traffic proxy**, not calibrated dynamic wheel scales.
3. **Roadside Drainage Ditch Blockage:** Direct telemetry on roadside ditch sediment buildup requires physical manual ditch surveys or specialized drone multispectral imagery. When unavailable, it must be flagged as `UNAVAILABLE_FIELD_SURVEY_REQUIRED`, never faked.
4. **Structural Rut Depth Profiling:** Millimeter-accurate rut depth requires transverse laser profilometers (e.g., Dynatest RSP). Camera dashcam imagery cannot accurately measure millimeter-level rutting depression without ground-truth profiling LiDAR. It is marked as unmeasured.
5. **Subgrade Resilient Modulus ($M_R$) & CBR:** Requires Falling Weight Deflectometer (FWD) or laboratory CBR testing of core samples.

---

## 4. Existing Code to be Replaced

The following specific code sections contain hardcoded multipliers or heuristic shortcuts and must be replaced:

1. **`risk/ml_deterioration_predictor.py` (Line 388):**
   ```python
   # CURRENT CODE (To be replaced):
   markov_pci = max(0.0, pci_val - (4.2 * (1.0 + (100.0 - pci_val) * 0.05)))
   ```
   *Action:* Replace with a discrete 5-state Markov transition rate matrix calibrated from observed segment condition shifts, with mathematically correct transition step scaling without heuristic multipliers.

2. **`risk/predictive_deterioration.py` (Lines 27, 290–297):**
   ```python
   # CURRENT CODE (To be refactored):
   "deterioration_multiplier": 4.2  # 420% accelerated deterioration
   cost_inaction_etb = round(cost_today_etb * (2.8 + 0.95 * m * t_months), 2)
   ```
   *Action:* Decouple the primary forecasting pipeline from `deterioration_multiplier: 4.2`. Relegate $4.20\times$ strictly to an optional comparison scenario labeled `"Regional Wet Season Simulation Scenario (HDM-4 Calibrated)"`. Replace arbitrary cost formulas with a transparent Bill of Quantities (BOQ) maintenance cost estimator based on ERA unit rates.

3. **`data/unified_dataset.py` (Lines 213, 243):**
   ```python
   # CURRENT CODE (To be refactored):
   step_weights.append(4.20 if is_k else 1.00)
   avg_depth_cm = round(3.5 + 4.2 * (1.0 - (current_pci / 100.0)), 1)
   ```
   *Action:* Remove fixed 4.2 weighting from data synthesis and feature definitions. Compute deterioration directly through empirical environmental and structural regressors.

4. **`web/templates/forecast.html`:**
   *Action:* Update executive KPI cards and projections so that the primary 7d, 30d, and 60d figures reflect the data-driven ML models (Ridge / Random Forest). Display the HDM-4 4.20× scenario in a secondary "Comparative Simulation Scenario" panel with explicit provenance badges (`[MEASURED]`, `[CAMERA-DETECTED]`, `[PROXY-DERIVED]`, `[PREDICTED]`, `[SIMULATED]`).

---

## 5. Architectural Upgrade Roadmap

```
Raw Observations:
[Open-Meteo Weather] + [YOLOv8 & 3D SfS Perception] + [ANPR Fleet Tracking] + [Corridor GIS]
                              ↓
Physical Deterioration Mechanisms:
• Moisture Exposure (7d/30d/60d Cumulative Rain, Rainy Days, Wet Spells)
• Heavy-Load Exposure Proxy (Truck Counts, Freight Ratio)
• Fatigue & Crack Progression (Crack Density, Longitudinal Strain)
• Pothole Cavity Growth (SfS Volume & Depth Expansion)
• Drainage Status (Explicitly Logged / Flagged as Field Survey Required)
                              ↓
Machine Learning Pipeline:
• Ridge L2 Regularized Regressor
• Random Forest Multi-Tree Regressor
• Markov Transition Rate Baseline (Calibrated Pⁿ)
                              ↓
Operational Outputs:
• 7-Day, 30-Day, 60-Day Pavement Condition Index (PCI) Forecasts with 95% CIs
• Pothole Count and Volume Expansion Predictions
• Transparent ERA Bill of Quantities (BOQ) Maintenance Cost Ledger
• REST API Suite & Glassmorphism Dashboard
```
