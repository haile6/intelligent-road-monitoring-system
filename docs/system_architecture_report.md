# RoadEye™ System Architecture Assessment Report
**Document ID:** ARCH-REP-2026-01  
**Project:** RoadEye — Intelligent Road Monitoring and Predictive Pavement Maintenance System  
**Corridor Scope:** Route A2 · Addis Ababa - Debre Berhan Highway (Debre Berhan Highland Zone)  
**Date:** November 19, 2026  
**Status:** Comprehensive Inspection Completed  

---

## 1. Executive Summary
RoadEye™ is an integrated, end-to-end intelligent transportation and pavement asset management system designed for municipal and federal road authorities (e.g., Ethiopian Roads Administration - ERA). This technical assessment inspects the existing software architecture, database layers, computer vision pipelines, pavement condition analytics, forecasting models, and dashboard interfaces.

---

## 2. Architecture Overview & Component Breakdown

```
+---------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER                                 |
|  - Operations Dashboard (/ )               - GIS Live Patrol Map (/map)               |
|  - Pavement Health & Budget (/pavement)    - Predictive Deterioration (/forecast)     |
|  - Incident Ledger (/incidents)            - Contractor DLP & Warranties (/warranty)  |
|  - Fleet Crowdsourcing (/fleet)            - Municipal Rates (/settings)              |
+---------------------------------------------------------------------------------------+
                                           |
                                  Flask WSGI REST APIs
                                           |
+---------------------------------------------------------------------------------------+
|                                  APPLICATION CORE                                     |
|  - web/app.py: 2,000+ LOC Flask Controller with 30+ RESTful Endpoints                 |
|  - risk/pavement_intelligence.py: ASTM D6433 PCI, World Bank HDM-4 IRI, ERA BoQ       |
|  - risk/volumetric_estimator.py: Monocular Shape-from-Shading (SfS) 3D Depth           |
|  - risk/predictive_deterioration.py: Markov-Chain 5-State Transition Engine & COI     |
|  - risk/fleet_crowdsourcing.py: Bayesian Fusion & Spatial De-Duplication              |
+---------------------------------------------------------------------------------------+
                      |                                            |
+-------------------------------------+   +---------------------------------------------+
|        DATA & STORAGE LAYER         |   |          COMPUTER VISION PIPELINE           |
|  - SQLite (database/road_monitoring |   |  - YOLOv8 Custom Defect Detection           |
|  - Tables: incidents, plate_readings|   |  - ByteTrack / Kalman Centroid Tracking     |
|    road_contractors, system_settings|   |  - Monocular 3D Depth & Simpson Volumetrics |
|    pipeline_runs                    |   |  - ANPR Vehicle License Plate OCR           |
+-------------------------------------+   +---------------------------------------------+
```

### 2.1 Backend Architecture (`web/app.py`)
- **Framework:** Flask 3.0+ running on Python 3.11 with WSGI architecture.
- **Routing:** 30+ modular RESTful endpoints serving both server-rendered Jinja2 templates and high-throughput JSON APIs.
- **State Management:** SQLite persistence through thread-safe connection factories with row factories returning dict-like records.
- **Caching:** In-memory caching for corridor pavement segmentation (`get_pavement_assessment_cache()`) and deterioration simulations (`get_forecast_simulation_cache()`) with dynamic invalidation upon parameter changes.

### 2.2 Database Layer (`database/db_manager.py`)
- **Engine:** SQLite 3 (`database/road_monitoring.db`).
- **Core Entities:**
  1. `incidents`: Primary defect repository (101 verified incidents across Route A2 corridor). Stores defect type, 2D coordinates, depth profile (`depth_max_cm`), surface area (`surface_area_m2`), volumetric cavity (`cavity_volume_liters`), risk level (`max_risk_level`), and dual risk factors.
  2. `plate_readings`: Vehicle traffic and commercial fleet observations (2,415 plate detections) with bounding boxes, OCR text, and confidence scores.
  3. `road_contractors`: Warranty and Defect Liability Period (DLP) registry covering Ethiopian road contractors, GPS boundaries, contract values, and warranty expiration dates.
  4. `system_settings`: Key-value configuration store preserving municipal unit rates for full reconstruction, milling, overlay, and routine maintenance.
  5. `pipeline_runs`: Operational audit journal logging video telemetry jobs, execution status, and exit codes.

### 2.3 Computer Vision & Sensing Pipeline (`pipeline/`)
- **Detection Model:** YOLOv8 trained specifically on Ethiopian road distress classes (potholes, alligator cracking, longitudinal/transverse cracks).
- **Spatial Tracking:** ByteTrack spatial tracker ensuring temporal continuity across frames with unique track IDs.
- **3D Depth Estimation:** Monocular Shape-from-Shading (SfS) algorithms in `risk/volumetric_estimator.py` that reconstruct 3D cavity profiles from 2D pixel intensity gradients without requiring expensive LiDAR hardware.
- **Volumetric Integration:** Double Simpson numerical integration ($\iint D(x,y)\,dx\,dy$) quantifying asphalt repair material in Liters and 25kg/50kg pre-mix asphalt bags.

### 2.4 Pavement Engineering & Forecasting Models (`risk/`)
- **ASTM D6433 PCI Calculator:** Evaluates corridor segments (KP-01 through KP-41 at 50m/100m/250m resolutions), calculating deduct values and mapping to international condition indices (0 to 100).
- **World Bank HDM-4 IRI Model:** Computes International Roughness Index ($m/km$) as a function of baseline road roughness, pothole cavities, and crack densities.
- **Markov-Chain Multi-State Engine:** Probabilistic transition matrices modeling pavement progression across 5 condition states: Intact (State 0), Alligator Cracking (State 1), Shallow Pothole (State 2), Deep Pothole (State 3), and Sub-Base Collapse (State 4).
- **Cost of Inaction (COI) Forecaster:** Compares immediate preventive patching costs (Routine Patching Today) versus delayed post-monsoon structural reconstruction costs over 30, 60, 90, and 180-day horizons.

---

## 3. Findings & Modernization Priorities

| Component | Current State | Modernization Needed for Unified ML Pipeline |
|---|---|---|
| **Weather Ingestion** | Static seasonal parameters (`CLIMATE_SEASONS` Kiremt 280mm, Belg 95mm, Bega 15mm) | Real-time & historical Open-Meteo API client with SQLite observation caching for Debre Berhan (`9.68`, `39.53`). |
| **Historical Pavement Data** | Single current snapshot of 101 incidents and 41 segments | Multi-date longitudinal time-series dataset linking weather, traffic, and pavement observations by segment and date. |
| **Deterioration Modeling** | Deterministic Markov transition matrix with climate multipliers | Multi-model machine learning regressors (Random Forest, Ridge) trained on chronological data with held-out validation. |
| **4.2x Seasonal Factor** | Standard simulation multiplier ($M = 4.2$) | Empirical historical rate calculation ($D_{\text{wet}} / D_{\text{dry}}$) with statistical uncertainty bounds and clear provenance labeling. |
| **Prediction APIs** | `/api/forecast/summary` and `/api/forecast/simulate` | Granular prediction endpoints for 7d, 30d, 60d horizons, per-segment forecasts, data-quality auditing, and model performance metrics. |
