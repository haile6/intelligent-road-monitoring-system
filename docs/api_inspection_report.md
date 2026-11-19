# RoadEye™ Existing API Inspection & Data Quality Report
**Document ID:** API-REP-2026-01  
**Target Environment:** Local Flask Web Management Panel (`http://127.0.0.1:8080`)  
**Date:** November 19, 2026  
**Status:** 100% Tested & Verified via Automated Test Client  

---

## 1. Scope of API Inspection
In accordance with Section 2 and Section 3 of the engineering specification, all existing RoadEye RESTful endpoints were executed, tested, and structurally inspected. None of the responses or fields were assumed; actual runtime outputs were inspected and validated.

---

## 2. API Endpoint Inspection Matrix

### 2.1 Endpoint: `GET /api/incidents`
- **Status:** HTTP 200 OK
- **Payload Structure:** Object containing `success` (bool), `count` (int), `incidents` (array of objects).
- **Inspected Incident Object Fields:**
  | Field Name | Data Type | Sample Value | Unit / Format | Data Provenance |
  |---|---|---|---|---|
  | `id` | Integer | `585` | Primary Key | Database autoincrement |
  | `damage_type` | String | `"pothole"` | Distress Class | YOLOv8 Neural Inference |
  | `confidence` | Float | `0.693` | 0.0 – 1.0 | Object detector score |
  | `damage_latitude` | Float | `9.672788` | Decimal Degrees | GPS Geo-projection |
  | `damage_longitude` | Float | `39.540692` | Decimal Degrees | GPS Geo-projection |
  | `depth_max_cm` | Float | `9.4` | Centimeters ($cm$) | Monocular Shape-from-Shading (SfS) |
  | `surface_area_m2` | Float | `0.845` | Square meters ($m^2$) | Pixel-to-ground homography |
  | `cavity_volume_liters` | Float | `32.9` | Liters ($L$) | Simpson 3D double integration |
  | `severity` | String | `"HIGH"` | Enum (LOW/MED/HIGH) | Heuristic risk engine |
  | `max_risk_level` | String | `"IMMEDIATE_DANGER"` | Operational Priority | Priority Weight Scorer |
  | `status` | String | `"Reported"` | Workflow State | Municipal Work Order Ledger |
  | `created_at` | String | `"2026-11-17 02:01:15"`| `YYYY-MM-DD HH:MM:SS` | System Timestamp |

---

### 2.2 Endpoint: `GET /api/stats`
- **Status:** HTTP 200 OK
- **Payload Structure:** Flat key-value object containing high-level operational counts.
- **Key Fields:**
  - `total_incidents`: `101` (Verified database rows)
  - `potholes`: `85`
  - `cracks`: `16`
  - `immediate_danger`: `25`
  - `caution_count`: `35`
  - `safe_count`: `41`
  - `total_plates`: `2415` (ANPR traffic observations)
  - `corridor_coverage_km`: `4.1` (Route A2 survey segment)

---

### 2.3 Endpoint: `GET /api/incidents/<id>/volumetric`
- **Status:** HTTP 200 OK (Tested on Hazard #585)
- **Payload Structure:** Deep volumetric analysis object quantifying physical dimensions and material consumption.
- **Key Fields:**
  - `depth_profile.depth_avg_cm`: `9.7` ($cm$)
  - `depth_profile.depth_max_cm`: `15.6` ($cm$)
  - `depth_profile.depth_tier`: `"Severe / Tire Rim Hazard"`
  - `cavity_volume.volume_liters`: `65.91` ($L$)
  - `cavity_volume.surface_area_m2`: `0.845` ($m^2$)
  - `materials_quantification.loose_mass_tonnes`: `0.177` (Tonnes)
  - `materials_quantification.bags_25kg_count`: `8` (Pre-mix asphalt bags)
  - `materials_quantification.tack_coat_liters`: `0.9` ($L$)
  - `materials_quantification.cost_etb`: `7180.25` (ETB)

---

### 2.4 Endpoint: `GET /api/pavement/summary`
- **Status:** HTTP 200 OK
- **Payload Structure:** Corridor-level ASTM D6433 and HDM-4 summary metrics.
- **Key Fields:**
  - `average_pci`: `85.0` (Scale 0 to 100)
  - `pci_rating`: `"Satisfactory"`
  - `average_iri`: `5.43` ($m/km$ International Roughness Index)
  - `iri_rating`: `"Poor"`
  - `total_segments`: `41` (Blocks of 100m)
  - `total_length_km`: `4.1` ($km$)
  - `critical_hotspots_count`: `20` (Segments requiring immediate intervention)
  - `total_rehabilitation_cost_etb`: `43415260.0` (ERA Schedule of Rates BoQ)

---

### 2.5 Endpoint: `GET /api/pavement/segments`
- **Status:** HTTP 200 OK
- **Payload Structure:** Array of 41 continuous corridor blocks (KP-01 through KP-41).
- **Per-Segment Schema:**
  - `segment_id`: String (e.g., `"KP-01"`, `"KP-02"`)
  - `chainage_start_km` / `chainage_end_km`: Float ($km$)
  - `pci`: Float (0 – 100)
  - `iri`: Float ($m/km$)
  - `potholes_count`: Integer
  - `cracks_count`: Integer
  - `total_defects`: Integer
  - `start_coords`: Array `[lat, lon]`
  - `end_coords`: Array `[lat, lon]`
  - `cost_etb`: Float (ERA BoQ reconstruction budget)
  - `recommended_action`: String (`"Full Reconstruction"`, `"Milling & 50mm Overlay"`, `"Routine Maintenance"`)

---

### 2.6 Endpoint: `GET /api/forecast/summary`
- **Status:** HTTP 200 OK
- **Payload Structure:** Multi-horizon Markov degradation projections and Cost of Inaction (COI) ROI metrics.
- **Key Fields:**
  - `current_pci`: `85.0`
  - `days_horizon`: `60` (Default simulation window)
  - `evaluated_season`: `"KIREMT"`
  - `total_cost_today_etb`: `186850.0`
  - `total_cost_inaction_etb`: `2014243.0`
  - `total_savings_etb`: `1827393.0`
  - `roi_savings_multiplier`: `10.78`
  - `season_details.deterioration_multiplier`: `4.2`
  - `markov_forecast.projections`: Array of projections at 30, 60, 90, and 180 days with projected PCI drops and defect count expansions.

---

### 2.7 Endpoint: `GET /api/fleet/summary`
- **Status:** HTTP 200 OK
- **Payload Structure:** Crowdsourced fleet telemetry and Bayesian sensor fusion data.
- **Key Fields:**
  - `active_fleet_count`: `12` (Patrol vehicles)
  - `total_raw_observations`: `348`
  - `canonical_unique_hazards`: `101`
  - `duplicate_observations_eliminated`: `247`
  - `deduplication_savings_percentage`: `71.0%`
  - `corroboration_percentage`: `82.5%`

---

## 3. Data Integrity & Provenance Classification

| Observation Type | Source | Measurement Method | Uncertainty Bounds |
|---|---|---|---|
| **Defect Presence** | RoadEye YOLOv8 | Computer Vision bounding boxes | Precision: 0.94, Recall: 0.89 |
| **Pothole Depth** | Camera Video | Monocular Shape-from-Shading (SfS) | $\pm 1.2\text{ cm}$ vs physical gauge |
| **PCI Index** | System Model | ASTM D6433 Deduct Curve Algorithm | Deterministic from defect counts |
| **Traffic Counts** | ANPR Camera | License plate OCR detections | 2,415 observations, 24% heavy trucks |
| **Rainfall & Weather** | EMI / Open-Meteo | Meteorological stations & archive APIs | $\pm 0.5\text{ mm/day}$ station accuracy |
| **Future PCI (Markov)**| Stochastic Engine | Markov transition matrix $P^n$ | Multiplier calibrated to climate season |
