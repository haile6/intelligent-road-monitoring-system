# RoadEye™ Machine Learning Feature Dictionary

## 1. Overview & Engineering Principles

The **RoadEye™ Machine Learning Pavement Deterioration Engine** maps multimodal sensor telemetry and atmospheric observations into physically grounded, mechanism-based feature vectors. 

Every feature strictly adheres to four scientific integrity rules:
1. **Explicit Physical Meaning**: Features quantify real thermodynamic, hydraulic, or mechanical stress phenomena (e.g., pore water pressure, shear fatigue, cumulative freight loading).
2. **Strict Provenance Categorization**: The data acquisition origin is explicitly logged (e.g., physical sensor, deep learning perception, atmospheric reanalysis, or traffic census).
3. **Transparent Missing Data Policy**: Unavailable variables (such as direct subsurface soil moisture or profilometer rutting) are flagged with explicit missingness indicators rather than being silently assumed to be optimal.
4. **Zero Lookahead Leakage**: Cumulative and rolling features utilize lookback windows strictly strictly bounded by $t \le t_{\text{observation}}$.

---

## 2. Complete Mechanism Feature Catalog

| Index | Feature Identifier | Physical Mechanism | Unit | Data Source / Provenance | Missing Data Strategy |
|---|---|---|---|---|---|
| 0 | `current_pci` | Current structural integrity | Index (0–100) | Derived (ASTM D6433 deduct engine) | Required field; baseline anchor |
| 1 | `moisture_rain_daily_mm` | Daily surface water delivery | $mm/\text{day}$ | Open-Meteo ERA5 / EMI station | Backward-fill nearest meteorological station |
| 2 | `moisture_rain_7d_cum_mm` | Short-term moisture exposure | $mm$ | 7-day rolling atmospheric sum | Sum over available days; zero if unobserved |
| 3 | `moisture_rain_30d_cum_mm` | Sub-base moisture saturation | $mm$ | 30-day rolling atmospheric sum | Sum over available days; zero if unobserved |
| 4 | `moisture_rain_60d_cum_mm` | Seasonal deep-subgrade softening | $mm$ | 60-day rolling atmospheric sum | Sum over available days; zero if unobserved |
| 5 | `moisture_rainy_days_30d` | Wet/dry stripping cycles | Days (count) | Threshold count ($\ge 1.0\,mm$) in 30d | Integer count over available days |
| 6 | `moisture_consecutive_wet_days` | Sustained pavement saturation | Days (count) | Run of consecutive days with rain $\ge 1.0\,mm$ | Reset to 0 upon dry day |
| 7 | `traffic_total_vpd` | Mechanical stress frequency | Vehicles/day | ERA census & roadside ANPR proxy | Corridor average baseline (8,500 vpd) |
| 8 | `traffic_truck_vpd` | Commercial heavy freight stress | Trucks/day | ERA commercial traffic classification | Corridor baseline ratio (24% of AADT) |
| 9 | `traffic_truck_ratio_pct` | Damaging commercial axle proportion | % (0–100) | Calculated ($N_{\text{truck}} / N_{\text{total}} \times 100$) | Corridor default (24.0%) |
| 10 | `fatigue_crack_count` | Surface structural discontinuities | Count | YOLOv8 visual defect segmentation | Zero if no cracks detected |
| 11 | `fatigue_crack_density_m_m2` | Structural alligator fatigue severity | $m / m^2$ | ASTM D6433 carriage area integration | Zero if no cracks detected |
| 12 | `cavity_pothole_count` | Localized sub-base punch-through | Count | YOLOv8 visual bounding-box detections | Zero if no potholes detected |
| 13 | `cavity_avg_depth_cm` | Cavity structural penetration | $cm$ | Monocular 3D Shape-from-Shading (SfS) | 0.0 if count=0; median baseline if uncalibrated |
| 14 | `cavity_max_depth_cm` | Subgrade breach severity ($>4.5\,cm$) | $cm$ | Peak depth gradient from 3D SfS surface | 0.0 if count=0; $1.5 \times \text{avg}$ if uncalibrated |
| 15 | `cavity_total_volume_liters` | Aggregate asphalt material loss | Liters | 2D Simpson numerical double integration | 0.0 if count=0; volumetric proxy if uncalibrated |
| 16 | `historical_pci_drop_velocity` | Recent degradation momentum | $\Delta\text{PCI} / 30\,\text{days}$ | Longitudinal difference between surveys | Default prior (-2.5 PCI/month) if 1st survey |
| 17 | `temperature_c` | Thermal expansion & bitumen stiffness | °C | Open-Meteo ERA5 / EMI station | Highland Debre Berhan baseline (16.5°C) |
| 18 | `relative_humidity_pct` | Evaporation & surface drying rate | % | Open-Meteo ERA5 station | Highland seasonal median (65.0%) |
| 19 | `antecedent_precip_index` | Soil moisture retention proxy | $mm$ | Hydrological API decay ($\sum_{i=1}^{30} 0.88^i P_{t-i}$) | Computed recursively over 30d weather series |
| 20 | `traffic_exposure_30d_kveh` | Cumulative 30-day vehicle exposure | Thousand vehicles | Rolling multiplication ($30 \times \text{vpd} / 1000$) | Derived from daily traffic proxy |
| 21 | `drainage_risk_score` | Side ditch obstruction & flow risk | Score (0.0–1.0) | Visual/field survey condition mapping | Explicit neutral prior (0.50) with missing flag |
| 22 | `water_accumulation_risk` | Ponding & hydraulic stripping risk | Score (0.0–1.0) | Hydrological interaction formula | Explicit neutral prior with missing flag |

---

## 3. Physical Deterioration Mechanism Subsystems

### 3.1. Moisture Ingress & Hydraulic Stripping
- **Atmospheric Rainfall vs Subgrade Pore Pressure**:
  Rainfall logged from weather stations measures atmospheric precipitation flux ($mm/\text{day}$). It is not a direct measure of in-situ subgrade suction or water table elevation.
- **Antecedent Precipitation Index (API)**:
  Quantifies residual moisture stored in the granular base course using standard hydrological recession:
  $$\text{API}_t = \sum_{i=1}^{30} k^i \cdot P_{t-i}, \quad k = 0.88$$
  where $P_{t-i}$ is the daily precipitation $i$ days prior to the inspection.
- **Consecutive Wet Days**:
  Captures prolonged saturation where the bituminous emulsion loses adhesion to mineral aggregates under repetitive dynamic hydraulic pulsing.

### 3.2. Heavy Freight Loading Dynamics (Axle Load Proxies)
- **Ethiopian Freight Corridor Context (Route A2)**:
  Route A2 (Addis Ababa – Debre Berhan – Dessie) experiences substantial heavy freight traffic, primarily dual- and tri-axle SinoTruck dumpers and fuel tankers transporting aggregate, cement, and goods.
- **AADT and Commercial Truck Percentage**:
  Commercial truck volume is derived from vehicle classification counts:
  $$\text{Truck Volume (vpd)} = \text{AADT} \times \frac{\text{Heavy Truck Ratio \%}}{100}$$
  *Scientific Qualification*: Commercial truck count functions as an operational proxy for equivalent single axle loads (ESALs); direct dynamic wheel weights require in-pavement Weigh-in-Motion (WIM) sensors.

### 3.3. Fatigue Cracking & Structural Discontinuity
- **Crack Density Computation**:
  Surface cracks detected by YOLOv8 deep learning models are quantified per standard 100-meter corridor segments:
  $$\text{Crack Density } (m/m^2) = \frac{\sum L_{\text{crack}}}{\text{Length} \times \text{Width}} = \frac{N_{\text{crack}} \times 3.5\,m}{100\,m \times 7.0\,m}$$
  where $3.5\,m$ represents the empirical average branch length of an alligator cracking distress cell under ASTM D6433 guidelines.

### 3.4. 3D Cavity Geometry & Volumetric Material Loss
- **Monocular 3D Shape-from-Shading (SfS)**:
  Pothole depth and volume are estimated using photometric intensity gradients:
  $$V = \iint_{\Omega} d(x, y) \, dx \, dy \approx \sum_{i} \sum_{j} \hat{d}_{i,j} \cdot \Delta x \cdot \Delta y$$
  *Scientific Qualification*: SfS yields camera-estimated visual volumes. Millimeter-accurate ground-truth requires physical depth gauge rulers or 3D LiDAR scanning.
- **Sub-Base Breach Flag**:
  Potholes exceeding $4.5\,cm$ in maximum depth have penetrated the wearing asphalt course and are excavating granular base materials, causing exponential acceleration in surrounding asphalt collapse.

### 3.5. Drainage Obstruction & Crossfall Vulnerability
- **Drainage Risk Score ($S_{\text{drain}}$)**:
  Field inspection states are mapped to standardized risk scores:
  - `ADEQUATE` / `GOOD`: $0.1$
  - `FAIR`: $0.4$
  - `POOR`: $0.8$
  - `BLOCKED`: $1.0$
  - `UNAVAILABLE / UNKNOWN`: $0.5$ (neutral uncertainty prior with `drainage_slope_missing_flag = 1`).
- **Water Accumulation (Ponding) Risk**:
  Pavement crossfall slope ($\theta$) accelerates or retards runoff:
  $$R_{\text{ponding}} = \min\left(1.0, \, \frac{P_{\text{daily}}}{50.0} \cdot S_{\text{drain}} \cdot \max\left(0.2, \min\left(2.0, \frac{3.0}{\max(0.5, \theta)}\right)\right)\right)$$

---

## 4. Provenance Taxonomy & Traceability

Every feature returned by the feature extractor includes explicit provenance metadata:

```json
{
  "feature_name": "cumulative_rainfall_30d_mm",
  "value": 142.6,
  "unit": "mm",
  "source": "WEATHER_REANALYSIS",
  "data_quality_tag": "MEASURED_ATMOSPHERIC",
  "calculation_method": "30-day rolling summation of Open-Meteo ERA5 precipitation",
  "physical_qualification": "Atmospheric surface precipitation flux; does NOT represent direct subgrade water pore pressure."
}
```

**Provenance Categories**:
- `MEASURED`: Physical field sensor, calibrated ruler, or rain gauge.
- `CAMERA_DETECTED`: YOLOv8 visual defect segmentation bounding boxes.
- `ESTIMATED_SFS`: Monocular 3D Shape-from-Shading numerical integration.
- `WEATHER_REANALYSIS`: Open-Meteo / ERA5 reanalysis and national meteorological archives.
- `TRAFFIC_PROXY`: Roadside camera ANPR counts and corridor census estimates.
- `DERIVED_FORMULA`: Mathematical combinations of primary physical features.
- `MISSING`: Unobserved variable handled via explicit imputation or neutral prior.
- `UNAVAILABLE`: Direct sensor absent in current deployment; requires future specialized instrumentation.
