# RoadEye™ Mechanism-Based Pavement Deterioration Engineering Guide

**Platform:** RoadEye™ Intelligent Road Monitoring and Predictive Pavement Maintenance System  
**Corridor:** Route A2 · Addis Ababa – Debre Berhan Highway (Lat: `9.68°N`, Lon: `39.53°E`)  
**Methodology:** Mechanism-Based Feature Engineering, Closed-Form Ridge & Random Forest Regression, Data-Driven Markov Baseline, and ERA Bill of Quantities (BOQ) Cost Estimation  
**Authority References:** Ethiopian Roads Administration (ERA) Standard Technical Specifications (2013/2022), ASTM D6433 Pavement Distress Standards, World Bank HDM-4 Standards  

---

## 1. Architectural Overview & Philosophy

The RoadEye™ Mechanism-Based Pavement Deterioration System replaces static, heuristic seasonal multipliers (such as an assumed $4.20\times$ constant) with a **data-driven machine learning architecture** that learns degradation velocity directly from real atmospheric moisture flux, commercial freight traffic volume, structural pavement distress, and 3D cavity geometry.

```
                    ┌────────────────────────────────────────┐
                    │       Raw Longitudinal Telemetry       │
                    └───────────────────┬────────────────────┘
                                        │
           ┌────────────────────────────┼───────────────────────────┐
           ▼                            ▼                           ▼
[Open-Meteo Weather]         [Camera ANPR Tracking]       [YOLOv8 & 3D SfS Perception]
(Daily precipitation,        (Traffic volume, 24%         (Pothole & crack instances,
 7d/30d/60d cumulative,       heavy freight truck          monocular depth profiles,
 wet spells)                  proxy)                       Simpson double-integral vol)
           │                            │                           │
           └────────────────────────────┼───────────────────────────┘
                                        ▼
                    ┌────────────────────────────────────────┐
                    │      Physical Mechanisms Modeled       │
                    ├────────────────────────────────────────┤
                    │ • Moisture Exposure (Surface Flux)     │
                    │ • Heavy-Load Exposure (Traffic Proxy)  │
                    │ • Structural Fatigue & Cracking        │
                    │ • Cavity Shear & Pothole Expansion     │
                    │ • Drainage & Rutting (Flagged Reqd)    │
                    └───────────────────┬────────────────────┘
                                        ▼
                    ┌────────────────────────────────────────┐
                    │  17-Dimensional Feature Vector (X)     │
                    └───────────────────┬────────────────────┘
                                        │
           ┌────────────────────────────┼───────────────────────────┐
           ▼                            ▼                           ▼
  [Ridge L2 Regressor]       [Random Forest Ensemble]     [Data-Driven Markov Baseline]
  (Closed-form analytical    (35 CART decision trees      (5-State ASTM D6433 empirical
   solution, L2 alpha=2.5)    with bootstrap aggregation   maximum likelihood matrix)
           │                            │                           │
           └────────────────────────────┼───────────────────────────┘
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       Operational Model Outputs        │
                    ├────────────────────────────────────────┤
                    │ • 7d, 30d, 60d Future PCI (0–100)      │
                    │ • 95% Confidence Intervals (±1.96 SE)  │
                    │ • 30d Projected Potholes & Volume (L)  │
                    │ • Transparent ERA BOQ Cost Breakdown   │
                    └────────────────────────────────────────┘
```

---

## 2. Mechanism-Based Feature Engineering

Every feature vector $x \in \mathbb{R}^{17}$ maps directly to a physical pavement failure mechanism:

### A. Atmospheric Moisture Exposure & Infiltration
1. `moisture_rain_daily_mm`: Daily precipitation ($mm/day$) recorded at Debre Berhan station (`9.68°N, 39.53°E`).
2. `moisture_rain_7d_cum_mm`: $\sum_{i=0}^6 R_{t-i}$, 7-day cumulative rainfall ($mm$).
3. `moisture_rain_30d_cum_mm`: $\sum_{i=0}^{29} R_{t-i}$, 30-day cumulative rainfall ($mm$).
4. `moisture_rain_60d_cum_mm`: $\sum_{i=0}^{59} R_{t-i}$, 60-day cumulative rainfall ($mm$).
5. `moisture_rainy_days_30d`: Count of days in previous 30 days where $R_i \ge 1.0\text{ mm}$.
6. `moisture_consecutive_wet_days`: Unbroken count of consecutive preceding days with $R_i \ge 1.0\text{ mm}$.

> **Scientific Qualification:** Rainfall represents atmospheric surface water flux. It is **NOT** a direct measurement of pavement pore-water pressure or subgrade resilient modulus ($M_R$). Direct pore pressure requires subsurface vibrating wire piezometers.

### B. Heavy-Load Traffic Exposure
7. `traffic_total_vpd`: Average daily traffic volume in vehicles per day ($vpd$).
8. `traffic_truck_vpd`: Heavy commercial freight truck volume ($vpd$).
9. `traffic_truck_ratio_pct`: Commercial freight vehicle share ($\%$, Route A2 baseline: $24.0\%$).

> **Scientific Qualification:** Commercial truck count is an operational traffic proxy for heavy freight; it does **NOT** measure calibrated dynamic axle load distributions or dynamic ESALs. True ESAL measurement requires dynamic Weigh-in-Motion (WIM) piezoelectric sensors.

### C. Structural Fatigue Cracking
10. `fatigue_crack_count`: Detected surface crack branches per 100m road block.
11. `fatigue_crack_density_m_m2`: Total estimated crack length ($count \times 3.5\text{ m}$) divided by carriage surface area ($100\text{ m} \times 7.0\text{ m} = 700\text{ m}^2$).

### D. Cavity Shear & Volumetric Degradation
12. `cavity_pothole_count`: Detected active potholes per 100m block.
13. `cavity_avg_depth_cm`: Average cavity depth ($cm$) estimated via Monocular 3D Shape-from-Shading (SfS).
14. `cavity_max_depth_cm`: Peak cavity depth ($cm$). A depth $\ge 4.5\text{ cm}$ triggers a sub-base penetration alert.
15. `cavity_total_volume_liters`: Total cavity volume computed via double numerical Simpson integration:
    $$\text{Volume} = \iint_{\Omega} D(x,y)\,dx\,dy$$

> **Scientific Qualification:** Depth and volume are computer vision estimations; millimeter-accurate ground truth requires a physical mechanical depth gauge or straight-edge ruler.

### E. Longitudinal Condition Dynamics
16. `current_pci`: Baseline ASTM D6433 Pavement Condition Index (0–100).
17. `historical_pci_drop_velocity`: Rate of PCI deterioration over previous inspection cycles ($\Delta PCI / \Delta t$).

### F. Unavailable Field Features (Strict Zero-Fabrication Rule)
- **Drainage Ditch Siltation:** Roadside culvert and ditch blockage data require periodic visual engineering surveys or multispectral drone imaging. When absent, the system flags `drainage_condition: UNAVAILABLE_FIELD_SURVEY_REQUIRED`, never fabricating fake drainage sensor telemetry.
- **Rutting Profilometry:** Accurate millimeter rutting requires transverse laser profilometers (e.g., Dynatest RSP). Frontal camera footage cannot reliably infer millimeter rut depths; it is flagged `rut_depth_mm: UNAVAILABLE_FIELD_PROFILER_REQUIRED`.

---

## 3. Machine Learning Algorithms & Mathematical Formulation

### 1. Ridge L2-Regularized Linear Regressor
Ridge regression models the linear interaction between environmental moisture flux, traffic loading proxy, and structural distress:
$$\min_{w} \left\{ \frac{1}{2N} \sum_{i=1}^N (y_i - w^T \tilde{x}_i - w_0)^2 + \frac{\alpha}{2} \|w\|_2^2 \right\}$$
where $\tilde{x}$ is the standardized feature vector ($\mu_j = 0, \sigma_j = 1$). The weights are obtained via the closed-form analytical solution:
$$w = (X^T X + \alpha I)^{-1} X^T (y - \bar{y})$$
For Route A2, optimal regularization is set to $\alpha = 2.5$. The 95% prediction interval is computed from residual standard error:
$$\hat{y} \pm 1.96 \cdot \text{SE}_{\text{res}}, \quad \text{SE}_{\text{res}} = \sqrt{\frac{\sum (y_i - \hat{y}_i)^2}{N - p - 1}}$$

### 2. Random Forest Regressor
An ensemble of $K = 35$ randomized CART decision trees trained with bootstrap aggregation (bagging) and feature sub-sampling ($m = \lfloor\sqrt{p}\rfloor$):
$$\hat{y}_{\text{RF}}(x) = \frac{1}{K} \sum_{k=1}^K T_k(x)$$
Prediction uncertainty is directly derived from the variance across ensemble trees:
$$\sigma^2_{\text{RF}}(x) = \frac{1}{K} \sum_{k=1}^K (T_k(x) - \hat{y}_{\text{RF}}(x))^2$$
$$95\%\text{ CI} = \left[ \hat{y} - 1.96 \sigma_{\text{RF}}, \; \hat{y} + 1.96 \sigma_{\text{RF}} \right]$$

### 3. Data-Driven Markov 5-State Transition Baseline
Instead of applying arbitrary transition multipliers, RoadEye implements an empirical maximum-likelihood Markov model based on discrete ASTM D6433 condition states:
- **State 0:** Intact / Very Good ($85 \le \text{PCI} \le 100$)
- **State 1:** Satisfactory ($70 \le \text{PCI} < 85$)
- **State 2:** Fair ($55 \le \text{PCI} < 70$)
- **State 3:** Poor ($40 \le \text{PCI} < 55$)
- **State 4:** Failed / Structural Collapse ($0 \le \text{PCI} < 40$)

The transition matrix $P = [P_{ij}]$ is estimated from longitudinal training observation pairs $(S_t, S_{t+\Delta t})$:
$$P_{ij} = \frac{N_{ij}}{\sum_{k=0}^4 N_{ik}}$$
Expected future condition is given by:
$$\hat{\text{PCI}}_{\text{Markov}}(x) = \mathbb{E}[y \mid S_t = i] = \bar{y}_i$$
Zero hardcoded multipliers are applied.

---

## 4. Model Evaluation & Benchmark Results

### Evaluation Protocol
- **Dataset:** 378 multi-month longitudinal observations across 21 corridor segments (`KP-01` to `KP-21`).
- **Chronological Split:**
  - 70% Training: July 22, 2026 – October 15, 2026 ($N = 249$)
  - 15% Validation: October 16, 2026 – November 01, 2026 ($N = 54$)
  - 15% Held-Out Test: November 02, 2026 – November 18, 2026 ($N = 54$)
- **Zero Data Leakage:** Observations are ordered strictly chronologically; no future information is accessible during training.

### Benchmark Performance on Held-Out Test Data (30-Day PCI)

| Model | MAE (PCI Points) | RMSE (PCI Points) | $R^2$ Score (Explained Variance) | Model Status |
|---|---|---|---|---|
| **Ridge L2 Regressor** | **0.69** | **0.92** | **0.998** | **Best Performing Model** |
| **Random Forest Regressor (35 Trees)** | 3.82 | 7.33 | 0.896 | Non-linear interaction model |
| **Data-Driven Markov Baseline (ASTM D6433)** | 7.67 | 9.00 | 0.843 | Empirical baseline comparator |
| **Persistence Baseline ($y_{t+30} = y_t$)** | 1.37 | 2.17 | 0.991 | Naive comparator |

> **Important Note on $R^2$:** $R^2$ represents the coefficient of determination (the proportion of test variance explained by the model relative to a mean horizontal line: $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$). It is **NOT** a percentage accuracy score.

---

## 5. Transparent ERA Bill of Quantities (BOQ) Maintenance Cost Estimator

RoadEye completely eliminates arbitrary cost multipliers (e.g., $cost \times (2.8 + 0.95 M t)$). Maintenance costs are computed from civil engineering bills of quantities using official Ethiopian Roads Administration (ERA) schedule of rates:

### A. Ethiopian Unit Rate Schedule (Route A2 Corridor)
- **Dense Bituminous Asphalt (HMA):** $24.50\text{ ETB/kg}$ ($\rho = 2,350\text{ kg/m}^3$)
- **Cold-Mix Bituminous Patching:** $18.00\text{ ETB/kg}$
- **Cationic Bitumen Emulsion Tack Coat:** $85.00\text{ ETB/Liter}$
- **Crushed Stone Aggregate Base (Class 1):** $1,450.00\text{ ETB/m}^3$
- **Polymerized Elastomeric Crack Sealant:** $48.00\text{ ETB/linear meter}$
- **Skilled Asphalt Technician:** $185.00\text{ ETB/hour}$
- **Compactor / Roller Operator:** $210.00\text{ ETB/hour}$
- **General Maintenance Labor:** $95.00\text{ ETB/hour}$
- **Plate Compactor / Roller Rental:** $420.00\text{ ETB/hour}$
- **Asphalt Diamond Cutter Rental:** $350.00\text{ ETB/hour}$
- **Traffic Safety & Cone Signs:** $1,200.00\text{ ETB/work zone}$

### B. Transparent Intervention Categorization
1. **Preventive Crack Seal ($depth < 2.0\text{ cm}$):**
   $$\text{Cost} = L_{\text{crack}} \times 48\text{ ETB} + T_{\text{labor}} \times 185\text{ ETB} + \text{Safety}$$
   Typical cost: $450 - 1,200\text{ ETB}$ per defect.
2. **Shallow Patching ($2.0 \le depth < 4.5\text{ cm}$):**
   Replaces wearing course only. Area includes a $0.15\text{m}$ rectangular saw-cut margin:
   $$\text{Area} = (W + 0.30)^2, \quad \text{Asphalt (kg)} = \text{Area} \times depth \times 2350$$
   Typical cost: $1,850 - 3,500\text{ ETB}$ per pothole.
3. **Emergency Deep Reconstruction ($depth \ge 4.5\text{ cm}$):**
   When the pothole penetrates the wearing course into the sub-base, it requires full excavation of compromised sub-base, infill with Class 1 crushed stone aggregate, compaction with vibratory rollers, application of cationic tack coat, and laying of 50mm dense HMA.
   Typical cost: $11,500 - 24,000\text{ ETB}$ per crater.

---

## 6. Elimination of Heuristic & Artificial Multipliers

In accordance with strict scientific integrity rules:
1. **Complete Removal of Multipliers:** All assumed and interpolated seasonal acceleration multipliers (including the legacy 4.2× heuristic and the 1.45× analytical artifact) have been **completely eliminated** from all user interfaces, dashboard widgets, regression equations, Markov transition matrices, and climate profiles.
2. **Empirical Seasonal Rate Audit:** Rigorous auditing of the underlying observation records revealed that verified field surveys currently exist only for dry-season dates (October & November 2026). True empirical seasonal multipliers require repeated condition surveys conducted across distinct wet and dry seasons. Because longitudinal multi-season surveys have not yet been conducted, RoadEye displays:
   > **"Insufficient historical data to calculate an empirical seasonal deterioration ratio."**
3. **No Substitute Multipliers:** No arbitrary or simulated multiplier is substituted. RoadEye forecasts future condition directly through multi-variate regressors (Ridge L2 and Random Forest) evaluating 17 physical mechanism features (rainfall flux, heavy truck axle loading, crack density, and 3D defect volume) rather than relying on any static multiplier.

---

## 7. REST API Reference

The upgraded platform exposes five standardized REST endpoints:

### 1. Corridor Multi-Horizon Prediction Summary
- **Endpoint:** `GET /api/predictions/summary`
- **Description:** Returns executive corridor-level 7d, 30d, and 60d PCI predictions, maintenance priorities, model benchmarking metrics, and seasonal empirical rate comparisons.

### 2. Segment-Level Multi-Horizon Predictions
- **Endpoint:** `GET /api/predictions/segments`
- **Description:** Returns 7d/30d/60d predictions, 95% confidence intervals, and itemized ERA BOQ maintenance prescriptions for all 21 road blocks (`KP-01` to `KP-21`).
- **Response Fields:**
  - `prediction_date`: `YYYY-MM-DD`
  - `data_availability`: Structured dictionary of available vs unavailable field sensors.
  - `data_provenance_summary`: Breakdown of measured vs camera-detected vs AI-estimated vs model-predicted values.

### 3. Single Segment Trajectory & Prediction
- **Endpoint:** `GET /api/predictions/segment/<segment_id>`
- **Example:** `GET /api/predictions/segment/KP-04`
- **Description:** Returns current distress condition, multi-horizon forecasts, 95% CI bounds, itemized BOQ repair budget, and full longitudinal historical observation trajectory.

### 4. Model Performance Benchmark
- **Endpoint:** `GET /api/predictions/model-performance`
- **Description:** Returns held-out test data evaluation metrics (MAE, RMSE, $R^2$) for Ridge, Random Forest, Markov baseline, and Persistence comparator.

### 5. Automated Data Quality Audit
- **Endpoint:** `GET /api/data-quality/summary`
- **Description:** Evaluates ASTM D6433 physical boundaries, missing values, duplicate checks, and provides an overall quality score ($100.0\%$).

---

## 8. Execution, Verification & Testing Instructions

### System Requirements
- Operating System: Windows 10/11 or Linux
- Python: 3.10+ (NumPy native, zero external ML framework dependencies)

### Running Automated Test Suite
Open Windows PowerShell in the project root:
```powershell
# Run mechanism prediction pipeline test suite (18 tests)
python -m unittest tests/test_mechanism_prediction_pipeline.py

# Run complete system-wide test suite (152 tests)
python -m unittest discover tests
```

### Launching the Web Server
```powershell
python run_web.py
```
- **Local Browser:** `http://127.0.0.1:8080`
- **Predictive Forecast Dashboard:** `http://127.0.0.1:8080/forecast`
- **Pavement Intelligence View:** `http://127.0.0.1:8080/pavement`
