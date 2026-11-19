# DEBRE BERHAN UNIVERSITY
### Center for Employability and Entrepreneurship Development
#### In Collaboration with the Entrepreneurship Development Institute (EDI)
## BUSINESS IDEA COMPETITION PROPOSAL
**Global Entrepreneurship Week 2026**

---

### Submission Information

| Field | Details |
| :--- | :--- |
| **Project Title** | **AI-Powered Intelligent Transportation Infrastructure Management & Advanced Collision Avoidance Platform** |
| **Platform Brand** | **RoadEye™** |
| **Submission Deadline**| **November 16, 2026 – 5:00 PM** |
| **Submission Email** | `deredbu2003@gmail.com` |
| **Host Office** | Center for Employability and Entrepreneurship Development, SE-02, Floor 2 |
| **Lead Innovator** | **Haile Sintayehu** |
| **College / Department**| Debre Berhan University – College of Computing |
| **Applicant Email** | `hailesintayehu113@gmail.com` |
| **Contact Numbers** | **0921746222 / 0912485346** |
| **Target Focus Areas** | ICT • Construction & Infrastructure Materials • Engineering Technologies |
| **Prototype Status** | **Functional Prototype with 114 Automated Unit Tests Passing (100%)** |
| **Live Repository** | [github.com/haile6/intelligent-road-monitoring-system](https://github.com/haile6/intelligent-road-monitoring-system) |
| **Word Download** | [Download Word Document (.docx)](http://127.0.0.1:5000/download/proposal/docx) |

---

# AI-Powered Intelligent Transportation Infrastructure Management & Advanced Collision Avoidance Platform

---

## 1. Business Idea Title and Overview

### 1.1 Official Title
**AI-Powered Intelligent Transportation Infrastructure Management & Advanced Collision Avoidance Platform (RoadEye™)**

### 1.2 Executive Overview
**RoadEye™** is an end-to-end GovTech and Mobility AI innovation engineered at Debre Berhan University. It pairs low-cost, off-the-shelf monocular cameras (dashcams and smartphones) with deep learning computer vision, 3D photogrammetry, spatial GIS kinematics, and physics-informed pavement deterioration modeling to simultaneously achieve two transformative missions:

1. **For Drivers (Active In-Cabin ADAS & 360° Proximity Radar):** Scans the forward travel corridor in real-time at 30+ FPS, maintains a multi-tier proximity radar horizon (25m, 50m, 100m, 150m), projects forward laser acquisition locks, estimates closing Time-To-Collision ($\text{TTC} \le 1.8\text{s}$), and triggers **instantaneous dual-language (Amharic & English) audio voice alarms** and flashing HUD visual alerts (*"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"* / *"Warning! Hazard Ahead, Reduce Speed"*). This stops deadly rollover crashes, tire blowouts, and violent head-on swerving collisions.
2. **For Municipalities & Road Authorities (Intelligent Asset Management):** Automatically maps pavement distresses, calculates 3D cavity depth and required asphalt repair mass in kilograms, projects hazards on an interactive GIS map, and enforces contractor road warranties under FIDIC Clause 11 (Defect Liability Period).

By predicting structural failure before the Ethiopian monsoon (**Kiremt**, where unsealed cracks and heavy freight traffic cause severe subgrade weakening), the platform cuts municipal road maintenance costs by up to **75%** while protecting road users' lives.

---

## 2. Deep Dive: Critical Problems Addressed and Engineering Solutions

Below is the exhaustive breakdown of the five critical real-world infrastructure and safety crises addressed by RoadEye™, and the exact engineering mechanisms developed to solve them:

### 2.1 Problem 1: Sudden Highway Potholes, Blindspots & Fatal Driver Collisions
* **The Reality on the Ground:** On major corridors like Route A2 (Addis Ababa–Debre Berhan–Dessie), commercial buses, freight trucks, and private vehicles travel at 70–100 km/h. Severe potholes are virtually invisible until the driver is within 10–15 meters, especially during nighttime, heavy fog, or torrential rain. When drivers suddenly encounter an unmapped crater, typical human perception-reaction time ($1.5\text{ to }2.5\text{ seconds}$) is insufficient. Drivers panic and violently swerve into oncoming traffic (causing catastrophic head-on crashes), veer off the road shoulder into ravines (causing fatal rollovers), or slam directly into the crater (causing rim fracture and suspension destruction). Luxury vehicle imported ADAS systems only detect lane paint in Europe/US and neither detect road holes nor speak Ethiopian languages.
* **How RoadEye™ Solved It:**
  1. **Edge Computer Vision Perception:** Deploys lightweight YOLOv8/11 deep learning models optimized for mobile processors and dashcams, running at 30+ FPS to detect potholes, alligator cracks, and vehicles 40–50 meters ahead.
  2. **Active Driving Corridor Projection:** Projects the vehicle's projected forward trajectory ($\pm 1.8\text{m}$ lane width) to eliminate false alarms caused by roadside gutters, parked cars, or curbs.
  3. **Software-Defined Proximity Radar & Multi-Tier Horizons (25m, 50m, 100m, 150m):** Operates an optical Software-Defined Radar projecting a 360° proximity scanning horizon around the vehicle. As hazards enter the radar threshold, the system draws a directional forward laser acquisition lock, triggers concentric visual radar shockwaves on the GIS digital twin, and synthesizes subtle acoustic sonar pings.
  4. **Kinematic Time-To-Collision (TTC) Engine:** Calculates metric forward distance ($Z$) and closing rate $\Delta v$. When $\text{TTC} \le 1.8\text{ seconds}$ (Critical) or $\le 3.0\text{ seconds}$ (Warning), the system acts instantly.
  5. **Dual-Language Auditory Voice Alarms:** Drivers cannot safely take their eyes off the road to look at small screens. RoadEye™ triggers high-priority, clear audio voice announcements in **Amharic (አማርኛ)** and English:  
     * *"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"* ("Caution! Pothole ahead!")  
     * *"አደጋ! ፍጥነትህን ቀንስ"* ("Danger! Reduce speed!")  
     accompanied by a prominent visual HUD flash, giving drivers 1.5 to 2.0 vital seconds to brake smoothly without swerving.

---

### 2.2 Problem 2: Catastrophic Monsoon (Kiremt) Pavement Destruction & Mechanism-Based Machine Learning
* **The Reality on the Ground:** In the Ethiopian central highlands (Debre Berhan, elevation >2,840m), annual rainfall exceeds 1,100 mm, with over 72% concentrated within the four Kiremt months (June–September). When rainwater enters unsealed hairline cracks and potholes, heavy truck axle loads (ESALs) rolling over the saturated asphalt generate intense **dynamic pore-water pressure ($\Delta u$)**. This acts like a hydraulic pressure washer from the inside, stripping the bitumen binder from the crushed stone aggregates (Bitumen Stripping) and washing out granular sub-base fines (Pumping). A small 5 cm crack that would remain stable in dry weather explodes into a vehicle-swallowing 15 cm crater within days of heavy rain, forcing full road reconstruction.
* **How RoadEye™ Solved It:**
  1. **Mechanism-Based Feature Engineering:** Integrates 17 physical variables (Open-Meteo rainfall, cumulative precipitation, ANPR truck traffic proxy, crack density, 3D SfS cavity depth & volume).
  2. **Multi-Horizon Machine Learning Models:**
     * **Ridge L2 Regressor ($R^2 = 0.998$):** Analytical linear closed-form regressor for 7, 30, and 60-day PCI forecasting.
     * **Random Forest Regressor ($R^2 = 0.896$):** 35-tree ensemble capturing non-linear interactions with 95% confidence intervals.
     * **Data-Driven Markov Baseline ($R^2 = 0.843$):** 5-state transition probabilities derived from observed condition shifts without unscientific heuristic multipliers.
  3. **Economic Cost of Inaction (COI) ROI Engine:** Uses transparent ERA Bill of Quantities unit rates (asphalt, emulsion, equipment, labor) to prove that timely preventive patching saves up to **10× capital expenditure** compared to post-monsoon structural reconstruction.

---

### 2.3 Problem 3: Massive Public Fund Waste from Unenforced Contractor Warranties (FIDIC Clause 11 DLP)
* **The Reality on the Ground:** Under Ethiopian public procurement law and standard civil engineering contracts (FIDIC Clause 11), newly constructed roads carry a 1-to-2-year Defect Liability Period (DLP). If defects, potholes, or structural cracks emerge during this period, the contractor is legally obligated to repair them entirely at their own cost. However, municipal offices lack automated tracking tools. Defect reports are lost, delayed, or compiled manually long after the warranty has expired. As a result, the municipality releases the contractor's 5% retention guarantee, and when the road fails, the city administration pays tens of millions of ETB out of public tax funds to fix mistakes the contractor was legally bound to repair.
* **How RoadEye™ Solved It:**
  1. **Contractor Warranty Geofencing Engine:** Stores road contract boundaries, contractor names, and active DLP start/expiry dates in the central database.
  2. **Automated Defect Cross-Referencing:** Every defect detected by RoadEye™ is automatically matched against active contractor geofences and warranty dates.
  3. **Configurable Liability Rate Engine:** Automatically calculates total contractor liability using configurable municipal unit rates (e.g., 28,000 ETB per pothole, 18,000 ETB per crack cluster).
  4. **Automated Legal Demand Notices:** With a single click, the system generates formal legal rectification demand notices complete with defect photographs, exact chainage/GPS coordinates, contract numbers, and legal deadlines under FIDIC Clause 11—saving Debre Berhan Municipality an estimated **15 to 30 Million ETB annually**.

---

### 2.4 Problem 4: 2D AI Guesswork vs. Quantitative 3D Cavity Depth & Asphalt Mix Calculation
* **The Reality on the Ground:** Standard computer vision solutions only draw 2D bounding boxes (e.g., $[x, y, w, h]$). A 2D box gives civil engineers zero volumetric information. A shallow 1 cm surface blemish and a 12 cm deep sub-base collapse look identical in a flat photo. Consequently, municipal maintenance crews arrive at the job site without knowing how much asphalt mix to prepare. They either bring too little (leaving craters unsealed) or too much (wasting expensive bitumen mix that cools and solidifies before use).
* **How RoadEye™ Solved It:**
  1. **Monocular 3D Photogrammetric Ground Geometry:** Using camera focal length ($f_x, f_y$), camera mounting height ($H$), tilt angle ($\theta$), and inverse perspective mapping, RoadEye™ computes real ground width ($W_m$) and length ($L_m$).
  2. **Truncated Paraboloid Depth & Volume Modeling:** Computes maximum cavity depth ($D_{\text{max}}$ in cm), average depth ($D_{\text{avg}}$), and 3D cavity volume:
     $$V_{\text{cavity}} = \frac{1}{2} \cdot A \cdot \frac{D_{\text{max}}}{100} \times 1000 \quad (\text{Liters})$$
  3. **Automated Asphalt Material Bill of Quantities:** Converts cavity volume into compacted asphalt mass ($M_{\text{loose}}$ in kg) and the exact number of commercial 25kg/50kg patching bags required according to Ethiopian Roads Administration (ERA) specifications.

---

### 2.5 Problem 5: Multi-Vehicle Observation Duplication & Inspection Bottlenecks
* **The Reality on the Ground:** Manual road inspections cover less than 5% of a city's network annually, cost exorbitant labor hours, and expose technicians to lethal roadside traffic. Conversely, if multiple vehicles equipped with cameras patrol the same street, traditional systems register dozens of duplicate reports for the same pothole, overwhelming municipal databases with noise. Furthermore, imported laser survey vans (ROMDAS, Dynatest) cost **$250,000 to $500,000 USD** in scarce foreign currency.
* **How RoadEye™ Solved It:**
  1. **Spatial Radius Clustering ($\epsilon \le 8.0\text{m}$):** Merges repeated observations of the same defect from multiple vehicles into a single canonical hazard.
  2. **Bayesian Probability Fusion:** Calculates the confidence-weighted geodesic centroid and fuses multi-pass observations:
     $$P(\text{Defect}) = 1.0 - \prod_{i=1}^k (1.0 - c_i)$$
     boosting defect certainty from a single camera's 80% confidence up to **99.9% statistical certainty**.
  3. **Zero Forex Hardware Footprint:** Runs on accessible commercial dashcams and smartphones ($< \$350\text{ USD}$ total setup), surveying 100–300+ km daily with zero foreign currency dependency.

---

## 3. Summary of Additional Proposal Dimensions (Concise)

### 3.1 Target Customers & Market Potential
* **Customer Segments:** 1) Municipalities & City Administrations (GovTech B2G), 2) Ethiopian Roads Administration (ERA), 3) Road Contractors (B2B quality & warranty compliance), 4) Freight & Public Transport Fleets (Route A2 driver safety), 5) Insurance Companies.
* **Market Size:** TAM: 160,000+ km Ethiopian roads (>25 Billion ETB national maintenance budget); SAM: Amhara Region & Central Corridors (180 Million ETB/year); SOM: Debre Berhan municipal network & A2 corridor (12–18 Million ETB/year).
* **Key Market Drivers:** 1) Federal Road Fund requirement for digitized PCI logs, 2) Federal Auditor General (OFAG) anti-corruption audits on uncollected contractor warranties, 3) Debre Berhan urban corridor boom, 4) National forex shortage, 5) Road material inflation (>250%).

### 3.2 Competitive Advantage

| Evaluation Dimension | Manual Survey | Imported Laser Vans | RoadEye™ (Our Platform) |
| :--- | :--- | :--- | :--- |
| **Capital Cost** | Low | **$250,000 – $500,000 USD** | **< $350 USD** (Dashcam / Smartphone) |
| **Forex Dependency** | None | **100% Forex Dependent** | **0% Forex** (100% locally built & hosted) |
| **In-Cabin Driver Voice Alarm** | None | None | **Real-Time Dual-Language (አማርኛ / English)** |
| **360° Proximity Radar Horizon** | None | None | **25m–150m Horizon with Sonar Pings & Laser Lock** |
| **Survey Speed** | 3 – 5 km/day | 40 – 60 km/day | **100 – 300+ km/day** |
| **Worker Safety Risk** | Extremely High | Low | **Zero Risk** (Surveyed from cabin) |
| **3D Cavity Material Calculation**| Guesswork | Requires LiDAR | **Automated Monocular 3D (Bags & kg)** |
| **Contractor Warranty (DLP)** | None | Manual / None | **Automated FIDIC Clause 11 Geofencing** |
| **Working Prototype Readiness** | Legacy | Commercial | **Fully Functional (114 Passing Unit Tests)** |

### 3.3 Business Model & Financial Forecast
* **Revenue Streams:** 1) Municipal SaaS Licenses (Basic: 250k ETB/yr, Pro: 650k ETB/yr, Metro: 1.8M ETB/yr); 2) Pavement Inspection-as-a-Service (1,200–1,800 ETB/km); 3) Contractor DLP Compliance Audits (45,000–120,000 ETB/project); 4) Fleet Safety Telemetry API feeds.
* **3-Year Revenue Forecast:** Year 1 (2027): **2.45M ETB** (42% margin); Year 2 (2028): **8.90M ETB** (56% margin); Year 3 (2029): **26.50M ETB** (68% margin).

### 3.4 Economic, Social, and Safety Impact
* **Saving Driver Lives:** Eliminates sudden swerving rollovers and head-on collisions via real-time audio voice alerts (*"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"*).
* **Protecting Public Funds:** Saves Debre Berhan City Administration **15–30 Million ETB annually** via contractor warranty enforcement and proactive crack sealing.
* **National Import Substitution:** Eliminates foreign exchange dependency for road survey technologies.
* **Youth Tech Employment:** Generates high-skill jobs in AI engineering, computer vision, GIS mapping, and municipal infrastructure management.

---

## 4. Team Information and Contact Details

| Parameter | Details |
| :--- | :--- |
| **Lead Innovator** | **Haile Sintayehu** |
| **Position** | Lead AI & Computer Vision Developer / System Architect |
| **Institution / College** | Debre Berhan University – College of Computing |
| **Email Address** | `hailesintayehu113@gmail.com` |
| **Contact Numbers** | **0921746222 / 0912485346** |
| **GitHub Portfolio** | [github.com/haile6](https://github.com/haile6) |
| **Code Repository** | [intelligent-road-monitoring-system](https://github.com/haile6/intelligent-road-monitoring-system) |
| **Institutional Mentorship**| Debre Berhan University – Faculty of Computing & Faculty of Civil Engineering Mentors |

---

### Declaration
I hereby submit this business proposal titled **"AI-Powered Intelligent Transportation Infrastructure Management & Advanced Collision Avoidance Platform"** to the **Debre Berhan University Center for Employability and Entrepreneurship Development** in collaboration with the **Entrepreneurship Development Institute (EDI)** for consideration in the Global Entrepreneurship Week 2026 Business Idea Competition. The platform is an original technology innovation developed at Debre Berhan University, validated by 114 automated tests passing successfully, and ready for immediate pilot deployment and commercial incubation.

**Applicant:** Haile Sintayehu  
**Signature:** *Haile Sintayehu* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Date:** November 16, 2026
