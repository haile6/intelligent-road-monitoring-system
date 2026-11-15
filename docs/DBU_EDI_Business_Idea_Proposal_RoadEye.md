# DEBRE BERHAN UNIVERSITY
### Center for Employability and Entrepreneurship Development
#### In Collaboration with the Entrepreneurship Development Institute (EDI)
## BUSINESS IDEA COMPETITION PROPOSAL
**Global Entrepreneurship Week 2026**

---

### Submission Information

| Field | Details |
| :--- | :--- |
| **Submission Deadline** | **November 16, 2026 – 5:00 PM** |
| **Submission Email** | `deredbu2003@gmail.com` |
| **Host Office** | Center for Employability and Entrepreneurship Development, SE-02, Floor 2 |
| **Lead Innovator** | **Haile Sintayehu** |
| **College / Department** | Debre Berhan University – College of Computing |
| **Applicant Email** | `hailesintayehu113@gmail.com` |
| **Contact Numbers** | **0921746222 / 0912485346** |
| **Target Focus Areas** | ICT • Construction & Infrastructure Materials • Engineering Technologies |
| **Prototype Status** | **Functional Prototype with 114 Automated Tests Passing (100%)** |
| **Live Repository** | [github.com/haile6/intelligent-road-monitoring-system](https://github.com/haile6/intelligent-road-monitoring-system) |

---

# 🛣️ RoadEye™
### Dual-Action Platform: Real-Time In-Cabin Driver Voice Warning & Autonomous Municipal Infrastructure Intelligence

---

## 1. Business Idea Title and Description

### 1.1 Business Idea Title
**RoadEye™: Edge-AI In-Cabin Driver Safety Voice Alarm & Municipal Road Asset Management Platform**

### 1.2 Executive Overview
**RoadEye™** is an innovative GovTech and Mobility AI platform engineered at Debre Berhan University. It converts affordable vehicle dashcams and smartphones into a **dual-action system**:

1. **For Drivers (Active In-Cabin ADAS Safety):** Continuously scans the driving corridor, calculates closing Time-To-Collision ($\text{TTC} \le 1.8\text{s}$), and sounds **real-time audio voice alarms and visual HUD alerts in both Amharic (አማርኛ) and English** (e.g., *"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"* / *"Warning! Pothole Ahead, Reduce Speed"*). This stops deadly rollover crashes, tire blowouts, and sudden swerving into oncoming traffic.
2. **For Municipalities & Road Authorities (GovTech SaaS):** Automatically maps pavement distresses (potholes, cracks), calculates 3D cavity depth and required asphalt repair kilograms, projects hazards on an interactive GIS map, and enforces contractor road warranties under FIDIC Clause 11 (Defect Liability Period).

By predicting road collapse before the Ethiopian monsoon (**Kiremt**, where damage accelerates by **4.2×**), RoadEye™ cuts municipal road repair costs by up to **75%** while protecting drivers' lives daily.

---

## 2. Problem Statement and Market Need

RoadEye™ directly solves four critical challenges across Ethiopia and the Route A2 corridor:

1. **Lethal Driver Accidents from Sudden Potholes:** Drivers traveling at high speeds suddenly encounter deep potholes, causing violent swerving into oncoming traffic, vehicle rollovers, or tire blowouts. Without an automated in-cabin audio alert, drivers cannot react safely at night or in rain.
2. **Slow, Lethal Manual Road Inspections:** Technicians walk along high-speed corridors with tape measures and clipboards. It is lethally dangerous, covers less than 5% of the network annually, and takes months to compile.
3. **Catastrophic Monsoon (Kiremt) Pavement Destruction:** Water trapped in micro-cracks experiences massive hydrodynamic pumping under heavy freight trucks, accelerating road deterioration by **4.2×** during Kiremt.
4. **Millions in Lost Public Funds from Unenforced Contractor Warranties:** When new roads fail during the 1–2 year Defect Liability Period (DLP under FIDIC Clause 11), municipalities lack digital evidence to hold contractors accountable, forcing taxpayers to pay tens of millions of ETB for contractor faults.
5. **The Foreign Currency Barrier:** Imported laser survey vans (ROMDAS, Dynatest) cost **$250,000–$500,000 USD**—completely unaffordable for Ethiopian municipalities facing acute forex shortages.

---

## 3. Proposed Product and Technical Solution

RoadEye™ delivers a cohesive three-tier architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RoadEye™ Platform Layers                        │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. In-Cabin Driver Safety &    │ • Real-Time 30+ FPS YOLOv8 Detection  │
│    Edge Sensor Layer           │ • Time-to-Collision (TTC) Lookahead   │
│                                │ • Dual-Language Voice Alarms (አማርኛ/EN)│
│                                │ • Works on any Dashcam or Smartphone  │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. Physics & 3D Geometry       │ • Monocular 3D Pothole Depth (cm)     │
│    Engine                      │ • Truncated Paraboloid Volume (Liters)│
│                                │ • Exact Asphalt Mix Mass (kg & Bags)  │
│                                │ • 5-State Markov Degradation Forecast │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Enterprise Web GIS          │ • Interactive Municipal GIS Heatmap   │
│    Municipal Dashboard         │ • ASTM D6433 Pavement Health (PCI)    │
│                                │ • Contractor FIDIC Warranty Tracker   │
│                                │ • Instant Legal Rectification Notices │
└────────────────────────────────┴───────────────────────────────────────┘
```

### Key Technical Capabilities:
* **Active Driver Voice Announcement & HUD Alarm:** Scans within the driving lane corridor and triggers loud in-cabin auditory warnings (e.g., *"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"* / *"Caution: Hazard Ahead"*) and flashing HUD displays when $\text{TTC} \le 1.8\text{ seconds}$, giving drivers critical reaction time.
* **Monocular 3D Cavity & Material Calculation:** Directly computes maximum depth ($D_{\text{max}}$ in cm), volume (Liters), and exact commercial asphalt mix mass (in kg and 25kg bags) compliant with ERA specifications.
* **Multi-Vehicle Fleet Fusion:** Merges duplicate detections from multiple passing vehicles within 8 meters, achieving 99.9% statistical certainty.
* **Contractor Warranty Engine (FIDIC Clause 11):** Automatically cross-references defect GPS coordinates with contractor warranty zones and unit rates (e.g., 28,000 ETB/pothole, 18,000 ETB/crack) to produce formal legal repair demand notices.
* **Rainy Season Cost-of-Inaction (COI) Forecaster:** Uses a Markov chain model to demonstrate that spending 2,850 ETB/m² on proactive crack sealing today saves 7,800 ETB/m² in post-monsoon reconstruction (**8.5×–11.2× ROI**).

---

## 4. Target Customers, Market Potential & Key Drivers

### 4.1 Target Customer Segments
* **Municipalities & City Administrations (GovTech B2G):** Debre Berhan City Administration, Zonal & Regional Urban Infrastructure Bureaus.
* **National & Regional Road Authorities:** Ethiopian Roads Administration (ERA) and Regional Road Bureaus.
* **Road Construction Contractors (B2B):** Defence Construction, Sur Construction, Midroc, Yencomad (for quality audits & warranty defense).
* **Public Transport & Commercial Freight Fleets:** Bus operators, freight trucks on Route A2, and ride-hailing services needing active driver safety alerts.
* **Automotive Insurance Underwriters:** Risk analytics and accident causation verification.

### 4.2 Market Opportunity (TAM, SAM, SOM)
* **Total Addressable Market (TAM):** 160,000+ km of Ethiopian roads; annual national maintenance budget > **25 Billion ETB**.
* **Serviceable Available Market (SAM):** Amhara Region & Central Highway Corridors (18,500+ km) valued at **180 Million ETB/year**.
* **Serviceable Obtainable Market (SOM):** Debre Berhan road network (180+ km) and adjacent A2 corridor, valued at **12–18 Million ETB/year**.

### 4.3 Key Market Drivers (Why Customers Buy Now)
1. **Federal Road Fund Digital Mandate:** The Ministry of Urban and Infrastructure (MUI) and ERA require digitized Pavement Condition Index (PCI) data before approving municipal road budgets.
2. **Strict Public Procurement & Anti-Corruption Audits (OFAG):** The Federal Auditor General holds municipal officials liable if they pay for repairs that contractors should fix for free under warranty.
3. **Urban Corridor Development Boom:** Rapid asphalt expansion in Debre Berhan and regional cities creates billions of ETB in new road assets currently in their 1–2 year warranty periods.
4. **Forex Shortage & Import Substitution:** Municipalities cannot access USD to buy $500,000 laser vans; RoadEye provides a 100% local-currency, low-cost alternative.
5. **Construction Material Inflation (>250%):** Rebuilding roads is unaffordable; cities must prioritize proactive, early-stage crack sealing.

---

## 5. Innovation and Competitive Advantage

| Feature | Manual Survey | Imported Laser Vans | RoadEye™ (Our Innovation) |
| :--- | :--- | :--- | :--- |
| **Capital Cost** | Low | **$250,000 – $500,000 USD** | **< $350 USD** (Standard dashcam / phone) |
| **Forex Dependency** | None | **100% Forex Dependent** | **0% Forex** (100% locally built & hosted) |
| **Driver Voice Alarm** | None | None | **Real-Time Dual-Language (አማርኛ / English)** |
| **Survey Speed** | 3 – 5 km/day | 40 – 60 km/day | **100 – 300+ km/day** |
| **Worker Risk** | Extremely High | Low | **Zero Risk** (Surveyed from inside vehicle) |
| **3D Cavity Material** | Guesswork | Requires LiDAR | **Automated Monocular 3D (Bags & kg)** |
| **Contractor DLP Tracker** | None | Manual / None | **Automated FIDIC Clause 11 Geofencing** |
| **Prototype Readiness** | Legacy | Commercial | **Working Prototype (114 Passing Unit Tests)** |

---

## 6. Business Model and Commercialization Strategy

RoadEye™ operates a hybrid **SaaS (Software-as-a-Service)** and **DaaS (Data-as-a-Service)** revenue model:

### 6.1 Revenue Streams
1. **Municipal SaaS Subscriptions:** Basic (250,000 ETB/yr), Pro/Debre Berhan (650,000 ETB/yr), Enterprise Metro (1,800,000 ETB/yr).
2. **Pavement Inspection-as-a-Service (PIaaS):** Turnkey 3D surveying at 1,200–1,800 ETB per kilometer.
3. **Contractor DLP Compliance Audits:** Pre-handover audit reports at 45,000–120,000 ETB per project.
4. **Fleet Safety & Telemetry Feeds:** Monthly subscription for in-cabin driver alerts and route hazard data.

### 6.2 Three-Year Financial Forecast

| Year | Strategic Milestone | Clients / Projects | Projected Revenue | Net Margin |
| :---: | :--- | :---: | :---: | :---: |
| **2027** | Debre Berhan Pilot & Fleet Launch | 1 Municipality + 4 Contractors | **2,450,000 ETB** | **42%** |
| **2028** | Expansion to 5 Amhara Cities + ERA | 6 Municipalities + 12 Contractors | **8,900,000 ETB** | **56%** |
| **2029** | National Corridor Expansion | 18 Municipalities + 35 Contractors | **26,500,000 ETB** | **68%** |

---

## 7. Expected Economic, Social, and Safety Impact

* **Saving Driver Lives & Preventing Crashes:** Real-time voice announcements (*"ጥንቃቄ! ከፊትህ ጉድጓድ አለ"*) prevent violent swerving, rollovers, and multi-vehicle highway collisions.
* **Massive Municipal Savings:** Proactive crack sealing and contractor warranty enforcement save **15–30 Million ETB** annually for Debre Berhan City Administration.
* **Preserving Foreign Currency:** Replaces scarce foreign currency imports with locally engineered software and standard commercial hardware.
* **Youth Employment & Tech Innovation:** Creates high-skill jobs in AI engineering, computer vision, GIS analysis, and municipal infrastructure management in Debre Berhan.

---

## 8. Team Information and Contact Details

| Parameter | Details |
| :--- | :--- |
| **Lead Innovator** | **Haile Sintayehu** |
| **Position** | Lead AI & Computer Vision Developer / System Architect |
| **Institution / College** | Debre Berhan University – College of Computing |
| **Email Address** | `hailesintayehu113@gmail.com` |
| **Contact Numbers** | **0921746222 / 0912485346** |
| **GitHub Portfolio** | [github.com/haile6](https://github.com/haile6) |
| **Repository** | [haile6/intelligent-road-monitoring-system](https://github.com/haile6/intelligent-road-monitoring-system) |
| **Technical Advisory** | Mentors from College of Computing & Faculty of Civil Engineering |

---

### Declaration
I hereby submit this business proposal to the **Debre Berhan University Center for Employability and Entrepreneurship Development** and the **Entrepreneurship Development Institute (EDI)** for the Global Entrepreneurship Week 2026 Business Idea Competition. RoadEye™ is an original innovation developed at Debre Berhan University, with a functional prototype validated by 114 automated passing tests, ready for pilot deployment and commercial incubation.

**Applicant:** Haile Sintayehu  
**Signature:** *Haile Sintayehu* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Date:** November 16, 2026
