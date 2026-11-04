# RoadEye™ — Ethiopian Road Contractors & DLP Warranty System Specification

## 1. Overview & Regulatory Framework
Under the **Ethiopian Roads Administration (ERA)** standard technical specifications and the **FIDIC Conditions of Contract for Construction (Red Book, Clause 11)**, contractors are legally bound to guarantee road infrastructure against structural failures and defects during the **Defect Liability Period (DLP)** (typically 2 to 3 years following substantial completion / provisional handover).

During this period:
- **100% Contractor Liability:** Any identified pavement defects (potholes, severe rutting, fatigue cracking) must be rectified at the contractor's sole expense.
- **Warranty Expiry Transition:** Upon DLP expiration and issuance of the Final Acceptance Certificate, maintenance responsibility transitions to municipal road authorities or ERA district offices.

---

## 2. Pavement Quality Scoring & Liability Logic

### Mathematical Model
RoadEye computes an dynamic quality score $Q \in [0, 100]$ for each road segment based on detected distress density:

$$D_{10} = \frac{N_{\text{incidents}}}{L_{\text{km}}} \times 10$$

Where:
- $N_{\text{incidents}}$ = Total number of computer vision validated defects
- $L_{\text{km}}$ = Corridor length in kilometers
- $D_{10}$ = Standardized defect rate per 10 km

The Quality Score is calculated as:
$$Q = \max\left(10, 100 - (D_{10} \times 3.5) - (N_{\text{high-risk}} \times 2.0)\right)$$

### Quality Grade Matrix
| Grade | Score Range | Color Indicator | Pavement Condition | Action Required |
|:---:|:---:|:---:|:---|:---|
| **A** | $90 - 100$ | 🟢 Green (`#10B981`) | Pristine / Good | Routine inspection |
| **B** | $75 - 89$ | 🔵 Blue (`#3B82F6`) | Satisfactory | Preventative maintenance |
| **C** | $60 - 74$ | 🟡 Yellow (`#F59E0B`) | Fair / Warning | Targeted patching |
| **D** | $< 60$ | 🔴 Red (`#EF4444`) | Severe / Critical | Immediate rectification order |

---

## 3. Financial Liability & Cost Estimation

When a defect is matched to a corridor under active warranty, estimated repair costs are calculated using standard Ethiopian road maintenance unit rates:
- **Pothole Repair (Deep patch asphalt concrete):** $\approx 28,000\text{ ETB per defect}$
- **Alligator Cracking (Milling and overlay):** $\approx 18,000\text{ ETB per incident}$
- **Longitudinal / Transverse Cracks (Crack sealing):** $\approx 12,000\text{ ETB per incident}$

---

## 4. Automated Legal Notice Generation
RoadEye automatically generates formal rectification demand notices complying with ERA FIDIC Clause 11 requirements.

### Notice Payload (`/api/contractors/<id>/claim-notice`)
```json
{
  "reference_no": "ERA-DLP-2026-0035",
  "authority": "Ethiopian Roads Administration & Municipal Infrastructure Division",
  "legal_basis": "ERA Standard Technical Specifications / FIDIC Defects Liability Clause 11",
  "generated_at": "November 04, 2026",
  "contractor": {
    "road_name": "Mehalmeda – Alemketema Asphalt Road",
    "contractor_name": "Gemshu Beyene Construction (GEBECON)",
    "is_under_warranty": true,
    "incident_count": 0,
    "repair_cost_etb": 0,
    "quality_grade": "A"
  }
}
```

---

## 5. Interactive GIS Map Visualization
The GIS Map (`/map`) renders road networks using vector polylines:
- **Green (`#10B981`):** Active warranty road corridors (Contractor Liability).
- **Indigo (`#6366F1`):** Expired warranty road corridors (Municipal Budget).
- **Cyan (`#0284C7`):** Urban OpenStreetMap asphalt segments (e.g. Debre Berhan network).
- **Red (`#EF4444`):** Real-time GPS patrol track.
- **Geotagged Markers:** Damage classification pins with imagery and depth estimates.
