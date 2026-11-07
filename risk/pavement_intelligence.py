"""
Pavement Intelligence Engine — ASTM D6433 PCI & World Bank IRI Analytics
==========================================================================
RoadEye™ Enterprise Road Intelligence Platform
Authority: Ethiopian Roads Administration (ERA) / World Bank HDM-4

Standards & Specifications:
- ASTM D6433-20: Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys.
- World Bank HDM-4: International Roughness Index (IRI in m/km) Deterioration Models.
- ERA Standard Technical Specifications & Schedule of Rates (2025/2026).
"""

import math
from typing import Dict, List, Tuple, Any, Optional

# ==============================================================================
# 1. ASTM D6433 PCI CLASSIFICATION & COLOR CODES
# ==============================================================================

PCI_RATING_THRESHOLDS = [
    (85.0, 100.0, "Good", "ጥሩ", "#10b981", "rgba(16, 185, 129, 0.85)", "Routine maintenance / preservation only"),
    (70.0, 84.99, "Satisfactory", "አጥጋቢ", "#84cc16", "rgba(132, 204, 22, 0.85)", "Preventive sealing, isolated crack fill"),
    (55.0, 69.99, "Fair", "መካከለኛ", "#eab308", "rgba(234, 179, 8, 0.85)", "Patching, thin surface treatment"),
    (40.0, 54.99, "Poor", "ዝቅተኛ", "#f97316", "rgba(249, 115, 22, 0.85)", "Cold milling & 50mm Asphalt Concrete overlay"),
    (25.0, 39.99, "Very Poor", "እጅግ ዝቅተኛ", "#ef4444", "rgba(239, 68, 68, 0.85)", "Structural overlay / partial base reconstruction"),
    (10.0, 24.99, "Serious", "አደገኛ", "#b91c1c", "rgba(185, 28, 28, 0.85)", "Major full-depth reconstruction required"),
    (0.0, 9.99, "Failed", "የወደቀ", "#7f1d1d", "rgba(127, 29, 29, 0.90)", "Complete structural failure / road impassable"),
]

IRI_RATING_THRESHOLDS = [
    (0.0, 2.0, "Smooth Highway", "ልስሉስ አውራ ጎዳና", "#10b981", "Optimal ride comfort, minimal vehicle operating cost"),
    (2.0, 3.5, "Good", "ጥሩ", "#84cc16", "Acceptable highway ride quality"),
    (3.5, 5.0, "Fair", "መካከለኛ", "#eab308", "Minor surface irregularities, slight vehicle vibration"),
    (5.0, 7.0, "Poor", "ዝቅተኛ", "#f97316", "Significant roughness, increased vehicle wear and fuel consumption"),
    (7.0, 9.0, "Very Poor", "እጅግ ዝቅተኛ", "#ef4444", "Severe bumpiness, reduced travel speeds required"),
    (9.0, 99.0, "Impassable / Extreme", "አደገኛ", "#7f1d1d", "Severe impact risk, structural breakdown"),
]


class PavementConditionIndex:
    """
    ASTM D6433 Pavement Condition Index (PCI) Calculation Engine.
    
    Computes deduct values for asphalt distresses (Potholes, Alligator Cracking)
    and produces a standardized 0-100 pavement health score.
    """

    @staticmethod
    def get_rating_info(pci_score: float) -> Dict[str, Any]:
        """Map a 0-100 PCI score to standardized ASTM condition classification."""
        score = max(0.0, min(100.0, float(pci_score)))
        for low, high, label_en, label_am, color_hex, color_rgba, action in PCI_RATING_THRESHOLDS:
            if low <= score <= high or (score >= high and high == 100.0):
                return {
                    "pci": round(score, 1),
                    "rating": label_en,
                    "rating_am": label_am,
                    "color": color_hex,
                    "color_rgba": color_rgba,
                    "recommended_action": action
                }
        return {
            "pci": round(score, 1),
            "rating": "Failed",
            "rating_am": "የወደቀ",
            "color": "#7f1d1d",
            "color_rgba": "rgba(127, 29, 29, 0.90)",
            "recommended_action": "Complete structural reconstruction"
        }

    @staticmethod
    def calculate_pothole_deduct(count: int, severity: str, sample_area_m2: float = 700.0) -> float:
        """
        Calculate ASTM D6433 Deduct Value for Potholes based on area density.
        ASTM curves parameterized by power-law fits.
        """
        if count <= 0 or sample_area_m2 <= 0:
            return 0.0

        # Pothole density = count / area (per 100 m2)
        density = (count / sample_area_m2) * 100.0
        sev = (severity or "MEDIUM").upper()

        if "IMMEDIATE" in sev or "HIGH" in sev or "DANGER" in sev:
            # High severity pothole: deep (> 5cm), sharp edges, tire blowout risk
            dv = 38.0 * (density ** 0.45)
            return min(95.0, max(8.0, dv))
        elif "CAUTION" in sev or "MEDIUM" in sev or "WARN" in sev:
            # Medium severity: moderate depth (2.5 - 5cm)
            dv = 24.0 * (density ** 0.40)
            return min(70.0, max(5.0, dv))
        else:
            # Low severity: shallow (< 2.5cm)
            dv = 14.0 * (density ** 0.35)
            return min(45.0, max(2.0, dv))

    @staticmethod
    def calculate_crack_deduct(length_or_count: int, severity: str, sample_area_m2: float = 700.0) -> float:
        """
        Calculate ASTM D6433 Deduct Value for Fatigue / Alligator Cracking.
        """
        if length_or_count <= 0 or sample_area_m2 <= 0:
            return 0.0

        density = (length_or_count / sample_area_m2) * 100.0
        sev = (severity or "MEDIUM").upper()

        if "IMMEDIATE" in sev or "HIGH" in sev or "DANGER" in sev:
            dv = 28.0 * (density ** 0.42)
            return min(85.0, max(6.0, dv))
        elif "CAUTION" in sev or "MEDIUM" in sev or "WARN" in sev:
            dv = 16.0 * (density ** 0.38)
            return min(60.0, max(3.5, dv))
        else:
            dv = 8.5 * (density ** 0.32)
            return min(38.0, max(1.5, dv))

    @classmethod
    def calculate_pci(cls, 
                      potholes: List[Dict[str, Any]], 
                      cracks: List[Dict[str, Any]], 
                      sample_area_m2: float = 700.0) -> Dict[str, Any]:
        """
        Compute ASTM D6433 PCI from individual observed defects in a sample segment.
        
        Args:
            potholes: List of pothole records (each with severity/risk)
            cracks: List of crack records (each with severity/risk)
            sample_area_m2: Surface area of the road segment in m2 (default 700m2 = 100m x 7m)
            
        Returns:
            Dict containing PCI score, condition rating, deduct breakdown, and maintenance action.
        """
        deduct_values: List[float] = []

        # 1. Potholes by severity
        p_high = sum(1 for p in potholes if "IMMEDIATE" in str(p.get("max_risk_level", "")).upper() or "DANGER" in str(p.get("max_risk_level", "")).upper())
        p_med = sum(1 for p in potholes if "CAUTION" in str(p.get("max_risk_level", "")).upper() or "WARN" in str(p.get("max_risk_level", "")).upper())
        p_low = len(potholes) - (p_high + p_med)

        if p_high > 0:
            deduct_values.append(cls.calculate_pothole_deduct(p_high, "HIGH", sample_area_m2))
        if p_med > 0:
            deduct_values.append(cls.calculate_pothole_deduct(p_med, "MEDIUM", sample_area_m2))
        if p_low > 0:
            deduct_values.append(cls.calculate_pothole_deduct(p_low, "LOW", sample_area_m2))

        # 2. Cracks by severity
        c_high = sum(1 for c in cracks if "IMMEDIATE" in str(c.get("max_risk_level", "")).upper() or "DANGER" in str(c.get("max_risk_level", "")).upper())
        c_med = sum(1 for c in cracks if "CAUTION" in str(c.get("max_risk_level", "")).upper() or "WARN" in str(c.get("max_risk_level", "")).upper())
        c_low = len(cracks) - (c_high + c_med)

        if c_high > 0:
            deduct_values.append(cls.calculate_crack_deduct(c_high, "HIGH", sample_area_m2))
        if c_med > 0:
            deduct_values.append(cls.calculate_crack_deduct(c_med, "MEDIUM", sample_area_m2))
        if c_low > 0:
            deduct_values.append(cls.calculate_crack_deduct(c_low, "LOW", sample_area_m2))

        # If zero defects found
        if not deduct_values:
            res = cls.get_rating_info(100.0)
            res.update({
                "total_deduct_value": 0.0,
                "corrected_deduct_value": 0.0,
                "deduct_count": 0,
                "potholes_count": 0,
                "cracks_count": 0
            })
            return res

        # Sort deduct values descending
        deduct_values.sort(reverse=True)
        tdv = sum(deduct_values)

        # ASTM D6433 Corrected Deduct Value (CDV) Iteration
        # q = number of individual deduct values > 5.0 (ASTM D6433 definition)
        q = sum(1 for d in deduct_values if d > 5.0)

        if q <= 1:
            cdv = tdv
        else:
            # Empirical ASTM adjustment for multiple distresses: CDV < TDV due to overlap
            # Standard ASTM curve approximation: CDV = TDV * (1 - (q - 1) * 0.085)
            attenuation = max(0.55, 1.0 - (q - 1) * 0.085)
            cdv = tdv * attenuation

        cdv = min(100.0, max(0.0, cdv))
        pci_score = round(max(0.0, 100.0 - cdv), 1)

        result = cls.get_rating_info(pci_score)
        result.update({
            "total_deduct_value": round(tdv, 2),
            "corrected_deduct_value": round(cdv, 2),
            "deduct_count": len(deduct_values),
            "potholes_count": len(potholes),
            "cracks_count": len(cracks)
        })
        return result


class InternationalRoughnessIndex:
    """
    World Bank HDM-4 International Roughness Index (IRI) Estimation Model.
    
    Relates observed roadway distresses (potholes, structural cracking, rutting)
    to the international roughness index (m/km).
    """

    BASE_HIGHWAY_IRI = 1.8  # Nominal newly paved asphalt roughness (m/km)

    @classmethod
    def get_rating_info(cls, iri_value: float) -> Dict[str, Any]:
        """Map IRI in m/km to standard World Bank road quality categories."""
        val = max(0.0, float(iri_value))
        for low, high, label_en, label_am, color_hex, impact in IRI_RATING_THRESHOLDS:
            if low <= val < high or (val >= high and high == 99.0):
                return {
                    "iri": round(val, 2),
                    "rating": label_en,
                    "rating_am": label_am,
                    "color": color_hex,
                    "impact": impact
                }
        return {
            "iri": round(val, 2),
            "rating": "Extreme",
            "rating_am": "አደገኛ",
            "color": "#7f1d1d",
            "impact": "Severe structural roughness"
        }

    @classmethod
    def estimate_iri(cls, 
                     potholes: List[Dict[str, Any]], 
                     cracks: List[Dict[str, Any]], 
                     segment_length_km: float = 0.25) -> Dict[str, Any]:
        """
        Estimate IRI (m/km) for a road corridor segment based on distress frequency.
        
        HDM-4 formulation:
        IRI = IRI_base + Delta_potholes + Delta_cracking
        """
        seg_km = max(0.05, float(segment_length_km))

        # Each severe pothole adds significant dynamic bump displacement
        p_high = sum(1 for p in potholes if "IMMEDIATE" in str(p.get("max_risk_level", "")).upper() or "DANGER" in str(p.get("max_risk_level", "")).upper())
        p_med = sum(1 for p in potholes if "CAUTION" in str(p.get("max_risk_level", "")).upper() or "WARN" in str(p.get("max_risk_level", "")).upper())
        p_low = len(potholes) - (p_high + p_med)

        c_high = sum(1 for c in cracks if "IMMEDIATE" in str(c.get("max_risk_level", "")).upper() or "DANGER" in str(c.get("max_risk_level", "")).upper())
        c_other = len(cracks) - c_high

        # Pothole contribution normalized to m/km
        # A single critical pothole in 250m causes vehicle impact of ~0.35m vertical bump equivalent
        delta_potholes = ((p_high * 0.32) + (p_med * 0.16) + (p_low * 0.07)) / seg_km

        # Cracking contribution to surface texture roughness
        delta_cracking = ((c_high * 0.12) + (c_other * 0.05)) / seg_km

        estimated_iri = min(16.0, max(cls.BASE_HIGHWAY_IRI, cls.BASE_HIGHWAY_IRI + delta_potholes + delta_cracking))
        
        result = cls.get_rating_info(estimated_iri)
        result.update({
            "base_iri": cls.BASE_HIGHWAY_IRI,
            "delta_potholes": round(delta_potholes, 2),
            "delta_cracking": round(delta_cracking, 2),
            "segment_length_km": seg_km
        })
        return result



# ==============================================================================
# 2. ERA SCHEDULE OF RATES & ASPHALT BUDGET ESTIMATOR
# ==============================================================================

class AsphaltBudgetEstimator:
    """
    Ethiopian Roads Administration (ERA) Standard Specifications & Schedule of Rates
    Asphalt Resurfacing & Rehabilitation Budget Estimator.
    
    Computes required maintenance treatments, Material Quantities (HMA tonnes, 
    Bitumen Emulsion liters), and Bill of Quantities (BoQ) in ETB and USD.
    """

    # Exchange Rate: 1 USD ≈ 128 ETB (Commercial Bank of Ethiopia / National Bank)
    ETB_TO_USD_RATE = 128.0

    # ERA Schedule of Unit Rates (ETB)
    RATE_POTHOLE_COLD_PATCH_PER_M2 = 2850.0      # Cold-mix asphalt patching & compaction
    RATE_CRACK_SEALING_PER_M = 480.0             # Hot-poured elastomeric polymer sealant
    RATE_MILLING_50MM_PER_M2 = 420.0             # Cold planar milling 50mm depth
    RATE_TACK_COAT_PER_M2 = 180.0                # Bitumen emulsion tack coat (0.55 L/m2)
    RATE_ASPHALT_OVERLAY_50MM_PER_M2 = 2650.0    # 50mm Hot Mix Asphalt Wearing Course
    RATE_FULL_RECONSTRUCTION_PER_M2 = 7800.0     # Sub-base, DBM binder 60mm & wearing 50mm

    # Material Constants
    DENSITY_HMA_TONNES_PER_M3 = 2.40             # Compacted Hot Mix Asphalt density
    DENSITY_COLD_PATCH_TONNES_PER_M3 = 2.20      # Cold Mix Asphalt density
    TACK_COAT_APPLICATION_RATE_L_M2 = 0.55       # Liters of emulsion per m2
    STANDARD_LANE_WIDTH_M = 3.50                 # 1 Lane width
    STANDARD_CORRIDOR_WIDTH_M = 7.00             # 2-lane carriageway width

    @classmethod
    def estimate_segment_budget(cls, 
                                pci_score: float, 
                                iri_score: float, 
                                potholes_count: int, 
                                cracks_count: int, 
                                segment_length_m: float = 250.0,
                                carriageway_width_m: float = 7.0) -> Dict[str, Any]:
        """
        Generate detailed Bill of Quantities (BoQ), treatment recommendations,
        and material estimates for a road segment based on PCI and IRI condition.
        """
        seg_len = max(10.0, float(segment_length_m))
        road_width = max(3.0, float(carriageway_width_m))
        total_surface_area_m2 = seg_len * road_width

        # Estimated distressed patch areas
        # Average pothole size: ~0.45 m2 surface area, depth ~0.06m
        pothole_repair_area_m2 = potholes_count * 0.45
        # Average crack length per detection: ~3.5 linear meters
        crack_sealing_meters = cracks_count * 3.5

        boq_items: List[Dict[str, Any]] = []
        treatment_en = ""
        treatment_am = ""
        urgency = "Scheduled Maintenance"
        intervention_level = "PREVENTIVE"

        # Determine Intervention Level based on ASTM D6433 & ERA triggers:
        # 1. PCI < 40 or IRI >= 7.0: Heavy Structural Failure -> Full-Depth Reconstruction
        # 2. 40 <= PCI < 70 or 4.0 <= IRI < 7.0: Functional Impairment -> Mill & 50mm Overlay
        # 3. 70 <= PCI < 85: Satisfactory -> Localized Patching & Crack Sealing
        # 4. PCI >= 85: Good -> Routine Preservation

        if pci_score < 40.0 or iri_score >= 7.0:
            intervention_level = "FULL_RECONSTRUCTION"
            treatment_en = "Full-Depth Carriageway Reconstruction (Crushed Rock Base + DBM + 50mm AC)"
            treatment_am = "ሙሉ የመንገድ መሰረት መልሶ ግንባታ (ሰብ-ቤዝ + ባይንደር + 50ሚሜ አስፋልት)"
            urgency = "Critical (Immediate Priority)"

            # Reconstruction of entire segment carriageway
            qty_reconstruct = total_surface_area_m2
            cost_reconstruct = qty_reconstruct * cls.RATE_FULL_RECONSTRUCTION_PER_M2

            boq_items.append({
                "item_code": "ERA-REC-01",
                "description": "Full-Depth Carriageway Reconstruction (Stabilized base + DBM + AC)",
                "unit": "m²",
                "quantity": round(qty_reconstruct, 1),
                "unit_rate_etb": cls.RATE_FULL_RECONSTRUCTION_PER_M2,
                "total_etb": round(cost_reconstruct, 2)
            })

            # Materials
            asphalt_tonnes = total_surface_area_m2 * 0.11 * cls.DENSITY_HMA_TONNES_PER_M3  # 60mm DBM + 50mm AC
            tack_coat_liters = total_surface_area_m2 * cls.TACK_COAT_APPLICATION_RATE_L_M2 * 2
            cold_patch_tonnes = 0.0

        elif pci_score < 70.0 or iri_score >= 4.0:
            intervention_level = "MILL_AND_OVERLAY"
            treatment_en = "Cold Planar Milling (50mm) + Tack Coat + 50mm Asphalt Concrete Overlay"
            treatment_am = "ሚሊንግ (50ሚሜ) + ታክ ኮት + 50ሚሜ የአስፋልት ኮንክሪት ሽፋን"
            urgency = "High (Quarterly Remediation)"

            cost_milling = total_surface_area_m2 * cls.RATE_MILLING_50MM_PER_M2
            cost_tack = total_surface_area_m2 * cls.RATE_TACK_COAT_PER_M2
            cost_overlay = total_surface_area_m2 * cls.RATE_ASPHALT_OVERLAY_50MM_PER_M2

            boq_items.append({
                "item_code": "ERA-MIL-01",
                "description": "Cold Milling of Existing Asphalt Surfacing (50mm depth)",
                "unit": "m²",
                "quantity": round(total_surface_area_m2, 1),
                "unit_rate_etb": cls.RATE_MILLING_50MM_PER_M2,
                "total_etb": round(cost_milling, 2)
            })
            boq_items.append({
                "item_code": "ERA-TC-02",
                "description": "Bituminous Emulsion Tack Coat (K1-60 @ 0.55 L/m²)",
                "unit": "m²",
                "quantity": round(total_surface_area_m2, 1),
                "unit_rate_etb": cls.RATE_TACK_COAT_PER_M2,
                "total_etb": round(cost_tack, 2)
            })
            boq_items.append({
                "item_code": "ERA-AC-03",
                "description": "50mm Hot-Mix Asphalt Concrete Wearing Course (AC 60/70)",
                "unit": "m²",
                "quantity": round(total_surface_area_m2, 1),
                "unit_rate_etb": cls.RATE_ASPHALT_OVERLAY_50MM_PER_M2,
                "total_etb": round(cost_overlay, 2)
            })

            asphalt_tonnes = total_surface_area_m2 * 0.05 * cls.DENSITY_HMA_TONNES_PER_M3
            tack_coat_liters = total_surface_area_m2 * cls.TACK_COAT_APPLICATION_RATE_L_M2
            cold_patch_tonnes = 0.0

        elif pci_score < 85.0 or potholes_count > 0 or cracks_count > 0:
            intervention_level = "LOCALIZED_REPAIR"
            treatment_en = "Emergency Cold-Mix Pothole Patching & Polymer Crack Sealing"
            treatment_am = "የጉድጓድ አስፋልት ማጣፊያ እና የስንጥቅ ማሸጊያ ጥገና"
            urgency = "Standard (Maintenance Patrol)"

            total_patch_cost = 0.0
            total_crack_cost = 0.0

            if pothole_repair_area_m2 > 0:
                cost_patch = max(pothole_repair_area_m2, 1.0) * cls.RATE_POTHOLE_COLD_PATCH_PER_M2
                total_patch_cost += cost_patch
                boq_items.append({
                    "item_code": "ERA-PTH-01",
                    "description": f"Cold-Mix Asphalt Pothole Patching ({potholes_count} verified potholes)",
                    "unit": "m²",
                    "quantity": round(max(pothole_repair_area_m2, 1.0), 2),
                    "unit_rate_etb": cls.RATE_POTHOLE_COLD_PATCH_PER_M2,
                    "total_etb": round(cost_patch, 2)
                })

            if crack_sealing_meters > 0:
                cost_crack = crack_sealing_meters * cls.RATE_CRACK_SEALING_PER_M
                total_crack_cost += cost_crack
                boq_items.append({
                    "item_code": "ERA-CRK-02",
                    "description": f"Polymer-Modified Bitumen Crack Sealing ({cracks_count} cracks)",
                    "unit": "m",
                    "quantity": round(crack_sealing_meters, 1),
                    "unit_rate_etb": cls.RATE_CRACK_SEALING_PER_M,
                    "total_etb": round(cost_crack, 2)
                })

            asphalt_tonnes = 0.0
            tack_coat_liters = pothole_repair_area_m2 * cls.TACK_COAT_APPLICATION_RATE_L_M2
            cold_patch_tonnes = pothole_repair_area_m2 * 0.06 * cls.DENSITY_COLD_PATCH_TONNES_PER_M3

        else:
            intervention_level = "PRESERVATION"
            treatment_en = "Routine Pavement Preservation & Drainage Clearing"
            treatment_am = "መደበኛ የውሃ ማፍሰሻ ማጽዳት እና ጥበቃ"
            urgency = "Routine (Quarterly Monitoring)"

            # Baseline monitoring fee per 250m
            baseline_fee = 18500.0
            boq_items.append({
                "item_code": "ERA-MON-00",
                "description": "Routine Corridor Inspection, Drainage Flushing & Edge Sweep",
                "unit": "segment",
                "quantity": 1.0,
                "unit_rate_etb": baseline_fee,
                "total_etb": baseline_fee
            })

            asphalt_tonnes = 0.0
            tack_coat_liters = 0.0
            cold_patch_tonnes = 0.0

        # Sum total ETB and USD
        total_etb = sum(item["total_etb"] for item in boq_items)
        total_usd = total_etb / cls.ETB_TO_USD_RATE

        # Recommend construction machinery fleet
        equipment = []
        if intervention_level in ["FULL_RECONSTRUCTION", "MILL_AND_OVERLAY"]:
            equipment = [
                "Wirtgen Cold Milling Machine (2.0m drum)",
                "Vögele Super 1800-3 Asphalt Paver",
                "HAMM HD+ 90i Tandem Vibratory Roller (9 tonnes)",
                "HAMM GRW 280 Pneumatic Tyre Roller",
                "Bitumen Emulsion Pressure Distributor (5000L)",
                "Heavy-Duty Tri-Axle Tipper Trucks (18m³)"
            ]
        elif intervention_level == "LOCALIZED_REPAIR":
            equipment = [
                "Walk-Behind Vibratory Compactor (Bomag 65cm)",
                "Asphalt Road Saw / Concrete Cutter",
                "Hot Air Lance for Crack Cleaning & Drying",
                "Rapid Cold-Mix Asphalt Patch Truck with Emulsion Wand"
            ]
        else:
            equipment = [
                "Inspection Drone / Patrol Vehicle Telemetry Unit",
                "Street Sweeper & Drainage Jetter"
            ]

        return {
            "intervention_level": intervention_level,
            "recommended_treatment": treatment_en,
            "recommended_treatment_am": treatment_am,
            "urgency": urgency,
            "total_surface_area_m2": round(total_surface_area_m2, 1),
            "total_cost_etb": round(total_etb, 2),
            "total_cost_usd": round(total_usd, 2),
            "materials": {
                "hot_mix_asphalt_tonnes": round(asphalt_tonnes, 1),
                "cold_patch_tonnes": round(cold_patch_tonnes, 2),
                "tack_coat_liters": round(tack_coat_liters, 1)
            },
            "equipment_fleet": equipment,
            "boq_items": boq_items
        }
