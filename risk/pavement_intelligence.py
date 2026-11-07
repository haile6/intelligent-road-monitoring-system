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
