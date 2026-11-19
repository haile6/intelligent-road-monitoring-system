"""
RoadEye™ Transparent Engineering Maintenance Cost Estimator
===========================================================
Replaces arbitrary heuristic cost multipliers with a transparent,
itemized Bill of Quantities (BOQ) maintenance cost estimation engine.

Authority Reference:
- Ethiopian Roads Administration (ERA) Standard Technical Specifications (2013/2022)
- ERA Schedule of Maintenance Rates & Price Escalation Indices (Route A2 Corridor)

Key Principles:
1. Costs are computed from physical defect dimensions, required repair depth,
   and standard civil engineering bills of quantities.
2. Separates estimated engineering repair costs from actual recorded historical contractor bills.
3. Completely eliminates arbitrary cost multipliers (e.g., no ad-hoc 4.2x cost scaling).
4. Provides full itemization: Materials, Labor, Equipment, Traffic Management.
"""

import math
from typing import Dict, List, Tuple, Any, Optional


class MaintenanceUnitRatesERA:
    """
    Standard unit rates for Ethiopian road maintenance (ETB, 2026 Price Index).
    Route A2 · Addis Ababa - Debre Berhan Highway corridor.
    """
    # Materials (ETB per unit)
    HOT_MIX_ASPHALT_PER_KG = 24.50         # Dense Bituminous Macadam / Asphaltic Concrete (~2,350 kg/m3)
    COLD_PATCH_ASPHALT_PER_KG = 18.00      # Cold Pre-mixed Bituminous Patching
    BITUMEN_EMULSION_TACK_PER_LITER = 85.0 # Cationic Rapid Setting (CRS-1) Bitumen Emulsion
    CRUSHED_AGGREGATE_BASE_PER_M3 = 1450.0 # Class 1 Crushed Stone Sub-Base Material
    CRACK_SEALANT_PER_METER = 48.0         # Hot-poured elastomeric polymer crack sealant

    # Labor (ETB per person-hour)
    SKILLED_ASPHALT_TECHNICIAN_PER_HR = 185.0
    ROLLER_COMPACTOR_OPERATOR_PER_HR = 210.0
    GENERAL_MAINTENANCE_LABOR_PER_HR = 95.0

    # Equipment & Plant (ETB per operating hour)
    ASPHALT_SAW_CUTTER_PER_HR = 350.0
    VIBRATORY_PLATE_COMPACTOR_PER_HR = 420.0
    DUMP_TRUCK_TRANSPORT_PER_HR = 850.0

    # Traffic Safety & Overhead (ETB per work zone setup)
    TRAFFIC_SAFETY_SIGNS_AND_CONES = 1200.0
    FLAGGER_CREW_PER_LOCATION = 1500.0

    # Currency exchange benchmark
    ETB_TO_USD_RATE = 128.0


class MaintenanceCostEstimator:
    """
    Computes transparent engineering maintenance budgets for individual defects
    and entire road corridors based on civil engineering physical requirements.
    """

    @classmethod
    def estimate_defect_repair_cost(cls,
                                    damage_type: str,
                                    width_cm: float,
                                    depth_cm: float,
                                    volume_liters: float,
                                    risk_level: str = "CAUTION") -> Dict[str, Any]:
        """
        Compute transparent itemized repair cost for a single surface defect.
        """
        dtype = (damage_type or "pothole").lower()
        is_pothole = "pothole" in dtype
        w_m = max(0.1, float(width_cm) / 100.0)
        d_cm = max(0.5, float(depth_cm))
        vol_m3 = max(0.001, float(volume_liters) / 1000.0)

        # Work area: add 0.15m rectangular saw-cut margin around perimeter for clean edges
        patch_area_m2 = round(max(0.05, (w_m + 0.30) * (w_m + 0.30)), 3)

        if not is_pothole:
            # Type 1: Crack Sealing (Hairline, Longitudinal, or Transverse)
            linear_m = max(1.0, w_m * 3.5)
            mat_sealant = linear_m * MaintenanceUnitRatesERA.CRACK_SEALANT_PER_METER
            mat_tack = 0.0
            mat_asphalt = 0.0
            mat_base = 0.0

            labor_hrs = max(0.5, linear_m * 0.12)
            labor_cost = labor_hrs * MaintenanceUnitRatesERA.SKILLED_ASPHALT_TECHNICIAN_PER_HR

            equip_cost = labor_hrs * MaintenanceUnitRatesERA.ASPHALT_SAW_CUTTER_PER_HR * 0.3
            safety_cost = 450.0  # Mobile work zone setup

            total_materials = round(mat_sealant + mat_tack + mat_asphalt + mat_base, 2)
            total_labor = round(labor_cost, 2)
            total_equipment = round(equip_cost, 2)
            total_safety = round(safety_cost, 2)
            repair_type = "Hot-Poured Elastomeric Crack Sealing"
            intervention_level = "PREVENTIVE_SURFACE_SEAL"

        elif d_cm < 4.5:
            # Type 2: Shallow Pothole Surface Patching (Depth < 4.5 cm)
            # Replaces only the wearing course layer
            asphalt_vol_m3 = patch_area_m2 * (d_cm / 100.0)
            asphalt_kg = asphalt_vol_m3 * 2350.0  # Density 2.35 t/m3
            mat_asphalt = asphalt_kg * MaintenanceUnitRatesERA.COLD_PATCH_ASPHALT_PER_KG
            mat_tack = (patch_area_m2 * 0.7) * MaintenanceUnitRatesERA.BITUMEN_EMULSION_TACK_PER_LITER
            mat_sealant = 0.0
            mat_base = 0.0

            labor_hrs = 1.5
            labor_cost = (
                labor_hrs * MaintenanceUnitRatesERA.SKILLED_ASPHALT_TECHNICIAN_PER_HR +
                labor_hrs * MaintenanceUnitRatesERA.ROLLER_COMPACTOR_OPERATOR_PER_HR +
                labor_hrs * MaintenanceUnitRatesERA.GENERAL_MAINTENANCE_LABOR_PER_HR
            )

            equip_hrs = 1.0
            equip_cost = (
                equip_hrs * MaintenanceUnitRatesERA.ASPHALT_SAW_CUTTER_PER_HR +
                equip_hrs * MaintenanceUnitRatesERA.VIBRATORY_PLATE_COMPACTOR_PER_HR
            )
            safety_cost = MaintenanceUnitRatesERA.TRAFFIC_SAFETY_SIGNS_AND_CONES

            total_materials = round(mat_asphalt + mat_tack + mat_sealant + mat_base, 2)
            total_labor = round(labor_cost, 2)
            total_equipment = round(equip_cost, 2)
            total_safety = round(safety_cost, 2)
            repair_type = "Cold-Mix Asphalt Surface Infill & Compaction"
            intervention_level = "INTERMEDIATE_SURFACE_PATCH"

        else:
            # Type 3: Deep Base Intrusion (Depth >= 4.5 cm)
            # Requires saw-cut, sub-base excavation, Class 1 aggregate infill, and HMA wearing course
            base_depth_m = max(0.10, (d_cm - 4.0) / 100.0)
            base_vol_m3 = patch_area_m2 * base_depth_m
            mat_base = base_vol_m3 * MaintenanceUnitRatesERA.CRUSHED_AGGREGATE_BASE_PER_M3

            wearing_depth_m = 0.05
            wearing_vol_m3 = patch_area_m2 * wearing_depth_m
            asphalt_kg = wearing_vol_m3 * 2350.0
            mat_asphalt = asphalt_kg * MaintenanceUnitRatesERA.HOT_MIX_ASPHALT_PER_KG
            mat_tack = (patch_area_m2 * 1.0) * MaintenanceUnitRatesERA.BITUMEN_EMULSION_TACK_PER_LITER
            mat_sealant = 0.0

            labor_hrs = 3.0
            labor_cost = (
                labor_hrs * MaintenanceUnitRatesERA.SKILLED_ASPHALT_TECHNICIAN_PER_HR * 2 +
                labor_hrs * MaintenanceUnitRatesERA.ROLLER_COMPACTOR_OPERATOR_PER_HR +
                labor_hrs * MaintenanceUnitRatesERA.GENERAL_MAINTENANCE_LABOR_PER_HR * 2
            )

            equip_hrs = 2.5
            equip_cost = (
                equip_hrs * MaintenanceUnitRatesERA.ASPHALT_SAW_CUTTER_PER_HR +
                equip_hrs * MaintenanceUnitRatesERA.VIBRATORY_PLATE_COMPACTOR_PER_HR +
                equip_hrs * MaintenanceUnitRatesERA.DUMP_TRUCK_TRANSPORT_PER_HR
            )
            safety_cost = (
                MaintenanceUnitRatesERA.TRAFFIC_SAFETY_SIGNS_AND_CONES +
                MaintenanceUnitRatesERA.FLAGGER_CREW_PER_LOCATION
            )

            total_materials = round(mat_base + mat_asphalt + mat_tack + mat_sealant, 2)
            total_labor = round(labor_cost, 2)
            total_equipment = round(equip_cost, 2)
            total_safety = round(safety_cost, 2)
            repair_type = "Deep Structural Patching & Sub-Base Recompaction"
            intervention_level = "EMERGENCY_DEEP_RECONSTRUCTION"

        total_direct_cost = total_materials + total_labor + total_equipment + total_safety
        # 10% municipal contingency for transport and logistics on Route A2
        contingency = round(total_direct_cost * 0.10, 2)
        total_estimated_cost_etb = round(total_direct_cost + contingency, 2)

        return {
            "repair_type": repair_type,
            "intervention_level": intervention_level,
            "patch_area_m2": patch_area_m2,
            "itemized_bill_of_quantities_etb": {
                "materials_cost": total_materials,
                "labor_cost": total_labor,
                "equipment_rental_cost": total_equipment,
                "traffic_control_and_safety_cost": total_safety,
                "logistics_contingency_10pct": contingency
            },
            "total_estimated_repair_cost_etb": total_estimated_cost_etb,
            "total_estimated_repair_cost_usd": round(total_estimated_cost_etb / MaintenanceUnitRatesERA.ETB_TO_USD_RATE, 2),
            "assumptions": {
                "specification_standard": "ERA Standard Technical Specifications Item 4200/5100",
                "saw_cut_margin_m": 0.15,
                "bituminous_density_kg_m3": 2350.0,
                "contingency_rate_pct": 10.0,
                "arbitrary_multiplier_used": False
            }
        }

    @classmethod
    def estimate_segment_maintenance_budget(cls,
                                            pothole_count: int,
                                            avg_pothole_depth_cm: float,
                                            crack_count: int,
                                            current_pci: float,
                                            projected_pci_30d: float) -> Dict[str, Any]:
        """
        Compute transparent maintenance budget for an entire 100m road segment.
        """
        potholes = max(0, int(pothole_count))
        cracks = max(0, int(crack_count))
        avg_d = max(0.0, float(avg_pothole_depth_cm))

        total_cost_etb = 0.0
        itemized_materials = 0.0
        itemized_labor = 0.0
        itemized_equipment = 0.0
        itemized_safety = 0.0

        # Pothole repair costs
        for _ in range(potholes):
            cost_res = cls.estimate_defect_repair_cost(
                damage_type="pothole",
                width_cm=45.0,
                depth_cm=avg_d if avg_d > 0 else 4.8,
                volume_liters=avg_d * 2.8,
                risk_level="CRITICAL" if avg_d >= 4.5 else "CAUTION"
            )
            boq = cost_res["itemized_bill_of_quantities_etb"]
            itemized_materials += boq["materials_cost"]
            itemized_labor += boq["labor_cost"]
            itemized_equipment += boq["equipment_rental_cost"]
            itemized_safety += boq["traffic_control_and_safety_cost"]
            total_cost_etb += cost_res["total_estimated_repair_cost_etb"]

        # Crack sealing costs
        for _ in range(cracks):
            cost_res = cls.estimate_defect_repair_cost(
                damage_type="crack",
                width_cm=12.0,
                depth_cm=1.2,
                volume_liters=0.5,
                risk_level="WARNING"
            )
            boq = cost_res["itemized_bill_of_quantities_etb"]
            itemized_materials += boq["materials_cost"]
            itemized_labor += boq["labor_cost"]
            itemized_equipment += boq["equipment_rental_cost"]
            itemized_safety += boq["traffic_control_and_safety_cost"]
            total_cost_etb += cost_res["total_estimated_repair_cost_etb"]

        # Base routine maintenance minimum if road is in fair condition with 0 defects
        if total_cost_etb == 0.0:
            if current_pci < 70.0:
                total_cost_etb = 8500.0  # Preventive shoulder clearing & minor crack sweep
                itemized_labor = 5000.0
                itemized_safety = 3500.0
            else:
                total_cost_etb = 0.0

        # Determine maintenance urgency level based on projected condition
        pci_val = float(projected_pci_30d) if projected_pci_30d is not None else float(current_pci)
        if pci_val < 50.0 or potholes >= 3:
            urgency_level = 1
            urgency_label = "Emergency Structural Intervention"
            urgency_color = "#ef4444"
            recommended_action = "Deep asphalt milling & aggregate base re-compaction"
        elif pci_val < 70.0 or potholes >= 1 or cracks >= 2:
            urgency_level = 2
            urgency_label = "High Priority Preventive Maintenance"
            urgency_color = "#f59e0b"
            recommended_action = "Cold-mix asphalt patching & elastomeric crack sealing"
        else:
            urgency_level = 3
            urgency_label = "Routine Scheduled Monitoring"
            urgency_color = "#10b981"
            recommended_action = "Routine road shoulder sweep & surface distress monitoring"

        return {
            "total_estimated_budget_etb": round(total_cost_etb, 2),
            "estimated_repair_cost_etb": round(total_cost_etb, 2),
            "total_estimated_budget_usd": round(total_cost_etb / MaintenanceUnitRatesERA.ETB_TO_USD_RATE, 2),
            "urgency_level": urgency_level,
            "urgency_label": urgency_label,
            "urgency": urgency_label,
            "color": urgency_color,
            "recommended_action": recommended_action,
            "cost_breakdown": {
                "materials_etb": round(itemized_materials, 2),
                "labor_etb": round(itemized_labor, 2),
                "equipment_etb": round(itemized_equipment, 2),
                "traffic_safety_etb": round(itemized_safety, 2)
            },
            "potholes_repaired_count": potholes,
            "cracks_sealed_count": cracks,
            "cost_basis": "ERA 2026 Schedule of Rates (Bill of Quantities Engineering Analysis)"
        }
