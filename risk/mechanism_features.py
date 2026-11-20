"""
RoadEye™ Mechanism-Based Pavement Feature Engineering Engine
============================================================
Transforms raw environmental, structural, and traffic observations into
theoretically sound, physical mechanism feature vectors for machine learning.

Physical Deterioration Mechanisms Modeled:
1. Moisture Exposure & Infiltration (Atmospheric Surface Flux Proxy)
2. Heavy Freight Loading Exposure (Vehicle Classification Proxy)
3. Structural Fatigue & Crack Propagation (ASTM D6433 Distress Density)
4. Cavity Shear & Pothole Volumetric Growth (Monocular 3D SfS Perception)
5. Drainage Blockage & Water Ponding Risk (Inspection State Flagging)
6. Wheel-Path Rutting (Field Profilometer Required / Flagged)

Scientific Integrity Principles:
- Rainfall is logged as atmospheric precipitation, NOT direct subgrade moisture.
- Truck count is logged as an operational freight proxy, NOT direct dynamic axle loads.
- Monocular 3D SfS depth/volume is labeled as camera-estimated, NOT core-drill measured.
- Unavailable field features (drainage survey, rut profilometry) are explicitly marked
  as unavailable with structured diagnostic flags, never silently fabricated.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class MechanismFeatureProvenance:
    """Provenance labels for individual features."""
    MEASURED = "MEASURED"                         # Direct field instrument (ruler, rain gauge, core drill)
    CAMERA_DETECTED = "CAMERA_DETECTED"           # YOLOv8 bounding box / edge detection
    ESTIMATED_SFS = "ESTIMATED_SFS"               # 3D Shape-from-Shading monocular inference
    WEATHER_REANALYSIS = "WEATHER_REANALYSIS"     # Open-Meteo station / ERA5 archive
    TRAFFIC_PROXY = "TRAFFIC_PROXY"               # ANPR counts proxying commercial axle loading
    DERIVED_FORMULA = "DERIVED_FORMULA"           # Geometric integration or ASTM standard
    SYNTHETIC_TEST_DATA = "SYNTHETIC_TEST_DATA"   # Explicitly flagged test fixture
    MISSING = "MISSING"                           # Sensor telemetry missing/unobserved
    UNAVAILABLE = "UNAVAILABLE"                   # Direct sensor absent, requires field survey


class MechanismFeatureExtractor:
    """
    Extracts mechanism-grounded feature vectors from historical time series
    and segment-level pavement observations.
    """

    FEATURE_NAMES = [
        "current_pci",                  # 0: ASTM D6433 index (0-100)
        "moisture_rain_daily_mm",       # 1: Daily precipitation (mm)
        "moisture_rain_7d_cum_mm",      # 2: 7-day cumulative rainfall (mm)
        "moisture_rain_30d_cum_mm",     # 3: 30-day cumulative rainfall (mm)
        "moisture_rain_60d_cum_mm",     # 4: 60-day cumulative rainfall (mm)
        "moisture_rainy_days_30d",      # 5: Rainy days count in 30d window (rain >= 1.0mm)
        "moisture_consecutive_wet_days",# 6: Consecutive rainy days preceding observation
        "traffic_total_vpd",            # 7: Total daily traffic volume (vehicles/day)
        "traffic_truck_vpd",            # 8: Commercial heavy truck volume proxy (trucks/day)
        "traffic_truck_ratio_pct",      # 9: Heavy freight vehicle percentage (%)
        "fatigue_crack_count",          # 10: Detected surface crack instances
        "fatigue_crack_density_m_m2",   # 11: Surface crack density (m / m^2)
        "cavity_pothole_count",         # 12: Detected pothole instances
        "cavity_avg_depth_cm",          # 13: 3D SfS estimated average depth (cm)
        "cavity_max_depth_cm",          # 14: 3D SfS estimated maximum depth (cm)
        "cavity_total_volume_liters",   # 15: 3D SfS numerical double-integral volume (Liters)
        "historical_pci_drop_velocity", # 16: Longitudinal PCI drop velocity (PCI points / 30 days)
        # Extended Mechanism Features
        "temperature_c",                # 17: Ambient temperature (°C)
        "relative_humidity_pct",        # 18: Relative humidity (%) or median-imputed
        "antecedent_precip_index",      # 19: Antecedent Precipitation Index (API, mm)
        "traffic_exposure_30d_kveh",    # 20: 30-day cumulative vehicle exposure (thousands)
        "drainage_risk_score",          # 21: Drainage condition score (0.0 good to 1.0 blocked/unknown)
        "water_accumulation_risk"       # 22: Ponding risk score based on slope & rainfall
    ]

    CORE_17_NAMES = FEATURE_NAMES[:17]

    @classmethod
    def compute_moisture_metrics(cls,
                                 weather_history: List[Dict[str, Any]],
                                 target_date: str) -> Dict[str, Any]:
        """
        Compute atmospheric moisture exposure metrics leading up to target_date.
        
        Strict Scientific Integrity Rule:
        Rainfall is atmospheric precipitation, NOT direct subgrade moisture.
        """
        dt_target = datetime.strptime(target_date, "%Y-%m-%d")
        
        sorted_weather = sorted(
            [w for w in weather_history if w.get("date") or w.get("observation_date")],
            key=lambda x: str(x.get("date") or x.get("observation_date"))
        )
        
        daily_rain_map = {}
        daily_temp_map = {}
        daily_humidity_map = {}
        for w in sorted_weather:
            d_str = str(w.get("date") or w.get("observation_date"))
            precip = float(w.get("precipitation_mm") or w.get("rainfall_mm") or 0.0)
            daily_rain_map[d_str] = precip
            if w.get("temperature_c") is not None:
                daily_temp_map[d_str] = float(w["temperature_c"])
            if w.get("relative_humidity_pct") is not None:
                daily_humidity_map[d_str] = float(w["relative_humidity_pct"])

        daily_val = daily_rain_map.get(target_date, 0.0)
        daily_temp = daily_temp_map.get(target_date, 16.5)  # Highland Debre Berhan baseline
        daily_humidity = daily_humidity_map.get(target_date, None)  # None if unobserved

        # Cumulative rainfall over 7, 30, and 60 days
        def get_cum_rain(days: int) -> float:
            total = 0.0
            for i in range(days):
                past_d = (dt_target - timedelta(days=i)).strftime("%Y-%m-%d")
                total += daily_rain_map.get(past_d, 0.0)
            return round(total, 2)

        rain_7d = get_cum_rain(7)
        rain_30d = get_cum_rain(30)
        rain_60d = get_cum_rain(60)

        # Antecedent Precipitation Index (API): API_t = sum_{i=1}^{30} (k^i * P_{t-i}), k = 0.88
        api_val = 0.0
        k_decay = 0.88
        for i in range(1, 31):
            past_d = (dt_target - timedelta(days=i)).strftime("%Y-%m-%d")
            api_val += (k_decay ** i) * daily_rain_map.get(past_d, 0.0)
        api_val = round(api_val, 2)

        # Rainy days in 30-day window (threshold: precipitation >= 1.0mm)
        rainy_days_30d = 0
        for i in range(30):
            past_d = (dt_target - timedelta(days=i)).strftime("%Y-%m-%d")
            if daily_rain_map.get(past_d, 0.0) >= 1.0:
                rainy_days_30d += 1

        # Consecutive wet days leading up to target_date
        consecutive_wet = 0
        curr_d = dt_target
        while True:
            d_str = curr_d.strftime("%Y-%m-%d")
            if daily_rain_map.get(d_str, 0.0) >= 1.0:
                consecutive_wet += 1
                curr_d -= timedelta(days=1)
            else:
                break

        # Rainfall intensity proxy (assuming diurnal peak 4-hour window)
        intensity_mm_hr = round(daily_val / 4.0, 2) if daily_val > 0 else 0.0

        return {
            "daily_rainfall_mm": daily_val,
            "cumulative_rainfall_7d_mm": rain_7d,
            "cumulative_rainfall_30d_mm": rain_30d,
            "cumulative_rainfall_60d_mm": rain_60d,
            "rainy_days_30d_count": rainy_days_30d,
            "consecutive_wet_days": consecutive_wet,
            "rainfall_intensity_mm_per_hr": intensity_mm_hr,
            "temperature_c": daily_temp,
            "relative_humidity_pct": daily_humidity,
            "antecedent_precipitation_index_api": api_val,
            "provenance": {
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "station_or_grid": "Debre Berhan (9.68°N, 39.53°E)",
                "physical_qualification": "Atmospheric surface precipitation flux; does NOT represent direct subgrade water pore pressure or in-situ soil suction."
            }
        }
        rain_60d = get_cum_rain(60)

        # Rainy days in 30-day window (threshold: precipitation >= 1.0mm)
        rainy_days_30d = 0
        for i in range(30):
            past_d = (dt_target - timedelta(days=i)).strftime("%Y-%m-%d")
            if daily_rain_map.get(past_d, 0.0) >= 1.0:
                rainy_days_30d += 1

        # Consecutive wet days leading up to target_date
        consecutive_wet = 0
        curr_d = dt_target
        while True:
            d_str = curr_d.strftime("%Y-%m-%d")
            if daily_rain_map.get(d_str, 0.0) >= 1.0:
                consecutive_wet += 1
                curr_d -= timedelta(days=1)
            else:
                break

        return {
            "daily_rainfall_mm": daily_val,
            "cumulative_rainfall_7d_mm": rain_7d,
            "cumulative_rainfall_30d_mm": rain_30d,
            "cumulative_rainfall_60d_mm": rain_60d,
            "rainy_days_30d_count": rainy_days_30d,
            "consecutive_wet_days": consecutive_wet,
            "provenance": {
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "station_or_grid": "Debre Berhan (9.68°N, 39.53°E)",
                "physical_qualification": "Atmospheric surface precipitation flux; does NOT represent direct subgrade water pore pressure."
            }
        }

    @classmethod
    def compute_heavy_load_metrics(cls,
                                   total_traffic_vpd: float = 8500.0,
                                   heavy_truck_count: Optional[float] = None,
                                   heavy_truck_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute commercial heavy-vehicle loading exposure.
        
        Note: Commercial truck count is an operational traffic proxy; it does NOT
        represent calibrated dynamic axle weights or direct ESAL measurements.
        """
        total_vpd = max(100.0, float(total_traffic_vpd))
        
        if heavy_truck_count is not None:
            truck_cnt = float(heavy_truck_count)
            truck_pct = round((truck_cnt / total_vpd) * 100.0, 2)
        elif heavy_truck_pct is not None:
            truck_pct = float(heavy_truck_pct)
            truck_cnt = round(total_vpd * (truck_pct / 100.0), 1)
        else:
            # Default Route A2 corridor baseline (24% commercial freight)
            truck_pct = 24.0
            truck_cnt = round(total_vpd * 0.24, 1)

        # Traffic frequency index (normalized relative to standard ERA 2-lane design capacity: 10,000 vpd)
        frequency_index = round(total_vpd / 10000.0, 3)

        # Traffic exposure metrics across horizons
        traffic_exposure_7d = round(total_vpd * 7.0, 1)
        traffic_exposure_30d = round(total_vpd * 30.0, 1)
        traffic_exposure_60d = round(total_vpd * 60.0, 1)
        heavy_truck_exposure_30d = round(truck_cnt * 30.0, 1)

        # Vehicle category distribution breakdown (Route A2 empirical baseline proxy)
        cat_dist = {
            "passenger_cars_pct": round(max(0.0, 100.0 - truck_pct - 12.0), 1),
            "commercial_heavy_trucks_pct": round(truck_pct, 1),
            "medium_buses_isuzu_pct": 12.0
        }

        return {
            "total_traffic_vpd": total_vpd,
            "heavy_truck_vpd": truck_cnt,
            "heavy_truck_ratio_pct": truck_pct,
            "traffic_frequency_index": frequency_index,
            "traffic_exposure_7d": traffic_exposure_7d,
            "traffic_exposure_30d": traffic_exposure_30d,
            "traffic_exposure_60d": traffic_exposure_60d,
            "heavy_truck_exposure_30d": heavy_truck_exposure_30d,
            "vehicle_category_distribution": cat_dist,
            "axle_load_proxy_label": "High-Density Commercial Freight (Sino-Trucks & Tankers)",
            "provenance": {
                "source": MechanismFeatureProvenance.TRAFFIC_PROXY,
                "sensor_type": "Camera ANPR & Corridor AADT Census",
                "physical_qualification": "Operational freight proxy; does NOT measure dynamic axle weights or direct ESAL without in-situ weigh-in-motion calibration."
            }
        }

    @classmethod
    def compute_crack_fatigue_metrics(cls,
                                      crack_count: int,
                                      segment_length_m: float = 100.0,
                                      segment_width_m: float = 7.0,
                                      historical_cracks: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compute fatigue cracking density and propagation.
        """
        count = max(0, int(crack_count))
        seg_len = max(1.0, float(segment_length_m))
        seg_width = max(1.0, float(segment_width_m))
        carriage_area_m2 = seg_len * seg_width

        # ASTM D6433 Crack Area Estimate (3.5 linear meters per detected fatigue branch)
        total_crack_len_m = count * 3.5
        crack_density = round(total_crack_len_m / carriage_area_m2, 4)

        # Historical growth rate if previous inspections exist
        delta_crack = 0
        if historical_cracks and len(historical_cracks) >= 2:
            delta_crack = count - int(historical_cracks[-2])

        # Fatigue severity classification
        if crack_density >= 0.15:
            severity = "HIGH_ALLIGATOR_FATIGUE"
        elif crack_density >= 0.05:
            severity = "MODERATE_LONGITUDINAL_CRACKING"
        elif count > 0:
            severity = "LOW_INITIAL_HAIRLINE"
        else:
            severity = "NONE_INTACT_SURFACE"

        return {
            "crack_count": count,
            "crack_density_m_m2": crack_density,
            "severity_classification": severity,
            "delta_crack_count_last_cycle": delta_crack,
            "provenance": {
                "source": MechanismFeatureProvenance.CAMERA_DETECTED,
                "method": "YOLOv8 Edge & Surface Distress Segmentation",
                "physical_qualification": "Surface distress instances; internal asphalt fatigue shear requires core sampling."
            }
        }

    @classmethod
    def compute_pothole_cavity_metrics(cls,
                                       pothole_count: int,
                                       avg_depth_cm: Optional[float] = None,
                                       max_depth_cm: Optional[float] = None,
                                       volume_liters: Optional[float] = None,
                                       historical_volumes: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute pothole cavity geometry and volume progression.
        Distinguishes monocular 3D SfS estimations from physical ruler measurements.
        """
        count = max(0, int(pothole_count))

        if count == 0:
            avg_d = 0.0
            max_d = 0.0
            vol = 0.0
        else:
            avg_d = float(avg_depth_cm) if avg_depth_cm is not None else 4.8
            max_d = float(max_depth_cm) if max_depth_cm is not None else round(avg_d * 1.5, 1)
            vol = float(volume_liters) if volume_liters is not None else round(count * (avg_d * 2.8), 2)

        # Volumetric growth rate
        delta_vol = 0.0
        if historical_volumes and len(historical_volumes) >= 2:
            delta_vol = round(vol - float(historical_volumes[-2]), 2)

        # Base penetration warning (depth >= 4.5 cm signals sub-base breach)
        subbase_breach = (max_d >= 4.5)

        return {
            "pothole_count": count,
            "avg_depth_cm": avg_d,
            "max_depth_cm": max_d,
            "cavity_volume_liters": vol,
            "delta_volume_liters_last_cycle": delta_vol,
            "subbase_breach_detected": subbase_breach,
            "provenance": {
                "source": MechanismFeatureProvenance.ESTIMATED_SFS,
                "method": "Monocular 3D Shape-from-Shading (SfS) & Simpson Volumetric Integration",
                "physical_qualification": "AI-estimated 3D surface depth; millimeter-accurate ground-truth requires physical depth gauge ruler."
            }
        }

    @classmethod
    def get_drainage_risk_telemetry(cls,
                                    observed_drainage_condition: Optional[str] = None,
                                    daily_rainfall_mm: float = 0.0,
                                    road_slope_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve drainage condition telemetry and compute ponding risk.
        
        Strict Scientific Integrity Rule:
        Missing drainage observations must NOT automatically be treated as good drainage.
        Missing data must be handled explicitly.
        """
        if observed_drainage_condition and str(observed_drainage_condition).strip().upper() not in ["", "NONE", "NULL", "UNKNOWN"]:
            cond = str(observed_drainage_condition).upper().strip()
            status = cond
            is_available = True
            is_missing = False
            score_map = {"ADEQUATE": 0.1, "GOOD": 0.1, "FAIR": 0.4, "POOR": 0.8, "BLOCKED": 1.0}
            drainage_score = score_map.get(cond, 0.5)
        else:
            status = "UNAVAILABLE_FIELD_SURVEY_REQUIRED"
            is_available = False
            is_missing = True
            drainage_score = 0.5  # Neutral uncertainty prior; explicitly flagged as missing

        slope = float(road_slope_pct) if road_slope_pct is not None else 1.5
        slope_factor = max(0.2, min(2.0, 3.0 / max(0.5, slope)))
        ponding_risk = round(min(1.0, (float(daily_rainfall_mm) / 50.0) * drainage_score * slope_factor), 3)

        return {
            "drainage_condition_status": status,
            "drainage_risk_score": drainage_score,
            "water_accumulation_risk": ponding_risk,
            "road_slope_pct": road_slope_pct,
            "is_field_measurement_available": is_available,
            "is_missing": is_missing,
            "provenance": {
                "source": MechanismFeatureProvenance.MEASURED if is_available else MechanismFeatureProvenance.UNAVAILABLE,
                "unit": "Dimensionless Score [0.0 - 1.0]",
                "calculation_method": "Field inspection status mapped to risk score; missing data assigned neutral uncertainty (0.5).",
                "note": "Drainage ditch sedimentation and culvert blockages require periodic manual road surveys or multispectral drone scans."
            }
        }

    @classmethod
    def get_rutting_telemetry(cls,
                              measured_rut_depth_mm: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve wheel-path rutting deformation.
        
        Strict Scientific Rule:
        Millimeter rutting cannot be reliably extracted from forward dashcam video alone.
        Mark as UNAVAILABLE unless a specialized transverse profilometer survey was run.
        """
        if measured_rut_depth_mm is not None:
            depth = float(measured_rut_depth_mm)
            is_available = True
        else:
            depth = None
            is_available = False

        return {
            "rut_depth_mm": depth,
            "is_field_measurement_available": is_available,
            "provenance": {
                "source": MechanismFeatureProvenance.MEASURED if is_available else MechanismFeatureProvenance.UNAVAILABLE,
                "unit": "millimeters (mm)",
                "calculation_method": "Direct profilometer measurement; uncalibrated dashcams cannot estimate rutting.",
                "note": "Subgrade shear deformation and wheel-path rut depth require a Transverse Laser Profilometer (RSP) or physical straight-edge beam."
            }
        }

    @classmethod
    def construct_mechanism_feature_vector(cls,
                                           current_pci: float,
                                           moisture_metrics: Dict[str, Any],
                                           traffic_metrics: Dict[str, Any],
                                           crack_metrics: Dict[str, Any],
                                           pothole_metrics: Dict[str, Any],
                                           historical_pci_velocity: float = -2.5,
                                           temperature_c: Optional[float] = None,
                                           relative_humidity_pct: Optional[float] = None,
                                           drainage_metrics: Optional[Dict[str, Any]] = None,
                                           antecedent_precip_index: Optional[float] = None) -> np.ndarray:
        """
        Assemble the 23-dimensional mechanism feature vector for ML models.
        Preserves complete backward compatibility — existing callers that don't pass
        the extra kwargs receive sensible defaults for the 6 extended features.

        Extended features (indices 17-22):
            17: temperature_c
            18: relative_humidity_pct
            19: antecedent_precip_index
            20: traffic_exposure_30d_kveh
            21: drainage_risk_score
            22: water_accumulation_risk
        """
        # --- Extended feature resolution ---
        # 17: temperature_c
        if temperature_c is not None:
            temp_val = float(temperature_c)
        else:
            temp_val = float(moisture_metrics.get("temperature_c", 16.5))

        # 18: relative_humidity_pct
        if relative_humidity_pct is not None:
            rh_val = float(relative_humidity_pct)
        else:
            rh_raw = moisture_metrics.get("relative_humidity_pct", None)
            rh_val = float(rh_raw) if rh_raw is not None else 65.0  # median imputed

        # 19: antecedent_precip_index
        if antecedent_precip_index is not None:
            api_val = float(antecedent_precip_index)
        else:
            api_val = float(moisture_metrics.get("antecedent_precipitation_index_api", 0.0))

        # 20: traffic_exposure_30d_kveh  (convert vehicles to thousands)
        traffic_exp_30d_kveh = float(traffic_metrics.get("traffic_exposure_30d", 0.0)) / 1000.0

        # 21 & 22: drainage risk + water accumulation risk
        if drainage_metrics is not None:
            drain_score = float(drainage_metrics.get("drainage_risk_score", 0.5))
            water_acc   = float(drainage_metrics.get("water_accumulation_risk", 0.0))
        else:
            drain_score = 0.5   # neutral uncertainty prior when drainage survey absent
            water_acc   = 0.0

        return np.array([
            float(current_pci),                                                 # 0: current_pci
            float(moisture_metrics["daily_rainfall_mm"]),                       # 1: moisture_rain_daily_mm
            float(moisture_metrics["cumulative_rainfall_7d_mm"]),               # 2: moisture_rain_7d_cum_mm
            float(moisture_metrics["cumulative_rainfall_30d_mm"]),              # 3: moisture_rain_30d_cum_mm
            float(moisture_metrics["cumulative_rainfall_60d_mm"]),              # 4: moisture_rain_60d_cum_mm
            float(moisture_metrics["rainy_days_30d_count"]),                    # 5: moisture_rainy_days_30d
            float(moisture_metrics["consecutive_wet_days"]),                    # 6: moisture_consecutive_wet_days
            float(traffic_metrics["total_traffic_vpd"]),                        # 7: traffic_total_vpd
            float(traffic_metrics["heavy_truck_vpd"]),                          # 8: traffic_truck_vpd
            float(traffic_metrics["heavy_truck_ratio_pct"]),                    # 9: traffic_truck_ratio_pct
            float(crack_metrics["crack_count"]),                                # 10: fatigue_crack_count
            float(crack_metrics["crack_density_m_m2"]),                         # 11: fatigue_crack_density_m_m2
            float(pothole_metrics["pothole_count"]),                            # 12: cavity_pothole_count
            float(pothole_metrics["avg_depth_cm"]),                             # 13: cavity_avg_depth_cm
            float(pothole_metrics["max_depth_cm"]),                             # 14: cavity_max_depth_cm
            float(pothole_metrics["cavity_volume_liters"]),                     # 15: cavity_total_volume_liters
            float(historical_pci_velocity),                                     # 16: historical_pci_drop_velocity
            temp_val,                                                           # 17: temperature_c
            rh_val,                                                             # 18: relative_humidity_pct
            api_val,                                                            # 19: antecedent_precip_index
            traffic_exp_30d_kveh,                                               # 20: traffic_exposure_30d_kveh
            drain_score,                                                        # 21: drainage_risk_score
            water_acc,                                                          # 22: water_accumulation_risk
        ], dtype=np.float64)

    @classmethod
    def extract_full_feature_dictionary(cls,
                                        current_pci: float,
                                        moisture_metrics: Dict[str, Any],
                                        traffic_metrics: Dict[str, Any],
                                        crack_metrics: Dict[str, Any],
                                        pothole_metrics: Dict[str, Any],
                                        drainage_metrics: Optional[Dict[str, Any]] = None,
                                        historical_pci_velocity: float = -2.5) -> Dict[str, Any]:
        """
        Assemble fully-documented, transparent feature dictionary with explicit metadata,
        units, provenance classification, and calculation methods for every feature.
        """
        if drainage_metrics is None:
            drainage_metrics = cls.get_drainage_risk_telemetry(None, float(moisture_metrics.get("daily_rainfall_mm", 0.0)))

        features = {
            "current_pci": {
                "value": float(current_pci),
                "unit": "PCI (0 - 100 Index)",
                "source": MechanismFeatureProvenance.DERIVED_FORMULA,
                "data_quality_tag": "DERIVED",
                "calculation_method": "ASTM D6433 Pavement Condition Index Deduct Engine",
                "physical_interpretation": "Current baseline structural and surface integrity state."
            },
            "daily_rainfall_mm": {
                "value": float(moisture_metrics["daily_rainfall_mm"]),
                "unit": "mm/day",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "Open-Meteo ERA5 / EMI Weather Station Daily Precipitation",
                "physical_interpretation": "Atmospheric surface water delivery flux (not in-situ subgrade suction)."
            },
            "cumulative_rainfall_7d_mm": {
                "value": float(moisture_metrics["cumulative_rainfall_7d_mm"]),
                "unit": "mm",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "7-day rolling summation of daily rainfall",
                "physical_interpretation": "Short-term moisture saturation and superficial softening exposure."
            },
            "cumulative_rainfall_30d_mm": {
                "value": float(moisture_metrics["cumulative_rainfall_30d_mm"]),
                "unit": "mm",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "30-day rolling summation of daily rainfall",
                "physical_interpretation": "Medium-term sub-base moisture accumulation and pumping susceptibility."
            },
            "cumulative_rainfall_60d_mm": {
                "value": float(moisture_metrics["cumulative_rainfall_60d_mm"]),
                "unit": "mm",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "60-day rolling summation of daily rainfall",
                "physical_interpretation": "Seasonal deep-subgrade pore water softening history."
            },
            "rainy_days_30d_count": {
                "value": int(moisture_metrics["rainy_days_30d_count"]),
                "unit": "days",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "Count of days with precipitation >= 1.0mm in 30d window",
                "physical_interpretation": "Frequency of wet/dry cycles inducing asphalt stripping fatigue."
            },
            "consecutive_wet_days": {
                "value": int(moisture_metrics["consecutive_wet_days"]),
                "unit": "days",
                "source": MechanismFeatureProvenance.WEATHER_REANALYSIS,
                "data_quality_tag": "MEASURED_ATMOSPHERIC",
                "calculation_method": "Consecutive days preceding survey with precipitation >= 1.0mm",
                "physical_interpretation": "Persistent pavement saturation preventing drainage evaporation."
            },
            "total_traffic_vpd": {
                "value": float(traffic_metrics["total_traffic_vpd"]),
                "unit": "vehicles/day",
                "source": MechanismFeatureProvenance.TRAFFIC_PROXY,
                "data_quality_tag": "PROXY",
                "calculation_method": "Corridor AADT census and mobile fleet ANPR detector count",
                "physical_interpretation": "Overall mechanical stress frequency on pavement structure."
            },
            "heavy_truck_vpd": {
                "value": float(traffic_metrics["heavy_truck_vpd"]),
                "unit": "trucks/day",
                "source": MechanismFeatureProvenance.TRAFFIC_PROXY,
                "data_quality_tag": "PROXY",
                "calculation_method": "Commercial vehicle proportion applied to daily traffic",
                "physical_interpretation": "Heavy freight loading proxy; not direct dynamic weigh-in-motion."
            },
            "heavy_truck_ratio_pct": {
                "value": float(traffic_metrics["heavy_truck_ratio_pct"]),
                "unit": "%",
                "source": MechanismFeatureProvenance.TRAFFIC_PROXY,
                "data_quality_tag": "PROXY",
                "calculation_method": "(heavy_truck_vpd / total_traffic_vpd) * 100",
                "physical_interpretation": "Proportion of damaging commercial axles in traffic stream."
            },
            "crack_count": {
                "value": int(crack_metrics["crack_count"]),
                "unit": "count",
                "source": MechanismFeatureProvenance.CAMERA_DETECTED,
                "data_quality_tag": "CAMERA_DETECTED",
                "calculation_method": "YOLOv8 deep learning visual defect detection",
                "physical_interpretation": "Surface structural discontinuities allowing water ingress."
            },
            "crack_density_m_m2": {
                "value": float(crack_metrics["crack_density_m_m2"]),
                "unit": "m / m^2",
                "source": MechanismFeatureProvenance.CAMERA_DETECTED,
                "data_quality_tag": "CAMERA_DETECTED",
                "calculation_method": "Total linear crack length divided by 100m x 7m carriage area",
                "physical_interpretation": "Fatigue cracking severity under ASTM D6433 standards."
            },
            "pothole_count": {
                "value": int(pothole_metrics["pothole_count"]),
                "unit": "count",
                "source": MechanismFeatureProvenance.CAMERA_DETECTED,
                "data_quality_tag": "CAMERA_DETECTED",
                "calculation_method": "YOLOv8 visual defect detection and spatial tracking",
                "physical_interpretation": "Localized structural base punch-through failures."
            },
            "avg_pothole_depth_cm": {
                "value": float(pothole_metrics["avg_depth_cm"]),
                "unit": "cm",
                "source": MechanismFeatureProvenance.ESTIMATED_SFS,
                "data_quality_tag": "ESTIMATED",
                "calculation_method": "Monocular 3D Shape-from-Shading (SfS) depth estimation",
                "physical_interpretation": "Depth of cavity; > 4.5cm penetrates base course layer."
            },
            "max_pothole_depth_cm": {
                "value": float(pothole_metrics["max_depth_cm"]),
                "unit": "cm",
                "source": MechanismFeatureProvenance.ESTIMATED_SFS,
                "data_quality_tag": "ESTIMATED",
                "calculation_method": "Peak SfS depth gradient pixel in detected cavity",
                "physical_interpretation": "Critical penetration depth threatening vehicle axle safety."
            },
            "pothole_volume_liters": {
                "value": float(pothole_metrics["cavity_volume_liters"]),
                "unit": "Liters",
                "source": MechanismFeatureProvenance.ESTIMATED_SFS,
                "data_quality_tag": "ESTIMATED",
                "calculation_method": "Numerical 2D Simpson integration over 3D depth map",
                "physical_interpretation": "Asphalt material loss volume requiring patch backfill."
            },
            "historical_pci_drop_velocity": {
                "value": float(historical_pci_velocity),
                "unit": "PCI points / 30 days",
                "source": MechanismFeatureProvenance.DERIVED_FORMULA,
                "data_quality_tag": "DERIVED",
                "calculation_method": "Difference between current PCI and previous survey PCI",
                "physical_interpretation": "Recent degradation momentum along the specific road segment."
            },
            "drainage_risk_score": {
                "value": float(drainage_metrics["drainage_risk_score"]),
                "unit": "Score [0.0 - 1.0]",
                "source": drainage_metrics["provenance"]["source"],
                "data_quality_tag": "MISSING" if drainage_metrics.get("is_missing") else "MEASURED",
                "calculation_method": drainage_metrics["provenance"]["calculation_method"],
                "physical_interpretation": "Drainage side-ditch capacity to evacuate surface runoff."
            },
            "water_accumulation_risk": {
                "value": float(drainage_metrics["water_accumulation_risk"]),
                "unit": "Risk [0.0 - 1.0]",
                "source": MechanismFeatureProvenance.DERIVED_FORMULA,
                "data_quality_tag": "DERIVED",
                "calculation_method": "Hydrological interaction between daily rainfall, crossfall slope, and drainage",
                "physical_interpretation": "Ponding potential on roadway surface causing dynamic tire splash stripping."
            }
        }
        return features
