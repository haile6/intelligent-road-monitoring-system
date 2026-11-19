"""
RoadEye™ Empirical Pothole Growth Prediction Engine
===================================================
Production-Grade Machine Learning Predictor for Individual Pavement Cavity Growth.
Predicts dimensional and volumetric defect expansion (length, width, depth, volume)
from repeated, calibrated multi-epoch visual sensor measurements.

Scientific Integrity Principles:
- Zero fixed-percentage growth formulas.
- Requires repeated, spatio-temporally matched defect observations across distinct survey dates.
- When repeated measurements are unavailable, returns explicit status:
  'INSUFFICIENT_POTHOLE_HISTORY'
- Never fabricates defect growth measurements or fills unobserved trends with static multipliers.
"""

import os
import math
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "road_monitoring.db")


class PotholeGrowthPredictor:
    """
    Predicts physical cavity geometry evolution:
    - Future length (cm)
    - Future width (cm)
    - Future depth (cm)
    - Future volume (Liters)
    
    Trained strictly on repeated defect observations across multi-epoch surveys.
    """

    MIN_REPEATED_PAIRS_REQUIRED = 20

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.is_trained: bool = False
        self.training_metadata: Dict[str, Any] = {}

    @classmethod
    def audit_defect_history(cls, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit the database to detect whether genuine repeated observations of the same
        physical pothole exist across distinct survey dates.
        """
        path = db_path or DB_PATH
        if not os.path.exists(path):
            return {
                "has_repeated_history": False,
                "repeated_pairs_count": 0,
                "survey_dates": [],
                "reason": "Database file not found."
            }

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query all pothole incidents with GPS coordinates and timestamps
        cursor.execute("""
            SELECT id, damage_type, latitude, longitude, timestamp,
                   depth_max_cm, surface_area_m2, cavity_volume_liters
            FROM incidents
            WHERE LOWER(damage_type) LIKE '%pothole%'
              AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY timestamp ASC
        """)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if len(rows) < 2:
            return {
                "has_repeated_history": False,
                "repeated_pairs_count": 0,
                "survey_dates": [],
                "reason": "Fewer than 2 pothole incidents found in database."
            }

        dates = sorted(list(set(r["timestamp"][:10] for r in rows if r.get("timestamp"))))

        # Check for multi-epoch temporal matching (within 3m radius on different dates)
        pairs_found = 0
        matched_pairs = []

        for i in range(len(rows)):
            d_i = rows[i]["timestamp"][:10] if rows[i].get("timestamp") else ""
            lat_i = float(rows[i]["latitude"])
            lon_i = float(rows[i]["longitude"])

            for j in range(i + 1, len(rows)):
                d_j = rows[j]["timestamp"][:10] if rows[j].get("timestamp") else ""
                if d_i == d_j:
                    continue  # Same day survey, not temporal longitudinal observation

                lat_j = float(rows[j]["latitude"])
                lon_j = float(rows[j]["longitude"])

                # Equirectangular approximation for small distance (meters)
                dlat = (lat_j - lat_i) * 111139.0
                dlon = (lon_j - lon_i) * 111139.0 * math.cos(math.radians(lat_i))
                dist_m = math.sqrt(dlat ** 2 + dlon ** 2)

                if dist_m <= 3.0:  # Matches same physical defect within 3.0 meters GPS error
                    pairs_found += 1
                    matched_pairs.append({
                        "initial_defect_id": rows[i]["id"],
                        "subsequent_defect_id": rows[j]["id"],
                        "distance_m": round(dist_m, 2),
                        "t1": d_i,
                        "t2": d_j
                    })

        is_sufficient = (pairs_found >= cls.MIN_REPEATED_PAIRS_REQUIRED)
        return {
            "has_repeated_history": is_sufficient,
            "repeated_pairs_count": pairs_found,
            "survey_dates": dates,
            "matched_pairs": matched_pairs,
            "required_pairs": cls.MIN_REPEATED_PAIRS_REQUIRED,
            "reason": "Sufficient repeated pairs found." if is_sufficient else (
                f"Found {pairs_found} repeated pairs; minimum {cls.MIN_REPEATED_PAIRS_REQUIRED} required to fit regression models."
            )
        }

    def predict_growth(self,
                       defect_record: Dict[str, Any],
                       days_horizon: int = 30,
                       cumulative_rain_mm: float = 0.0,
                       heavy_truck_vpd: float = 2000.0,
                       allow_synthetic_test: bool = False) -> Dict[str, Any]:
        """
        Predict dimensional and volumetric expansion of a specific pothole.
        
        Strict Scientific Rule:
        If genuine repeated defect measurements are unavailable, return:
        status: 'INSUFFICIENT_POTHOLE_HISTORY'
        Do NOT apply a fixed multiplier or percentage growth.
        """
        audit = self.audit_defect_history()

        if not audit["has_repeated_history"] and not allow_synthetic_test:
            return {
                "status": "INSUFFICIENT_POTHOLE_HISTORY",
                "message": (
                    "Insufficient repeated calibrated defect measurements across survey epochs "
                    "to train an empirical defect growth model. No fixed multiplier is applied."
                ),
                "is_data_sufficient": False,
                "days_horizon": days_horizon,
                "repeated_pairs_found": audit["repeated_pairs_count"],
                "required_pairs_count": self.MIN_REPEATED_PAIRS_REQUIRED,
                "projected_dimensions": None,
                "data_quality_tag": "MISSING_LONGITUDINAL_GROUND_TRUTH"
            }

        # Current observed dimensions
        cur_w = float(defect_record.get("width_cm") or (float(defect_record.get("estimated_width_m") or 0.45) * 100.0))
        cur_l = float(defect_record.get("length_cm") or cur_w)
        cur_d = float(defect_record.get("depth_cm") or (float(defect_record.get("estimated_depth_m") or 0.05) * 100.0))
        cur_vol = float(defect_record.get("volume_liters") or (float(defect_record.get("estimated_volume_m3") or 0.006) * 1000.0))

        # If trained models exist from genuine repeated data or calibrated benchmark
        if self.is_trained:
            feat_vec = np.array([[cur_w, cur_l, cur_d, cur_vol, float(days_horizon), float(cumulative_rain_mm), float(heavy_truck_vpd)]])
            pred_w = round(float(self.models["width"].predict(feat_vec)[0]), 1)
            pred_l = round(float(self.models["length"].predict(feat_vec)[0]), 1)
            pred_d = round(float(self.models["depth"].predict(feat_vec)[0]), 1)
            pred_vol = round(float(self.models["volume"].predict(feat_vec)[0]), 2)
            model_used = "Random Forest Regressor (Trained on Repeated Defect Panel)"
        else:
            # When running in test mode with benchmark data
            return {
                "status": "INSUFFICIENT_POTHOLE_HISTORY",
                "message": "Pothole growth model not yet fitted on repeated panel observations.",
                "is_data_sufficient": False,
                "projected_dimensions": None
            }

        return {
            "status": "TRAINED_MODEL_PREDICTION",
            "is_data_sufficient": True,
            "days_horizon": days_horizon,
            "model_name": model_used,
            "current_dimensions": {
                "width_cm": cur_w,
                "length_cm": cur_l,
                "depth_cm": cur_d,
                "volume_liters": cur_vol
            },
            "projected_dimensions": {
                "width_cm": pred_w,
                "length_cm": pred_l,
                "depth_cm": pred_d,
                "volume_liters": pred_vol
            },
            "dimensional_expansion": {
                "delta_width_cm": round(pred_w - cur_w, 1),
                "delta_depth_cm": round(pred_d - cur_d, 1),
                "delta_volume_liters": round(pred_vol - cur_vol, 2)
            }
        }
