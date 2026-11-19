"""
RoadEye™ Machine Learning Dataset Builder & Integrity Auditor
=============================================================
Constructs time-aligned training datasets for multi-horizon pavement deterioration models.

Scientific Integrity Principles:
- Strictly distinguishes GENUINE field survey panels from SYNTHETIC benchmark test fixtures.
- Synthetic interpolated records are NEVER treated as real training measurements.
- If genuine longitudinal observations across 7d, 30d, 60d horizons are insufficient, returns:
  'INSUFFICIENT_TRAINING_DATA'
- Never fabricates ground truth future PCI targets.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "road_monitoring.db")

from risk.mechanism_features import MechanismFeatureExtractor, MechanismFeatureProvenance


class MLDatasetBuilder:
    """
    Constructs time-aligned feature matrices X and multi-horizon target vectors y
    from longitudinal corridor observation panels.
    """

    FEATURE_NAMES = MechanismFeatureExtractor.FEATURE_NAMES
    MIN_GENUINE_SAMPLES_REQUIRED = 30
    MIN_GENUINE_EPOCHS_REQUIRED = 3

    @classmethod
    def audit_genuine_dataset(cls, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit the database to verify whether genuine longitudinal panel observations
        exist across 7-day, 30-day, and 60-day horizons.
        """
        path = db_path or DB_PATH
        if not os.path.exists(path):
            return {
                "status": "INSUFFICIENT_TRAINING_DATA",
                "is_sufficient": False,
                "reason": "Database file not found.",
                "genuine_dates_count": 0,
                "genuine_survey_dates": [],
                "temporal_pairs_7d": 0,
                "temporal_pairs_30d": 0,
                "temporal_pairs_60d": 0
            }

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check incidents table for distinct survey dates
        cursor.execute("SELECT DISTINCT date(timestamp) as dt FROM incidents WHERE timestamp IS NOT NULL ORDER BY dt ASC")
        incident_dates = [r["dt"] for r in cursor.fetchall() if r["dt"]]

        # Check unified observations table for non-synthetic records
        cursor.execute("""
            SELECT DISTINCT observation_date
            FROM unified_road_observations
            WHERE provenance_label NOT LIKE '%synthetic%'
              AND provenance_label NOT LIKE '%calibrated%'
            ORDER BY observation_date ASC
        """)
        verified_unified_dates = [r["observation_date"] for r in cursor.fetchall() if r["observation_date"]]
        conn.close()

        # Genuine dates are those from actual field survey videos
        genuine_dates = sorted(list(set(incident_dates + verified_unified_dates)))
        n_epochs = len(genuine_dates)

        # Count multi-horizon interval pairs between genuine survey dates
        pairs_7d = 0
        pairs_30d = 0
        pairs_60d = 0

        for i in range(len(genuine_dates)):
            d1 = datetime.strptime(genuine_dates[i], "%Y-%m-%d")
            for j in range(i + 1, len(genuine_dates)):
                d2 = datetime.strptime(genuine_dates[j], "%Y-%m-%d")
                delta_days = (d2 - d1).days
                if 5 <= delta_days <= 10:
                    pairs_7d += 1
                elif 25 <= delta_days <= 35:
                    pairs_30d += 1
                elif 50 <= delta_days <= 70:
                    pairs_60d += 1

        is_sufficient = (
            n_epochs >= cls.MIN_GENUINE_EPOCHS_REQUIRED and
            pairs_7d >= 1 and pairs_30d >= 1 and pairs_60d >= 1
        )

        reason = (
            "Genuine multi-horizon panel dataset is validated."
            if is_sufficient else
            f"Insufficient genuine historical data: Found {n_epochs} survey dates ({', '.join(genuine_dates)}). "
            f"Multi-horizon ML training requires repeated panel surveys across distinct 7d, 30d, and 60d intervals."
        )

        return {
            "status": "DATA_SUFFICIENT" if is_sufficient else "INSUFFICIENT_TRAINING_DATA",
            "is_sufficient": is_sufficient,
            "reason": reason,
            "genuine_dates_count": n_epochs,
            "genuine_survey_dates": genuine_dates,
            "temporal_pairs_7d": pairs_7d,
            "temporal_pairs_30d": pairs_30d,
            "temporal_pairs_60d": pairs_60d,
            "required_epochs": cls.MIN_GENUINE_EPOCHS_REQUIRED
        }

    @classmethod
    def build_dataset(cls,
                      mode: str = "genuine",
                      db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build the time-aligned machine learning dataset.
        
        Parameters:
        - mode: 'genuine' (default, enforces strict scientific validation)
                'synthetic_benchmark' (for software testing, integration tests, and pipeline verification)
        """
        audit = cls.audit_genuine_dataset(db_path)

        if mode == "genuine":
            if not audit["is_sufficient"]:
                return {
                    "success": False,
                    "status": "INSUFFICIENT_TRAINING_DATA",
                    "reason": audit["reason"],
                    "is_sufficient": False,
                    "audit": audit,
                    "X": None,
                    "targets": None,
                    "dates": [],
                    "segments": [],
                    "data_provenance": "GENUINE_FIELD_SURVEYS_ONLY"
                }

        # If mode == 'synthetic_benchmark' (Explicitly labeled for automated software testing)
        return cls._build_benchmark_synthetic_dataset(db_path)

    @classmethod
    def _build_benchmark_synthetic_dataset(cls, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Constructs a reproducible synthetic benchmark dataset for unit testing,
        model fitting verification, and cross-validation integrity checking.
        Clearly tagged with provenance: 'SYNTHETIC_BENCHMARK_TEST_DATA'.
        """
        from data.unified_dataset import UnifiedDatasetManager
        mgr = UnifiedDatasetManager()
        records = mgr.get_records()

        if len(records) < 50:
            mgr.populate_longitudinal_corridor_history(num_historical_days=120)
            records = mgr.get_records()

        # Group chronologically by road segment
        by_segment: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            seg = r["segment_id"]
            if seg not in by_segment:
                by_segment[seg] = []
            by_segment[seg].append(r)

        for seg in by_segment:
            by_segment[seg].sort(key=lambda x: x["observation_date"])

        # Weather lookup
        path = db_path or DB_PATH
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        w_rows = conn.cursor().execute("SELECT date, precipitation_mm, temperature_mean_c as temperature_c, relative_humidity_pct FROM weather_observations").fetchall()
        conn.close()
        weather_history = [dict(r) for r in w_rows]

        X_rows = []
        y_7d = []
        y_30d = []
        y_60d = []
        dates = []
        segments = []

        step_7d = 1
        step_30d = 4
        step_60d = 8

        for seg_id, traj in by_segment.items():
            T = len(traj)
            for t in range(T - step_7d):
                curr = traj[t]
                obs_date = curr["observation_date"]

                moisture = MechanismFeatureExtractor.compute_moisture_metrics(weather_history, obs_date)
                traffic = MechanismFeatureExtractor.compute_heavy_load_metrics(
                    total_traffic_vpd=float(curr.get("total_traffic_vpd", 8500)),
                    heavy_truck_count=float(curr.get("heavy_truck_count", 2040)),
                    heavy_truck_pct=float(curr.get("heavy_truck_pct", 24.0))
                )
                cracks = MechanismFeatureExtractor.compute_crack_fatigue_metrics(
                    crack_count=int(round(float(curr.get("crack_density_m_per_m2", 0.02)) * 700 / 3.5)),
                    segment_length_m=100.0
                )
                potholes = MechanismFeatureExtractor.compute_pothole_cavity_metrics(
                    pothole_count=int(curr["pothole_count"]),
                    avg_depth_cm=float(curr["avg_pothole_depth_cm"]),
                    max_depth_cm=float(curr["max_pothole_depth_cm"]),
                    volume_liters=float(curr["pothole_volume_liters"])
                )
                velocity = float(curr["pci"]) - float(curr.get("previous_pci", curr["pci"]))

                feat_vec = MechanismFeatureExtractor.construct_mechanism_feature_vector(
                    current_pci=float(curr["pci"]),
                    moisture_metrics=moisture,
                    traffic_metrics=traffic,
                    crack_metrics=cracks,
                    pothole_metrics=potholes,
                    historical_pci_velocity=velocity
                )

                t_7 = min(T - 1, t + step_7d)
                t_30 = min(T - 1, t + step_30d)
                t_60 = min(T - 1, t + step_60d)

                X_rows.append(feat_vec)
                y_7d.append(float(traj[t_7]["pci"]))
                y_30d.append(float(traj[t_30]["pci"]))
                y_60d.append(float(traj[t_60]["pci"]))
                dates.append(obs_date)
                segments.append(seg_id)

        X = np.array(X_rows, dtype=np.float64)
        targets = {
            "pci_7d": np.array(y_7d, dtype=np.float64),
            "pci_30d": np.array(y_30d, dtype=np.float64),
            "pci_60d": np.array(y_60d, dtype=np.float64)
        }

        return {
            "success": True,
            "status": "SYNTHETIC_BENCHMARK_READY",
            "is_sufficient": True,
            "data_provenance": "SYNTHETIC_BENCHMARK_TEST_DATA",
            "warning": "This dataset contains synthetic backward-interpolated records. Intended for software testing only.",
            "total_samples": len(X),
            "X": X,
            "targets": targets,
            "dates": dates,
            "segments": segments,
            "feature_names": cls.FEATURE_NAMES
        }
