"""
RoadEye™ Seasonal Pavement Deterioration Integrity & Evaluation Engine
======================================================================
Strict Scientific Integrity Module for Evaluating Seasonal Deterioration Claims.

Audit Finding:
- The database contains genuine CV dashcam survey incidents from 2026-10-20 and 2026-11-16 (dry season).
- Genuine repeated field condition surveys across a full wet season (Kiremt, June–September)
  and dry season (Bega) do NOT currently exist in the database.
- The 378 records in `unified_road_observations` were mathematically synthesized
  via backwards interpolation from current PCI weighted by atmospheric rainfall reanalysis.
- The previously calculated 1.45x ratio was an analytical mathematical artifact of the
  interpolation function: 1.0 + 0.18 * ln(1 + rain_mm).
- It was NOT an empirical field observation.

In accordance with strict scientific integrity principles:
If genuine multi-season field survey observations are unavailable, the system displays:
"Insufficient historical data to calculate an empirical seasonal deterioration ratio."
No arbitrary multiplier is assumed, simulated, or substituted.
"""

import os
import sys
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "road_monitoring.db")


class SeasonalHypothesisInvestigator:
    """
    Evaluates whether genuine repeated field pavement condition observations exist
    across distinct seasons to compute an empirical seasonal deterioration ratio.
    """

    def __init__(self):
        pass

    def check_genuine_seasonal_observations(self) -> Dict[str, Any]:
        """
        Check database for genuine repeated field inspection surveys spanning
        both wet season (June-September) and dry season (October-May).
        """
        if not os.path.exists(DB_PATH):
            return {"has_genuine_data": False, "reason": "Database file not found"}

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check genuine incident survey dates from field dashcam runs
        cursor.execute("SELECT DISTINCT date(timestamp) FROM incidents WHERE timestamp IS NOT NULL")
        raw_dates = [r[0] for r in cursor.fetchall() if r[0]]

        # Check video runs
        cursor.execute("SELECT DISTINCT video_name FROM incidents WHERE video_name IS NOT NULL")
        videos = [r[0] for r in cursor.fetchall() if r[0]]
        conn.close()

        wet_dates = []
        dry_dates = []

        for d_str in raw_dates:
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                if dt.month in [6, 7, 8, 9]:
                    wet_dates.append(d_str)
                else:
                    dry_dates.append(d_str)
            except Exception:
                continue

        has_both_seasons = len(wet_dates) >= 2 and len(dry_dates) >= 2

        return {
            "has_genuine_multi_season_surveys": has_both_seasons,
            "genuine_survey_dates_found": raw_dates,
            "wet_season_survey_dates": wet_dates,
            "dry_season_survey_dates": dry_dates,
            "videos_cataloged": videos,
            "provenance_note": (
                "Verified field inspection surveys require dashcam video runs recorded across "
                "distinct wet (June-Sept) and dry (Oct-May) seasons. Current recorded incidents "
                "originate only from dry season dates (October/November 2026)."
            )
        }

    def evaluate_seasonal_rates(self) -> Dict[str, Any]:
        """
        Evaluate seasonal deterioration rates strictly from genuine field observations.
        If sufficient observations across both seasons do not exist, report honestly:
        "Insufficient historical data to calculate an empirical seasonal deterioration ratio."
        """
        check = self.check_genuine_seasonal_observations()

        if not check["has_genuine_multi_season_surveys"]:
            return {
                "success": True,
                "status": "INSUFFICIENT_HISTORICAL_DATA",
                "summary_status": "Insufficient historical data to calculate an empirical seasonal deterioration ratio.",
                "message": "Insufficient historical data to calculate an empirical seasonal deterioration ratio.",
                "empirical_multiplier": None,
                "is_data_sufficient": False,
                "genuine_field_data_audit": {
                    "wet_season_field_surveys_count": len(check["wet_season_survey_dates"]),
                    "dry_season_field_surveys_count": len(check["dry_season_survey_dates"]),
                    "available_dates": check["genuine_survey_dates_found"],
                    "required_condition": "At least 2 repeated field condition surveys in Kiremt and 2 in Bega"
                },
                "scientific_integrity_declaration": {
                    "was_1_45_empirical": False,
                    "provenance_of_1_45": (
                        "The previously reported 1.45x value was an artifact of backwards mathematical "
                        "interpolation using the formula 1.0 + 0.18 * ln(1 + rain_mm). It was NOT an "
                        "empirical measurement from repeated field condition surveys. In accordance with "
                        "scientific integrity principles, it has been completely removed."
                    ),
                    "policy": "No arbitrary or simulated multiplier is substituted."
                }
            }

        # If genuine multi-season data were available, calculate rates here (placeholder for future real data)
        return {
            "success": True,
            "status": "CALCULATED_FROM_GENUINE_DATA",
            "empirical_multiplier": None
        }
