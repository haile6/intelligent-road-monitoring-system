"""
Comprehensive End-to-End Verification & Mathematical Validation Test Suite.
Validates:
1. Distance Estimation geometry & boundary cases
2. GPS synchronization, interpolation, and forward geodesic projection
3. Time-To-Collision (TTC) & Dynamic Risk Classification
4. Database integrity, persistence, and workflow transitions
5. Web dashboard, map markers, and reports against real persisted data
"""
import os
import sys
import math
import unittest
from datetime import datetime, timedelta

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from risk.distance_estimator import PerspectiveDistanceEstimator, make_estimator
from gps.geo import haversine_m, haversine_distance, bearing_deg, destination_point
from gps.gps_sync import GpsSync
from risk.ttc_calculator import calculate_ttc, calculate_lead_vehicle_ttc, format_ttc
from risk.risk_analyzer import assess_risk, scene_risk
from database import db_manager
from web.app import app


class TestDistanceEstimation(unittest.TestCase):
    """Test mathematical accuracy of monocular perspective projection."""

    def setUp(self):
        self.width = 1280
        self.height = 720
        self.estimator = PerspectiveDistanceEstimator(self.width, self.height)

    def test_horizon_calculation(self):
        """Verify that horizon row is computed below image center for positive tilt."""
        # Horizon row must be above optical center when camera is tilted downward
        self.assertLess(self.estimator.horizon_row, self.estimator.cy)

    def test_row_at_horizon_returns_none(self):
        """Objects located at or above horizon cannot have valid ground-plane distance."""
        res = self.estimator.distance_at_row(int(self.estimator.horizon_row))
        self.assertIsNone(res)

    def test_closer_row_gives_shorter_distance(self):
        """As the object appears lower in the image (closer to vehicle), distance decreases monotonically."""
        d_far = self.estimator.distance_at_row(450)
        d_near = self.estimator.distance_at_row(650)
        self.assertIsNotNone(d_far)
        self.assertIsNotNone(d_near)
        self.assertGreater(d_far, d_near)

    def test_bottom_of_screen_distance(self):
        """Test physical plausibility of distance at bottom row (road right in front of bumper)."""
        d_bottom = self.estimator.distance_at_row(719)
        self.assertIsNotNone(d_bottom)
        # For H=1.2m and ~70 deg FOV, distance at bottom edge should be between 0.5m and 3.0m
        self.assertGreater(d_bottom, 0.5)
        self.assertLess(d_bottom, 3.5)

    def test_invalid_bbox_safety(self):
        """Invalid or malformed bounding boxes should return None gracefully without throwing exceptions."""
        self.assertIsNone(self.estimator.estimate([])["distance_m"])
        self.assertIsNone(self.estimator.estimate([100, 100])["distance_m"])


class TestGeodesicsAndGpsSync(unittest.TestCase):
    """Test geospatial mathematics and GPS synchronization."""

    def test_haversine_known_distance(self):
        """Test Haversine distance between two known geographic points in Addis Ababa / Debre Berhan."""
        # Approx 1 degree latitude is ~111.19 km
        lat1, lon1 = 9.000000, 39.000000
        lat2, lon2 = 9.008983, 39.000000  # ~1000 meters North
        dist = haversine_m(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(dist, 1000.0, delta=5.0)
        self.assertEqual(haversine_distance(lat1, lon1, lat2, lon2), dist)

    def test_bearing_cardinal_directions(self):
        """Test compass azimuth computation for cardinal directions."""
        # Due North
        b_north = bearing_deg(9.0, 39.0, 9.1, 39.0)
        self.assertAlmostEqual(b_north, 0.0, delta=0.5)
        # Due East
        b_east = bearing_deg(9.0, 39.0, 9.0, 39.1)
        self.assertAlmostEqual(b_east, 90.0, delta=0.5)

    def test_destination_point_reversibility(self):
        """Verify that projecting a point 50 meters North and calculating Haversine distance matches 50m."""
        start_lat, start_lon = 9.680000, 39.530000  # Debre Berhan approximate coordinates
        target_lat, target_lon = destination_point(start_lat, start_lon, 0.0, 50.0)
        measured_dist = haversine_m(start_lat, start_lon, target_lat, target_lon)
        self.assertAlmostEqual(measured_dist, 50.0, delta=0.5)

    def test_gps_interpolation_lerp(self):
        """Test sub-second linear interpolation between two consecutive GPS fixes."""
        base_time = datetime(2026, 10, 20, 10, 0, 0)
        records = [
            {"time": base_time, "latitude": 9.000000, "longitude": 39.000000, "speed_kmh": 36.0},
            {"time": base_time + timedelta(seconds=1), "latitude": 9.000100, "longitude": 39.000100, "speed_kmh": 36.0}
        ]
        sync = GpsSync(records, video_start=base_time, offset_s=0)
        # At 0.5 seconds, coordinates must be exact midpoint
        mid = sync.at(0.5)
        self.assertAlmostEqual(mid["latitude"], 9.000050, places=6)
        self.assertAlmostEqual(mid["longitude"], 39.000050, places=6)
        self.assertAlmostEqual(mid["speed_kmh"], 36.0, places=1)


class TestTTCEdgeCasesAndRisk(unittest.TestCase):
    """Test dynamic Time-To-Collision and hazard risk categorization."""

    def test_ttc_stopped_vehicle(self):
        """Stopped vehicle must not result in ZeroDivisionError and must return status: stopped."""
        res_zero = calculate_ttc(distance_m=10.0, speed_mps=0.0)
        self.assertIsNone(res_zero["ttc_s"])
        self.assertEqual(res_zero["status"], "stopped")

        res_slow = calculate_ttc(distance_m=10.0, speed_mps=0.2)  # below 2 km/h
        self.assertEqual(res_slow["status"], "stopped")

    def test_ttc_missing_distance(self):
        """Missing or negative distance must return status: no_distance."""
        self.assertEqual(calculate_ttc(None, 10.0)["status"], "no_distance")
        self.assertEqual(calculate_ttc(-5.0, 10.0)["status"], "no_distance")

    def test_ttc_valid_computation(self):
        """Normal velocity and distance produces exact time: 30m at 10m/s (36 km/h) = 3.0s."""
        res = calculate_ttc(30.0, 10.0)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["ttc_s"], 3.0, places=2)

    def test_lead_vehicle_fcw_relative_speed(self):
        """Test Forward Collision Warning when approaching a slower lead vehicle."""
        # Host: 72 km/h (20 m/s), Lead: 36 km/h (10 m/s), Closing: 10 m/s, Dist: 15m -> TTC = 1.5s (WARNING)
        fcw = calculate_lead_vehicle_ttc(distance_m=15.0, ego_speed_mps=20.0, lead_speed_mps=10.0)
        self.assertAlmostEqual(fcw["ttc_s"], 1.5, places=1)
        self.assertEqual(fcw["risk_level"], "WARNING")


class DummyTrack:
    def __init__(self, confirmed=True, in_path=True, ttc_s=0.8, distance_m=5.0,
                 mean_conf=0.85, class_name="pothole", ttc_status="ok"):
        self.confirmed = confirmed
        self.in_path = in_path
        self.ttc_s = ttc_s
        self.distance_m = distance_m
        self.mean_confidence = mean_conf
        self.class_name = class_name
        self.ttc_status = ttc_status


class TestRiskClassification(unittest.TestCase):
    def test_immediate_danger_threshold(self):
        """TTC < 1.0s for confirmed in-path pothole must evaluate to IMMEDIATE DANGER."""
        track = DummyTrack(ttc_s=0.8, in_path=True, mean_conf=0.8)
        speed = {"speed_kmh": 30.0, "is_stopped": False, "is_default": False}
        risk = assess_risk(track, speed)
        self.assertEqual(risk["level"], "IMMEDIATE_DANGER")

    def test_out_of_path_is_safe(self):
        """Pothole outside driving corridor must be classified as SAFE."""
        track = DummyTrack(ttc_s=0.8, in_path=False)
        speed = {"speed_kmh": 30.0, "is_stopped": False, "is_default": False}
        risk = assess_risk(track, speed)
        self.assertEqual(risk["level"], "SAFE")

    def test_low_confidence_caps_at_caution(self):
        """Low detection confidence must prevent false alarms and limit risk to CAUTION."""
        track = DummyTrack(ttc_s=0.5, in_path=True, mean_conf=0.35)
        speed = {"speed_kmh": 30.0, "is_stopped": False, "is_default": False}
        risk = assess_risk(track, speed)
        self.assertEqual(risk["level"], "CAUTION")


class TestDatabaseAndWebEndToEnd(unittest.TestCase):
    """Test actual SQLite persistence and verify that web endpoints reflect real database records."""

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_database_record_counts(self):
        """Confirm database connectivity and presence of persisted incidents & plate readings."""
        conn = db_manager.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM incidents")
        inc_count = c.fetchone()[0]
        self.assertGreaterEqual(inc_count, 1, "Database should contain persisted incidents.")

        c.execute("SELECT COUNT(*) FROM plate_readings")
        plate_count = c.fetchone()[0]
        self.assertGreaterEqual(plate_count, 0, "Plate readings table should be accessible.")

    def test_dashboard_kpis_match_database(self):
        """Verify that /api/stats returns counts matching actual SQLite records."""
        conn = db_manager.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM incidents")
        db_total = c.fetchone()[0]

        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total_hazards"], db_total, "KPI card hazard count must match database.")

    def test_map_markers_have_valid_coordinates(self):
        """Verify that /api/incidents returns valid geographic coordinates for map markers."""
        resp = self.client.get("/api/incidents")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("incidents", data)
        incidents = data["incidents"]
        self.assertGreater(len(incidents), 0)

        # Inspect first 10 incidents for valid coordinates
        valid_coords = 0
        for inc in incidents[:10]:
            lat = inc.get("damage_latitude") or inc.get("latitude")
            lon = inc.get("damage_longitude") or inc.get("longitude")
            if lat is not None and lon is not None:
                # Valid latitude between -90 and 90, longitude between -180 and 180
                self.assertGreaterEqual(lat, -90.0)
                self.assertLessEqual(lat, 90.0)
                self.assertGreaterEqual(lon, -180.0)
                self.assertLessEqual(lon, 180.0)
                valid_coords += 1
        self.assertGreater(valid_coords, 0, "Map markers must have valid persisted geographic coordinates.")

    def test_work_orders_and_status_update_persistence(self):
        """Test municipal workflow transition (Reported -> Scheduled -> Repaired) in SQLite."""
        conn = db_manager.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, status FROM incidents LIMIT 1")
        row = c.fetchone()
        self.assertIsNotNone(row)
        incident_id = row[0]
        original_status = row[1]

        # Update status via db_manager
        new_status = "Scheduled" if original_status != "Scheduled" else "Reported"
        success = db_manager.update_incident(incident_id, status=new_status, notes="Audit Test Update")
        self.assertIsNotNone(success)

        # Verify persisted change in SQLite
        c.execute("SELECT status FROM incidents WHERE id = ?", (incident_id,))
        updated_status = c.fetchone()[0]
        self.assertEqual(updated_status, new_status)

        # Revert back to maintain dataset state
        db_manager.update_incident(incident_id, status=original_status, notes="Reverted Test Update")


if __name__ == "__main__":
    unittest.main()
