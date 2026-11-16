"""
Unit tests for Dual-Layer Multi-Factor Risk Assessment (Physical Structural + Kinematic Exposure).
"""
import unittest
from risk.risk_analyzer import assess_risk


class MockTrack:
    def __init__(self, in_path=True, ttc_s=0.8, distance_m=10.0, class_name="pothole", 
                 severity=None, depth_max_cm=None, area_m2=None, mean_confidence=0.90):
        self.confirmed = True
        self.in_path = in_path
        self.ttc_s = ttc_s
        self.ttc_status = "closing"
        self.distance_m = distance_m
        self.class_name = class_name
        self.severity = severity
        self.depth_max_cm = depth_max_cm
        self.area_m2 = area_m2
        self.mean_confidence = mean_confidence


class TestDualLayerRiskAssessment(unittest.TestCase):
    def setUp(self):
        self.speed = {"speed_kmh": 40.0, "is_stopped": False, "is_default": False}

    def test_minor_crack_in_path_downgraded_from_immediate_danger(self):
        """Minor crack in vehicle path should be softened so it does not trigger panic stop."""
        track = MockTrack(in_path=True, ttc_s=0.6, class_name="alligator_crack", severity="Low")
        res = assess_risk(track, self.speed)
        self.assertIn(res["level"], ["WARNING", "CAUTION"])
        self.assertNotEqual(res["level"], "IMMEDIATE_DANGER")
        self.assertTrue(any("minor" in r for r in res["reasons"]))

    def test_deep_severe_pothole_escalates_to_danger(self):
        """Deep severe pothole (depth >= 8cm) in path escalates to IMMEDIATE_DANGER even with 2.0s TTC."""
        track = MockTrack(in_path=True, ttc_s=2.0, class_name="pothole", severity="Severe", depth_max_cm=9.5)
        res = assess_risk(track, self.speed)
        self.assertEqual(res["level"], "IMMEDIATE_DANGER")
        self.assertTrue(any("deep severe cavity" in r for r in res["reasons"]))

    def test_off_path_severe_pothole_records_municipal_asset_warning(self):
        """Off-path severe pothole is SAFE for driver collision but WARNING for municipal repairs."""
        track = MockTrack(in_path=False, ttc_s=1.5, class_name="pothole", severity="Severe", depth_max_cm=10.0)
        res = assess_risk(track, self.speed)
        self.assertEqual(res["level"], "SAFE")
        self.assertEqual(res["asset_risk_level"], "WARNING")
        self.assertGreater(res["unified_score"], 50.0)
        self.assertTrue(any("municipal" in r for r in res["reasons"]))

    def test_unified_score_combines_structural_and_kinematic(self):
        """Unified score weights both structural physical severity and kinematic exposure."""
        track = MockTrack(in_path=True, ttc_s=0.5, class_name="pothole", severity="High")
        res = assess_risk(track, self.speed)
        self.assertEqual(res["level"], "IMMEDIATE_DANGER")
        self.assertGreaterEqual(res["unified_score"], 80.0)
        self.assertIn("unified_score", res)
        self.assertIn("structural_severity", res)


if __name__ == "__main__":
    unittest.main()
