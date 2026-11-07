"""
Unit Tests for Multi-Vehicle Fleet Crowdsourcing & De-duplication Engine
"""

import unittest
from risk.fleet_crowdsourcing import (
    BayesianConfidenceFusion,
    SpatialDeDuplicationEngine,
    FleetVehicleManager
)


class TestBayesianConfidenceFusion(unittest.TestCase):
    """Test Bayesian multi-observation probability fusion."""

    def test_single_sighting_confidence(self):
        """Single sighting should equal original confidence with zero gain."""
        res = BayesianConfidenceFusion.fuse_confidences([0.80])
        self.assertAlmostEqual(res["fused_confidence"], 0.80, places=2)
        self.assertEqual(res["confidence_gain"], 0.0)
        self.assertEqual(res["observations_count"], 1)
        self.assertIn("Single", res["verification_status"])

    def test_dual_sighting_boost(self):
        """Two independent sightings of 75% should boost joint certainty to > 93%."""
        res = BayesianConfidenceFusion.fuse_confidences([0.75, 0.75])
        # P = 1 - (1 - 0.75)^2 = 1 - 0.0625 = 0.9375
        self.assertAlmostEqual(res["fused_confidence"], 0.9375, delta=0.01)
        self.assertGreater(res["confidence_gain"], 0.15)
        self.assertEqual(res["observations_count"], 2)
        self.assertIn("Dual", res["verification_status"])
        self.assertIn("ሁለት", res["verification_status_am"])

    def test_multi_vehicle_consensus_cap(self):
        """Three high-confidence sightings should reach 99%+ certainty."""
        res = BayesianConfidenceFusion.fuse_confidences([0.85, 0.88, 0.82])
        self.assertGreater(res["fused_confidence"], 0.99)
        self.assertLessEqual(res["fused_confidence"], 0.999)
        self.assertEqual(res["observations_count"], 3)
        self.assertIn("Consensus", res["verification_status"])


class TestSpatialDeDuplicationEngine(unittest.TestCase):
    """Test spatial radius clustering and weighted centroid geometry."""

    def setUp(self):
        # 3 observations of the same physical pothole within 4 meters
        self.near_potholes = [
            {"damage_latitude": 9.68000, "damage_longitude": 39.53000, "damage_type": "pothole", "confidence": 0.75, "track_id": 101},
            {"damage_latitude": 9.68002, "damage_longitude": 39.53002, "damage_type": "pothole", "confidence": 0.85, "track_id": 102},
            {"damage_latitude": 9.68001, "damage_longitude": 39.52999, "damage_type": "pothole", "confidence": 0.90, "track_id": 103},
        ]
        # 1 distant pothole 50 meters away
        self.distant_pothole = {
            "damage_latitude": 9.68045, "damage_longitude": 39.53000, "damage_type": "pothole", "confidence": 0.80, "track_id": 201
        }
        # 1 nearby crack at the same coords (different defect type)
        self.nearby_crack = {
            "damage_latitude": 9.68001, "damage_longitude": 39.53001, "damage_type": "alligator_crack", "confidence": 0.82, "track_id": 301
        }

    def test_weighted_centroid_calculation(self):
        """Weighted centroid must fall between input coordinates, biased toward higher confidence."""
        lat, lon, spread = SpatialDeDuplicationEngine.calculate_weighted_centroid(self.near_potholes)
        self.assertAlmostEqual(lat, 9.68001, places=4)
        self.assertAlmostEqual(lon, 39.53000, places=4)
        self.assertLess(spread, 5.0)  # Spread within 5m

    def test_clustering_merges_near_potholes(self):
        """3 near observations should collapse into exactly 1 canonical hazard with 3 sightings."""
        clusters = SpatialDeDuplicationEngine.cluster_raw_incidents(self.near_potholes, epsilon_m=8.0)
        self.assertEqual(len(clusters), 1)
        can = clusters[0]
        self.assertEqual(can["sightings_count"], 3)
        self.assertTrue(can["is_deduplicated"])
        self.assertGreater(can["bayesian_fusion"]["fused_confidence"], 0.98)

    def test_distinct_defect_types_not_merged(self):
        """Pothole and Alligator Crack at same location should NOT be merged together."""
        incidents = [self.near_potholes[0], self.nearby_crack]
        clusters = SpatialDeDuplicationEngine.cluster_raw_incidents(incidents, epsilon_m=8.0)
        self.assertEqual(len(clusters), 2)

    def test_distant_potholes_remain_separate(self):
        """Defect 50m away should form its own separate cluster."""
        incidents = [self.near_potholes[0], self.distant_pothole]
        clusters = SpatialDeDuplicationEngine.cluster_raw_incidents(incidents, epsilon_m=8.0)
        self.assertEqual(len(clusters), 2)


class TestFleetVehicleManager(unittest.TestCase):
    """Test fleet crowdsourcing processing and vehicle telemetry."""

    def test_registered_fleet_profile(self):
        """Verify registered fleet contains 4 standard patrol vehicles."""
        fleet = FleetVehicleManager.get_registered_fleet()
        self.assertEqual(len(fleet), 4)
        ids = [v["vehicle_id"] for v in fleet]
        self.assertIn("ERA-PATROL-01", ids)
        self.assertIn("POLICE-CRUISER-07", ids)

    def test_process_fleet_crowdsourcing_summary(self):
        """Test full crowdsourcing summary metrics calculation."""
        mock_raw = [
            {"damage_latitude": 9.68000, "damage_longitude": 39.53000, "damage_type": "pothole", "confidence": 0.70},
            {"damage_latitude": 9.68002, "damage_longitude": 39.53001, "damage_type": "pothole", "confidence": 0.85},
            {"damage_latitude": 9.68200, "damage_longitude": 39.53100, "damage_type": "alligator_crack", "confidence": 0.80},
        ]
        res = FleetVehicleManager.process_fleet_crowdsourcing(mock_raw, epsilon_m=8.0)
        self.assertTrue(res["success"])
        self.assertEqual(res["total_raw_observations"], 3)
        self.assertEqual(res["canonical_unique_hazards"], 2)  # 2 merged into 1 + 1 separate crack
        self.assertEqual(res["duplicate_observations_eliminated"], 1)
        self.assertGreater(res["deduplication_savings_percentage"], 30.0)
        self.assertEqual(res["active_fleet_count"], 4)


if __name__ == "__main__":
    unittest.main()
