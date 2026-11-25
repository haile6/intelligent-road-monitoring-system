"""
Unit Tests for Pavement Intelligence Engine (ASTM D6433 PCI & World Bank IRI)
"""

import unittest
from risk.pavement_intelligence import (
    PavementConditionIndex,
    InternationalRoughnessIndex,
    AsphaltBudgetEstimator,
    CorridorPavementSegmenter,
    PCI_RATING_THRESHOLDS,
    IRI_RATING_THRESHOLDS
)


class TestPavementConditionIndex(unittest.TestCase):
    """Test ASTM D6433 Pavement Condition Index logic."""

    def test_perfect_pavement(self):
        """Zero distresses should yield PCI = 100.0 and Good rating."""
        res = PavementConditionIndex.calculate_pci([], [], sample_area_m2=700.0)
        self.assertEqual(res["pci"], 100.0)
        self.assertEqual(res["rating"], "Good")
        self.assertEqual(res["total_deduct_value"], 0.0)
        self.assertEqual(res["potholes_count"], 0)

    def test_pothole_deduct_severity_scaling(self):
        """Higher severity potholes should produce higher deduct values."""
        dv_low = PavementConditionIndex.calculate_pothole_deduct(2, "LOW", 700.0)
        dv_med = PavementConditionIndex.calculate_pothole_deduct(2, "MEDIUM", 700.0)
        dv_high = PavementConditionIndex.calculate_pothole_deduct(2, "IMMEDIATE_DANGER", 700.0)

        self.assertGreater(dv_med, dv_low)
        self.assertGreater(dv_high, dv_med)

    def test_crack_deduct_severity_scaling(self):
        """Higher severity alligator cracking should produce higher deducts."""
        dv_low = PavementConditionIndex.calculate_crack_deduct(3, "LOW", 700.0)
        dv_med = PavementConditionIndex.calculate_crack_deduct(3, "CAUTION", 700.0)
        dv_high = PavementConditionIndex.calculate_crack_deduct(3, "HIGH", 700.0)

        self.assertGreater(dv_med, dv_low)
        self.assertGreater(dv_high, dv_med)

    def test_severe_distress_degrades_pci(self):
        """Multiple critical potholes should drop PCI to Poor or Serious."""
        potholes = [
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "CAUTION"},
        ]
        cracks = [
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "CAUTION"},
        ]
        res = PavementConditionIndex.calculate_pci(potholes, cracks, sample_area_m2=700.0)
        self.assertLess(res["pci"], 60.0)
        self.assertIn(res["rating"], ["Fair", "Poor", "Very Poor", "Serious", "Failed"])
        self.assertGreater(res["corrected_deduct_value"], 30.0)

    def test_pci_rating_bounds(self):
        """Ensure all PCI ranges properly map to labels and colors."""
        for score, expected_label in [(95.0, "Good"), (75.0, "Satisfactory"), (60.0, "Fair"),
                                      (45.0, "Poor"), (30.0, "Very Poor"), (15.0, "Serious"), (5.0, "Failed")]:
            info = PavementConditionIndex.get_rating_info(score)
            self.assertEqual(info["rating"], expected_label)
            self.assertTrue(info["color"].startswith("#"))


class TestInternationalRoughnessIndex(unittest.TestCase):
    """Test World Bank HDM-4 IRI estimation logic."""

    def test_baseline_highway_roughness(self):
        """Smooth road with no defects should return nominal highway baseline."""
        res = InternationalRoughnessIndex.estimate_iri([], [], segment_length_km=0.25)
        self.assertAlmostEqual(res["iri"], 1.8, delta=0.05)
        self.assertEqual(res["rating"], "Smooth Highway")

    def test_potholes_increase_iri(self):
        """Potholes should elevate the IRI roughness value."""
        potholes = [
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "CAUTION"},
        ]
        res = InternationalRoughnessIndex.estimate_iri(potholes, [], segment_length_km=0.25)
        self.assertGreater(res["iri"], 3.5)
        self.assertGreater(res["delta_potholes"], 0.0)

    def test_iri_threshold_mapping(self):
        """Test standard World Bank IRI rating bins."""
        self.assertEqual(InternationalRoughnessIndex.get_rating_info(1.5)["rating"], "Smooth Highway")
        self.assertEqual(InternationalRoughnessIndex.get_rating_info(3.0)["rating"], "Good")
        self.assertEqual(InternationalRoughnessIndex.get_rating_info(4.5)["rating"], "Fair")
        self.assertEqual(InternationalRoughnessIndex.get_rating_info(6.5)["rating"], "Poor")
        self.assertEqual(InternationalRoughnessIndex.get_rating_info(8.5)["rating"], "Very Poor")

    def test_iri_calibration_weights_override(self):
        """Updating IRI bump weights in db_manager should dynamically affect estimate_iri."""
        from database import db_manager

        potholes = [
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "CAUTION"},
            {"max_risk_level": "SAFE"},
        ]
        cracks = [
            {"max_risk_level": "IMMEDIATE_DANGER"},
            {"max_risk_level": "SAFE"},
        ]

        baseline = db_manager.get_pci_calibration()
        try:
            db_manager.update_pci_calibration({
                "base_highway_iri": 2.0,
                "iri_pothole_high_bump": 0.50,
                "iri_pothole_med_bump": 0.25,
                "iri_pothole_low_bump": 0.10,
                "iri_crack_high_bump": 0.20,
                "iri_crack_other_bump": 0.05,
            })
            res = InternationalRoughnessIndex.estimate_iri(potholes, cracks, segment_length_km=0.25)
            # delta_potholes = (0.50 + 0.25 + 0.10) / 0.25 = 3.40
            # delta_cracking = (0.20 + 0.05) / 0.25 = 1.00
            # total = 2.0 + 3.40 + 1.00 = 6.40
            self.assertAlmostEqual(res["delta_potholes"], 3.40, places=2)
            self.assertAlmostEqual(res["delta_cracking"], 1.00, places=2)
            self.assertAlmostEqual(res["iri"], 6.40, places=2)
        finally:
            db_manager.update_pci_calibration(baseline)


class TestAsphaltBudgetEstimator(unittest.TestCase):
    """Test ERA schedule of rates and rehabilitation budget estimation."""

    def test_good_condition_preservation_budget(self):
        """PCI >= 85 should trigger routine maintenance with minimal budget."""
        budget = AsphaltBudgetEstimator.estimate_segment_budget(
            pci_score=92.0,
            iri_score=2.1,
            potholes_count=0,
            cracks_count=0,
            segment_length_m=250.0
        )
        self.assertEqual(budget["intervention_level"], "PRESERVATION")
        self.assertGreater(budget["total_cost_etb"], 0.0)
        self.assertEqual(budget["materials"]["hot_mix_asphalt_tonnes"], 0.0)

    def test_fair_condition_mill_and_overlay(self):
        """PCI between 40 and 70 should trigger 50mm AC Mill & Overlay."""
        budget = AsphaltBudgetEstimator.estimate_segment_budget(
            pci_score=52.0,
            iri_score=5.2,
            potholes_count=4,
            cracks_count=3,
            segment_length_m=250.0,
            carriageway_width_m=7.0
        )
        self.assertEqual(budget["intervention_level"], "MILL_AND_OVERLAY")
        self.assertIn("Milling", budget["recommended_treatment"])
        self.assertGreater(budget["total_cost_etb"], 1_000_000.0)  # > 1M ETB for 250m carriageway
        self.assertGreater(budget["materials"]["hot_mix_asphalt_tonnes"], 100.0)
        self.assertGreater(budget["materials"]["tack_coat_liters"], 500.0)

    def test_failed_condition_full_reconstruction(self):
        """PCI < 40 or IRI >= 7.0 should trigger Full-Depth Reconstruction."""
        budget = AsphaltBudgetEstimator.estimate_segment_budget(
            pci_score=28.0,
            iri_score=7.8,
            potholes_count=8,
            cracks_count=6,
            segment_length_m=250.0,
            carriageway_width_m=7.0
        )
        self.assertEqual(budget["intervention_level"], "FULL_RECONSTRUCTION")
        self.assertIn("Full-Depth", budget["recommended_treatment"])
        # Full reconstruction rate is 7,800 ETB/m2 * (250 * 7 = 1750m2) = ~13,650,000 ETB
        self.assertGreater(budget["total_cost_etb"], 10_000_000.0)
        self.assertGreater(budget["total_cost_usd"], 80_000.0)

    def test_etb_to_usd_currency_consistency(self):
        """Verify USD matches ETB / conversion rate."""
        budget = AsphaltBudgetEstimator.estimate_segment_budget(
            pci_score=78.0,
            iri_score=3.2,
            potholes_count=2,
            cracks_count=1,
            segment_length_m=250.0
        )
        expected_usd = budget["total_cost_etb"] / AsphaltBudgetEstimator.ETB_TO_USD_RATE
        self.assertAlmostEqual(budget["total_cost_usd"], expected_usd, delta=0.1)

    def test_mill_and_overlay_boq_bills_by_liters_and_tonnes(self):
        """Mill & Overlay BoQ should bill Tack Coat by Liters and HMA by Tonnes using Settings rates."""
        from database import db_manager

        baseline = db_manager.get_material_rates()
        try:
            db_manager.update_material_rates({
                "milling_rate_m2_etb": 420.0,
                "tack_coat_per_liter_etb": 120.0,
                "hot_mix_ton_etb": 14500.0,
                "labor_compaction_rate_m2_etb": 450.0,
                "etb_to_usd_rate": 128.0,
            })
            budget = AsphaltBudgetEstimator.estimate_segment_budget(
                pci_score=85.5,
                iri_score=5.0,
                potholes_count=1,
                cracks_count=0,
                segment_length_m=50.0,
                carriageway_width_m=7.0,
            )
            items = {item["item_code"]: item for item in budget["boq_items"]}
            self.assertEqual(items["ERA-MIL-01"]["unit"], "m²")
            self.assertEqual(items["ERA-MIL-01"]["quantity"], 350.0)
            self.assertEqual(items["ERA-MIL-01"]["total_etb"], 147000.0)

            self.assertEqual(items["ERA-TC-02"]["unit"], "L")
            self.assertEqual(items["ERA-TC-02"]["quantity"], 192.5)
            self.assertEqual(items["ERA-TC-02"]["unit_rate_etb"], 120.0)
            self.assertEqual(items["ERA-TC-02"]["total_etb"], 23100.0)

            self.assertEqual(items["ERA-AC-03"]["unit"], "Tonnes")
            self.assertEqual(items["ERA-AC-03"]["quantity"], 42.0)
            self.assertEqual(items["ERA-AC-03"]["unit_rate_etb"], 14500.0)
            self.assertEqual(items["ERA-AC-03"]["total_etb"], 609000.0)

            self.assertEqual(items["ERA-LAY-04"]["unit"], "m²")
            self.assertEqual(items["ERA-LAY-04"]["quantity"], 350.0)
            self.assertEqual(items["ERA-LAY-04"]["total_etb"], 157500.0)

            self.assertEqual(budget["total_cost_etb"], 936600.0)

            # Verify Settings rate updates propagate dynamically
            db_manager.update_material_rates({
                "tack_coat_per_liter_etb": 180.0,
                "hot_mix_ton_etb": 15000.0,
            })
            updated = AsphaltBudgetEstimator.estimate_segment_budget(
                pci_score=85.5,
                iri_score=5.0,
                potholes_count=1,
                cracks_count=0,
                segment_length_m=50.0,
                carriageway_width_m=7.0,
            )
            updated_items = {item["item_code"]: item for item in updated["boq_items"]}
            self.assertEqual(updated_items["ERA-TC-02"]["total_etb"], 34650.0)
            self.assertEqual(updated_items["ERA-AC-03"]["total_etb"], 630000.0)
        finally:
            db_manager.update_material_rates(baseline)


class TestCorridorPavementSegmenter(unittest.TestCase):
    """Test corridor segmentation and spatial incident association."""

    def setUp(self):
        # 1 km test GPS trace (Debre Berhan Route A2 corridor)
        self.mock_gps = [
            {"lat": 9.6800, "lon": 39.5300},
            {"lat": 9.6820, "lon": 39.5315},
            {"lat": 9.6845, "lon": 39.5330},
            {"lat": 9.6870, "lon": 39.5350},
            {"lat": 9.6890, "lon": 39.5370},
        ]
        self.mock_incidents = [
            {"damage_latitude": 9.6802, "damage_longitude": 39.5301, "damage_type": "pothole", "max_risk_level": "IMMEDIATE_DANGER"},
            {"damage_latitude": 9.6821, "damage_longitude": 39.5316, "damage_type": "pothole", "max_risk_level": "CAUTION"},
            {"damage_latitude": 9.6844, "damage_longitude": 39.5329, "damage_type": "alligator_crack", "max_risk_level": "SAFE"},
        ]

    def test_haversine_distance(self):
        """Test accurate geodesic distance calculation."""
        dist = CorridorPavementSegmenter.haversine_distance_m(9.6800, 39.5300, 9.6800, 39.5310)
        self.assertGreater(dist, 100.0)
        self.assertLess(dist, 120.0)

    def test_corridor_segmentation_output(self):
        """Verify corridor is properly divided into segments with full metrics."""
        result = CorridorPavementSegmenter.segment_corridor(
            gps_points=self.mock_gps,
            incidents=self.mock_incidents,
            target_segment_length_m=300.0
        )
        self.assertIn("corridor_name", result)
        self.assertGreater(result["total_length_km"], 0.5)
        self.assertGreater(result["total_segments"], 1)
        self.assertGreater(result["average_pci"], 0.0)
        self.assertGreater(result["average_iri"], 0.0)
        self.assertGreater(result["total_rehabilitation_cost_etb"], 0.0)
        self.assertTrue(len(result["segments"]) > 0)

        # Check segment 1 structure
        seg1 = result["segments"][0]
        self.assertIn("pci", seg1)
        self.assertIn("iri", seg1)
        self.assertIn("cost_etb", seg1)
        self.assertIn("cost_usd", seg1)
        self.assertIn("polyline", seg1)
        self.assertIn("materials", seg1)


if __name__ == "__main__":
    unittest.main()

