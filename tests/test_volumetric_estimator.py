"""
Unit Tests for 3D Depth & Volumetric Estimator (Pothole Cavity Geometry & Asphalt Material)
"""

import unittest
from risk.volumetric_estimator import (
    Pothole3DDepthEstimator,
    PotholeVolumetricCalculator
)


class TestPothole3DDepthEstimator(unittest.TestCase):
    """Test monocular 3D depth and physical dimension estimation."""

    def test_physical_dimensions_scaling(self):
        """Bounding box at larger distance should correspond to larger physical width."""
        bbox = [500.0, 400.0, 600.0, 480.0]
        dim_near = Pothole3DDepthEstimator.estimate_physical_dimensions(bbox, distance_m=3.0)
        dim_far = Pothole3DDepthEstimator.estimate_physical_dimensions(bbox, distance_m=12.0)

        self.assertGreater(dim_far["width_m"], dim_near["width_m"])
        self.assertGreater(dim_far["surface_area_m2"], dim_near["surface_area_m2"])
        self.assertGreater(dim_near["major_axis_cm"], 10.0)

    def test_depth_estimation_severity_tiers(self):
        """Immediate Danger potholes should produce deeper cavity estimates than Caution."""
        dim = {
            "width_m": 0.65,
            "length_m": 0.50,
            "major_axis_cm": 65.0,
            "minor_axis_cm": 50.0,
            "surface_area_m2": 0.255
        }
        depth_safe = Pothole3DDepthEstimator.estimate_depth(dim, risk_level="SAFE")
        depth_caution = Pothole3DDepthEstimator.estimate_depth(dim, risk_level="CAUTION")
        depth_danger = Pothole3DDepthEstimator.estimate_depth(dim, risk_level="IMMEDIATE_DANGER")

        self.assertGreater(depth_caution["depth_max_cm"], depth_safe["depth_max_cm"])
        self.assertGreater(depth_danger["depth_max_cm"], depth_caution["depth_max_cm"])
        self.assertGreater(depth_danger["depth_max_cm"], depth_danger["depth_avg_cm"])

    def test_depth_clamping_bounds(self):
        """Ensure depth is clamped between physical limits 1.5 cm and 18.0 cm."""
        dim_tiny = {"width_m": 0.05, "length_m": 0.05, "major_axis_cm": 5.0, "minor_axis_cm": 5.0, "surface_area_m2": 0.002}
        dim_huge = {"width_m": 3.0, "length_m": 3.0, "major_axis_cm": 300.0, "minor_axis_cm": 300.0, "surface_area_m2": 7.0}

        depth_tiny = Pothole3DDepthEstimator.estimate_depth(dim_tiny, "SAFE")
        depth_huge = Pothole3DDepthEstimator.estimate_depth(dim_huge, "IMMEDIATE_DANGER")

        self.assertGreaterEqual(depth_tiny["depth_max_cm"], 1.5)
        self.assertLessEqual(depth_huge["depth_max_cm"], 18.0)


class TestPotholeVolumetricCalculator(unittest.TestCase):
    """Test 3D cavity integration and asphalt material quantification."""

    def test_cavity_volume_paraboloid(self):
        """Paraboloid cavity volume should be half of cylinder volume."""
        area = 0.50  # 0.5 m2
        depth_cm = 8.0  # 0.08 m
        vol_para = PotholeVolumetricCalculator.calculate_cavity_volume(area, depth_cm, "paraboloid")
        vol_cyl = PotholeVolumetricCalculator.calculate_cavity_volume(area, depth_cm, "cylinder")

        self.assertAlmostEqual(vol_para["volume_m3"], vol_cyl["volume_m3"] * 0.5, places=4)
        self.assertAlmostEqual(vol_para["volume_liters"], vol_para["volume_m3"] * 1000.0, places=2)
        self.assertGreater(vol_para["volume_liters"], 1.0)

    def test_asphalt_material_quantification(self):
        """Verify loose mass calculation with compaction factor and bag counts."""
        vol_data = {
            "volume_m3": 0.025,  # 25 Liters
            "volume_liters": 25.0,
            "depth_m": 0.06,
            "surface_area_m2": 0.40
        }
        mat = PotholeVolumetricCalculator.calculate_asphalt_material_requirements(
            volume_data=vol_data,
            major_axis_cm=70.0,
            minor_axis_cm=55.0
        )

        # Compacted cold patch mass = 0.025 * 2200 = 55.0 kg
        self.assertAlmostEqual(mat["compacted_mass_kg"], 55.0, delta=0.5)
        # Loose mass = 55.0 * 1.22 = 67.1 kg
        self.assertGreater(mat["loose_mass_kg"], mat["compacted_mass_kg"])
        # Bags = ceil(67.1 / 25) = 3 bags
        self.assertEqual(mat["bags_25kg_count"], 3)
        self.assertGreaterEqual(mat["bags_50kg_count"], 1)
        self.assertGreater(mat["tack_coat_liters"], 0.2)
        self.assertGreater(mat["cost_etb"], 1000.0)

    def test_estimate_repair_package_end_to_end(self):
        """Test full package output with bilingual summaries."""
        pkg = PotholeVolumetricCalculator.estimate_repair_package(
            bbox=[500.0, 420.0, 680.0, 560.0],
            distance_m=4.2,
            risk_level="IMMEDIATE_DANGER",
            confidence=0.88
        )
        self.assertTrue(pkg["success"])
        self.assertIn("physical_dimensions", pkg)
        self.assertIn("depth_profile", pkg)
        self.assertIn("cavity_volume", pkg)
        self.assertIn("materials_quantification", pkg)
        self.assertIn("summary_pill", pkg)
        self.assertIn("summary_pill_am", pkg)
        self.assertIn("ሊትር", pkg["summary_pill_am"])
        self.assertIn("ቦርሳ", pkg["summary_pill_am"])


if __name__ == "__main__":
    unittest.main()
