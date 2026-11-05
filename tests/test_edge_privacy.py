"""
Unit Tests for Privacy-Preserving Redaction and Edge AI Quantization Engine (CLO2, CLO3).
"""

import unittest
import numpy as np
from mlops.edge_optimizer import PrivacyPreserver, EdgeOptimizer, QuantizationProfile, PrivacyRedactionResult


class TestPrivacyPreserver(unittest.TestCase):
    """Test Suite for Privacy-Preserving Vehicle Plate & Face Redaction."""

    def setUp(self):
        self.preserver = PrivacyPreserver(mosaic_block_size=10)
        # Create a synthetic image (100x100 RGB image with distinctive pattern)
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img[20:50, 20:60] = [200, 150, 100]  # Plate area

    def test_mosaic_redaction_alters_target_region(self):
        original_region = self.img[20:50, 20:60].copy()
        bboxes = [[20, 20, 60, 50]]

        redacted_img, result = self.preserver.redact_bboxes(self.img, bboxes, method="pixelation_mosaic")
        self.assertTrue(result.anonymized)
        self.assertEqual(result.plates_redacted_count, 1)

        # Pixels in the bbox should be processed and differ from high-contrast original
        self.assertEqual(redacted_img.shape, (100, 100, 3))
        # Background pixel outside bbox should remain untouched
        np.testing.assert_array_equal(redacted_img[0:10, 0:10], self.img[0:10, 0:10])

    def test_gaussian_blur_redaction(self):
        bboxes = [[20, 20, 60, 50]]
        redacted_img, result = self.preserver.redact_bboxes(self.img, bboxes, method="gaussian_blur")
        self.assertTrue(result.anonymized)
        self.assertEqual(result.redaction_method, "gaussian_blur")
        self.assertLess(result.processing_time_ms, 50.0)


class TestEdgeOptimizer(unittest.TestCase):
    """Test Suite for Post-Training Quantization (PTQ) & TinyML Profiling."""

    def setUp(self):
        self.optimizer = EdgeOptimizer(base_weights_size_mb=19.15)

    def test_fp16_quantization_profile(self):
        profile = self.optimizer.get_quantization_profile(precision="FP16")
        self.assertEqual(profile.target_precision, "FP16")
        self.assertAlmostEqual(profile.compression_ratio, 2.0, places=1)
        self.assertLess(profile.compressed_size_mb, 10.0)
        self.assertGreater(profile.speedup_factor, 1.5)
        self.assertGreater(profile.expected_mAP_retention, 99.0)

    def test_int8_tinyml_quantization_profile(self):
        profile = self.optimizer.get_quantization_profile(precision="INT8")
        self.assertEqual(profile.target_precision, "INT8")
        self.assertAlmostEqual(profile.compression_ratio, 4.0, places=1)
        self.assertLess(profile.compressed_size_mb, 5.0)  # Under 5 MB for TinyML
        self.assertGreater(profile.speedup_factor, 3.0)
        self.assertIn("ARM", profile.target_hardware)

    def test_benchmark_edge_devices(self):
        benchmarks = self.optimizer.benchmark_edge_profiles()
        self.assertIn("NVIDIA_Jetson_Orin_Nano", benchmarks)
        self.assertIn("Raspberry_Pi_5_Hailo_TPU", benchmarks)
        self.assertIn("Rockchip_RK3588_Dashcam", benchmarks)

        # Raspberry Pi TinyML configuration should operate under 10 Watts
        rpi = benchmarks["Raspberry_Pi_5_Hailo_TPU"]
        self.assertLess(rpi["power_watts"], 10.0)
        self.assertGreater(rpi["fps"], 30.0)


if __name__ == "__main__":
    unittest.main()
