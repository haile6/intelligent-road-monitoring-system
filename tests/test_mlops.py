"""
Unit Tests for MLOps Experiment Tracking, Model Registry, and Drift Monitoring (CLO1, CLO2, CLO3).
"""

import os
import shutil
import tempfile
import unittest
import datetime

from mlops.experiment_tracker import ExperimentTracker, ModelArtifact, ExperimentRun
from mlops.drift_monitor import DriftMonitor, DriftReport


class TestMLOpsExperimentTracker(unittest.TestCase):
    """Test Suite for Model Registry and Experiment Versioning."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.registry_dir = os.path.join(self.test_dir, "registry")
        self.runs_dir = os.path.join(self.test_dir, "runs")
        self.tracker = ExperimentTracker(registry_dir=self.registry_dir, runs_dir=self.runs_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_catalog_initialization(self):
        models = self.tracker.list_models()
        self.assertIn("road_damage_detector", models)
        self.assertIn("vehicle_detector", models)
        self.assertIn("license_plate_detector", models)

        prod_road = self.tracker.get_production_model("road_damage_detector")
        self.assertIsNotNone(prod_road)
        self.assertEqual(prod_road["task"], "pavement_distress")
        self.assertIn("mAP50", prod_road["metrics"])

    def test_register_new_model_version(self):
        dummy_weights = os.path.join(self.test_dir, "dummy_model.pt")
        with open(dummy_weights, "wb") as f:
            f.write(b"ROADEYE_DUMMY_WEIGHTS_FOR_TESTING")

        checksum = ExperimentTracker.compute_sha256(dummy_weights)
        self.assertTrue(len(checksum) == 64)

        artifact = ModelArtifact(
            model_name="road_damage_detector",
            version="v2.0.0-rc1",
            task="pavement_distress",
            framework="Ultralytics YOLOv11",
            weights_path="models/road_damage_v2.pt",
            sha256_checksum=checksum,
            size_mb=18.5,
            registered_at=datetime.datetime.utcnow().isoformat() + "Z",
            metrics={"mAP50": 0.902, "precision": 0.915, "recall": 0.880},
            parameters={"imgsz": 640, "batch": 32, "epochs": 150},
            status="production"
        )
        res = self.tracker.register_model(artifact)
        self.assertEqual(res["version"], "v2.0.0-rc1")

        latest_prod = self.tracker.get_production_model("road_damage_detector")
        self.assertEqual(latest_prod["version"], "v2.0.0-rc1")
        self.assertEqual(latest_prod["metrics"]["mAP50"], 0.902)

    def test_log_experiment_run(self):
        run = ExperimentRun(
            run_id="run-exp-2026-001",
            experiment_name="yolov11-pavement-hyperband",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            model_version="v2.0.0-rc1",
            dataset_version="dvc://v2.1-debre-berhan",
            hyperparameters={"lr0": 0.002, "momentum": 0.937, "weight_decay": 0.0005},
            metrics={"mAP50": 0.902, "val_loss": 0.034},
            hardware_env={"gpu": "NVIDIA RTX 4090", "cuda": "12.2"},
            tags=["debre_berhan", "potholes", "benchmark"]
        )
        run_id = self.tracker.log_experiment(run)
        self.assertEqual(run_id, "run-exp-2026-001")

        runs = self.tracker.list_experiments()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["experiment_name"], "yolov11-pavement-hyperband")

    def test_dataset_version_metadata(self):
        meta = self.tracker.get_dataset_version_metadata()
        self.assertIn("dataset_name", meta)
        self.assertIn("classes", meta)
        self.assertEqual(meta["splits"]["train"]["percentage"], 80)
        self.assertTrue(len(meta["hash_sha256"]) > 0)


class TestMLOpsDriftMonitor(unittest.TestCase):
    """Test Suite for Covariate & Concept Drift Monitoring."""

    def setUp(self):
        self.monitor = DriftMonitor(window_size=50)

    def test_nominal_distribution_healthy_report(self):
        # Feed 30 nominal daylight samples
        for _ in range(30):
            self.monitor.record_inference(
                latency_ms=16.5,
                confidences=[0.75, 0.82, 0.68],
                classes=["pothole", "alligator_crack"],
                luminance=125.0,
                contrast=52.0
            )

        report = self.monitor.evaluate()
        self.assertEqual(report.status, "HEALTHY")
        self.assertFalse(report.covariate_drift["is_drifted"])
        self.assertFalse(report.concept_drift["is_drifted"])
        self.assertLess(report.drift_score, 0.2)
        self.assertGreater(report.latency_profile["p50_ms"], 0)

    def test_low_light_covariate_drift_detection(self):
        # Feed 20 night/tunnel low luminance samples
        for _ in range(20):
            self.monitor.record_inference(
                latency_ms=18.0,
                confidences=[0.60, 0.55],
                classes=["pothole"],
                luminance=35.0,  # Far below baseline 128.5 (Z-score > 2.0)
                contrast=20.0
            )

        report = self.monitor.evaluate()
        self.assertTrue(report.covariate_drift["is_drifted"])
        self.assertIn(report.status, ["WARNING", "DRIFT_DETECTED"])
        self.assertTrue(any("Low-Light" in a for a in report.anomalies))

    def test_concept_drift_degradation(self):
        # Feed severely degraded confidence scores
        for _ in range(25):
            self.monitor.record_inference(
                latency_ms=22.0,
                confidences=[0.31, 0.28, 0.35],  # Below min acceptable 0.45
                classes=["alligator_crack"],
                luminance=120.0,
                contrast=50.0
            )

        report = self.monitor.evaluate()
        self.assertTrue(report.concept_drift["is_drifted"])
        self.assertIn(report.status, ["WARNING", "DRIFT_DETECTED"])
        self.assertTrue(any("Concept Drift" in a for a in report.anomalies))
        self.assertTrue(any("retraining" in r.lower() for r in report.recommendations))

    def test_latency_spike_anomaly(self):
        # Feed high latency samples
        for _ in range(15):
            self.monitor.record_inference(
                latency_ms=125.0,  # Exceeds max P95 60ms
                confidences=[0.75],
                classes=["pothole"]
            )

        report = self.monitor.evaluate()
        self.assertGreater(report.latency_profile["p95_ms"], 60.0)
        self.assertTrue(any("Latency Spike" in a for a in report.anomalies))


if __name__ == "__main__":
    unittest.main()
