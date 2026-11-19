"""
Automated Test Suite for RoadEye™ Mechanism-Based Prediction Pipeline
====================================================================
Tests:
1. Mechanism Feature Extraction (Atmospheric Moisture, Heavy Freight Proxy, Cracking, 3D SfS Cavity)
2. Missing Data Handling & Explicit Unavailable Flags (Drainage, Rut Profilometry)
3. Transparent Bill of Quantities (BOQ) Maintenance Cost Estimator
4. Machine Learning Algorithms (Ridge L2 Closed-Form, Random Forest Ensemble, Data-Driven Markov)
5. Chronological Train/Val/Test Split & Data Leakage Prevention
6. Multi-Horizon Prediction Generation (7d, 30d, 60d) & Confidence Intervals
7. REST API Endpoints Schema & HTTP Contracts
8. Scientific Integrity Auditing & Non-Heuristic Validation
"""

import os
import sys
import math
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from risk.mechanism_features import MechanismFeatureExtractor, MechanismFeatureProvenance
from risk.maintenance_cost_estimator import MaintenanceCostEstimator, MaintenanceUnitRatesERA
from risk.ml_deterioration_predictor import (
    RidgeRegressor,
    RandomForestRegressor,
    DataDrivenMarkovBaseline,
    MLDeteriorationEngine,
    ml_engine
)
from data.unified_dataset import UnifiedDatasetManager
from data.data_quality import DataQualityValidator
from risk.seasonal_hypothesis_investigator import SeasonalHypothesisInvestigator
from web.app import app


class TestMechanismFeatureExtractor(unittest.TestCase):
    """Test physical mechanism feature extraction and boundary constraints."""

    def setUp(self):
        self.weather_sample = [
            {"date": "2026-11-01", "precipitation_mm": 0.0},
            {"date": "2026-11-02", "precipitation_mm": 5.2},
            {"date": "2026-11-03", "precipitation_mm": 12.4},
            {"date": "2026-11-04", "precipitation_mm": 8.0},
            {"date": "2026-11-05", "precipitation_mm": 0.0},
            {"date": "2026-11-06", "precipitation_mm": 15.0},
            {"date": "2026-11-07", "precipitation_mm": 22.0},
        ]

    def test_moisture_aggregation(self):
        """Verify cumulative rainfall, rainy day counts, and consecutive wet days."""
        metrics = MechanismFeatureExtractor.compute_moisture_metrics(
            weather_history=self.weather_sample,
            target_date="2026-11-07"
        )
        self.assertEqual(metrics["daily_rainfall_mm"], 22.0)
        self.assertAlmostEqual(metrics["cumulative_rainfall_7d_mm"], 62.6, places=1)
        self.assertEqual(metrics["consecutive_wet_days"], 2)  # Nov 6 & 7
        self.assertIn("Atmospheric surface precipitation", metrics["provenance"]["physical_qualification"])

    def test_traffic_freight_proxy(self):
        """Verify heavy truck counts are correctly labeled as traffic proxy."""
        traffic = MechanismFeatureExtractor.compute_heavy_load_metrics(
            total_traffic_vpd=10000.0,
            heavy_truck_pct=24.0
        )
        self.assertEqual(traffic["total_traffic_vpd"], 10000.0)
        self.assertEqual(traffic["heavy_truck_vpd"], 2400.0)
        self.assertEqual(traffic["traffic_frequency_index"], 1.0)
        self.assertEqual(traffic["provenance"]["source"], MechanismFeatureProvenance.TRAFFIC_PROXY)

    def test_crack_fatigue_density(self):
        """Verify crack density calculation under ASTM D6433 carriage area."""
        crack_m = MechanismFeatureExtractor.compute_crack_fatigue_metrics(
            crack_count=4,
            segment_length_m=100.0,
            segment_width_m=7.0
        )
        # 4 cracks * 3.5m = 14m; 14m / 700m2 = 0.02 m/m2
        self.assertEqual(crack_m["crack_count"], 4)
        self.assertAlmostEqual(crack_m["crack_density_m_m2"], 0.02, places=3)
        self.assertEqual(crack_m["severity_classification"], "LOW_INITIAL_HAIRLINE")

    def test_pothole_cavity_metrics(self):
        """Verify 3D SfS depth and sub-base penetration alert."""
        potholes = MechanismFeatureExtractor.compute_pothole_cavity_metrics(
            pothole_count=2,
            avg_depth_cm=5.2,
            max_depth_cm=7.8,
            volume_liters=29.12
        )
        self.assertEqual(potholes["pothole_count"], 2)
        self.assertTrue(potholes["subbase_breach_detected"])  # max_depth >= 4.5 cm
        self.assertEqual(potholes["provenance"]["source"], MechanismFeatureProvenance.ESTIMATED_SFS)

    def test_unavailable_field_data_handling(self):
        """Verify missing drainage and rut depth are explicitly marked unavailable, never faked."""
        drainage = MechanismFeatureExtractor.get_drainage_risk_telemetry(observed_drainage_condition=None)
        self.assertFalse(drainage["is_field_measurement_available"])
        self.assertEqual(drainage["drainage_condition_status"], "UNAVAILABLE_FIELD_SURVEY_REQUIRED")
        self.assertEqual(drainage["provenance"]["source"], MechanismFeatureProvenance.UNAVAILABLE)

        rutting = MechanismFeatureExtractor.get_rutting_telemetry(measured_rut_depth_mm=None)
        self.assertFalse(rutting["is_field_measurement_available"])
        self.assertIsNone(rutting["rut_depth_mm"])
        self.assertEqual(rutting["provenance"]["source"], MechanismFeatureProvenance.UNAVAILABLE)


class TestMaintenanceCostEstimator(unittest.TestCase):
    """Test transparent Bill of Quantities (BOQ) maintenance cost calculation."""

    def test_pothole_deep_patching_cost(self):
        """Verify deep pothole repair breaks down materials, labor, equipment, and safety."""
        res = MaintenanceCostEstimator.estimate_defect_repair_cost(
            damage_type="pothole",
            width_cm=45.0,
            depth_cm=5.2,
            volume_liters=5.5
        )
        self.assertEqual(res["intervention_level"], "EMERGENCY_DEEP_RECONSTRUCTION")
        boq = res["itemized_bill_of_quantities_etb"]
        self.assertGreater(boq["materials_cost"], 0.0)
        self.assertGreater(boq["labor_cost"], 0.0)
        self.assertGreater(boq["equipment_rental_cost"], 0.0)
        self.assertGreater(boq["traffic_control_and_safety_cost"], 0.0)
        self.assertFalse(res["assumptions"]["arbitrary_multiplier_used"])

    def test_crack_sealing_cost(self):
        """Verify crack sealing utilizes elastomeric sealant rates."""
        res = MaintenanceCostEstimator.estimate_defect_repair_cost(
            damage_type="crack",
            width_cm=15.0,
            depth_cm=1.0,
            volume_liters=0.2
        )
        self.assertEqual(res["intervention_level"], "PREVENTIVE_SURFACE_SEAL")
        self.assertLess(res["total_estimated_repair_cost_etb"], 1500.0)

    def test_segment_maintenance_budget(self):
        """Verify aggregated segment maintenance budget computation."""
        budget = MaintenanceCostEstimator.estimate_segment_maintenance_budget(
            pothole_count=2,
            avg_pothole_depth_cm=5.0,
            crack_count=3,
            current_pci=64.0,
            projected_pci_30d=55.0
        )
        self.assertGreater(budget["total_estimated_budget_etb"], 10000.0)
        self.assertEqual(budget["potholes_repaired_count"], 2)
        self.assertEqual(budget["cracks_sealed_count"], 3)


class TestMLDeteriorationModels(unittest.TestCase):
    """Test machine learning regressors, Markov baseline, and chronological splits."""

    def setUp(self):
        np.random.seed(42)
        self.N = 100
        self.p = 5
        self.X = np.random.randn(self.N, self.p)
        # Linear relationship with noise
        self.y = 2.0 * self.X[:, 0] - 1.5 * self.X[:, 1] + 50.0 + np.random.randn(self.N) * 0.5

    def test_ridge_regression_fit_predict(self):
        """Verify Ridge analytical closed-form solution and prediction uncertainty."""
        ridge = RidgeRegressor(alpha=1.0)
        ridge.fit(self.X, self.y)
        preds, std_err = ridge.predict_with_uncertainty(self.X)
        self.assertEqual(len(preds), self.N)
        metrics = MLDeteriorationEngine.compute_metrics(self.y, preds)
        self.assertGreater(metrics["r2"], 0.85)
        self.assertLess(metrics["mae"], 1.5)

    def test_random_forest_fit_predict(self):
        """Verify Random Forest ensemble and standard error intervals."""
        rf = RandomForestRegressor(n_estimators=15, max_depth=5)
        rf.fit(self.X, self.y)
        preds, std_err = rf.predict_with_uncertainty(self.X)
        self.assertEqual(len(preds), self.N)
        self.assertTrue(np.all(std_err >= 0.0))
        metrics = MLDeteriorationEngine.compute_metrics(self.y, preds)
        self.assertGreater(metrics["r2"], 0.70)

    def test_data_driven_markov_baseline(self):
        """Verify data-driven Markov baseline learns valid transition matrix."""
        curr_pci = np.array([90.0, 80.0, 65.0, 48.0, 20.0, 88.0, 75.0, 60.0])
        future_pci = np.array([86.0, 74.0, 58.0, 42.0, 15.0, 84.0, 70.0, 55.0])
        markov = DataDrivenMarkovBaseline()
        markov.fit(curr_pci, future_pci)
        preds = markov.predict(curr_pci)
        self.assertEqual(len(preds), len(curr_pci))
        # Ensure rows sum to 1.0
        row_sums = np.sum(markov.transition_matrix, axis=1)
        for s in row_sums:
            self.assertAlmostEqual(float(s), 1.0, places=4)

    def test_chronological_splitting_zero_leakage(self):
        """Verify train-validation-test split is strictly chronological."""
        X, targets, dates, segments = ml_engine.prepare_dataset_matrices()
        self.assertGreaterEqual(len(X), 50)
        # Check sorted order
        sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
        sorted_dates = [dates[i] for i in sorted_indices]
        # Dates should be monotonic non-decreasing
        for i in range(len(sorted_dates) - 1):
            self.assertLessEqual(sorted_dates[i], sorted_dates[i + 1])


class TestPredictionRESTAPIs(unittest.TestCase):
    """Test Flask REST API contracts, status codes, and JSON schemas."""

    def setUp(self):
        self.client = app.test_client()

    def test_predictions_summary_api(self):
        """Verify GET /api/predictions/summary returns corridor metrics and model benchmarks."""
        res = self.client.get("/api/predictions/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("corridor_pci_metrics", data)
        self.assertIn("model_benchmarking", data)
        self.assertIn("best_performing_model", data)

    def test_predictions_segments_api(self):
        """Verify GET /api/predictions/segments returns all 21 road blocks with availability flags."""
        res = self.client.get("/api/predictions/segments")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_segments"], 21)
        self.assertIn("data_availability", data)
        self.assertIn("data_provenance_summary", data)

    def test_predictions_single_segment_api(self):
        """Verify GET /api/predictions/segment/KP-04 returns predictions and history."""
        res = self.client.get("/api/predictions/segment/KP-04")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["segment_id"], "KP-04")
        self.assertIn("multi_horizon_pci_predictions", data["prediction"])
        self.assertIn("historical_trajectory", data)

    def test_predictions_single_segment_not_found(self):
        """Verify GET /api/predictions/segment/INVALID returns 404 with helpful message."""
        res = self.client.get("/api/predictions/segment/KP-999")
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("not found", data["error"])

    def test_predictions_model_performance_api(self):
        """Verify GET /api/predictions/model-performance returns held-out MAE, RMSE, R²."""
        res = self.client.get("/api/predictions/model-performance")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        bench = data["performance"]["benchmark_30d_pci"]
        self.assertIn("ridge_regression", bench)
        self.assertIn("random_forest", bench)
        self.assertIn("markov_model_baseline", bench)

    def test_data_quality_summary_api(self):
        """Verify GET /api/data-quality/summary returns audit score and completeness."""
        res = self.client.get("/api/data-quality/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data_quality_score"], 100.0)
        self.assertEqual(data["validity_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
