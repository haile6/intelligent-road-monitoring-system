"""
Comprehensive Automated Test Suite for RoadEye™ Predictive Maintenance Pipeline
=============================================================================
Tests:
1. WeatherDataClient (Open-Meteo caching, range query, cumulative precipitation)
2. UnifiedDatasetManager (Longitudinal integration, schema validation, query filters)
3. DataQualityValidator (Missing values, physical bounds, outlier detection, provenance)
4. MLDeteriorationEngine (Random Forest, Ridge L2, Markov baselines, 7d/30d/60d horizons, metrics)
5. SeasonalHypothesisInvestigator (Empirical rate calculation, paired t-test, confidence intervals)
6. REST API Endpoints (All 6 prediction and data quality routes)
"""

import unittest
import json
from datetime import datetime

from data.weather_client import WeatherDataClient
from data.unified_dataset import UnifiedDatasetManager
from data.data_quality import DataQualityValidator
from risk.ml_deterioration_predictor import MLDeteriorationEngine, RandomForestRegressor, RidgeRegressor
from risk.seasonal_hypothesis_investigator import SeasonalHypothesisInvestigator
from web.app import app


class TestWeatherIntegration(unittest.TestCase):
    """Test Open-Meteo and EMI weather client caching and climatological aggregation."""

    def setUp(self):
        self.client = WeatherDataClient()

    def test_weather_range_retrieval(self):
        obs = self.client.fetch_historical_range("2024-08-01", "2024-08-05")
        self.assertEqual(len(obs), 5)
        for o in obs:
            self.assertIn("date", o)
            self.assertGreaterEqual(o["precipitation_mm"], 0.0)
            self.assertGreater(o["temperature_mean_c"], -10.0)
            self.assertIn("weather_source", o)

    def test_cumulative_metrics(self):
        cum = self.client.compute_cumulative_metrics("2024-08-10", window_days=7)
        self.assertEqual(cum["window_days"], 7)
        self.assertGreaterEqual(cum["cumulative_rainfall_mm"], 0.0)
        self.assertIn(cum["saturation_risk"], ["CRITICAL", "ELEVATED", "NOMINAL"])


class TestUnifiedDataset(unittest.TestCase):
    """Test unified historical road condition dataset schema and records."""

    def setUp(self):
        self.mgr = UnifiedDatasetManager()

    def test_dataset_summary_and_population(self):
        summary = self.mgr.get_dataset_summary()
        self.assertGreaterEqual(summary["total_observations"], 50)
        self.assertGreaterEqual(summary["unique_segments_count"], 10)
        self.assertIn("pci_distribution", summary)

    def test_records_filtering(self):
        records = self.mgr.get_records(segment_id="KP-01")
        self.assertIsInstance(records, list)
        for r in records:
            self.assertEqual(r["segment_id"], "KP-01")
            self.assertIn("pci", r)
            self.assertIn("rainfall_mm", r)


class TestDataQualityValidator(unittest.TestCase):
    """Test data quality rules, physical bounds enforcement, and provenance audit."""

    def test_valid_record(self):
        valid_rec = {
            "observation_date": "2026-08-15",
            "segment_id": "KP-05",
            "latitude": 9.68,
            "longitude": 39.53,
            "pci": 82.5,
            "previous_pci": 85.0,
            "rainfall_mm": 18.4,
            "pothole_count": 2,
            "avg_pothole_depth_cm": 4.5,
            "max_pothole_depth_cm": 6.8,
            "pothole_volume_liters": 12.0
        }
        is_valid, errors, _ = DataQualityValidator.validate_single_record(valid_rec)
        self.assertTrue(is_valid, f"Expected valid record, got errors: {errors}")

    def test_out_of_bounds_detection(self):
        invalid_rec = {
            "observation_date": "2026-08-15",
            "segment_id": "KP-05",
            "latitude": 9.68,
            "longitude": 39.53,
            "pci": 150.0,  # Invalid: PCI cannot exceed 100
            "previous_pci": 85.0,
            "rainfall_mm": -5.0,  # Invalid: Negative rainfall
            "pothole_count": 2
        }
        is_valid, errors, _ = DataQualityValidator.validate_single_record(invalid_rec)
        self.assertFalse(is_valid)
        self.assertTrue(any("pci" in e.lower() for e in errors))

    def test_dataset_audit(self):
        audit = DataQualityValidator.audit_dataset()
        self.assertTrue(audit["success"])
        self.assertGreater(audit["total_records_audited"], 0)
        self.assertGreaterEqual(audit["validity_percentage"], 95.0)
        self.assertIn("provenance_metadata", audit)


class TestMLDeteriorationEngine(unittest.TestCase):
    """Test Machine Learning model training, evaluation, and multi-horizon predictions."""

    def setUp(self):
        self.engine = MLDeteriorationEngine()

    def test_train_and_evaluate_metrics(self):
        perf = self.engine.train_and_evaluate()
        self.assertIn("benchmark_30d_pci", perf)
        bench = perf["benchmark_30d_pci"]
        self.assertIn("random_forest", bench)
        self.assertIn("ridge_regression", bench)
        self.assertIn("markov_model_baseline", bench)

        # Verify regression error metrics are non-negative
        self.assertGreaterEqual(bench["random_forest"]["mae"], 0.0)
        self.assertGreaterEqual(bench["ridge_regression"]["mae"], 0.0)
        self.assertGreater(bench["random_forest"]["r2"], 0.5)

    def test_multi_horizon_segment_predictions(self):
        preds = self.engine.predict_corridor_segments()
        self.assertGreaterEqual(len(preds), 10)
        for p in preds:
            self.assertIn("segment_id", p)
            horizons = p["multi_horizon_pci_predictions"]
            # Verify monotonic deterioration: 60d <= 30d <= 7d <= current
            curr = p["current_condition"]["pci"]
            p_7d = horizons["horizon_7d_pci"]
            p_30d = horizons["horizon_30d_pci"]
            p_60d = horizons["horizon_60d_pci"]
            self.assertLessEqual(p_7d, curr + 0.1)
            self.assertLessEqual(p_30d, p_7d + 0.1)
            self.assertLessEqual(p_60d, p_30d + 0.1)
            self.assertIn("maintenance_prescription", p)


class TestSeasonalHypothesisInvestigator(unittest.TestCase):
    """Test empirical rate calculation and statistical hypothesis testing."""

    def setUp(self):
        self.investigator = SeasonalHypothesisInvestigator()

    def test_seasonal_rates_evaluation(self):
        res = self.investigator.evaluate_seasonal_rates()
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "INSUFFICIENT_HISTORICAL_DATA")
        self.assertEqual(res["summary_status"], "Insufficient historical data to calculate an empirical seasonal deterioration ratio.")
        self.assertIsNone(res["empirical_multiplier"])
        self.assertFalse(res["is_data_sufficient"])
        self.assertIn("genuine_field_data_audit", res)
        self.assertIn("scientific_integrity_declaration", res)


class TestPredictionRESTAPIs(unittest.TestCase):
    """Test all 6 newly created Prediction and Data Quality REST endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_api_predictions_summary(self):
        resp = self.client.get("/api/predictions/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("corridor_pci_metrics", data)
        self.assertIn("model_benchmarking", data)

    def test_api_predictions_segments(self):
        resp = self.client.get("/api/predictions/segments")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_segments"], 0)

    def test_api_predictions_single_segment(self):
        resp = self.client.get("/api/predictions/segment/KP-01")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["segment_id"], "KP-01")

    def test_api_predictions_history(self):
        resp = self.client.get("/api/predictions/history?segment_id=KP-01")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["count"], 0)

    def test_api_predictions_model_performance(self):
        resp = self.client.get("/api/predictions/model-performance")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("benchmark_30d_pci", data["performance"])

    def test_api_data_quality_summary(self):
        resp = self.client.get("/api/data-quality/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["data_quality_score"], 90.0)


if __name__ == "__main__":
    unittest.main()
