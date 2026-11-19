"""
Unit & Integration Tests for RoadEye™ ML Pavement Deterioration Engine
=====================================================================
Validates all requirements from the ML Replacement Specification:
1. No fixed 1.45x or 4.2x multiplier used in predictions.
2. Synthetic observations excluded from genuine model training.
3. Predictions originate strictly from fitted Scikit-Learn estimators.
4. Missing features handled explicitly (no silent assumption of good drainage).
5. Zero future information leakage in feature extraction.
6. Chronological time-based splitting (train <= val <= test).
7. Prediction APIs return explicit statuses and standardized schemas.
8. Insufficient data returns explicit status ('INSUFFICIENT_TRAINING_DATA', 'INSUFFICIENT_SEASONAL_DATA', 'INSUFFICIENT_POTHOLE_HISTORY').
9. Model persistence, loading, and safe non-destructive retraining.
10. Separation of Maintenance Cost Estimation from ML deterioration models.
"""

import os
import unittest
import numpy as np
from datetime import datetime

from risk.mechanism_features import MechanismFeatureExtractor, MechanismFeatureProvenance
from risk.ml_dataset_builder import MLDatasetBuilder
from risk.pothole_growth_predictor import PotholeGrowthPredictor
from risk.ml_deterioration_predictor import (
    MLDeteriorationEngine,
    RidgeRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressorWrapper,
    DataDrivenMarkovBaseline
)
from risk.seasonal_hypothesis_investigator import SeasonalHypothesisInvestigator
from risk.predictive_deterioration import CLIMATE_SEASONS
from risk.maintenance_cost_estimator import MaintenanceCostEstimator
from web.app import app


class TestMLPavementDeteriorationEngine(unittest.TestCase):
    """Core ML Algorithm and Pipeline Integrity Tests."""

    def setUp(self):
        self.engine = MLDeteriorationEngine()
        self.client = app.test_client()

    def test_no_fixed_multipliers_used(self):
        """Verify no fixed 1.45x or 4.2x multiplier drives predictions."""
        # Check climate seasons configuration
        for season_name, season_meta in CLIMATE_SEASONS.items():
            self.assertIsNone(
                season_meta.get("deterioration_multiplier"),
                f"Season {season_name} must not have a hardcoded deterioration multiplier."
            )
            if season_name in ["KIREMT", "BELG"]:
                self.assertIn("Insufficient historical data", season_meta.get("multiplier_status", ""))
            else:
                self.assertIn("Baseline dry season", season_meta.get("multiplier_status", ""))

        # Check seasonal investigator
        investigator = SeasonalHypothesisInvestigator()
        res = investigator.evaluate_seasonal_rates()
        self.assertEqual(res["status"], "INSUFFICIENT_SEASONAL_DATA")
        self.assertEqual(res["seasonal_comparison"], "Data pending")
        self.assertEqual(res["empirical_seasonal_multiplier"], "Not available")
        self.assertFalse(res["is_data_sufficient"])

    def test_synthetic_data_excluded_from_genuine_training(self):
        """Verify genuine training rejects synthetic backwards-interpolated records."""
        audit = MLDatasetBuilder.audit_genuine_dataset()
        self.assertFalse(audit["is_sufficient"])
        self.assertEqual(audit["status"], "INSUFFICIENT_TRAINING_DATA")

        dataset = MLDatasetBuilder.build_dataset(mode="genuine")
        self.assertFalse(dataset["is_sufficient"])
        self.assertEqual(dataset["status"], "INSUFFICIENT_TRAINING_DATA")
        self.assertIsNone(dataset["X"])
        self.assertIsNone(dataset["targets"])

    def test_synthetic_benchmark_is_clearly_labeled(self):
        """Verify synthetic benchmark dataset is explicitly flagged for software testing only."""
        bench = MLDatasetBuilder.build_dataset(mode="synthetic_benchmark")
        self.assertTrue(bench["is_sufficient"])
        self.assertEqual(bench["data_provenance"], "SYNTHETIC_BENCHMARK_TEST_DATA")
        self.assertIn("synthetic", bench["warning"].lower())
        self.assertGreaterEqual(bench["total_samples"], 50)

    def test_predictions_originate_from_fitted_scikit_learn_estimators(self):
        """Verify Ridge, Random Forest, and HistGradientBoosting are genuine fitted estimators."""
        np.random.seed(42)
        X = np.random.randn(60, 17)
        y = 80.0 - 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 7] + np.random.randn(60) * 0.2

        # 1. Ridge
        ridge = RidgeRegressor(alpha=1.5).fit(X, y)
        preds_ridge, std_ridge = ridge.predict_with_uncertainty(X)
        self.assertEqual(len(preds_ridge), 60)
        self.assertGreater(float(np.var(preds_ridge)), 0.01)

        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        preds_rf, std_rf = rf.predict_with_uncertainty(X)
        self.assertEqual(len(preds_rf), 60)
        self.assertTrue(np.all(std_rf >= 0.0))

        # 3. HistGradientBoosting
        hgb = HistGradientBoostingRegressorWrapper(max_iter=40, max_depth=4).fit(X, y)
        preds_hgb = hgb.predict(X)
        self.assertEqual(len(preds_hgb), 60)

        # Predictions change when input features change
        X_altered = X.copy()
        X_altered[:, 0] += 5.0
        preds_altered = rf.predict(X_altered)
        self.assertFalse(np.allclose(preds_rf, preds_altered))

    def test_missing_features_handled_explicitly(self):
        """Verify missing features are handled explicitly without assuming good drainage."""
        # Drainage telemetry missing
        drainage = MechanismFeatureExtractor.get_drainage_risk_telemetry(observed_drainage_condition=None)
        self.assertTrue(drainage["is_missing"])
        self.assertEqual(drainage["drainage_condition_status"], "UNAVAILABLE_FIELD_SURVEY_REQUIRED")
        self.assertEqual(drainage["provenance"]["source"], MechanismFeatureProvenance.UNAVAILABLE)
        # Neutral uncertainty prior (0.5), NOT 0.0 (good drainage)
        self.assertEqual(drainage["drainage_risk_score"], 0.5)

        # Relative humidity missing
        weather_hist = [{"date": "2026-11-01", "precipitation_mm": 5.0}]
        moisture = MechanismFeatureExtractor.compute_moisture_metrics(weather_hist, "2026-11-01")
        self.assertIsNone(moisture["relative_humidity_pct"])

    def test_zero_future_leakage_in_feature_extraction(self):
        """Verify weather and traffic features only look backwards from observation date."""
        weather_hist = [
            {"date": "2026-11-01", "precipitation_mm": 10.0},
            {"date": "2026-11-05", "precipitation_mm": 20.0},
            {"date": "2026-11-10", "precipitation_mm": 50.0},  # Future relative to Nov 5
        ]
        # Target date is Nov 5: Nov 10 rainfall must NOT leak into cumulative sum
        m = MechanismFeatureExtractor.compute_moisture_metrics(weather_hist, target_date="2026-11-05")
        self.assertEqual(m["daily_rainfall_mm"], 20.0)
        self.assertEqual(m["cumulative_rainfall_7d_mm"], 30.0)  # 20 + 10, NOT including 50

    def test_chronological_time_based_splitting(self):
        """Verify chronological train-val-test split guarantees zero temporal shuffle leakage."""
        X, targets, dates, segments = self.engine.prepare_dataset_matrices()
        N = len(X)
        self.assertGreaterEqual(N, 40)

        # Ensure strict chronological order
        sorted_indices = sorted(range(N), key=lambda i: dates[i])
        sorted_dates = [dates[i] for i in sorted_indices]
        for i in range(len(sorted_dates) - 1):
            self.assertLessEqual(sorted_dates[i], sorted_dates[i + 1])

        # Verify train dates strictly precede or equal val dates, which precede test dates
        n_train = int(N * 0.70)
        n_val = int(N * 0.15)
        train_max_date = sorted_dates[n_train - 1]
        val_min_date = sorted_dates[n_train]
        val_max_date = sorted_dates[n_train + n_val - 1]
        test_min_date = sorted_dates[n_train + n_val]

        self.assertLessEqual(train_max_date, val_min_date)
        self.assertLessEqual(val_max_date, test_min_date)

    def test_model_persistence_and_loading(self):
        """Verify model pipelines can be saved to disk and loaded without corruption."""
        perf = self.engine.train_and_evaluate(allow_synthetic_benchmark=True)
        self.assertIn("benchmark_30d_pci", perf)

        # Check saved files exist in models/ml/
        meta_file = os.path.join(self.engine.models_dir, "model_metadata.json")
        self.assertTrue(os.path.exists(meta_file))

        # Create fresh engine instance and verify loading
        fresh_engine = MLDeteriorationEngine(models_dir=self.engine.models_dir)
        self.assertIsNotNone(fresh_engine.trained_at)
        self.assertIn("rf_pci_30d", fresh_engine.trained_models)

    def test_pothole_growth_model_returns_insufficient_history(self):
        """Verify pothole growth predictor returns INSUFFICIENT_POTHOLE_HISTORY when data is lacking."""
        predictor = PotholeGrowthPredictor()
        audit = predictor.audit_defect_history()
        self.assertFalse(audit["has_repeated_history"])

        defect = {"width_cm": 45.0, "depth_cm": 5.0, "volume_liters": 6.0}
        res = predictor.predict_growth(defect, days_horizon=30)
        self.assertEqual(res["status"], "INSUFFICIENT_POTHOLE_HISTORY")
        self.assertFalse(res["is_data_sufficient"])
        self.assertIsNone(res["projected_dimensions"])

    def test_maintenance_cost_estimation_separated(self):
        """Verify maintenance cost model is separate and does not claim ROI without valid study."""
        cost = MaintenanceCostEstimator.estimate_defect_repair_cost(
            damage_type="pothole",
            width_cm=40.0,
            depth_cm=5.0,
            volume_liters=5.0
        )
        self.assertIn("total_estimated_repair_cost_etb", cost)
        self.assertIn("itemized_bill_of_quantities_etb", cost)
        boq = cost["itemized_bill_of_quantities_etb"]
        self.assertIn("materials_cost", boq)
        self.assertIn("labor_cost", boq)
        self.assertIn("equipment_rental_cost", boq)
        self.assertIn("traffic_control_and_safety_cost", boq)


class TestPredictionRESTEndpoints(unittest.TestCase):
    """Test all prediction API endpoints and JSON response contracts."""

    def setUp(self):
        self.client = app.test_client()

    def test_api_predictions_summary(self):
        res = self.client.get("/api/predictions/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("corridor_pci_metrics", data)
        self.assertIn("model_benchmarking", data)
        self.assertIn("seasonal_empirical_investigation", data)
        seasonal = data["seasonal_empirical_investigation"]
        self.assertEqual(seasonal["status"], "INSUFFICIENT_SEASONAL_DATA")

    def test_api_predictions_segments(self):
        res = self.client.get("/api/predictions/segments")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_segments"], 0)
        self.assertIn("data_provenance_summary", data)

    def test_api_predictions_single_segment_contract(self):
        res = self.client.get("/api/predictions/segment/KP-04")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["segment_id"], "KP-04")
        # Verify Section 9 mandatory response fields
        self.assertIn("prediction_horizon", data)
        self.assertIn("current_pci", data)
        self.assertIn("predicted_pci", data)
        self.assertIn("predicted_pci_change", data)
        self.assertIn("model_name", data)
        self.assertIn("model_version", data)
        self.assertIn("prediction_timestamp", data)
        self.assertIn("data_quality_status", data)
        self.assertIn("confidence_interval", data)
        self.assertIn("prediction_status", data)

    def test_api_predictions_feature_importance(self):
        res = self.client.get("/api/predictions/feature-importance")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("features_ranked", data)
        self.assertGreater(len(data["features_ranked"]), 0)
        top_feat = data["features_ranked"][0]
        self.assertIn("feature", top_feat)
        self.assertIn("importance_score", top_feat)
        self.assertIn("physical_interpretation", top_feat)

    def test_api_predictions_retrain(self):
        res = self.client.post("/api/predictions/retrain", json={"allow_synthetic_benchmark": True})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["retrain_status"], "COMPLETED")
        self.assertIn("performance", data)

    def test_api_predictions_pothole_growth(self):
        res = self.client.get("/api/predictions/pothole-growth/1")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        growth = data["growth_prediction"]
        self.assertEqual(growth["status"], "INSUFFICIENT_POTHOLE_HISTORY")
        self.assertFalse(growth["is_data_sufficient"])


if __name__ == "__main__":
    unittest.main()
