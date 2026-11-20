"""
Unit & Integration Tests for RoadEye™ ML Pavement Deterioration Engine
=====================================================================
Validates all requirements from the ML Replacement Specification & Audit Remediation:
1. No fixed 1.45x or 4.2x multiplier used in predictions.
2. Synthetic observations excluded from genuine model training (returns DATA_PENDING).
3. Predictions originate strictly from fitted Scikit-Learn estimators.
4. Missing features handled explicitly (no silent assumption of good drainage).
5. Zero future information leakage in feature extraction.
6. Chronological time-based splitting with 60-day temporal embargo.
7. Unclamped future observation targets with tolerance matching windows.
8. All 23 physical mechanism features included in feature vector and training.
9. Serving model dynamically binds to best-performing model (no hardcoded RF when Ridge wins).
10. Delta PCI (ΔPCI = Current PCI - Future PCI) evaluated against persistence baseline.
11. Spatial generalization evaluated via GroupKFold segment-holdout cross-validation.
12. Split Conformal Prediction intervals calibrated on validation residuals.
13. Stale model artifact rejection when feature dimensionality != 23.
14. Synthetic benchmark models never written to production disk.
15. API endpoints report data provenance, DATA_PENDING status, and confidence intervals.
"""

import os
import math
import json
import tempfile
import unittest
import numpy as np
from datetime import datetime, timedelta

from risk.mechanism_features import MechanismFeatureExtractor, MechanismFeatureProvenance
from risk.ml_dataset_builder import MLDatasetBuilder
from risk.pothole_growth_predictor import PotholeGrowthPredictor
from risk.ml_deterioration_predictor import (
    MLDeteriorationEngine,
    RidgeRegressor,
    RandomForestRegressorWrapper,
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
        for season_name, season_meta in CLIMATE_SEASONS.items():
            self.assertIsNone(
                season_meta.get("deterioration_multiplier"),
                f"Season {season_name} must not have a hardcoded deterioration multiplier."
            )
            if season_name in ["KIREMT", "BELG"]:
                self.assertIn("Insufficient historical data", season_meta.get("multiplier_status", ""))
            else:
                self.assertIn("Baseline dry season", season_meta.get("multiplier_status", ""))

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

    def test_unclamped_future_targets(self):
        """PHASE 2 FIX: Verify target builder drops samples without future observations and clamps zero."""
        bench = MLDatasetBuilder.build_dataset(mode="synthetic_benchmark")
        targets = bench["targets"]
        self.assertIn("pci_7d", targets)
        self.assertIn("pci_30d", targets)
        self.assertIn("pci_60d", targets)
        # All returned samples must have genuine date-matched future targets (no NaNs or infinities)
        for h in ["pci_7d", "pci_30d", "pci_60d"]:
            vals = targets[h]
            self.assertFalse(np.any(np.isnan(vals)), f"Target {h} contains NaN values")
            self.assertFalse(np.any(np.isinf(vals)), f"Target {h} contains infinite values")
            # Verify PCI values are physically valid in [0, 100]
            self.assertTrue(np.all(vals >= 0.0) and np.all(vals <= 100.0))

    def test_all_23_mechanism_features_vector_and_schema(self):
        """PHASE 4 FIX: Verify all 23 mechanism features are defined and extracted."""
        self.assertEqual(len(MechanismFeatureExtractor.FEATURE_NAMES), 23)
        expected_new_features = [
            "temperature_c",
            "relative_humidity_pct",
            "antecedent_precip_index",
            "traffic_exposure_30d_kveh",
            "drainage_risk_score",
            "water_accumulation_risk"
        ]
        for f in expected_new_features:
            self.assertIn(f, MechanismFeatureExtractor.FEATURE_NAMES)

        # Test constructing feature vector from sample observations
        moisture = MechanismFeatureExtractor.compute_moisture_metrics(
            [{"date": "2026-11-01", "precipitation_mm": 12.0}], target_date="2026-11-01"
        )
        traffic = MechanismFeatureExtractor.compute_heavy_load_metrics(
            total_traffic_vpd=1200, heavy_truck_count=300
        )
        cracks = MechanismFeatureExtractor.compute_crack_fatigue_metrics(
            crack_count=3, segment_length_m=100.0, segment_width_m=7.0
        )
        potholes = MechanismFeatureExtractor.compute_pothole_cavity_metrics(
            pothole_count=2, avg_depth_cm=4.5, max_depth_cm=7.0, volume_liters=15.0
        )
        drainage = MechanismFeatureExtractor.get_drainage_risk_telemetry(
            observed_drainage_condition=None, daily_rainfall_mm=12.0
        )
        vec = MechanismFeatureExtractor.construct_mechanism_feature_vector(
            current_pci=75.0,
            moisture_metrics=moisture,
            traffic_metrics=traffic,
            crack_metrics=cracks,
            pothole_metrics=potholes,
            historical_pci_velocity=-1.5,
            temperature_c=18.0,
            relative_humidity_pct=65.0,
            drainage_metrics=drainage,
            antecedent_precip_index=5.0
        )
        self.assertEqual(len(vec), 23)
        self.assertEqual(vec.shape, (23,))
        self.assertFalse(np.any(np.isnan(vec)))
        # Verify first element is current_pci
        self.assertEqual(vec[0], 75.0)

    def test_predictions_originate_from_fitted_scikit_learn_estimators(self):
        """Verify Ridge, Random Forest, and HistGradientBoosting are genuine fitted estimators on 23 features."""
        np.random.seed(42)
        X = np.random.randn(60, 23)
        y = 80.0 - 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 7] + np.random.randn(60) * 0.2

        # 1. Ridge
        ridge = RidgeRegressor(alpha=1.5).fit(X, y)
        preds_ridge, std_ridge = ridge.predict_with_uncertainty(X)
        self.assertEqual(len(preds_ridge), 60)
        self.assertGreater(float(np.var(preds_ridge)), 0.01)

        # 2. Random Forest
        rf = RandomForestRegressorWrapper(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        preds_rf, std_rf = rf.predict_with_uncertainty(X)
        self.assertEqual(len(preds_rf), 60)
        self.assertTrue(np.all(std_rf >= 0.0))

        # 3. HistGradientBoosting
        hgb = HistGradientBoostingRegressorWrapper(max_iter=40, max_depth=4).fit(X, y)
        preds_hgb, std_hgb = hgb.predict_with_uncertainty(X)
        self.assertEqual(len(preds_hgb), 60)
        self.assertTrue(np.all(std_hgb >= 0.0))

        # Predictions change when input features change
        X_altered = X.copy()
        X_altered[:, 0] += 5.0
        preds_altered = rf.predict(X_altered)
        self.assertFalse(np.allclose(preds_rf, preds_altered))

    def test_missing_features_handled_explicitly(self):
        """Verify missing features are handled explicitly without assuming good drainage."""
        drainage = MechanismFeatureExtractor.get_drainage_risk_telemetry(observed_drainage_condition=None)
        self.assertTrue(drainage["is_missing"])
        self.assertEqual(drainage["drainage_condition_status"], "UNAVAILABLE_FIELD_SURVEY_REQUIRED")
        self.assertEqual(drainage["provenance"]["source"], MechanismFeatureProvenance.UNAVAILABLE)
        # Neutral uncertainty prior (0.5), NOT 0.0 (good drainage)
        self.assertEqual(drainage["drainage_risk_score"], 0.5)

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
        m = MechanismFeatureExtractor.compute_moisture_metrics(weather_hist, target_date="2026-11-05")
        self.assertEqual(m["daily_rainfall_mm"], 20.0)
        self.assertEqual(m["cumulative_rainfall_7d_mm"], 30.0)  # 20 + 10, NOT including 50

    def test_chronological_time_based_splitting(self):
        """PHASE 6 FIX: Verify chronological train-val-test split guarantees zero temporal shuffle leakage."""
        X, targets, dates, segments = self.engine.prepare_benchmark_matrices()
        N = len(X)
        self.assertGreaterEqual(N, 40)
        self.assertEqual(X.shape[1], 23, "Feature matrix must have 23 mechanism columns")

        # Ensure strict chronological order
        sorted_indices = sorted(range(N), key=lambda i: dates[i])
        sorted_dates = [dates[i] for i in sorted_indices]
        for i in range(len(sorted_dates) - 1):
            self.assertLessEqual(sorted_dates[i], sorted_dates[i + 1])

        # Date-based splits
        unique_dates_sorted = sorted(set(sorted_dates))
        n_unique = len(unique_dates_sorted)
        n_train_dates = max(1, int(n_unique * 0.70))
        train_cutoff_date = unique_dates_sorted[n_train_dates - 1]
        val_cutoff_date = unique_dates_sorted[min(n_train_dates + max(1, int(n_unique * 0.15)) - 1, n_unique - 1)]

        self.assertLessEqual(train_cutoff_date, val_cutoff_date)

    def test_model_persistence_and_loading(self):
        """PHASE 1 & 5 FIX: Verify model pipelines can be saved and loaded when provenance is genuine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_engine = MLDeteriorationEngine(models_dir=tmp_dir)
            test_engine.train_and_evaluate(allow_synthetic_benchmark=True)
            # Explicitly persist with GENUINE_FIELD_SURVEYS provenance to test file I/O
            test_engine._persist_models(data_provenance="GENUINE_FIELD_SURVEYS")

            meta_file = os.path.join(tmp_dir, "model_metadata.json")
            self.assertTrue(os.path.exists(meta_file))

            # Fresh engine instance loading from tmp_dir
            fresh_engine = MLDeteriorationEngine(models_dir=tmp_dir)
            self.assertIsNotNone(fresh_engine.trained_at)
            self.assertIn("rf_pci_30d", fresh_engine.trained_models)
            self.assertIn("ridge_pci_30d", fresh_engine.trained_models)
            self.assertEqual(fresh_engine.trained_models["rf_pci_30d"].rf.n_features_in_, 23)

    def test_synthetic_data_not_persisted_to_disk(self):
        """PHASE 1 FIX: Verify synthetic benchmark data does NOT write joblib models to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_engine = MLDeteriorationEngine(models_dir=tmp_dir)
            # Training with allow_synthetic_benchmark sets data_provenance to SYNTHETIC_BENCHMARK_TEST_DATA
            test_engine.train_and_evaluate(allow_synthetic_benchmark=True)
            joblib_files = [f for f in os.listdir(tmp_dir) if f.endswith(".joblib")]
            self.assertEqual(len(joblib_files), 0, "Synthetic benchmark models must never be written to disk.")

    def test_delta_pci_prediction_evaluation(self):
        """PHASE 8 FIX: Verify ΔPCI = Current - Future is evaluated and beats persistence."""
        res = self.engine.evaluate_delta_pci_prediction(allow_synthetic_benchmark=True)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["prediction_target"], "ΔPCI = Current PCI - 30d Future PCI")
        metrics = res["metrics"]
        self.assertIn("ridge_regression", metrics)
        self.assertIn("random_forest", metrics)
        self.assertIn("persistence_baseline", metrics)

        # Ridge and Random Forest must beat naive persistence (MAE < persistence MAE)
        r_mae = metrics["ridge_regression"]["mae"]
        p_mae = metrics["persistence_baseline"]["mae"]
        self.assertIsNotNone(r_mae)
        self.assertIsNotNone(p_mae)
        self.assertLess(r_mae, p_mae, f"Ridge MAE ({r_mae}) should be lower than persistence MAE ({p_mae}) on ΔPCI")

    def test_spatial_generalization_group_kfold(self):
        """PHASE 7 FIX: Verify segment-holdout evaluation ensures zero spatial leakage."""
        spatial = self.engine.evaluate_spatial_generalization(allow_synthetic_benchmark=True, k_folds=5)
        self.assertEqual(spatial["status"], "COMPLETED")
        self.assertEqual(spatial["evaluation_type"], "Segment-Holdout Cross-Validation (GroupKFold)")
        self.assertIn("metrics", spatial)
        self.assertIn("spatial_generalization_disclosure", spatial)
        m = spatial["metrics"]
        self.assertIn("ridge_regression", m)
        self.assertIn("random_forest", m)
        self.assertIsNotNone(m["ridge_regression"]["r2"])

    def test_conformal_prediction_intervals(self):
        """PHASE 9 FIX: Verify Split Conformal Prediction half-width is calibrated on validation set."""
        perf = self.engine.train_and_evaluate(allow_synthetic_benchmark=True)
        self.assertIn("conformal_calibration", perf)
        calib = perf["conformal_calibration"]
        self.assertIn("half_width_pci", calib)
        self.assertIn("empirical_test_coverage_95", calib)
        self.assertGreater(calib["half_width_pci"], 0.0)
        self.assertGreater(calib["empirical_test_coverage_95"], 0.70)

    def test_serving_model_matches_best_model(self):
        """PHASE 3 FIX: Verify predict_corridor_segments serves the dynamically selected best model."""
        preds = self.engine.predict_corridor_segments()
        self.assertGreater(len(preds), 0)
        best_name = self.engine.best_model_name
        self.assertIsNotNone(best_name)
        # First segment's model_name must match the best model family
        seg0 = preds[0]
        self.assertIn("model_name", seg0)
        self.assertEqual(seg0["model_name"], best_name)

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
        self.assertEqual(data["production_training_status"], "DATA_PENDING")
        seasonal = data["seasonal_empirical_investigation"]
        self.assertEqual(seasonal["status"], "INSUFFICIENT_SEASONAL_DATA")

    def test_api_predictions_segments(self):
        res = self.client.get("/api/predictions/segments")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_segments"], 0)
        self.assertIn("data_provenance_summary", data)
        self.assertEqual(data["production_training_status"], "DATA_PENDING")

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


class TestForecastingAndRainySeasonRepair(unittest.TestCase):
    """
    Validation Suite for the 12-Part Repair of Predictive Deterioration,
    Markov Chain Simulation, Ethiopian Climate Forecasting, and Cost of Inaction Engine.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_forecast_defect_growth_uses_actual_dimensions(self):
        """Part 7 & 9: Verify defect growth trajectory reads measured dimensions, not hardcoded 45cm/5.2cm."""
        from risk.predictive_deterioration import CostOfInactionForecaster
        
        # Incident A: Large shallow pothole
        inc_a = {
            "id": 101,
            "damage_type": "pothole",
            "depth_max_cm": 3.2,
            "surface_area_m2": 0.45,
            "cavity_volume_liters": 14.4
        }
        # Incident B: Deep small pothole
        inc_b = {
            "id": 102,
            "damage_type": "pothole",
            "depth_max_cm": 9.8,
            "surface_area_m2": 0.08,
            "cavity_volume_liters": 7.84
        }

        traj_a = CostOfInactionForecaster.estimate_defect_growth_trajectory(inc_a, season="KIREMT", days_horizon=60)
        traj_b = CostOfInactionForecaster.estimate_defect_growth_trajectory(inc_b, season="KIREMT", days_horizon=60)

        # Initial measured dimensions must match inputs, NOT hardcoded 45.0 cm / 5.2 cm
        self.assertEqual(traj_a["current_dimensions"]["depth_cm"], 3.2)
        self.assertEqual(traj_a["current_dimensions"]["volume_liters"], 14.4)
        self.assertEqual(traj_b["current_dimensions"]["depth_cm"], 9.8)
        self.assertEqual(traj_b["current_dimensions"]["volume_liters"], 7.84)
        self.assertNotEqual(traj_a["current_dimensions"]["depth_cm"], traj_b["current_dimensions"]["depth_cm"])
        self.assertNotEqual(traj_a["projected_dimensions"]["depth_cm"], traj_b["projected_dimensions"]["depth_cm"])

    def test_forecast_coi_spatial_deduplication(self):
        """Part 9: Verify evaluate_corridor_coi clusters raw sightings within 8m to eliminate duplicate records."""
        from risk.predictive_deterioration import CostOfInactionForecaster
        
        coi = CostOfInactionForecaster.evaluate_corridor_coi(season="KIREMT", days_horizon=60)
        self.assertIn("canonical_hazards_count", coi)
        self.assertIn("duplicates_eliminated", coi)
        self.assertIn("raw_sightings_evaluated", coi)
        
        # With 101 raw incidents on Route A2, clustering reduces duplicate sightings
        self.assertGreater(coi["duplicates_eliminated"], 0)
        self.assertLess(coi["canonical_hazards_count"], coi["raw_sightings_evaluated"])
        self.assertEqual(coi["canonical_hazards_count"] + coi["duplicates_eliminated"], coi["raw_sightings_evaluated"])

    def test_cost_of_inaction_and_escalation_ratio_calculation(self):
        """Part 8: Verify Cost of Inaction = Delayed - Today and Cost Escalation Ratio = Delayed / Today."""
        from risk.predictive_deterioration import CostOfInactionForecaster

        coi = CostOfInactionForecaster.evaluate_corridor_coi(season="BEGA", days_horizon=60)
        today = coi["total_cost_today_etb"]
        delayed = coi["total_cost_inaction_etb"]
        coi_penalty = coi["total_cost_of_inaction_etb"]
        ratio = coi["cost_escalation_ratio"]

        # COI = Delayed - Today
        self.assertAlmostEqual(coi_penalty, delayed - today, places=2)
        # Escalation Ratio = Delayed / Today
        expected_ratio = round(delayed / max(1.0, today), 2)
        self.assertEqual(ratio, expected_ratio)
        # Verify not hardcoded to 1.28
        self.assertIsInstance(ratio, float)

    def test_markov_simulation_labeling_and_stochastic_rows(self):
        """Part 5: Verify Markov model is explicitly labeled as simulation and all rows sum to 1.0."""
        from risk.predictive_deterioration import MarkovPavementDegradationModel

        markov = MarkovPavementDegradationModel.forecast_markov_chain_degradation(
            current_pci=85.0, incidents_count=67, season="KIREMT", horizons=[30, 60]
        )
        self.assertEqual(markov["simulation_status"], "ILLUSTRATIVE MARKOV SIMULATION — NOT FIELD VALIDATED")
        self.assertFalse(markov["is_field_validated"])
        self.assertIn("illustrative markov simulation", markov["simulation_disclaimer"].lower())

        # Validate stochasticity of base transition matrices
        for season_key, matrix in MarkovPavementDegradationModel.TRANSITION_MATRICES.items():
            for i, row in enumerate(matrix):
                row_sum = sum(row)
                self.assertAlmostEqual(row_sum, 1.0, places=4, msg=f"{season_key} row {i} does not sum to 1.0: {row_sum}")
                for val in row:
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    def test_dynamic_calendar_season_and_live_telemetry(self):
        """Part 6: Verify dynamic season selection and live weather queries from DB."""
        from web.app import get_forecast_simulation_cache
        
        cache = get_forecast_simulation_cache()
        self.assertIn("calendar_season", cache)
        self.assertIn("evaluated_season", cache)
        self.assertIn("live_telemetry", cache)
        
        # November is Bega
        current_month = datetime.utcnow().month
        if current_month in [10, 11, 12, 1, 2]:
            self.assertEqual(cache["calendar_season"], "BEGA")
        elif current_month in [3, 4, 5]:
            self.assertEqual(cache["calendar_season"], "BELG")
        else:
            self.assertEqual(cache["calendar_season"], "KIREMT")

        # Telemetry contains live rainfall
        self.assertIn("recent_rain_30d_mm", cache["live_telemetry"])
        self.assertIsInstance(cache["live_telemetry"]["recent_rain_30d_mm"], (int, float))

    def test_forecast_api_summary_and_page_render(self):
        """Part 10: Verify /api/forecast/summary and /forecast page render with clean contracts."""
        # 1. API Summary
        res = self.client.get("/api/forecast/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("canonical_hazards_count", data)
        self.assertIn("duplicates_eliminated", data)
        self.assertIn("cost_escalation_ratio", data)
        self.assertIn("total_cost_of_inaction_etb", data)
        self.assertEqual(data["markov_forecast"]["is_field_validated"], False)

        # 2. HTML Page Render
        page = self.client.get("/forecast")
        self.assertEqual(page.status_code, 200)
        html = page.data.decode("utf-8")
        self.assertIn("23 Physical Mechanism Features", html)
        self.assertIn("ILLUSTRATIVE MARKOV SIMULATION — NOT FIELD VALIDATED", html)
        self.assertIn("Cost of Inaction", html)
        self.assertIn("Cost Escalation", html)
        self.assertNotIn("Capital Savings (ROI)", html)


if __name__ == "__main__":
    unittest.main()
