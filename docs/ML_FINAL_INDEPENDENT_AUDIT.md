# RoadEye™ Final Independent Machine Learning Validation Audit

**Audit Date**: November 20, 2026  
**Auditor**: Senior Machine Learning & Pavement Engineering Validation Specialist  
**Audited Commit**: `c798fdc` (`feat(ml): complete genuine machine learning pavement deterioration replacement`)  
**Target Repository**: [intelligent-road-monitoring-system](https://github.com/haile6/intelligent-road-monitoring-system)  
**Final Audit Verdict**: **NOT YET SCIENTIFICALLY VALIDATED**

---

## 1. Executive Summary

This independent scientific audit evaluated the pavement deterioration machine learning pipeline implemented in commit `c798fdc`. The audit examined mathematical formulations, data provenance, temporal splitting, target alignment, feature vectors, uncertainty quantification, and model serving.

### High-Level Audit Findings:
1. **Scikit-Learn Implementation**: The system successfully discarded fixed heuristic multipliers ($4.2\times$ and $1.45\times$) and built real Scikit-Learn wrappers (`Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`).
2. **Data Provenance**: The platform honestly reports `INSUFFICIENT_TRAINING_DATA` and `INSUFFICIENT_SEASONAL_DATA` when queried in genuine mode. However, the models currently serialized in `models/ml/` and serving predictions on the dashboard were trained and evaluated on **synthetic backwards-interpolated benchmark data** (`SYNTHETIC_BENCHMARK_TEST_DATA`).
3. **Critical Target Alignment Flaw**: In the synthetic dataset, **100% of 60-day targets and 100% of 30-day targets in the test split were clamped** to the final trajectory index ($T-1$) because lookahead was not truncated before the end of the time series. The reported 60-day test metrics evaluate a 7-to-21-day forecast disguised as 60-day.
4. **Model Serving Divergence**: The API summary states that `Ridge Regression` is the active best model (MAE 0.64), but the segment prediction loop in `predict_corridor_segments()` hardcodes calls to `Random Forest` (MAE 4.88, which is **256% worse than naive persistence**).
5. **Feature Omission**: While 23 features are documented in `model_metadata.json`, the actual feature vector passed to estimators contains only 17 features. Critical physical parameters—including `drainage_risk_score` and `antecedent_precip_index`—are omitted from model training and inference.
6. **Spatial Generalization Collapse**: In a segment-holdout experiment on unobserved road sections, Random Forest performance collapsed from $R^2 = 0.876$ down to $R^2 = 0.076$ (MAE = 14.98), demonstrating that tree splits memorized segment-specific condition tiers rather than generalizable degradation physics.

---

## 2. Comprehensive Audit Matrix (14 Validation Criteria)

| # | Validation Item | Scientific Status | Severity | Primary File & Function |
|---|---|---|---|---|
| **1** | Genuine Scikit-Learn Estimators | **VERIFIED** | Low | `risk/ml_deterioration_predictor.py` (`RidgeRegressor`, `RandomForestRegressorWrapper`) |
| **2** | Synthetic Data Exclusion in Production | **FAILED** | **CRITICAL** | `risk/ml_deterioration_predictor.py` (`prepare_dataset_matrices`) |
| **3** | Chronological 70/15/15 Split & Leakage | **DEFICIENT** | **HIGH** | `risk/ml_deterioration_predictor.py` (`train_and_evaluate`) |
| **4** | Spatial Segment Overlap & Holdout | **DEFICIENT** | **HIGH** | `risk/ml_deterioration_predictor.py` (`train_and_evaluate`) |
| **5** | Current PCI Autoregressive Level Effect | **PARTIAL** | **MEDIUM** | `risk/mechanism_features.py` (`construct_mechanism_feature_vector`) |
| **6** | Ridge vs Persistence Sample Alignment | **VERIFIED** | Low | `risk/ml_deterioration_predictor.py` (`train_and_evaluate`) |
| **7** | ML Improvement Over Persistence | **CONTRADICTORY** | **CRITICAL** | `risk/ml_deterioration_predictor.py` (`predict_corridor_segments`) |
| **8** | Full Mechanism Feature Usage | **FAILED** | **HIGH** | `risk/mechanism_features.py` (`construct_mechanism_feature_vector`) |
| **9** | Permutation Importance Split & Model | **PARTIAL** | **MEDIUM** | `risk/ml_deterioration_predictor.py` (`_compute_feature_importance`) |
| **10** | 95% Prediction Interval Calibration | **DEFICIENT** | **MEDIUM** | `risk/ml_deterioration_predictor.py` (`predict_with_uncertainty`) |
| **11** | Multi-Horizon Target Alignment & Clamping | **FAILED** | **CRITICAL** | `risk/ml_dataset_builder.py` (`_build_benchmark_synthetic_dataset`) |
| **12** | Model Artifact vs Metadata Consistency | **FAILED** | **HIGH** | `models/ml/model_metadata.json` |
| **13** | Safe Handling of Missing Observations | **VERIFIED** | Low | `risk/mechanism_features.py`, `risk/pothole_growth_predictor.py` |
| **14** | Seasonal Investigator Reporting | **VERIFIED** | Low | `risk/seasonal_hypothesis_investigator.py` (`evaluate_seasonal_rates`) |

---

## 3. Detailed Audit Findings & Forensic Evidence

### Finding 1: Estimator Authenticity
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `RidgeRegressor`, `RandomForestRegressorWrapper`, `HistGradientBoostingRegressorWrapper`
- **Evidence**:
  - `models/ml/ridge_pci_30d.joblib`: Contains `sklearn.pipeline.Pipeline` with `SimpleImputer`, `StandardScaler`, and `sklearn.linear_model.Ridge(alpha=2.0)`. Coefficient vector has 17 fitted values.
  - `models/ml/rf_pci_30d.joblib`: Contains `RandomForestRegressor` with 40 genuinely fitted decision trees (`len(rf.estimators_) == 40`).
  - `models/ml/hgb_pci_30d.joblib`: Contains `HistGradientBoostingRegressor` with `n_iter_ == 80`.
- **Scientific Conclusion**: Models are genuine Scikit-Learn estimators. No hardcoded prediction formulas or mock classes exist.
- **Severity**: Low.

---

### Finding 2: Synthetic Data Contamination in Persisted Production Models
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `prepare_dataset_matrices()` (line 257) & `train_and_evaluate()`
- **Evidence**:
  ```python
  def prepare_dataset_matrices(self):
      bench = MLDatasetBuilder.build_dataset(mode="synthetic_benchmark")
      X = bench["X"]
      ...
  ```
  `train_and_evaluate(allow_synthetic_benchmark=True)` is called during startup, background retraining, and CLI execution. Because SQLite contains genuine observations for only 2 dry-season dates (`2026-10-20` and `2026-11-16`), the models serialized to `models/ml/*.joblib` and served on the dashboard were trained on `SYNTHETIC_BENCHMARK_TEST_DATA`.
- **Scientific Issue**: Synthetic data generated via backward interpolation ($1.0 + 0.18 \times \ln(1 + \text{rain})$) was ingested into the deployed model weights. Software tests enforce labeling, but the deployed `.joblib` files contain weights learned from synthetic trajectories.
- **Severity**: **CRITICAL**.
- **Recommended Correction**: When genuine repeated survey data is insufficient, the system must not write trained weights to production `.joblib` files. The dashboard should clearly lock prediction inference in `DATA_PENDING` mode rather than executing synthetic-trained models.

---

### Finding 3: Chronological Split Boundary Collision & Lookahead Leakage
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `train_and_evaluate()` (lines 320–337)
- **Evidence**:
  ```python
  sorted_indices = sorted(range(N), key=lambda i: dates[i])
  n_train = int(N * 0.70)
  n_val = int(N * 0.15)
  ```
  1. **Date Collision**: `sorted_dates[n_train - 1] == '2026-10-07'` and `sorted_dates[n_train] == '2026-10-07'`. Splitting by sample index splits the date `2026-10-07` across both train and validation splits.
  2. **Lookahead Horizon Overlap**: For training observations on `2026-10-07`, the 30-day future target $y_{t+30}$ falls on `2026-11-04`. However, validation set features $X_{\text{val}}$ span `2026-10-07` to `2026-10-28`. Training targets extend temporally past the validation features.
- **Scientific Issue**: Lack of temporal embargo buffer creates lookahead leakage between training targets and validation inputs.
- **Severity**: **HIGH**.
- **Recommended Correction**: Split strictly on unique calendar dates with an embargo buffer equal to the forecast horizon ($t_{\text{train\_max}} + H_{\text{days}} \le t_{\text{val\_min}}$).

---

### Finding 4: Spatial Generalization Failure on Unseen Road Segments
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `train_and_evaluate()`
- **Evidence**:
  In the current split, all 21 road segments (`KP-01` to `KP-21`) are present in training (21/21), validation (21/21), and testing (21/21).
  
  **Independent Segment-Holdout Experiment**:
  Holding out 4 segments (`KP-18`, `KP-19`, `KP-20`, `KP-21`) completely from training yielded:
  - **Random Forest**: Test $R^2$ collapsed from **0.876 to 0.076**, and MAE surged from **4.88 to 14.98 PCI points**.
  - **Ridge Regression**: Maintained strong generalization ($R^2 = 0.995$, $\text{MAE} = 1.03$).
- **Scientific Issue**: The Random Forest model memorizes segment-specific condition tiers rather than learning generalizable deterioration dynamics. It cannot reliably predict pavement degradation on a newly monitored road corridor.
- **Severity**: **HIGH**.
- **Recommended Correction**: Implement GroupKFold cross-validation grouped by road segment. Disclose spatial generalization limitations for non-linear tree models.

---

### Finding 5: Autoregressive Level Dominance vs Deterioration Change
- **File & Function**: `risk/mechanism_features.py` -> `construct_mechanism_feature_vector()` (Feature 0)
- **Evidence**:
  Feature 0 is `current_pci`. The models predict future absolute PCI ($y_{t+30}$).
  Because pavement condition changes slowly, $y_t \approx y_{t+30}$, allowing a naive persistence model ($\hat{y}_{t+30} = y_t$) to achieve $R^2 = 0.991$.
  
  **Independent $\Delta\text{PCI}$ Prediction Experiment**:
  When trained to predict the actual condition change ($\Delta\text{PCI} = y_t - y_{t+30}$):
  - **Ridge Regression**: $R^2 = 0.328$ ($\text{MAE} = 1.19$, $\text{RMSE} = 1.37$).
  - **Random Forest**: $R^2 = -0.916$ ($\text{MAE} = 1.61$, $\text{RMSE} = 2.37$).
- **Scientific Issue**: The reported $R^2 = 0.999$ reflects persistence of the condition level, not high precision in forecasting deterioration deltas. Random Forest performs worse than predicting the mean condition drop.
- **Severity**: **MEDIUM**.
- **Recommended Correction**: Reframe the modeling objective to predict $\Delta\text{PCI}$ directly, or explicitly clarify in documentation that $R^2 = 0.999$ is driven by baseline PCI level persistence.

---

### Finding 6: Comparative Sample Integrity
- **File & Function**: `risk/ml_deterioration_predictor.py` -> lines 345–370
- **Evidence**:
  `test_pred_ridge = ridge.predict(X_test)` and `current_pci_test = X_test[:, 0]` were both evaluated against `y_test_30 = targets["pci_30d"][n_train + n_val:]` on the exact same 55 test samples.
- **Scientific Conclusion**: The comparison between Ridge and Persistence was conducted on identical test samples.
- **Severity**: Low.

---

### Finding 7: Model Serving Discrepancy & Random Forest Sub-Baseline Performance
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `predict_corridor_segments()` (lines 545–586)
- **Evidence**:
  ```python
  # Best model selected based on validation RMSE is Ridge Regression:
  best_name = min(model_val_rmses, key=model_val_rmses.get) # -> "Ridge Regression"
  ...
  # But the segment prediction loop hardcodes Random Forest:
  rf_30 = self.trained_models.get("rf_pci_30d")
  pred_30d_arr, std_30d = rf_30.predict_with_uncertainty(feat_vec)
  ```
  On the held-out test split:
  - **Persistence Baseline**: $\text{MAE} = 1.37$, $\text{RMSE} = 2.17$, $R^2 = 0.991$
  - **Ridge Regression**: $\text{MAE} = 0.64$, $\text{RMSE} = 0.87$, $R^2 = 0.999$
  - **Random Forest**: $\text{MAE} = 4.88$, $\text{RMSE} = 7.98$, $R^2 = 0.876$
- **Scientific Issue**:
  1. The API claims `model_name: "Ridge Regression"`, but the actual numbers served to the user are generated by `Random Forest`.
  2. The Random Forest model being served has **more than triple the error of a simple persistence forecast** (4.88 vs 1.37 MAE).
- **Severity**: **CRITICAL**.
- **Recommended Correction**: Dynamically bind `predict_corridor_segments()` to use the model identified in `self.best_model_name`. If Ridge is best, serve Ridge predictions.

---

### Finding 8: Omission of Extended Physical Mechanism Features
- **File & Function**: `risk/mechanism_features.py` -> `construct_mechanism_feature_vector()` (lines 433–451)
- **Evidence**:
  `MechanismFeatureExtractor.FEATURE_NAMES` defines 23 features.
  However, `construct_mechanism_feature_vector()` only returns 17 features (indices 0 to 16).
  The following 6 features are omitted from the input matrix $X$:
  - `temperature_c`
  - `relative_humidity_pct`
  - `antecedent_precip_index`
  - `traffic_exposure_30d_kveh`
  - `drainage_risk_score`
  - `water_accumulation_risk`
- **Scientific Issue**: `drainage_risk_score`—a central requirement for mechanism-based modeling—is completely missing from model inputs. The models predict without drainage awareness.
- **Severity**: **HIGH**.
- **Recommended Correction**: Update `construct_mechanism_feature_vector()` to return all 23 features, update `SimpleImputer` to handle missing values, and retrain models.

---

### Finding 9: Permutation Importance Execution
- **File & Function**: `risk/ml_deterioration_predictor.py` -> line 398
- **Evidence**:
  `feat_importance = self._compute_feature_importance(rf_30.rf, X_val, y_val_30)`
  Permutation importance is calculated on validation data (`X_val, y_val_30`), which matches the documentation in `docs/ML_MODEL_EVALUATION.md`.
- **Scientific Issue**: Importance is calculated exclusively on `rf_30.rf` (Random Forest), even though `Ridge Regression` was selected as the superior model.
- **Severity**: **MEDIUM**.
- **Recommended Correction**: Compute and report permutation feature importance for the active best model (Ridge) or report across both estimators.

---

### Finding 10: Statistical Deficiencies in Prediction Intervals
- **File & Function**: `risk/ml_deterioration_predictor.py` -> `RandomForestRegressorWrapper.predict_with_uncertainty()` (lines 115–121)
- **Evidence**:
  ```python
  tree_preds = np.array([tree.predict(X_imp) for tree in self.rf.estimators_])
  std_pred = np.std(tree_preds, axis=0)
  ```
  The interval is set to $\hat{y} \pm 1.96 \cdot \sigma_{\text{trees}}$.
  - This captures ensemble variance $\text{Var}(\hat{f}(X))$, but omits residual observation variance $\sigma_{\epsilon}^2$.
  - In `predict_corridor_segments()`, post-processing overrides the center: `pred_30d = min(pred_7d, float(pred_30d_arr[0]))`, centering the interval on a shifted point without adjusting bounds.
- **Scientific Issue**: Ensemble standard deviation is not a calibrated prediction interval. If individual trees agree on an erroneous forecast, the interval collapses to zero width.
- **Severity**: **MEDIUM**.
- **Recommended Correction**: Replace heuristic tree standard deviation with **Split Conformal Prediction** or **Quantile Regression Forests** ($Q_{0.025}, Q_{0.975}$) to guarantee valid coverage.

---

### Finding 11: 100% Target Clamping in Held-Out Test Splits
- **File & Function**: `risk/ml_dataset_builder.py` -> `_build_benchmark_synthetic_dataset()` (lines 197–235)
- **Evidence**:
  ```python
  step_7d = 1
  step_30d = 4
  step_60d = 8

  for seg_id, traj in by_segment.items():
      T = len(traj)
      for t in range(T - step_7d):
          ...
          t_30 = min(T - 1, t + step_30d)
          t_60 = min(T - 1, t + step_60d)
  ```
  - Total trajectory length per segment is $T = 18$ weekly observations.
  - The loop runs up to $t = T - 2 = 16$.
  - For $t \ge 14$, $t + 4 \ge 18 \implies t_{30} = 17$ (clamped).
  - For $t \ge 10$, $t + 8 \ge 18 \implies t_{60} = 17$ (clamped).
  - **Empirical Clamping Audit**:
    - Entire dataset: **17.6%** of 30d targets and **41.2%** of 60d targets are clamped.
    - **Held-Out Test Set** (last 15% of samples): **100.0% of 30d targets and 100.0% of 60d targets are clamped to $T-1$**.
- **Scientific Issue**: In the test set, the "60-day prediction" is evaluated against ground truth only 7 to 21 days ahead. The reported multi-horizon evaluation is invalid.
- **Severity**: **CRITICAL**.
- **Recommended Correction**: Truncate the outer loop to $t \in [0, T - 1 - \text{step\_max}]$ so that every sample in every split has a genuine, unclamped future observation.

---

### Finding 12: Dimensionality Discrepancy Between Metadata and Model Artifacts
- **File & Function**: `models/ml/model_metadata.json` vs `models/ml/*.joblib`
- **Evidence**:
  - `model_metadata.json` lists 23 features in `feature_names`.
  - Serialized models `rf_pci_30d.joblib` and `ridge_pci_30d.joblib` have `n_features_in_ = 17`.
- **Scientific Issue**: Passing the documented feature dictionary into the persisted `.joblib` files triggers an immediate Scikit-Learn `ValueError: X has 23 features, but estimator is expecting 17 features`.
- **Severity**: **HIGH**.
- **Recommended Correction**: Reconcile metadata to reflect the actual 17 input dimensions, or retrain models with the full 23-feature vector.

---

### Finding 13: Robust Handling of Missing Field Data
- **File & Function**: `risk/mechanism_features.py` -> `get_drainage_risk_telemetry()` & `risk/pothole_growth_predictor.py`
- **Evidence**:
  - Missing drainage survey data sets `drainage_condition_status: "UNAVAILABLE_FIELD_SURVEY_REQUIRED"`, `is_missing: True`, and applies a neutral prior (0.50).
  - `PotholeGrowthPredictor` audits GPS coordinates and emits `status: "INSUFFICIENT_POTHOLE_HISTORY"` when repeated defect passes are absent.
- **Scientific Conclusion**: Missing data handling and insufficiency reporting are safe, explicit, and non-destructive.
- **Severity**: Low.

---

### Finding 14: Seasonal Deterioration Investigator Integrity
- **File & Function**: `risk/seasonal_hypothesis_investigator.py` -> `evaluate_seasonal_rates()`
- **Evidence**:
  - Audits database for genuine dashcam video runs across seasons.
  - Correctly detects that incidents originate only from dry-season dates (`2026-10-20` and `2026-11-16`).
  - Emits `status: "INSUFFICIENT_SEASONAL_DATA"`, `seasonal_comparison: "Data pending"`, `empirical_seasonal_multiplier: "Not available"`.
  - Formally repudiates the legacy $1.45\times$ formula as a mathematical interpolation artifact.
- **Scientific Conclusion**: The seasonal investigator adheres to strict scientific integrity standards.
- **Severity**: Low.

---

## 4. Final Verdict & Scientific Conclusion

### Verdict: **NOT YET SCIENTIFICALLY VALIDATED**

#### Justification:
While the codebase establishes a robust software architecture, introduces genuine Scikit-Learn models, eliminates hardcoded multipliers, and provides safe fallback reporting, the system **cannot be certified as scientifically validated** at this time due to five critical deficiencies:
1. **Model Weights Learned from Synthetic Data**: Production models are trained on backwards-interpolated synthetic trajectories rather than repeated empirical field surveys.
2. **100% Test-Target Horizon Clamping**: Test set targets for 30-day and 60-day horizons are clamped to the end of the time series, invalidating reported multi-horizon error metrics.
3. **Model Serving Inconsistency**: Predictions served on the dashboard are generated by Random Forest (MAE 4.88, which underperforms naive persistence), while the API metadata claims Ridge Regression (MAE 0.64) is active.
4. **Physical Feature Omission**: Six key features—most critically `drainage_risk_score`—are omitted from the model vector.
5. **Spatial Generalization Collapse**: On unobserved road segments, Random Forest degrades to $R^2 = 0.076$, indicating severe memorization of corridor-specific baseline condition tiers.

---

## 5. Prioritized Remediation Roadmap

```
Step 1: Code Fixes (Immediate)
  ├── 1. Truncate MLDatasetBuilder lookahead (eliminate target clamping)
  ├── 2. Pack all 23 features into construct_mechanism_feature_vector()
  ├── 3. Bind predict_corridor_segments() dynamically to best_model_name
  └── 4. Reconcile model_metadata.json feature_names (23 features)

Step 2: Splitting & Evaluation (Short-Term)
  ├── 5. Add 60-day temporal embargo buffer between train, val, and test splits
  ├── 6. Implement GroupKFold (Segment Holdout) cross-validation
  └── 7. Add Delta-PCI (ΔPCI) prediction benchmarks alongside absolute PCI

Step 3: Field Calibration (Long-Term)
  ├── 8. Complete bi-weekly dashcam survey passes along Route A2
  └── 9. Transition from synthetic benchmark training to verified field observations
```
