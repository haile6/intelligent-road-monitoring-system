"""
MLOps and Production AI Engineering Package for RoadEye.

Modules:
- experiment_tracker: ML lifecycle tracking, model registry, and dataset versioning (CLO1, CLO2).
- drift_monitor: Performance tracking, covariate data drift, and concept drift detection (CLO3).
- explainability: Explainable AI (XAI) attribution maps and fairness/bias auditing (CLO2).
- edge_optimizer: Privacy-preserving redaction and TinyML/Edge AI quantization (CLO2, CLO3).
"""

from mlops.experiment_tracker import ExperimentTracker, ModelArtifact
from mlops.drift_monitor import DriftMonitor, DriftReport
from mlops.explainability import XAIExplainer, FairnessAuditor, AttributionMap, FairnessAuditReport
from mlops.edge_optimizer import PrivacyPreserver, EdgeOptimizer, QuantizationProfile, PrivacyRedactionResult

__all__ = [
    "ExperimentTracker",
    "ModelArtifact",
    "DriftMonitor",
    "DriftReport",
    "XAIExplainer",
    "FairnessAuditor",
    "AttributionMap",
    "FairnessAuditReport",
    "PrivacyPreserver",
    "EdgeOptimizer",
    "QuantizationProfile",
    "PrivacyRedactionResult"
]
