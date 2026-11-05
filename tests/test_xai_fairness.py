"""
Unit Tests for Explainable AI (XAI) Attribution & Fairness Auditing Engine (CLO2).
"""

import unittest
from mlops.explainability import XAIExplainer, FairnessAuditor, AttributionMap, FairnessAuditReport


class TestXAIExplainer(unittest.TestCase):
    """Test Suite for Model Interpretability & Attribution Maps."""

    def setUp(self):
        self.explainer = XAIExplainer(grid_rows=4, grid_cols=4)

    def test_explain_pothole_detection(self):
        attr = self.explainer.explain_detection(
            class_name="pothole",
            confidence=0.88,
            bbox=[100, 200, 300, 400]
        )
        self.assertEqual(len(attr.attributions), 4)
        self.assertEqual(len(attr.attributions[0]), 4)
        self.assertEqual(attr.grid_size, (4, 4))
        self.assertEqual(attr.class_name, "pothole")
        self.assertGreater(attr.original_confidence, 0.8)
        self.assertIn("saliency", attr.explanation_text.lower())

        # Check normalization (max value is 1.0)
        flat_vals = [v for row in attr.attributions for v in row]
        self.assertAlmostEqual(max(flat_vals), 1.0, places=2)
        self.assertGreaterEqual(min(flat_vals), 0.0)

    def test_explain_alligator_crack_detection(self):
        attr = self.explainer.explain_detection(
            class_name="alligator_crack",
            confidence=0.75,
            bbox=[50, 150, 250, 350]
        )
        self.assertEqual(attr.class_name, "alligator_crack")
        self.assertIn("rim_perimeter_gradient", attr.feature_importance_summary)
        self.assertTrue(len(attr.top_contributing_quadrant) > 0)


class TestFairnessAuditor(unittest.TestCase):
    """Test Suite for Algorithmic Fairness & Bias Parity Audits."""

    def setUp(self):
        self.auditor = FairnessAuditor()

    def test_default_cohort_audit_compliance(self):
        default_data = self.auditor.get_default_audit_data()
        report = self.auditor.audit_cohort_performance(default_data)

        self.assertIn(report.compliance_status, ["COMPLIANT", "ACCEPTABLE_PARITY"])
        self.assertGreaterEqual(report.overall_fairness_score, 0.70)
        self.assertLessEqual(report.equal_opportunity_difference, 0.15)
        self.assertGreater(len(report.mitigation_recommendations), 0)

    def test_severe_disparity_bias_detection(self):
        # Biased data with massive recall gap and severe false discovery in shadow
        biased_data = {
            "direct_sunlight": {"precision": 0.95, "recall": 0.92, "false_discovery_rate": 0.02},
            "shadow_cast": {"precision": 0.60, "recall": 0.68, "false_discovery_rate": 0.18}
        }
        report = self.auditor.audit_cohort_performance(biased_data)

        self.assertEqual(report.compliance_status, "BIAS_DETECTED")
        self.assertGreater(report.equal_opportunity_difference, 0.12)
        self.assertGreater(report.false_discovery_parity, 0.08)
        self.assertTrue(any("Disparate Recall Gap" in d for d in report.detected_disparities))
        self.assertTrue(any("shadow" in r.lower() for r in report.mitigation_recommendations))


if __name__ == "__main__":
    unittest.main()
