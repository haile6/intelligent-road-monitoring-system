"""
Unit Tests for Predictive Pavement Deterioration & Rainy Season Forecast Engine
"""

import unittest
from risk.predictive_deterioration import (
    MarkovPavementDeteriorationEngine,
    CostOfInactionForecaster,
    CLIMATE_SEASONS
)


class TestMarkovPavementDeteriorationEngine(unittest.TestCase):
    """Test Markov transition matrices, seasonal climate multipliers, and degradation curves."""

    def test_transition_matrix_row_stochasticity(self):
        """Every row in the adjusted transition matrix must sum to 1.0."""
        for season in ["KIREMT", "BELG", "BEGA"]:
            matrix = MarkovPavementDeteriorationEngine.get_adjusted_transition_matrix(season=season)
            self.assertEqual(len(matrix), 5)
            for row_idx, row in enumerate(matrix):
                self.assertEqual(len(row), 5)
                row_sum = sum(row)
                self.assertAlmostEqual(row_sum, 1.0, places=3,
                                       msg=f"Row {row_idx} in season {season} does not sum to 1.0 (sum={row_sum})")
                for val in row:
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    def test_absorbing_state(self):
        """State 4 (Structural Collapse) must be an absorbing state."""
        matrix = MarkovPavementDeteriorationEngine.get_adjusted_transition_matrix("KIREMT")
        self.assertEqual(matrix[4], [0.0, 0.0, 0.0, 0.0, 1.0])

    def test_kiremt_accelerates_deterioration(self):
        """Kiremt rainy season must have lower stay probability than Bega dry season."""
        mat_kiremt = MarkovPavementDeteriorationEngine.get_adjusted_transition_matrix("KIREMT")
        mat_bega = MarkovPavementDeteriorationEngine.get_adjusted_transition_matrix("BEGA")

        # Probability of staying in State 1 (Alligator crack)
        stay_kiremt = mat_kiremt[1][1]
        stay_bega = mat_bega[1][1]
        self.assertLess(stay_kiremt, stay_bega)

    def test_multi_horizon_pci_decline(self):
        """Projected PCI must decline monotonically over 30, 60, 90, and 180 days."""
        res = MarkovPavementDeteriorationEngine.forecast_corridor_deterioration(
            current_pci=75.0,
            incidents_count=20,
            season="KIREMT",
            horizons=[30, 60, 90, 180]
        )
        projs = res["projections"]
        self.assertEqual(len(projs), 4)

        prev_pci = res["current_pci"]
        for p in projs:
            self.assertLessEqual(p["projected_pci"], prev_pci)
            self.assertGreaterEqual(p["projected_defects_count"], 20)
            prev_pci = p["projected_pci"]


class TestCostOfInactionForecaster(unittest.TestCase):
    """Test defect growth models and Cost of Inaction (COI) ROI calculations."""

    def test_pothole_growth_trajectory(self):
        """Pothole dimensions and volume must expand over 60-day horizon."""
        inc = {
            "id": 101,
            "damage_type": "pothole",
            "max_risk_level": "CAUTION"
        }
        res = CostOfInactionForecaster.estimate_defect_growth_trajectory(inc, season="KIREMT", days_horizon=60)
        self.assertGreater(res["projected_dimensions"]["volume_liters"], res["current_dimensions"]["volume_liters"])
        self.assertGreater(res["projected_dimensions"]["depth_cm"], res["current_dimensions"]["depth_cm"])
        self.assertGreater(res["cost_inaction_etb"], res["cost_today_etb"])
        self.assertGreater(res["roi_multiplier"], 2.0)

    def test_crack_bursts_into_pothole(self):
        """Alligator cracks under Kiremt rains should predict deeper failure."""
        inc = {
            "id": 102,
            "damage_type": "alligator_crack",
            "max_risk_level": "CAUTION"
        }
        res = CostOfInactionForecaster.estimate_defect_growth_trajectory(inc, season="KIREMT", days_horizon=90)
        self.assertIn("Pothole", res["projected_state"])
        self.assertGreater(res["savings_etb"], 1000.0)

    def test_evaluate_corridor_coi_aggregation(self):
        """Test full corridor COI aggregation."""
        mock_incidents = [
            {"id": 1, "damage_type": "pothole", "max_risk_level": "IMMEDIATE_DANGER"},
            {"id": 2, "damage_type": "pothole", "max_risk_level": "CAUTION"},
            {"id": 3, "damage_type": "alligator_crack", "max_risk_level": "SAFE"},
        ]
        res = CostOfInactionForecaster.evaluate_corridor_coi(
            incidents=mock_incidents,
            current_pci=62.0,
            season="KIREMT",
            days_horizon=60
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["total_defects_evaluated"], 3)
        self.assertGreater(res["total_cost_inaction_etb"], res["total_cost_today_etb"])
        self.assertGreater(res["total_savings_etb"], 0.0)
        self.assertGreater(res["roi_savings_multiplier"], 2.5)
        self.assertEqual(len(res["defect_trajectories"]), 3)


if __name__ == "__main__":
    unittest.main()
