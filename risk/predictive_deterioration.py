"""
Predictive Pavement Deterioration & Rainy Season Forecast Engine
=================================================================
RoadEye™ Enterprise Road Intelligence Platform
Authority: Ethiopian Roads Administration (ERA) / HDM-4 Deterioration Models

Physics, Markovian & Climate Foundations:
- Markov-Chain Multi-State Pavement Transition Matrices (5 Condition States)
- Ethiopian Climate Season Modeling (Kiremt ክረምት vs Belg በልግ vs Bega በጋ)
- Hydrodynamic Pore-Water Pressure Infiltration & Asphalt Stripping Dynamics
- Equivalent Single Axle Loads (ESAL) & Traffic Loading Impact (Route A2 Corridor)
"""

import math
from typing import Dict, List, Tuple, Any, Optional

# ==============================================================================
# 1. ETHIOPIAN CLIMATE SEASONS & DISTRESS STATES
# ==============================================================================

CLIMATE_SEASONS = {
    "KIREMT": {
        "season_en": "Kiremt (Heavy Monsoon Season)",
        "season_am": "ክረምት (ከባድ የዝናብ ወቅት)",
        "months": "June - September",
        "rainfall_mm_month": 280.0,
        "deterioration_multiplier": 4.2,  # 420% accelerated deterioration
        "water_stripping_risk": "CRITICAL",
        "color": "#ef4444"
    },
    "BELG": {
        "season_en": "Belg (Moderate Autumn Rains)",
        "season_am": "በልግ (መካከለኛ ዝናብ)",
        "months": "March - May",
        "rainfall_mm_month": 95.0,
        "deterioration_multiplier": 1.8,
        "water_stripping_risk": "MODERATE",
        "color": "#f59e0b"
    },
    "BEGA": {
        "season_en": "Bega (Dry Sunny Season)",
        "season_am": "በጋ (ደረቅ የበጋ ወቅት)",
        "months": "October - February",
        "rainfall_mm_month": 15.0,
        "deterioration_multiplier": 1.0,  # Baseline dry pavement wear
        "water_stripping_risk": "MINIMAL",
        "color": "#10b981"
    }
}

PAVEMENT_STATES = [
    {"index": 0, "name": "Intact / Micro-Crack", "name_am": "ጤነኛ / ጥቃቅን ስንጥቅ", "pci_range": (85, 100), "color": "#10b981"},
    {"index": 1, "name": "Alligator Cracking", "name_am": "የአዞ ቆዳ ስንጥቅ", "pci_range": (55, 84), "color": "#eab308"},
    {"index": 2, "name": "Shallow Pothole (<3cm)", "name_am": "ቀላል ጉድጓድ (<3 ሳ.ሜ)", "pci_range": (40, 54), "color": "#f97316"},
    {"index": 3, "name": "Deep Pothole (3-6cm)", "name_am": "ጥልቅ ጉድጓድ (3-6 ሳ.ሜ)", "pci_range": (25, 39), "color": "#ef4444"},
    {"index": 4, "name": "Sub-Base Collapse (>6cm)", "name_am": "የመሰረት መደርመስ (>6 ሳ.ሜ)", "pci_range": (0, 24), "color": "#7f1d1d"},
]


class MarkovPavementDeteriorationEngine:
    """
    Markov-Chain Pavement Condition Deterioration Forecaster.
    
    Models the probabilistic transition of road surface defects across 5 discrete
    structural states as a function of time, rainfall, and freight axle loads.
    """

    # Baseline 30-Day Transition Matrix P (Dry Season, Standard AADT 8,500)
    # Rows: Current State S_i (0 to 4), Columns: Future State S_j (0 to 4)
    # Properties: sum of each row == 1.0, P_ij >= 0, upper triangular (degradation is monotonic)
    BASE_TRANSITION_MATRIX_30D = [
        [0.86, 0.11, 0.03, 0.00, 0.00],  # State 0: Intact
        [0.00, 0.76, 0.18, 0.06, 0.00],  # State 1: Alligator Crack
        [0.00, 0.00, 0.68, 0.24, 0.08],  # State 2: Shallow Pothole
        [0.00, 0.00, 0.00, 0.58, 0.42],  # State 3: Deep Pothole
        [0.00, 0.00, 0.00, 0.00, 1.00],  # State 4: Structural Collapse (absorbing)
    ]

    @classmethod
    def get_adjusted_transition_matrix(cls, 
                                       season: str = "KIREMT", 
                                       heavy_truck_pct: float = 24.0) -> List[List[float]]:
        """
        Adjust Markov transition probabilities based on seasonal rainfall and truck loading.
        """
        season_key = (season or "KIREMT").upper()
        climate = CLIMATE_SEASONS.get(season_key, CLIMATE_SEASONS["KIREMT"])
        rain_mult = climate["deterioration_multiplier"]

        # Truck factor: baseline 20% heavy freight; higher loading increases shear failure
        truck_factor = max(0.6, min(2.0, float(heavy_truck_pct) / 20.0))
        total_stress_multiplier = rain_mult * truck_factor

        matrix = []
        for row_idx, row in enumerate(cls.BASE_TRANSITION_MATRIX_30D):
            # If absorbing state (State 4)
            if row_idx == 4:
                matrix.append([0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            stay_prob = row[row_idx]
            degrade_prob = 1.0 - stay_prob

            # Accelerated degradation under rain & heavy axle load
            accelerated_degrade = min(0.92, degrade_prob * (1.0 + (total_stress_multiplier - 1.0) * 0.38))
            new_stay_prob = 1.0 - accelerated_degrade

            new_row = [0.0] * 5
            new_row[row_idx] = new_stay_prob

            # Distribute degradation probability across forward states proportional to base rates
            orig_forward_probs = row[row_idx + 1:]
            sum_orig = sum(orig_forward_probs) if sum(orig_forward_probs) > 0 else 1.0

            for col_idx in range(row_idx + 1, 5):
                share = row[col_idx] / sum_orig
                new_row[col_idx] = round(accelerated_degrade * share, 4)

            # Ensure strict sum to 1.0
            new_row[row_idx] = round(1.0 - sum(new_row[row_idx + 1:]), 4)
            matrix.append(new_row)

        return matrix

    @staticmethod
    def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Square matrix multiplication 5x5."""
        size = len(a)
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                result[i][j] = sum(a[i][k] * b[k][j] for k in range(size))
        return result

    @classmethod
    def compute_horizon_matrix(cls, 
                               base_matrix: List[List[float]], 
                               days_horizon: int = 60) -> List[List[float]]:
        """
        Compute P^n for arbitrary forecast horizon (30, 60, 90, 180 days).
        """
        steps = max(1, round(days_horizon / 30.0))
        current = [row[:] for row in base_matrix]
        for _ in range(steps - 1):
            current = cls.matrix_multiply(current, base_matrix)

        # Normalize rows to ensure mathematical integrity
        normalized = []
        for row in current:
            total = sum(row)
            if total > 0:
                normalized.append([round(x / total, 4) for x in row])
            else:
                normalized.append(row)
        return normalized

    @classmethod
    def forecast_corridor_deterioration(cls, 
                                        current_pci: float, 
                                        incidents_count: int, 
                                        season: str = "KIREMT", 
                                        horizons: List[int] = [30, 60, 90, 180]) -> Dict[str, Any]:
        """
        Generate multi-horizon corridor-level PCI deterioration curves.
        """
        matrix_30d = cls.get_adjusted_transition_matrix(season=season)
        season_info = CLIMATE_SEASONS.get(season.upper(), CLIMATE_SEASONS["KIREMT"])

        projections = []
        base_pci = float(current_pci)

        for days in horizons:
            p_n = cls.compute_horizon_matrix(matrix_30d, days_horizon=days)
            
            # Weighted expected state transition starting from current average pavement state
            # If current PCI is 60, initial state is State 1 (Alligator Crack)
            if base_pci >= 85:
                initial_dist = [0.80, 0.15, 0.05, 0.0, 0.0]
            elif base_pci >= 55:
                initial_dist = [0.10, 0.70, 0.15, 0.05, 0.0]
            elif base_pci >= 40:
                initial_dist = [0.0, 0.20, 0.60, 0.15, 0.05]
            elif base_pci >= 25:
                initial_dist = [0.0, 0.05, 0.25, 0.55, 0.15]
            else:
                initial_dist = [0.0, 0.0, 0.05, 0.35, 0.60]

            # Future state distribution pi_t = pi_0 * P^n
            future_dist = [0.0] * 5
            for j in range(5):
                future_dist[j] = sum(initial_dist[i] * p_n[i][j] for i in range(5))

            # Expected PCI = sum(pi_t(j) * mean_pci(j))
            mean_pcis = [92.0, 70.0, 47.0, 32.0, 12.0]
            projected_pci = round(max(0.0, sum(future_dist[j] * mean_pcis[j] for j in range(5))), 1)

            # Defect count growth factor (cracks turn into potholes, potholes enlarge)
            growth_factor = 1.0 + (1.0 - (projected_pci / max(1.0, base_pci))) * 1.5
            projected_defects = round(incidents_count * growth_factor)
            severe_collapse_pct = round(future_dist[4] * 100.0, 1)

            projections.append({
                "days_horizon": days,
                "projected_pci": projected_pci,
                "pci_drop": round(base_pci - projected_pci, 1),
                "projected_defects_count": projected_defects,
                "severe_collapse_probability_pct": severe_collapse_pct,
                "state_probabilities": [round(p * 100.0, 1) for p in future_dist],
                "urgency": "Immediate Emergency Action" if projected_pci < 35.0 else ("High Priority" if projected_pci < 55.0 else "Scheduled Maintenance")
            })

        return {
            "current_pci": base_pci,
            "season": season.upper(),
            "season_details": season_info,
            "projections": projections
        }
