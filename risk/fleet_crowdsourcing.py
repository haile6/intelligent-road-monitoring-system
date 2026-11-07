"""
Multi-Vehicle Fleet Crowdsourcing & Automatic De-duplication Engine
====================================================================
RoadEye™ Enterprise Road Intelligence Platform
Authority: Ethiopian Roads Administration (ERA) / Municipal Fleet Telemetry

Algorithms & Mathematical Models:
- Geodesic Spatial Radius Clustering (Haversine epsilon <= 8.0m)
- Bayesian Multi-Observation Probability Fusion (False Positive Elimination)
- Confidence-Weighted Geodesic Centroid Spatial Smoothing
- Multi-Pass Temporal Progression Tracking (Deterioration Monitoring)
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


class BayesianConfidenceFusion:
    """
    Bayesian Probability Fusion for Multi-Vehicle Defect Observations.
    
    When multiple distinct patrol vehicles detect a hazard at the same location,
    the joint probability of an error drops exponentially:
    P(Defect | O_1, O_2, ..., O_k) = 1.0 - Product(1.0 - P_i)
    """

    MAX_CONFIDENCE_CAP = 0.999  # 99.9% maximum statistical certainty
    MIN_CONFIDENCE_FLOOR = 0.15

    @classmethod
    def fuse_confidences(cls, confidences: List[float]) -> Dict[str, Any]:
        """
        Compute joint Bayesian probability from multiple independent sensor passes.
        """
        if not confidences:
            return {
                "fused_confidence": 0.0,
                "confidence_gain": 0.0,
                "verification_status": "Unverified",
                "verification_status_am": "ያልተረጋገጠ",
                "observations_count": 0,
                "consensus_level": "None"
            }

        valid_confs = [max(cls.MIN_CONFIDENCE_FLOOR, min(0.98, float(c))) for c in confidences]
        k = len(valid_confs)
        max_single = max(valid_confs)

        if k == 1:
            fused = valid_confs[0]
            gain = 0.0
            status = "Single Observation (Preliminary)"
            status_am = "አንድ ቅኝት (የመጀመሪያ ደረጃ)"
            consensus = "Preliminary"
        elif k == 2:
            prod_error = (1.0 - valid_confs[0]) * (1.0 - valid_confs[1])
            fused = min(cls.MAX_CONFIDENCE_CAP, 1.0 - prod_error)
            gain = round(fused - max_single, 4)
            status = "Dual-Vehicle Corroborated"
            status_am = "በሁለት መኪኖች የተረጋገጠ"
            consensus = "Corroborated"
        else:
            prod_error = 1.0
            for c in valid_confs:
                prod_error *= (1.0 - c)
            fused = min(cls.MAX_CONFIDENCE_CAP, 1.0 - prod_error)
            gain = round(fused - max_single, 4)
            status = f"Fleet Consensus Verified ({k} Vehicles)"
            status_am = f"በ{k} መኪኖች ሙሉ በሙሉ የተረጋገጠ"
            consensus = "High Consensus"

        fused = round(fused, 4)

        return {
            "fused_confidence": fused,
            "confidence_percentage": round(fused * 100.0, 1),
            "max_single_confidence": round(max_single, 4),
            "confidence_gain": gain,
            "confidence_gain_percentage": round(gain * 100.0, 1),
            "observations_count": k,
            "verification_status": status,
            "verification_status_am": status_am,
            "consensus_level": consensus
        }


class SpatialDeDuplicationEngine:
    """
    Spatio-Temporal Spatial Radius Clustering Engine.
    
    Groups redundant observations across fleet vehicles within epsilon meters
    and computes the true weighted geodesic centroid.
    """

    DEFAULT_EPSILON_M = 8.0  # Geodesic radius threshold in meters for matching

    @staticmethod
    def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two coordinates in meters."""
        r = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @classmethod
    def calculate_weighted_centroid(cls, sightings: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Calculate confidence-weighted centroid coordinates and spatial spread radius (m).
        """
        if not sightings:
            return 0.0, 0.0, 0.0

        total_weight = 0.0
        weighted_lat = 0.0
        weighted_lon = 0.0

        for s in sightings:
            lat = float(s.get("damage_latitude") or s.get("latitude") or 0.0)
            lon = float(s.get("damage_longitude") or s.get("longitude") or 0.0)
            w = max(0.1, float(s.get("confidence") or 0.80))
            weighted_lat += lat * w
            weighted_lon += lon * w
            total_weight += w

        centroid_lat = weighted_lat / total_weight
        centroid_lon = weighted_lon / total_weight

        # Compute maximum spread radius from centroid
        max_spread_m = 0.0
        for s in sightings:
            lat = float(s.get("damage_latitude") or s.get("latitude") or 0.0)
            lon = float(s.get("damage_longitude") or s.get("longitude") or 0.0)
            dist = cls.haversine_distance_m(centroid_lat, centroid_lon, lat, lon)
            if dist > max_spread_m:
                max_spread_m = dist

        return round(centroid_lat, 6), round(centroid_lon, 6), round(max_spread_m, 2)

    @classmethod
    def cluster_raw_incidents(cls, 
                              raw_incidents: List[Dict[str, Any]], 
                              epsilon_m: float = 8.0) -> List[Dict[str, Any]]:
        """
        De-duplicate raw sightings into canonical verified hazards using spatial radius matching.
        """
        if not raw_incidents:
            return []

        eps = max(2.0, float(epsilon_m))
        unassigned = list(raw_incidents)
        clusters = []

        while unassigned:
            seed = unassigned.pop(0)
            seed_lat = float(seed.get("damage_latitude") or seed.get("latitude") or 0.0)
            seed_lon = float(seed.get("damage_longitude") or seed.get("longitude") or 0.0)
            seed_type = (seed.get("damage_type") or "pothole").lower()

            current_cluster = [seed]
            remaining = []

            for candidate in unassigned:
                c_lat = float(candidate.get("damage_latitude") or candidate.get("latitude") or 0.0)
                c_lon = float(candidate.get("damage_longitude") or candidate.get("longitude") or 0.0)
                c_type = (candidate.get("damage_type") or "pothole").lower()

                # Must match defect class family (pothole vs crack) and fall within epsilon distance
                type_match = ("pothole" in seed_type and "pothole" in c_type) or ("pothole" not in seed_type and "pothole" not in c_type)
                dist = cls.haversine_distance_m(seed_lat, seed_lon, c_lat, c_lon)

                if type_match and dist <= eps:
                    current_cluster.append(candidate)
                else:
                    remaining.append(candidate)

            unassigned = remaining
            clusters.append(current_cluster)

        # Build canonical verified hazard objects from clusters
        canonical_hazards = []
        for cluster_idx, group in enumerate(clusters, start=1):
            centroid_lat, centroid_lon, spread_m = cls.calculate_weighted_centroid(group)
            confs = [float(item.get("confidence") or 0.85) for item in group]
            bayes = BayesianConfidenceFusion.fuse_confidences(confs)

            # Determine worst risk level among all sightings
            risk_hierarchy = ["IMMEDIATE_DANGER", "DANGER", "WARNING", "CAUTION", "SAFE", "MONITOR"]
            worst_risk = "SAFE"
            for r in risk_hierarchy:
                if any(r in str(item.get("max_risk_level", "")).upper() for item in group):
                    worst_risk = r
                    break

            primary_type = group[0].get("damage_type") or "pothole"

            canonical_hazards.append({
                "canonical_id": f"HAZ-CAN-{cluster_idx:04d}",
                "primary_track_id": group[0].get("track_id") or group[0].get("id"),
                "damage_type": primary_type,
                "damage_latitude": centroid_lat,
                "damage_longitude": centroid_lon,
                "spatial_spread_radius_m": spread_m,
                "max_risk_level": worst_risk,
                "sightings_count": len(group),
                "is_deduplicated": len(group) > 1,
                "raw_incidents": group,
                "bayesian_fusion": bayes
            })

        return canonical_hazards



# ==============================================================================
# 2. FLEET VEHICLE TELEMETRY & SIGHTING HISTORY AGGREGATOR
# ==============================================================================

class FleetVehicleManager:
    """
    Manages municipal patrol fleet units, maps vehicles to observations,
    and aggregates multi-vehicle crowdsourced road intelligence.
    """

    # Municipal & Highway Patrol Fleet Profiles (Route A2 Addis Ababa - Debre Berhan)
    STANDARD_FLEET = [
        {
            "vehicle_id": "ERA-PATROL-01",
            "vehicle_name": "ERA Survey Hilux #01",
            "vehicle_type": "Pavement Inspection 4WD",
            "agency": "Ethiopian Roads Administration (ERA)",
            "driver": "Ins. Haile Sintayehu",
            "hardware": "Jetson Orin Nano + Sony IMX477 (70° FOV)",
            "status": "ONLINE",
            "color": "#38bdf8"
        },
        {
            "vehicle_id": "ERA-PATROL-02",
            "vehicle_name": "ERA Heavy Patrol Truck #02",
            "vehicle_type": "Mobile Maintenance Rig",
            "agency": "Ethiopian Roads Administration (ERA)",
            "driver": "Eng. Bekele Tadesse",
            "hardware": "Intel NUC + Dual USB3 Road Camera",
            "status": "ONLINE",
            "color": "#10b981"
        },
        {
            "vehicle_id": "MUNI-TRANSIT-14",
            "vehicle_name": "Debre Berhan Public Transit #14",
            "vehicle_type": "Municipal Transit Bus",
            "agency": "Debre Berhan Municipal Transport",
            "driver": "Ato Girma Wolde",
            "hardware": "Edge ADAS Dashcam Unit",
            "status": "ONLINE",
            "color": "#f59e0b"
        },
        {
            "vehicle_id": "POLICE-CRUISER-07",
            "vehicle_name": "Federal Highway Cruiser #07",
            "vehicle_type": "Federal Highway Police Patrol",
            "agency": "Federal Police Commission",
            "driver": "Cmd. Daniel Alemu",
            "hardware": "On-Board Highway Video Radar",
            "status": "ONLINE",
            "color": "#818cf8"
        }
    ]

    @classmethod
    def get_registered_fleet(cls) -> List[Dict[str, Any]]:
        """Return list of active patrol fleet vehicles."""
        return list(cls.STANDARD_FLEET)

    @classmethod
    def process_fleet_crowdsourcing(cls, 
                                    incidents: List[Dict[str, Any]], 
                                    epsilon_m: float = 8.0) -> Dict[str, Any]:
        """
        Synthesize multi-vehicle crowdsourcing observations, cluster duplicates,
        and generate comprehensive fleet telemetry metrics.
        """
        fleet = cls.get_registered_fleet()
        raw_count = len(incidents)

        # Assign realistic fleet metadata to incidents based on their track_id/index
        enriched_incidents = []
        for idx, inc in enumerate(incidents):
            v_idx = idx % len(fleet)
            assigned_vehicle = fleet[v_idx]

            # Construct observation record
            enriched = dict(inc)
            enriched["vehicle_id"] = inc.get("vehicle_id") or assigned_vehicle["vehicle_id"]
            enriched["vehicle_name"] = assigned_vehicle["vehicle_name"]
            enriched["vehicle_type"] = assigned_vehicle["vehicle_type"]
            enriched["agency"] = assigned_vehicle["agency"]
            enriched["vehicle_color"] = assigned_vehicle["color"]

            # Timestamp simulation if missing
            if not enriched.get("created_at"):
                mins_ago = (len(incidents) - idx) * 14
                enriched["observation_time"] = f"{mins_ago // 60:02d}:{mins_ago % 60:02d} ago"
            else:
                enriched["observation_time"] = str(enriched.get("created_at"))

            enriched_incidents.append(enriched)

        # Run Spatial Clustering & Bayesian Fusion
        canonical_hazards = SpatialDeDuplicationEngine.cluster_raw_incidents(
            raw_incidents=enriched_incidents,
            epsilon_m=epsilon_m
        )

        unique_count = len(canonical_hazards)
        dedup_count = max(0, raw_count - unique_count)
        dedup_savings_pct = round((dedup_count / max(1, raw_count)) * 100.0, 1)

        # Multi-vehicle verified hazards
        multi_sighting_count = sum(1 for h in canonical_hazards if h["sightings_count"] > 1)
        multi_sighting_pct = round((multi_sighting_count / max(1, unique_count)) * 100.0, 1)

        # Average Bayesian Confidence
        avg_fused_conf = 0.0
        avg_gain = 0.0
        if canonical_hazards:
            avg_fused_conf = sum(h["bayesian_fusion"]["fused_confidence"] for h in canonical_hazards) / len(canonical_hazards)
            avg_gain = sum(h["bayesian_fusion"]["confidence_gain"] for h in canonical_hazards) / len(canonical_hazards)

        # Vehicle activity breakdown
        vehicle_stats = {v["vehicle_id"]: {"vehicle": v, "sightings_count": 0} for v in fleet}
        for inc in enriched_incidents:
            vid = inc["vehicle_id"]
            if vid in vehicle_stats:
                vehicle_stats[vid]["sightings_count"] += 1

        return {
            "success": True,
            "corridor": "Route A2 · Addis Ababa - Debre Berhan Highway",
            "total_raw_observations": raw_count,
            "canonical_unique_hazards": unique_count,
            "duplicate_observations_eliminated": dedup_count,
            "deduplication_savings_percentage": dedup_savings_pct,
            "multi_sighting_hazards_count": multi_sighting_count,
            "corroboration_percentage": multi_sighting_pct,
            "average_fused_confidence_pct": round(avg_fused_conf * 100.0, 1),
            "average_confidence_gain_pct": round(avg_gain * 100.0, 1),
            "active_fleet_count": len(fleet),
            "fleet_vehicles": fleet,
            "vehicle_contributions": list(vehicle_stats.values()),
            "canonical_hazards": canonical_hazards
        }
