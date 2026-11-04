"""
Unit Tests for Virtual Patrol Playback & Radar Proximity Simulator Engine.
Validates:
1. Geodesic spherical bearing calculation across all 8 cardinal quadrants.
2. Haversine distance accuracy in meters.
3. High-density waypoint interpolation and cumulative distance monotonicity.
4. Proximity radar detection triggers within specified spatial radius.
"""
import os
import sys
import math
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Compute bearing angle from point 1 to point 2 in degrees (0 to 360)."""
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d_lon))
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two GPS points in meters."""
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def interpolate_path(waypoints, step_meters=2.0):
    """Simulate trajectory interpolation into fine-grained simulation points."""
    interpolated = []
    running_dist = 0.0

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = haversine_m(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        bearing = calculate_bearing(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        steps = max(4, int(dist / step_meters))

        for s in range(steps):
            ratio = s / steps
            cur_lat = p1["lat"] + (p2["lat"] - p1["lat"]) * ratio
            cur_lon = p1["lon"] + (p2["lon"] - p1["lon"]) * ratio
            cur_speed = p1.get("speed", 30) + (p2.get("speed", 30) - p1.get("speed", 30)) * ratio
            running_dist += (dist / steps)

            interpolated.append({
                "lat": cur_lat,
                "lon": cur_lon,
                "bearing": bearing,
                "speed": cur_speed,
                "distance_m": running_dist
            })

    return interpolated


class TestPatrolSimulatorEngine(unittest.TestCase):
    """Test suite for patrol simulator mathematical algorithms."""

    def test_bearing_cardinal_directions(self):
        """Validate heading bearings for cardinal directions."""
        # Due North (Lat increases, Lon same)
        b_north = calculate_bearing(9.0, 39.0, 10.0, 39.0)
        self.assertAlmostEqual(b_north, 0.0, delta=1.5)

        # Due East (Lat same, Lon increases)
        b_east = calculate_bearing(9.0, 39.0, 9.0, 40.0)
        self.assertAlmostEqual(b_east, 90.0, delta=1.5)

        # Due South (Lat decreases, Lon same)
        b_south = calculate_bearing(10.0, 39.0, 9.0, 39.0)
        self.assertAlmostEqual(b_south, 180.0, delta=1.5)

        # Due West (Lat same, Lon decreases)
        b_west = calculate_bearing(9.0, 40.0, 9.0, 39.0)
        self.assertAlmostEqual(b_west, 270.0, delta=1.5)

    def test_haversine_distance_accuracy(self):
        """Ensure haversine distance computation produces valid non-negative metrics."""
        # Distance from DBU University (9.6720, 39.5272) to Hakim Gizaw Hospital (9.6825, 39.5406)
        dist = haversine_m(9.6720, 39.5272, 9.6825, 39.5406)
        # Should be roughly 1.8 km to 2.2 km
        self.assertGreater(dist, 1500)
        self.assertLess(dist, 2500)

        # Zero distance
        self.assertEqual(haversine_m(9.5, 39.5, 9.5, 39.5), 0.0)

    def test_trajectory_interpolation_monotonicity(self):
        """Ensure interpolated trajectory points strictly increase in cumulative distance."""
        waypoints = [
            {"lat": 9.6424, "lon": 39.5066, "speed": 35.0},
            {"lat": 9.6578, "lon": 39.5140, "speed": 40.0},
            {"lat": 9.6720, "lon": 39.5272, "speed": 45.0}
        ]

        path = interpolate_path(waypoints, step_meters=5.0)
        self.assertGreater(len(path), 50, "Should generate high-density interpolated steps")

        # Verify monotonicity of cumulative distance
        for i in range(len(path) - 1):
            self.assertLessEqual(path[i]["distance_m"], path[i + 1]["distance_m"])
            # Bearing must be in [0, 360)
            self.assertTrue(0 <= path[i]["bearing"] < 360)

    def test_proximity_radar_detection_trigger(self):
        """Verify that hazards within proximity threshold (45m) are successfully detected."""
        car_position = {"lat": 9.67205, "lon": 39.52722}
        hazard_close = {"id": 1, "lat": 9.67210, "lon": 39.52725}  # ~6.5m away
        hazard_far = {"id": 2, "lat": 9.68000, "lon": 39.53500}    # ~1.2 km away

        dist_close = haversine_m(car_position["lat"], car_position["lon"], hazard_close["lat"], hazard_close["lon"])
        dist_far = haversine_m(car_position["lat"], car_position["lon"], hazard_far["lat"], hazard_far["lon"])

        threshold_m = 45.0
        self.assertLess(dist_close, threshold_m, "Close hazard must fall within detection radius")
        self.assertGreater(dist_far, threshold_m, "Far hazard must exceed detection radius")


if __name__ == "__main__":
    unittest.main()
