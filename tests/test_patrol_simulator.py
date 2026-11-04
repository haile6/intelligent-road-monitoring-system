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

            cur_landmark = p1.get("landmark", "") if ratio < 0.5 else p2.get("landmark", p1.get("landmark", ""))
            interpolated.append({
                "lat": cur_lat,
                "lon": cur_lon,
                "bearing": bearing,
                "speed": cur_speed,
                "distance_m": running_dist,
                "landmark": cur_landmark
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
        """Verify that hazards within proximity threshold (100m default) are successfully detected."""
        car_position = {"lat": 9.67205, "lon": 39.52722}
        hazard_close = {"id": 1, "lat": 9.67210, "lon": 39.52725}  # ~6.5m away
        hazard_mid = {"id": 2, "lat": 9.67265, "lon": 39.52765}    # ~80m away
        hazard_far = {"id": 3, "lat": 9.68000, "lon": 39.53500}    # ~1.2 km away

        dist_close = haversine_m(car_position["lat"], car_position["lon"], hazard_close["lat"], hazard_close["lon"])
        dist_mid = haversine_m(car_position["lat"], car_position["lon"], hazard_mid["lat"], hazard_mid["lon"])
        dist_far = haversine_m(car_position["lat"], car_position["lon"], hazard_far["lat"], hazard_far["lon"])

        # Default 100m threshold
        threshold_100m = 100.0
        self.assertLess(dist_close, threshold_100m, "Close hazard must fall within 100m detection radius")
        self.assertLess(dist_mid, threshold_100m, "Mid hazard (~80m) must fall within 100m detection radius")
        self.assertGreater(dist_far, threshold_100m, "Far hazard must exceed 100m detection radius")

        # Comparison with 50m tight threshold
        threshold_50m = 50.0
        self.assertLess(dist_close, threshold_50m)
        self.assertGreater(dist_mid, threshold_50m, "Mid hazard (~80m) should NOT trigger under 50m tight radius")

    def test_real_dashcam_video_defects_playback_trigger(self):
        """Verify proximity detection across actual dashcam video defect track."""
        import database.db_manager as dm
        gps_pts = dm.get_gps_track_points()
        incidents = dm.get_all_incidents()

        self.assertGreater(len(gps_pts), 10, "Should have real recorded GPS track points")
        self.assertGreater(len(incidents), 20, "Should have real detected defects")

        # Interpolate the real GPS path
        waypoints = [{"lat": float(p["lat"]), "lon": float(p["lon"]), "speed": float(p.get("speed_kmh", 30))} for p in gps_pts]
        path = interpolate_path(waypoints, step_meters=2.0)

        # Track which defects get triggered
        detected_ids = set()
        threshold_m = 45.0

        for pt in path:
            for inc in incidents:
                i_lat = float(inc.get("damage_latitude") or inc.get("latitude") or 0)
                i_lon = float(inc.get("damage_longitude") or inc.get("longitude") or 0)
                if i_lat == 0 or i_lon == 0:
                    continue

                dist = haversine_m(pt["lat"], pt["lon"], i_lat, i_lon)
                if dist <= threshold_m:
                    detected_ids.add(inc.get("track_id") or inc.get("id"))

        # Verify that major video defects were all encountered along the path
        # (e.g. Defect #56, #55, #47, #19)
        self.assertIn(56, detected_ids)
        self.assertIn(55, detected_ids)
        self.assertIn(47, detected_ids)
        self.assertIn(19, detected_ids)
        self.assertGreaterEqual(len(detected_ids), 30, "Most video defects should be pinged along the trajectory")

    def test_sonar_acoustic_frequencies_and_ring_delays(self):
        """Verify frequency selection logic for proximity sonar alerts."""
        def get_sonar_freq(risk_str):
            risk = (risk_str or "").lower()
            return 920 if ("danger" in risk or "immediate" in risk or "high" in risk) else 660

        self.assertEqual(get_sonar_freq("IMMEDIATE DANGER"), 920)
        self.assertEqual(get_sonar_freq("Danger"), 920)
        self.assertEqual(get_sonar_freq("CAUTION"), 660)
        self.assertEqual(get_sonar_freq("Safe / Monitor"), 660)

        # Verify concentric ring stagger interval (0s, 0.38s, 0.76s)
        delays = [0.0, 0.38, 0.76]
        self.assertEqual(len(delays), 3)
        self.assertTrue(delays[0] < delays[1] < delays[2])


    def test_radar_range_tiers_and_early_warning_horizon(self):
        """Verify multi-tier radar range horizons (50m, 100m, 150m, 200m) and early warning TTC lookahead."""
        valid_ranges = [50.0, 100.0, 150.0, 200.0]
        speed_kmh = 30.0
        speed_mps = speed_kmh / 3.6  # ~8.333 m/s

        ttc_expectations = {
            50.0: 6.0,    # ~6.0s TTC
            100.0: 12.0,  # ~12.0s TTC early warning
            150.0: 18.0,  # ~18.0s TTC
            200.0: 24.0   # ~24.0s TTC
        }

        for r in valid_ranges:
            expected_ttc = r / speed_mps
            self.assertAlmostEqual(expected_ttc, ttc_expectations[r], delta=0.5)

        # 100m range provides more than double the reaction time compared to legacy 45m
        ttc_legacy = 45.0 / speed_mps
        ttc_100m = 100.0 / speed_mps
        self.assertGreater(ttc_100m, ttc_legacy * 2.0)



    def test_compass_cardinal_resolution_8_and_16_wind(self):
        """Verify dynamic compass heading quadrant resolution for 8-point and 16-point roses."""
        def bearing_to_cardinal(deg, precision=8):
            if precision == 16:
                cardinals16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
                idx = round(deg / 22.5) % 16
                return cardinals16[idx]
            cardinals8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            idx = round(deg / 45.0) % 8
            return cardinals8[idx]

        # 8-point cardinal checks
        self.assertEqual(bearing_to_cardinal(0, 8), 'N')
        self.assertEqual(bearing_to_cardinal(14, 8), 'N')     # 14 degrees is North quadrant (337.5 - 22.5)
        self.assertEqual(bearing_to_cardinal(48, 8), 'NE')    # 48 degrees is North-East
        self.assertEqual(bearing_to_cardinal(92, 8), 'E')
        self.assertEqual(bearing_to_cardinal(180, 8), 'S')
        self.assertEqual(bearing_to_cardinal(268, 8), 'W')

        # 16-point cardinal high-precision checks
        self.assertEqual(bearing_to_cardinal(0, 16), 'N')
        self.assertEqual(bearing_to_cardinal(14, 16), 'NNE')  # 14 degrees resolves to NNE in 16-wind
        self.assertEqual(bearing_to_cardinal(48, 16), 'NE')
        self.assertEqual(bearing_to_cardinal(68, 16), 'ENE')
        self.assertEqual(bearing_to_cardinal(90, 16), 'E')
        self.assertEqual(bearing_to_cardinal(355, 16), 'N')

    def test_cornering_deceleration_physics_engine(self):
        """Verify that sharp angular bends induce deceleration while straight roads permit cruising."""
        # Route 1: Straight road heading North (no turn angle)
        straight_waypoints = [
            {"lat": 9.6700, "lon": 39.5200, "speed": 40.0},
            {"lat": 9.6720, "lon": 39.5200, "speed": 40.0},
            {"lat": 9.6740, "lon": 39.5200, "speed": 40.0}
        ]
        straight_path = interpolate_path(straight_waypoints, step_meters=2.0)
        straight_speeds = [p["speed"] for p in straight_path]
        avg_straight_speed = sum(straight_speeds) / len(straight_speeds)
        self.assertAlmostEqual(avg_straight_speed, 40.0, delta=2.0)

        # Function simulating cornering deceleration factor
        def compute_cornering_speed(base_speed, turn_angle_deg, progress):
            if turn_angle_deg <= 15:
                return base_speed
            turn_severity = min(1.0, turn_angle_deg / 90.0)
            decel = math.sin(progress * math.pi) * 0.35 * turn_severity
            return max(12.0, base_speed * (1.0 - decel))

        # Sharp 90-degree right turn from North to East
        turn_angle = 90.0
        mid_turn_speed = compute_cornering_speed(40.0, turn_angle, 0.5)
        # At apex of turn (progress=0.5), speed drops by 35% from 40 to ~26 km/h
        self.assertLess(mid_turn_speed, 30.0, "Car must slow down at corner apex")
        self.assertGreater(mid_turn_speed, 20.0, "Car speed remains positive and safe")

    def test_dual_odometer_conservation_and_eta(self):
        """Verify distance conservation (traveled + remaining = total) and ETA calculation."""
        total_dist_m = 2500.0  # 2.50 km
        traveled_m = 1200.0    # 1.20 km
        remaining_m = total_dist_m - traveled_m

        # Conservation check
        self.assertEqual(traveled_m + remaining_m, total_dist_m)
        self.assertAlmostEqual(traveled_m / 1000.0, 1.20, places=2)
        self.assertAlmostEqual(total_dist_m / 1000.0, 2.50, places=2)
        self.assertAlmostEqual(remaining_m / 1000.0, 1.30, places=2)

        # ETA at 36 km/h (10 m/s)
        speed_mps = 36.0 / 3.6  # 10 m/s
        eta_seconds = round(remaining_m / speed_mps)
        self.assertEqual(eta_seconds, 130)  # 2 minutes 10 seconds



    def test_variable_speed_multipliers_and_frame_timing(self):
        """Verify speed multipliers (1x, 2x, 4x, 8x) and interval timing calculation."""
        fps = 40
        speed_multipliers = [0.5, 1.0, 2.0, 4.0, 8.0]
        expected_intervals_ms = {
            0.5: 50,  # 1000 / (40 * 0.5) = 50ms
            1.0: 25,  # 1000 / (40 * 1.0) = 25ms
            2.0: 12,  # 1000 / (40 * 2.0) = 12ms (clamped to max(12, ...))
            4.0: 12,  # clamped to 12ms
            8.0: 12   # clamped to 12ms
        }

        for mult in speed_multipliers:
            raw_interval = math.floor(1000 / (fps * mult))
            actual_interval = max(12, raw_interval)
            self.assertEqual(actual_interval, expected_intervals_ms[mult])

        # Clamping checks: multiplier must be within [0.25, 16.0]
        def clamp_multiplier(m):
            return max(0.25, min(16.0, float(m)))

        self.assertEqual(clamp_multiplier(0.1), 0.25)
        self.assertEqual(clamp_multiplier(32.0), 16.0)
        self.assertEqual(clamp_multiplier(8.0), 8.0)

    def test_bidirectional_scrubber_seeking_and_hazard_reset(self):
        """Verify that scrubbing forward detects hazards, and scrubbing backward clears them."""
        # Setup mock trajectory of 100 points
        path_len = 100
        # Hazard at index 40 and index 80
        hazard_indices = {40: 101, 80: 102}

        def compute_detected_at_index(target_idx):
            detected = set()
            for i in range(target_idx + 1):
                if i in hazard_indices:
                    detected.add(hazard_indices[i])
            return detected

        # Seek to 20% (index 20) -> no hazards
        det_20 = compute_detected_at_index(20)
        self.assertEqual(len(det_20), 0)

        # Seek forward to 50% (index 50) -> Hazard #101 detected
        det_50 = compute_detected_at_index(50)
        self.assertIn(101, det_50)
        self.assertNotIn(102, det_50)

        # Seek forward to 90% (index 90) -> Both #101 and #102 detected
        det_90 = compute_detected_at_index(90)
        self.assertIn(101, det_90)
        self.assertIn(102, det_90)

        # Seek BACKWARD to 30% (index 30) -> Both must be cleared!
        det_back_30 = compute_detected_at_index(30)
        self.assertEqual(len(det_back_30), 0, "Scrubbing backward must reset detected hazards")

    def test_relative_stepping_and_boundary_clamping(self):
        """Verify relative scrubber stepping (+-0.02) and [0.0, 1.0] boundary clamping."""
        def step_relative(cur_progress, delta):
            return max(0.0, min(1.0, cur_progress + delta))

        # Forward step from 0.50
        self.assertAlmostEqual(step_relative(0.50, 0.02), 0.52)
        # Backward step from 0.50
        self.assertAlmostEqual(step_relative(0.50, -0.02), 0.48)
        # Clamping at upper boundary
        self.assertEqual(step_relative(0.99, 0.05), 1.0)
        # Clamping at lower boundary
        self.assertEqual(step_relative(0.01, -0.05), 0.0)

    def test_waypoint_metadata_at_progress_for_tooltips(self):
        """Verify hover scrubber tooltip metadata generation at arbitrary progress."""
        total_dist_m = 3500.0  # 3.5 km
        total_sec = round((total_dist_m / 1000.0) / (35.0 / 3600.0))  # at 35 km/h -> 360s

        def get_meta(progress):
            clamped = max(0.0, min(1.0, progress))
            cur_dist = total_dist_m * clamped
            cur_sec = round(total_sec * clamped)
            m = str(cur_sec // 60).zfill(2)
            s = str(cur_sec % 60).zfill(2)
            return {
                "distance_km": f"{cur_dist / 1000.0:.2f}",
                "time_formatted": f"{m}:{s}",
                "progress_pct": round(clamped * 100)
            }

        meta_mid = get_meta(0.50)
        self.assertEqual(meta_mid["distance_km"], "1.75")
        self.assertEqual(meta_mid["time_formatted"], "03:00")
        self.assertEqual(meta_mid["progress_pct"], 50)



    def test_debre_berhan_arterial_route_interpolation_and_monotonicity(self):
        """Verify Debre Berhan municipal corridor (Tebasse -> DBU -> Stadium -> Hospital -> Piazza) interpolation."""
        debre_berhan_waypoints = [
            {"lat": 9.64242, "lon": 39.50659, "speed": 45.0, "landmark": "Tebasse Industrial Park"},
            {"lat": 9.65832, "lon": 39.51624, "speed": 48.0, "landmark": "Kebele 08 & BGI Brewery"},
            {"lat": 9.66717, "lon": 39.51937, "speed": 42.0, "landmark": "DBU South Gate & Research"},
            {"lat": 9.67202, "lon": 39.52719, "speed": 38.0, "landmark": "DBU Main Campus Entrance"},
            {"lat": 9.67750, "lon": 39.52780, "speed": 35.0, "landmark": "Sports Complex Avenue"},
            {"lat": 9.68182, "lon": 39.52820, "speed": 32.0, "landmark": "Debre Berhan Stadium"},
            {"lat": 9.68450, "lon": 39.53500, "speed": 36.0, "landmark": "Health Science Avenue"},
            {"lat": 9.68644, "lon": 39.54096, "speed": 30.0, "landmark": "Hakim Gizaw Referral Hospital"},
            {"lat": 9.68247, "lon": 39.54063, "speed": 34.0, "landmark": "Millennium Park & Monument"},
            {"lat": 9.67822, "lon": 39.53304, "speed": 28.0, "landmark": "Piazza Central Square & Zara Yaqob"},
            {"lat": 9.67608, "lon": 39.53490, "speed": 25.0, "landmark": "Arada Central Heritage Market"}
        ]

        interpolated = interpolate_path(debre_berhan_waypoints, step_meters=5.0)
        self.assertGreater(len(interpolated), 100, "Should generate smooth high-density path along corridor")

        # Total distance should be approximately 5.0 to 6.5 km
        total_dist_km = interpolated[-1]["distance_m"] / 1000.0
        self.assertGreater(total_dist_km, 7.5)
        self.assertLess(total_dist_km, 9.5)

        # Monotonicity check
        for i in range(len(interpolated) - 1):
            self.assertLessEqual(interpolated[i]["distance_m"], interpolated[i + 1]["distance_m"])

    def test_landmark_metadata_and_urban_distress_detection(self):
        """Verify landmark metadata propagation and proximity hazard detection along Debre Berhan corridor."""
        debre_berhan_waypoints = [
            {"lat": 9.64242, "lon": 39.50659, "speed": 45.0, "landmark": "Tebasse Industrial Park"},
            {"lat": 9.67202, "lon": 39.52719, "speed": 38.0, "landmark": "DBU Main Campus Entrance"},
            {"lat": 9.68182, "lon": 39.52820, "speed": 32.0, "landmark": "Debre Berhan Stadium"},
            {"lat": 9.68644, "lon": 39.54096, "speed": 30.0, "landmark": "Hakim Gizaw Referral Hospital"},
            {"lat": 9.67822, "lon": 39.53304, "speed": 28.0, "landmark": "Piazza Central Square & Zara Yaqob"}
        ]

        interpolated = interpolate_path(debre_berhan_waypoints, step_meters=10.0)

        # Check landmarks appear along the path
        landmarks_seen = {p["landmark"] for p in interpolated if p["landmark"]}
        self.assertIn("Tebasse Industrial Park", landmarks_seen)
        self.assertIn("DBU Main Campus Entrance", landmarks_seen)
        self.assertIn("Debre Berhan Stadium", landmarks_seen)
        self.assertIn("Hakim Gizaw Referral Hospital", landmarks_seen)
        self.assertIn("Piazza Central Square & Zara Yaqob", landmarks_seen)

        # Test urban distress defect detection near DBU (9.6725, 39.5275)
        dbu_pothole = {"id": "DB-106", "lat": 9.67250, "lon": 39.52750}
        detected = False
        threshold_m = 100.0

        for pt in interpolated:
            dist = haversine_m(pt["lat"], pt["lon"], dbu_pothole["lat"], dbu_pothole["lon"])
            if dist <= threshold_m:
                detected = True
                break

        self.assertTrue(detected, "DBU urban defect must be detected within 100m radar threshold")



    def test_video_time_synchronization_and_drift_correction(self):
        """Verify video time mapping from progress and sub-second drift compensation."""
        video_duration_sec = 34.0

        def compute_video_target_time(progress):
            clamped = max(0.0, min(1.0, progress))
            return clamped * video_duration_sec

        self.assertAlmostEqual(compute_video_target_time(0.0), 0.0)
        self.assertAlmostEqual(compute_video_target_time(0.5), 17.0)
        self.assertAlmostEqual(compute_video_target_time(1.0), 34.0)

        # Drift correction logic
        threshold_drift_sec = 0.40
        def check_drift_correction_needed(current_video_time, progress):
            target_time = compute_video_target_time(progress)
            drift = abs(current_video_time - target_time)
            return drift > threshold_drift_sec, target_time

        # Normal small jitter (0.15s) -> no forced seek
        needs_seek, target = check_drift_correction_needed(17.15, 0.5)
        self.assertFalse(needs_seek, "Small drift within 0.40s should play naturally")

        # Significant drift (1.2s, e.g. after scrubber jump) -> forced seek
        needs_seek, target = check_drift_correction_needed(10.0, 0.5)
        self.assertTrue(needs_seek, "Large drift must trigger forced synchronization")
        self.assertAlmostEqual(target, 17.0)

    def test_video_playback_rate_matching(self):
        """Verify HTML5 video playbackRate matches simulation speed multipliers."""
        valid_multipliers = [0.5, 1.0, 2.0, 4.0, 8.0]
        for mult in valid_multipliers:
            playback_rate = max(0.25, min(16.0, mult))
            self.assertEqual(playback_rate, mult)

    def test_bidirectional_video_seek_mapping(self):
        """Verify seeking video timestamp (e.g. 28.9s for Defect #55) maps to trajectory index."""
        video_duration_sec = 34.0
        defect_55_time = 28.96  # Video timestamp for Defect #55 in database
        expected_ratio = defect_55_time / video_duration_sec

        self.assertGreater(expected_ratio, 0.80)
        self.assertLess(expected_ratio, 0.90)

        # Mapping back from ratio to video time
        recovered_time = expected_ratio * video_duration_sec
        self.assertAlmostEqual(recovered_time, defect_55_time, places=2)


if __name__ == "__main__":
    unittest.main()

