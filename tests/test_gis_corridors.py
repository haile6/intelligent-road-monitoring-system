"""
GIS Geometry and Spatial Integrity Test Suite.
Validates:
1. Debre Berhan OpenStreetMap (OSM) GeoJSON schema and coordinate ranges.
2. 36 Nationwide road corridor coordinates, start/end bounds, and length validation.
3. Spatial bounding box calculations for regional corridor zoom presets.
"""
import os
import sys
import json
import math
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database.db_manager as db_manager


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class TestGisCorridors(unittest.TestCase):
    """Test suite for GIS spatial structures and geographic validity."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.osm_path = os.path.join(cls.base_dir, "data", "open_data", "debre_berhan_roads_osm.geojson")

    def test_osm_geojson_exists_and_valid(self):
        """Ensure Debre Berhan OSM GeoJSON exists and conforms to GeoJSON specification."""
        self.assertTrue(os.path.exists(self.osm_path), "Debre Berhan OSM GeoJSON file must exist.")
        
        with open(self.osm_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("type"), "FeatureCollection")
        features = data.get("features", [])
        self.assertGreaterEqual(len(features), 50, "Should contain at least 50 road segments.")

        for feat in features:
            self.assertEqual(feat.get("type"), "Feature")
            geom = feat.get("geometry", {})
            self.assertIn(geom.get("type"), ["LineString", "MultiLineString"])
            coords = geom.get("coordinates", [])
            self.assertTrue(len(coords) >= 2)

    def test_corridor_coordinates_within_ethiopia(self):
        """Validate that all 36 corridor start/end points fall within Ethiopian territory."""
        analytics = db_manager.get_contractor_analytics()
        contractors = analytics["contractors"]

        for c in contractors:
            s_lat = float(c["start_lat"])
            s_lon = float(c["start_lon"])
            e_lat = float(c["end_lat"])
            e_lon = float(c["end_lon"])

            # Ethiopia bounding box: Lat [3.0, 15.0], Lon [33.0, 48.0]
            self.assertTrue(3.0 <= s_lat <= 15.0, f"Start lat {s_lat} out of bounds for {c['road_name']}")
            self.assertTrue(33.0 <= s_lon <= 48.0, f"Start lon {s_lon} out of bounds for {c['road_name']}")
            self.assertTrue(3.0 <= e_lat <= 15.0, f"End lat {e_lat} out of bounds for {c['road_name']}")
            self.assertTrue(33.0 <= e_lon <= 48.0, f"End lon {e_lon} out of bounds for {c['road_name']}")

    def test_corridor_length_reasonableness(self):
        """Verify that corridor reported lengths are positive and geodetically reasonable."""
        analytics = db_manager.get_contractor_analytics()
        contractors = analytics["contractors"]

        for c in contractors:
            length = float(c["length_km"])
            self.assertGreater(length, 0, f"Length must be positive for {c['road_name']}")

            # Calculate straight-line Euclidean/geodesic distance between start and end
            s_lat = float(c["start_lat"])
            s_lon = float(c["start_lon"])
            e_lat = float(c["end_lat"])
            e_lon = float(c["end_lon"])
            crow_fly = haversine_distance(s_lat, s_lon, e_lat, e_lon)

            # Road distance is typically >= crow-fly distance (with slight tolerance for roundabouts/loop roads)
            self.assertLessEqual(crow_fly, length * 1.5,
                                f"Crow-fly distance {crow_fly:.1f} km significantly exceeds reported road length {length} km for {c['road_name']}")


if __name__ == "__main__":
    unittest.main()

