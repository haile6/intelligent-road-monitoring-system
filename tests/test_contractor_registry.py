"""
Unit and Integration Tests for Ethiopian Road Contractors Registry,
Defect Liability Period (DLP) Warranty Analytics, and Web API Endpoints.
"""
import os
import sys
import unittest
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database.db_manager as db_manager
import web.app as wa


class TestContractorRegistry(unittest.TestCase):
    """Test suite for contractor analytics, warranty logic, and API endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = wa.app.test_client() if hasattr(wa, 'app') else wa.test_client()
        cls.client.testing = True

    def test_database_contractors_populated(self):
        """Verify that the database contains the 36 nationwide road contractors benchmark."""
        analytics = db_manager.get_contractor_analytics()
        self.assertIn("contractors", analytics)
        contractors = analytics["contractors"]
        self.assertGreaterEqual(len(contractors), 36, "Should contain at least 36 nationwide road corridors.")

    def test_contractor_fields_integrity(self):
        """Ensure all required engineering and GIS fields are present and valid."""
        analytics = db_manager.get_contractor_analytics()
        contractors = analytics["contractors"]

        required_keys = [
            "id", "contractor_name", "road_name", "length_km",
            "pavement_type", "handover_date", "warranty_expiry_date",
            "warranty_years", "is_under_warranty", "liability_party",
            "start_lat", "start_lon", "end_lat", "end_lon"
        ]

        for c in contractors:
            for k in required_keys:
                self.assertIn(k, c, f"Missing field '{k}' in project {c.get('road_name')}")
            
            # Check GIS coordinates validity for Ethiopia (approx Lat 3 to 15, Lon 33 to 48)
            start_lat = float(c["start_lat"])
            start_lon = float(c["start_lon"])
            self.assertTrue(3.0 <= start_lat <= 15.0, f"Latitude {start_lat} out of Ethiopia bounds")
            self.assertTrue(33.0 <= start_lon <= 48.0, f"Longitude {start_lon} out of Ethiopia bounds")

    def test_warranty_status_consistency(self):
        """Verify warranty calculation logic and liability classification."""
        analytics = db_manager.get_contractor_analytics()
        contractors = analytics["contractors"]

        for c in contractors:
            if c["is_under_warranty"]:
                self.assertIn("Contractor", c["liability_party"])
            else:
                self.assertIn("Municipality", c["liability_party"])

    def test_quality_distribution_metrics(self):
        """Verify summary metrics and analytics keys."""
        analytics = db_manager.get_contractor_analytics()
        self.assertIn("total_contractors", analytics)
        self.assertIn("active_warranties", analytics)
        self.assertIn("total_roads_km", analytics)
        self.assertEqual(analytics["total_contractors"], 36)

    def test_api_contractors_endpoint(self):
        """Test GET /api/contractors returns valid JSON with 36 projects."""
        response = self.client.get("/api/contractors")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        contractors_data = data.get("data", {}).get("contractors", [])
        self.assertGreaterEqual(len(contractors_data), 36)

    def test_api_claim_notice_endpoint(self):
        """Test GET /api/contractors/<id>/claim-notice generates valid legal claim notices."""
        analytics = db_manager.get_contractor_analytics()
        contractors = analytics["contractors"]
        first_id = contractors[0]["id"]

        response = self.client.get(f"/api/contractors/{first_id}/claim-notice")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn("reference_no", data)
        self.assertIn("legal_basis", data)
        self.assertIn("authority", data)
        self.assertIn("contractor", data)

    def test_web_routes_render_success(self):
        """Verify web HTML templates render correctly without errors."""
        res_contractors = self.client.get("/contractors")
        self.assertEqual(res_contractors.status_code, 200)
        self.assertIn(b"Road Contractors &amp; Warranty Registry", res_contractors.data)

        res_map = self.client.get("/map")
        self.assertEqual(res_map.status_code, 200)
        self.assertIn(b"Geographic Road Intelligence Map", res_map.data)


if __name__ == "__main__":
    unittest.main()
