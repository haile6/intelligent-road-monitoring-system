"""
Automated script to capture high-resolution demo screenshots of the running RoadEye web platform.
Saves directly into docs/screenshots/ for inclusion in README.md and GitHub showcase.
"""
import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By

BASE_URL = "http://127.0.0.1:8080"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pages = [
    ("/", "01_executive_dashboard.png", "Executive KPI Dashboard"),
    ("/studio", "02_telemetry_studio.png", "Synchronized Telemetry Studio"),
    ("/map", "03_gis_road_map.png", "Interactive GIS Pavement Map"),
    ("/incidents", "04_incident_ledger.png", "Municipal Incident Ledger & Workflow"),
    ("/plates", "05_anpr_plates_gallery.png", "ANPR License Plate Evidence Gallery"),
    ("/reports", "06_work_orders_reports.png", "Municipal Work Orders & Inspection Reports"),
]

def main():
    print("[*] Initializing Edge WebDriver in headless mode...")
    options = EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,1000")
    options.add_argument("--hide-scrollbars")
    
    driver = None
    try:
        driver = webdriver.Edge(options=options)
        driver.set_window_size(1600, 1000)

        for route, filename, label in pages:
            url = BASE_URL + route
            target_path = os.path.join(OUTPUT_DIR, filename)
            print(f"[*] Navigating to {label} ({url})...")
            driver.get(url)
            time.sleep(2.5)  # Allow Leaflet tiles, Chart.js, and telemetry elements to render completely
            driver.save_screenshot(target_path)
            size_kb = os.path.getsize(target_path) / 1024
            print(f"[+] Saved {filename} ({size_kb:.1f} KB)")

        print("\n[+] All demo screenshots successfully captured!")
    except Exception as e:
        print(f"[!] Error capturing screenshots: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
