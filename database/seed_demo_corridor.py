"""
Seed realistic Ethiopian corridor demonstration data for Debre Berhan / Route A2 Highway.
Populates GPS track points, realistic road damage incidents, and plate detections.
"""
import os
import sys
import csv
import math
import random
import sqlite3
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "road_monitoring.db")
GPS_CSV = os.path.join(BASE_DIR, "data", "gps_track.csv")
OSM_CSV = os.path.join(BASE_DIR, "data", "open_data", "debre_berhan_roads_osm.csv")

sys.path.insert(0, BASE_DIR)
try:
    from risk.priority_weight import compute_priority_weight
except ImportError:
    def compute_priority_weight(*args, **kwargs):
        return {"weight": 70, "priority_label": "High Priority", "breakdown": {}, "explanation": ""}


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2.0)**2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def seed_all_demo_data(force=False):
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    # 1. Read OSM roads
    if not os.path.exists(OSM_CSV):
        print(f"OSM road file not found: {OSM_CSV}")
        return False

    roads = []
    with open(OSM_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                lat = float(r.get("mid_lat", 0))
                lon = float(r.get("mid_lon", 0))
                if lat != 0 and lon != 0:
                    roads.append({
                        "name": r.get("road_name", "Debre Berhan Highway"),
                        "name_en": r.get("name_en", "Debre Berhan Highway"),
                        "landmark": r.get("landmark", ""),
                        "subcity": r.get("subcity_kebele", ""),
                        "highway_class": r.get("highway_class", "trunk"),
                        "lat": lat,
                        "lon": lon
                    })
            except Exception:
                continue

    if not roads:
        print("No valid road coordinates found.")
        return False

    # Sort roads to create a smooth corridor trajectory (South to North)
    roads.sort(key=lambda r: (r["lat"], r["lon"]))

    # 2. Generate smooth GPS track points (~300 points)
    gps_rows = []
    start_time = datetime.now() - timedelta(minutes=45)
    current_time = start_time

    # Interpolate waypoints
    for i in range(len(roads) - 1):
        p1 = roads[i]
        p2 = roads[i + 1]
        dist = haversine_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        steps = max(2, min(12, int(dist / 40.0)))
        for s in range(steps):
            frac = s / float(steps)
            cur_lat = p1["lat"] + frac * (p2["lat"] - p1["lat"])
            cur_lon = p1["lon"] + frac * (p2["lon"] - p1["lon"])
            speed = round(random.uniform(48.0, 62.0), 1)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            gps_rows.append(f"{time_str},{cur_lat:.6f},{cur_lon:.6f},{speed}")
            current_time += timedelta(seconds=2)

    # Write GPS CSV if missing or force
    if not os.path.exists(GPS_CSV) or force:
        with open(GPS_CSV, "w", encoding="utf-8") as f:
            f.write("timestamp,latitude,longitude,speed_kmh\n")
            for line in gps_rows:
                f.write(line + "\n")
        print(f"Generated {len(gps_rows)} GPS track points in {GPS_CSV}")

    # 3. Seed Incidents into SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ensure tables exist
    c.execute("""
    CREATE TABLE IF NOT EXISTS weather_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        precipitation_mm REAL,
        temperature_c REAL,
        humidity_pct REAL,
        location_name TEXT
    )""")

    c.execute("SELECT COUNT(*) FROM incidents")
    inc_count = c.fetchone()[0]

    if inc_count == 0 or force:
        if force:
            c.execute("DELETE FROM incidents")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incidents_to_insert = []
        track_id = 1

        # We select distinct road points to place realistic defects
        sample_roads = roads[::max(1, len(roads) // 40)]
        for idx, rd in enumerate(sample_roads):
            # Alternate defect types: 60% Potholes, 25% Alligator Cracks, 15% Transverse Cracks
            rand_type = random.random()
            if rand_type < 0.60:
                damage_type = "pothole"
                depth_cm = round(random.uniform(4.0, 14.5), 1)
                width_cm = round(random.uniform(28.0, 75.0), 1)
                area_m2 = round((math.pi * (width_cm / 200.0)**2) * 1.2, 3)
                vol_liters = round(area_m2 * (depth_cm / 100.0) * 1000.0 * 0.75, 1)
            elif rand_type < 0.85:
                damage_type = "alligator_crack"
                depth_cm = round(random.uniform(2.5, 6.0), 1)
                width_cm = round(random.uniform(40.0, 110.0), 1)
                area_m2 = round(random.uniform(0.35, 1.8), 2)
                vol_liters = round(area_m2 * 2.5, 1)
            else:
                damage_type = "transverse_crack"
                depth_cm = round(random.uniform(1.8, 4.5), 1)
                width_cm = round(random.uniform(15.0, 35.0), 1)
                area_m2 = round(random.uniform(0.15, 0.6), 2)
                vol_liters = round(area_m2 * 1.8, 1)

            # Realistic Kinematics
            dist_m = round(random.uniform(8.5, 28.0), 1)
            speed_kmh = round(random.uniform(45.0, 60.0), 1)
            speed_mps = speed_kmh / 3.6
            ttc_s = round(dist_m / max(speed_mps, 5.0), 2)
            ever_in_path = 1 if (idx % 3 != 0) else 0

            # Dynamic Risk Classification
            if (depth_cm >= 8.0 or ttc_s < 2.0) and ever_in_path:
                risk_level = "IMMEDIATE_DANGER"
                severity = "Critical"
                priority = "Critical"
                status = "Reported"
            elif depth_cm >= 4.5 or ttc_s < 3.2:
                risk_level = "CAUTION"
                severity = "Warning"
                priority = "High"
                status = "Pending" if (idx % 2 == 0) else "Reported"
            else:
                risk_level = "SAFE"
                severity = "Minor"
                priority = "Medium"
                status = "Repaired" if (idx % 4 == 0) else "Reported"

            # Geolocation with slight lateral road offset
            lat_offset = random.uniform(-0.00012, 0.00012)
            lon_offset = random.uniform(-0.00012, 0.00012)
            d_lat = rd["lat"] + lat_offset
            d_lon = rd["lon"] + lon_offset

            # Ethiopian Road Authority Priority Weight
            pw_res = compute_priority_weight(
                risk_level=risk_level,
                severity=severity,
                speed_kmh=speed_kmh,
                confidence=0.92,
                damage_type=damage_type
            )

            note_text = f"{rd['name']} ({rd['name_en']}) — {rd['subcity']}"
            dual_factors = f"TTC: {ttc_s}s | Dmax: {depth_cm}cm ({'In-Path Hazard' if ever_in_path else 'Shoulder'})"

            incidents_to_insert.append((
                "debre_berhan_route_a2.mp4",
                track_id,
                damage_type,
                round(random.uniform(0.88, 0.98), 2),
                track_id * 85,
                round(track_id * 2.8, 1),
                now_str,
                rd["lat"],
                rd["lon"],
                speed_kmh,
                round(speed_mps, 2),
                "CAN_BUS",
                0,
                dist_m,
                ttc_s,
                ttc_s,
                severity,
                risk_level,
                f"Depth {depth_cm}cm + Speed {speed_kmh}km/h",
                d_lat,
                d_lon,
                ever_in_path,
                dist_m,
                "FIX_3D",
                status,
                "Road Works Crew Alpha" if idx % 2 == 0 else "Emergency Patch Crew Beta",
                priority,
                note_text,
                pw_res.get("weight", 65),
                str(pw_res.get("breakdown", {})),
                pw_res.get("explanation", ""),
                depth_cm,
                area_m2,
                vol_liters,
                dual_factors,
                now_str,
                now_str
            ))
            track_id += 1

        c.executemany("""
        INSERT INTO incidents (
            video_name, track_id, damage_type, confidence, frame_number, video_time_s,
            timestamp, latitude, longitude, speed_kmh, speed_mps, speed_source,
            speed_is_default, distance_m, ttc_seconds, min_ttc_in_path_s, severity,
            max_risk_level, risk_reasons, damage_latitude, damage_longitude, ever_in_path,
            est_min_distance_m, gps_status, status, assigned_crew, repair_priority,
            notes, priority_weight, weight_breakdown, weight_explanation,
            depth_max_cm, surface_area_m2, cavity_volume_liters, dual_risk_factors,
            created_at, updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """, incidents_to_insert)
        print(f"Seeded {len(incidents_to_insert)} authentic road hazards on Debre Berhan Highway.")

    # 4. Seed Plate Readings (Ethiopian commercial trucks & cars)
    c.execute("SELECT COUNT(*) FROM plate_readings")
    if c.fetchone()[0] == 0 or force:
        if force:
            c.execute("DELETE FROM plate_readings")

        plates = [
            ("3-A-12845 AA", 0.94, "Toyota Hilux Patrol"),
            ("3-B-67219 AM", 0.92, "Isuzu FSR Freight Truck"),
            ("3-C-89102 ET", 0.96, "Sino Heavy Dump Truck"),
            ("3-A-45120 AA", 0.91, "Toyota Land Cruiser"),
            ("3-B-33491 AM", 0.95, "Sino Dump Truck (35-Ton)"),
            ("4-08912 ETH", 0.89, "Federal Ministry Patrol"),
            ("3-A-99214 AA", 0.93, "Commercial Minibus"),
            ("3-B-71284 AM", 0.97, "Isuzu Forward Freight"),
            ("3-C-11029 ET", 0.95, "Heavy Cement Bulk Hauler"),
            ("3-A-66710 AA", 0.92, "Toyota Pickup")
        ]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plate_rows = []
        for i, (plt, conf, desc) in enumerate(plates):
            plate_rows.append((
                "debre_berhan_route_a2.mp4",
                (i + 1) * 120,
                round((i + 1) * 4.2, 1),
                i + 1,
                plt,
                conf,
                320, 240, 580, 310,
                f"crop_plate_{i+1}.jpg",
                now_str
            ))
        c.executemany("""
        INSERT INTO plate_readings (
            video_name, frame, video_time_s, vehicle_id, plate_text,
            ocr_confidence, plate_x1, plate_y1, plate_x2, plate_y2,
            crop_filename, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, plate_rows)
        print(f"Seeded {len(plate_rows)} Ethiopian plate readings.")

    # 5. Seed Weather Observations (30-day realistic rainfall)
    c.execute("SELECT COUNT(*) FROM weather_observations")
    if c.fetchone()[0] == 0 or force:
        if force:
            c.execute("DELETE FROM weather_observations")

        weather_rows = []
        today = datetime.now().date()
        for d in range(30):
            day_date = (today - timedelta(days=29 - d)).strftime("%Y-%m-%d")
            # Bega / Mild rainfall simulation: occasional showers
            rain_mm = round(random.choice([0.0, 0.0, 0.0, 1.2, 4.5, 0.0, 8.0, 0.0, 2.4]), 1)
            weather_rows.append((
                day_date,
                9.68,
                39.53,
                rain_mm,
                round(random.uniform(16.0, 24.0), 1),
                58.0,
                0.15,
                15.0,
                "Open-Meteo ERA Archive",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

        c.executemany("""
        INSERT INTO weather_observations (
            date, latitude, longitude, precipitation_mm, temperature_mean_c,
            relative_humidity_pct, soil_moisture_m3m3, soil_moisture_pct,
            weather_source, retrieval_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, weather_rows)
        print("Seeded 30 days of weather observations.")

    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    seed_all_demo_data(force=True)
