"""
Database manager for Road Monitoring System Web Management Panel.
Handles SQLite persistence for damage incidents, vehicle/plate detections, and pipeline jobs.
"""
import os
import sqlite3
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "road_monitoring.db")
PHASE9_CSV = os.path.join(BASE_DIR, "outputs", "phase9_damage_events.csv")
PLATE_CSV = os.path.join(BASE_DIR, "outputs", "phase8_plate_readings.csv")
GPS_CSV = os.path.join(BASE_DIR, "data", "gps_track.csv")


def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Incidents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER,
        damage_type TEXT,
        confidence REAL,
        frame_number INTEGER,
        video_time_s REAL,
        timestamp TEXT,
        latitude REAL,
        longitude REAL,
        speed_kmh REAL,
        speed_mps REAL,
        speed_source TEXT,
        speed_is_default INTEGER,
        distance_m REAL,
        ttc_seconds REAL,
        min_ttc_in_path_s REAL,
        severity TEXT,
        max_risk_level TEXT,
        risk_reasons TEXT,
        damage_latitude REAL,
        damage_longitude REAL,
        ever_in_path INTEGER,
        est_min_distance_m REAL,
        gps_status TEXT,
        status TEXT DEFAULT 'Reported',
        assigned_crew TEXT DEFAULT 'Road Works Crew Alpha',
        repair_priority TEXT DEFAULT 'Medium',
        notes TEXT DEFAULT '',
        created_at TEXT,
        updated_at TEXT
    )
    """)

    # Vehicle / License Plate table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plate_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame INTEGER,
        video_time_s REAL,
        vehicle_id INTEGER,
        plate_text TEXT,
        ocr_confidence REAL,
        plate_x1 INTEGER,
        plate_y1 INTEGER,
        plate_x2 INTEGER,
        plate_y2 INTEGER,
        crop_filename TEXT,
        created_at TEXT
    )
    """)

    # Pipeline Runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phase_name TEXT,
        command TEXT,
        status TEXT,
        logs TEXT,
        started_at TEXT,
        finished_at TEXT,
        exit_code INTEGER
    )
    """)

    # System Settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT,
        description TEXT,
        updated_at TEXT
    )
    """)

    conn.commit()
    conn.close()

    # Auto import existing data if empty
    sync_from_csv_if_empty()


def sync_from_csv_if_empty():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM incidents")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0 and os.path.exists(PHASE9_CSV):
        import_incidents_from_csv(PHASE9_CSV)
    
    # Check plate readings
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM plate_readings")
    plate_count = cursor.fetchone()[0]
    conn.close()

    if plate_count == 0 and os.path.exists(PLATE_CSV):
        import_plates_from_csv(PLATE_CSV)


def import_incidents_from_csv(csv_path, clear_existing=False):
    if not os.path.exists(csv_path):
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()
    if clear_existing:
        cursor.execute("DELETE FROM incidents")
    imported_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row.get("track_id", 0)) if row.get("track_id") else None
            # Check if already exists by track_id and frame
            frame_num = int(row.get("frame_number", 0)) if row.get("frame_number") else None
            
            cursor.execute("SELECT id FROM incidents WHERE track_id = ? AND frame_number = ?", (track_id, frame_num))
            existing = cursor.fetchone()
            if existing:
                continue

            max_risk = row.get("max_risk_level", "SAFE / MONITOR").strip()
            # Assign priority based on risk level
            if "IMMEDIATE" in max_risk.upper() or "DANGER" in max_risk.upper():
                priority = "Critical"
            elif "WARNING" in max_risk.upper():
                priority = "High"
            elif "CAUTION" in max_risk.upper():
                priority = "Medium"
            else:
                priority = "Low"

            cursor.execute("""
            INSERT INTO incidents (
                track_id, damage_type, confidence, frame_number, video_time_s,
                timestamp, latitude, longitude, speed_kmh, speed_mps, speed_source,
                speed_is_default, distance_m, ttc_seconds, min_ttc_in_path_s, severity,
                max_risk_level, risk_reasons, damage_latitude, damage_longitude,
                ever_in_path, est_min_distance_m, gps_status, status, assigned_crew,
                repair_priority, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                row.get("damage_type"),
                float(row.get("confidence", 0)) if row.get("confidence") else None,
                frame_num,
                float(row.get("video_time_s", 0)) if row.get("video_time_s") else None,
                row.get("timestamp"),
                float(row.get("latitude", 0)) if row.get("latitude") else None,
                float(row.get("longitude", 0)) if row.get("longitude") else None,
                float(row.get("speed_kmh", 0)) if row.get("speed_kmh") else None,
                float(row.get("speed_mps", 0)) if row.get("speed_mps") else None,
                row.get("speed_source"),
                1 if str(row.get("speed_is_default", "")).lower() == "true" else 0,
                float(row.get("distance_m", 0)) if row.get("distance_m") else None,
                float(row.get("ttc_seconds", 0)) if row.get("ttc_seconds") else None,
                float(row.get("min_ttc_in_path_s", 0)) if row.get("min_ttc_in_path_s") else None,
                row.get("severity") or "",
                max_risk,
                row.get("risk_reasons") or "",
                float(row.get("damage_latitude", 0)) if row.get("damage_latitude") else None,
                float(row.get("damage_longitude", 0)) if row.get("damage_longitude") else None,
                1 if str(row.get("ever_in_path", "")).lower() == "true" else 0,
                float(row.get("est_min_distance_m", 0)) if row.get("est_min_distance_m") else None,
                row.get("gps_status") or "ok",
                "Reported",
                "Road Works Crew Alpha",
                priority,
                f"Auto-imported from Phase 9 analysis. {row.get('risk_reasons', '')}".strip(),
                now_str,
                now_str
            ))
            imported_count += 1

    conn.commit()
    conn.close()
    return imported_count


def import_plates_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return 0

    crops_dir = os.path.join(BASE_DIR, "outputs", "plate_crops")
    crop_files = set(os.listdir(crops_dir)) if os.path.exists(crops_dir) else set()

    conn = get_db_connection()
    cursor = conn.cursor()
    imported_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate_text = (row.get("plate_text") or "").strip()
            # Only import rows with valid detections or valid coordinates
            x1 = row.get("plate_x1")
            if not x1:
                continue

            frame = int(row.get("frame", 0))
            # Try to find corresponding crop file: frame_{frame:05d}_...
            crop_name = None
            prefix = f"frame_{frame:05d}_plate_"
            for cf in crop_files:
                if cf.startswith(prefix):
                    crop_name = cf
                    break

            cursor.execute("""
            INSERT INTO plate_readings (
                frame, video_time_s, vehicle_id, plate_text, ocr_confidence,
                plate_x1, plate_y1, plate_x2, plate_y2, crop_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                frame,
                float(row.get("video_time_s", 0)) if row.get("video_time_s") else None,
                int(row.get("vehicle_id", 0)) if row.get("vehicle_id") else None,
                plate_text if plate_text else "UNREADABLE / LOW CONF",
                float(row.get("ocr_confidence", 0)) if row.get("ocr_confidence") else None,
                int(x1),
                int(row.get("plate_y1", 0)),
                int(row.get("plate_x2", 0)),
                int(row.get("plate_y2", 0)),
                crop_name,
                now_str
            ))
            imported_count += 1

    conn.commit()
    conn.close()
    return imported_count


def get_all_incidents(status=None, risk_level=None, damage_type=None, search=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    if risk_level and risk_level != "All":
        query += " AND max_risk_level LIKE ?"
        params.append(f"%{risk_level}%")

    if damage_type and damage_type != "All":
        query += " AND damage_type = ?"
        params.append(damage_type)

    if search:
        query += " AND (notes LIKE ? OR track_id LIKE ? OR risk_reasons LIKE ? OR assigned_crew LIKE ?)"
        s = f"%{search}%"
        params.extend([s, s, s, s])

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_incident_by_id(incident_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_incident(incident_id, status=None, priority=None, assigned_crew=None, notes=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if priority is not None:
        updates.append("repair_priority = ?")
        params.append(priority)
    if assigned_crew is not None:
        updates.append("assigned_crew = ?")
        params.append(assigned_crew)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)

    if updates:
        updates.append("updated_at = ?")
        params.append(now_str)
        params.append(incident_id)
        sql = f"UPDATE incidents SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(sql, params)
        conn.commit()

    conn.close()
    return get_incident_by_id(incident_id)


def get_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM incidents")
    total_incidents = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE max_risk_level LIKE '%IMMEDIATE%' OR max_risk_level LIKE '%DANGER%'")
    immediate_danger = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE max_risk_level LIKE '%WARNING%'")
    warning_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE max_risk_level LIKE '%CAUTION%'")
    caution_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE max_risk_level LIKE '%SAFE%' OR max_risk_level LIKE '%MONITOR%'")
    safe_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE damage_type = 'pothole'")
    potholes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE damage_type = 'alligator_crack'")
    cracks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE status = 'Repaired'")
    repaired_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM incidents WHERE status IN ('Reported', 'In Review')")
    pending_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM plate_readings")
    total_plates = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(speed_kmh) FROM incidents WHERE speed_kmh IS NOT NULL")
    avg_speed_row = cursor.fetchone()[0]
    avg_speed = round(avg_speed_row, 1) if avg_speed_row else 0.0

    conn.close()

    return {
        "total_incidents": total_incidents,
        "immediate_danger": immediate_danger,
        "warning_count": warning_count,
        "caution_count": caution_count,
        "safe_count": safe_count,
        "potholes": potholes,
        "cracks": cracks,
        "repaired_count": repaired_count,
        "pending_count": pending_count,
        "total_plates": total_plates,
        "avg_speed": avg_speed
    }


def get_gps_track_points():
    points = []
    if os.path.exists(GPS_CSV):
        with open(GPS_CSV, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("timestamp"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        points.append({
                            "timestamp": parts[0].strip(),
                            "lat": float(parts[1].strip()),
                            "lon": float(parts[2].strip()),
                            "speed": float(parts[3].strip()) if len(parts) > 3 else 0.0
                        })
                    except ValueError:
                        continue
    return points


def get_all_plates():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plate_readings ORDER BY frame ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def log_pipeline_run(phase_name, command, status="running", logs=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO pipeline_runs (phase_name, command, status, logs, started_at)
    VALUES (?, ?, ?, ?, ?)
    """, (phase_name, command, status, logs, now_str))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def update_pipeline_run(run_id, status, logs, exit_code):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    UPDATE pipeline_runs
    SET status = ?, logs = ?, exit_code = ?, finished_at = ?
    WHERE id = ?
    """, (status, logs, exit_code, now_str, run_id))
    conn.commit()
    conn.close()


def get_recent_pipeline_runs(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
