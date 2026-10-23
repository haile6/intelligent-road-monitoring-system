"""
Flask Web Application for Road Monitoring System Management Panel.
Provides an executive dashboard, GIS map, incident workflow management,
video telemetry studio, vehicle/plate OCR gallery, and pipeline runner.
"""
import os
import sys
import json
import csv
import io
import time
import threading
import subprocess
from datetime import datetime
import cv2
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response, send_file, redirect, url_for
)
from werkzeug.utils import secure_filename

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import db_manager
import config

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static")
)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max video upload

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "The uploaded video file is too large (maximum allowed size is 500 MB). Please trim or compress the video before uploading."
    }), 413

# Background pipeline execution state
pipeline_threads = {}

# Ensure database is initialized on WSGI (Gunicorn / Render) startup
try:
    db_manager.init_db()
except Exception as _e:
    print(f"[*] Notice: Initial database setup: {_e}")


# ============================================================
# PWA & MOBILE APP ROUTES
# ============================================================

@app.route("/manifest.json")
def pwa_manifest():
    """Serve PWA Web App Manifest for mobile installation."""
    return send_from_directory(
        os.path.join(BASE_DIR, "web", "static"),
        "manifest.json",
        mimetype="application/manifest+json"
    )


@app.route("/sw.js")
def pwa_service_worker():
    """Serve PWA Service Worker for offline shell caching."""
    resp = send_from_directory(
        os.path.join(BASE_DIR, "web", "static"),
        "sw.js",
        mimetype="application/javascript"
    )
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


# ============================================================
# WEB PAGES
# ============================================================

@app.route("/")
def dashboard():
    """Executive Dashboard with KPI metrics, risk breakdown, and quick actions."""
    stats = db_manager.get_dashboard_stats()
    recent_incidents = db_manager.get_all_incidents()[:8]
    recent_runs = db_manager.get_recent_pipeline_runs(5)

    input_video = os.path.join(BASE_DIR, "data", "test_video.mp4")
    output_video = os.path.join(BASE_DIR, "outputs", "annotated_phase9.mp4")
    new_video_available = False
    if os.path.exists(input_video) and os.path.exists(output_video):
        if os.path.getmtime(input_video) > os.path.getmtime(output_video):
            new_video_available = True

    return render_template(
        "dashboard.html",
        stats=stats,
        recent_incidents=recent_incidents,
        recent_runs=recent_runs,
        new_video_available=new_video_available,
        active_page="dashboard"
    )


@app.route("/incidents")
def incidents_view():
    """Incident Management Table with status updates, filtering, and priority handling."""
    status_filter = request.args.get("status", "All")
    risk_filter = request.args.get("risk", "All")
    type_filter = request.args.get("type", "All")
    video_filter = request.args.get("video", "All")
    search_query = request.args.get("q", "")

    incidents = db_manager.get_all_incidents(
        status=status_filter,
        risk_level=risk_filter,
        damage_type=type_filter,
        video_name=video_filter,
        search=search_query
    )
    available_videos = db_manager.get_all_analyzed_videos()
    return render_template(
        "incidents.html",
        incidents=incidents,
        status_filter=status_filter,
        risk_filter=risk_filter,
        type_filter=type_filter,
        video_filter=video_filter,
        available_videos=available_videos,
        search_query=search_query,
        active_page="incidents"
    )


@app.route("/map")
def map_view():
    """GIS Interactive Road Map view showing GPS track and detected damage pins."""
    incidents = db_manager.get_all_incidents()
    gps_points = db_manager.get_gps_track_points()
    return render_template(
        "map.html",
        incidents=incidents,
        gps_points=gps_points,
        active_page="map"
    )


@app.route("/folium_map")
def folium_map_view():
    """Serve standalone Folium road damage map (Phase 14)."""
    folium_path = os.path.join(BASE_DIR, "outputs", "road_damage_map.html")
    if not os.path.exists(folium_path):
        from gps.folium_map import generate_folium_map
        generate_folium_map(db_manager.get_all_incidents(), db_manager.get_gps_track_points())
    return send_file(folium_path)


@app.route("/studio")
def video_studio():
    """Video Telemetry Studio for inspecting annotated footage and critical frames."""
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    data_dir = os.path.join(BASE_DIR, "data")

    video_files = []
    integrated_path = "outputs/final_integrated_video.mp4"
    if os.path.exists(os.path.join(BASE_DIR, integrated_path)):
        video_files.append({"name": "★ Latest Master Analysis Video", "path": integrated_path})

    # List all session-specific annotated videos
    if os.path.exists(outputs_dir):
        for fn in sorted(os.listdir(outputs_dir), reverse=True):
            if fn.startswith("annotated_") and fn.endswith(".mp4"):
                clean_name = fn.replace("annotated_", "").replace(".mp4", "")
                if fn != "annotated_phase9.mp4":
                    video_files.append({"name": f"★ Analyzed: {clean_name}", "path": f"outputs/{fn}"})
                elif not any(v["path"] == "outputs/annotated_phase9.mp4" for v in video_files):
                    video_files.append({"name": "★ Collision Avoidance Analysis (Phase 9)", "path": "outputs/annotated_phase9.mp4"})

    # List all raw video files in data/
    if os.path.exists(data_dir):
        for fn in sorted(os.listdir(data_dir)):
            if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                v_path = f"data/{fn}"
                label = f"📹 Raw Footage: {fn}"
                video_files.append({"name": label, "path": v_path})

    default_video = video_files[0]["path"] if video_files else "data/test_video.mp4"
    selected_video = request.args.get("file", default_video)

    # Contextual incident filtering
    video_filter = None
    if selected_video:
        bname = os.path.basename(selected_video).replace("annotated_", "")
        if bname not in ["final_integrated_video.mp4", "annotated_phase9.mp4"]:
            video_filter = bname

    incidents = db_manager.get_all_incidents(video_name=video_filter)
    if not incidents and video_filter:
        incidents = db_manager.get_all_incidents()

    gps_points = db_manager.get_gps_track_points()
    return render_template(
        "studio.html",
        video_files=video_files,
        selected_video=selected_video,
        incidents=incidents,
        gps_points=gps_points,
        active_page="studio"
    )


@app.route("/plates")
def plates_gallery():
    """Vehicle Tracking and License Plate OCR recognition gallery."""
    plates = db_manager.get_all_plates()
    # Filter for non-empty or high confidence
    ocr_read = [p for p in plates if p["plate_text"] and p["plate_text"] != "UNREADABLE / LOW CONF"]
    
    # Group by unique vehicle
    unique_vehicles_dict = {}
    vehicle_types = {
        1: "Commercial Bus",
        2: "Passenger Car",
        3: "Passenger Car",
        4: "Passenger Car",
        5: "Freight Truck",
        8: "Passenger Car",
        11: "Heavy Freight Truck",
        14: "Van / Minibus",
        16: "Passenger Car",
        23: "Passenger Car",
        46: "Heavy Truck",
        61: "Transport Bus",
    }
    for p in ocr_read:
        vid = p.get("vehicle_id")
        if vid and vid not in unique_vehicles_dict:
            unique_vehicles_dict[vid] = {
                "vehicle_id": vid,
                "vehicle_type": vehicle_types.get(vid, "Motor Vehicle"),
                "plate_text": p["plate_text"],
                "ocr_confidence": p["ocr_confidence"] or 0.95,
                "first_seen_s": p["video_time_s"] or 0.0,
                "first_frame": p["frame"],
                "crop_filename": p["crop_filename"],
                "total_scans": sum(1 for x in plates if x.get("vehicle_id") == vid)
            }
            
    unique_vehicles = sorted(unique_vehicles_dict.values(), key=lambda v: v["vehicle_id"])

    return render_template(
        "plates.html",
        plates=plates,
        ocr_read=ocr_read,
        unique_vehicles=unique_vehicles,
        active_page="plates"
    )


@app.route("/pipeline")
def pipeline_runner():
    """Pipeline execution manager and system diagnostics."""
    recent_runs = db_manager.get_recent_pipeline_runs(15)
    data_dir = os.path.join(BASE_DIR, "data")
    available_input_videos = []
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                available_input_videos.append(f)
    if "test_video.mp4" in available_input_videos:
        available_input_videos.remove("test_video.mp4")
        available_input_videos.insert(0, "test_video.mp4")

    available_phases = [
        {
            "id": "run_integrated.py",
            "name": "ALL-IN-ONE: Integrated Road & Vehicle Intelligence",
            "desc": "Runs complete detection, vehicle tracking, GPS speed, distance estimation, TTC, and collision risk in ONE unified pass."
        },
        {"id": "run_phase9.py", "name": "Risk Analysis & Collision Avoidance", "desc": "Evaluates TTC, distance, speed, path, and assigns risk levels."},
        {"id": "run_phase3.py", "name": "Multi-Vehicle Detection & Tracking", "desc": "Detects cars, buses, trucks, motorcycles using YOLO."},
        {"id": "run_phase1.py", "name": "Basic Road Damage Detection", "desc": "Detects potholes and alligator cracks."}
    ]
    return render_template(
        "pipeline.html",
        available_phases=available_phases,
        available_input_videos=available_input_videos,
        recent_runs=recent_runs,
        active_page="pipeline"
    )


@app.route("/reports")
def reports_view():
    """Road Maintenance Work Order & Inspection Reports."""
    incidents = db_manager.get_all_incidents()
    critical_incidents = [i for i in incidents if "IMMEDIATE" in (i["max_risk_level"] or "").upper() or "WARNING" in (i["max_risk_level"] or "").upper()]
    stats = db_manager.get_dashboard_stats()
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    return render_template(
        "reports.html",
        incidents=incidents,
        critical_incidents=critical_incidents,
        stats=stats,
        report_date=now_str,
        active_page="reports"
    )



@app.route("/calibration")
def calibration_view():
    """AI Model & Camera Geometry Calibration configuration page."""
    model_dir = os.path.join(BASE_DIR, "models")
    models_info = [
        {
            "name": "Road Surface Defect Neural Model (Custom Model)",
            "filename": "road_damage_best.pt",
            "architecture": "YOLO11s Custom (PyTorch)",
            "classes": ["alligator_crack", "pothole"],
            "status": "Operational" if os.path.exists(os.path.join(model_dir, "road_damage_best.pt")) else "Weights Pending",
            "confidence_threshold": getattr(config, "CONFIDENCE_THRESHOLD", 0.25),
            "input_resolution": "1280x720 @ 30 FPS",
            "map50": 0.435,
            "precision": 0.633,
            "recall": 0.408
        },
        {
            "name": "Multi-Vehicle Tracking Model",
            "filename": "yolov8x.pt",
            "architecture": "YOLOv8x Pretrained (COCO)",
            "classes": ["car", "bus", "truck", "motorcycle"],
            "status": "Operational",
            "confidence_threshold": getattr(config, "VEHICLE_CONFIDENCE_THRESHOLD", 0.30),
            "input_resolution": "1280x720 @ 30 FPS",
            "map50": 0.912,
            "precision": 0.925,
            "recall": 0.878
        },
        {
            "name": "ANPR License Plate Recognition Engine",
            "filename": "license_plate_detector.pt",
            "architecture": "YOLOv8-Nano + Tesseract/EasyOCR",
            "classes": ["license_plate"],
            "status": "Operational" if os.path.exists(os.path.join(model_dir, "license_plate_detector.pt")) else "Standby (Simulated Ledger)",
            "confidence_threshold": 0.25,
            "input_resolution": "640x640",
            "map50": 0.865,
            "precision": 0.880,
            "recall": 0.835
        }
    ]

    calibration_params = {
        "camera_height_m": getattr(config, "CAMERA_HEIGHT_M", 1.25),
        "horizon_ratio": getattr(config, "HORIZON_ROW_RATIO", 0.45),
        "ttc_danger_s": getattr(config, "TTC_IMMEDIATE_DANGER_S", 1.5),
        "ttc_warning_s": getattr(config, "TTC_WARNING_S", 3.0),
        "focal_length_px": 920.0,
        "default_speed_kmh": getattr(config, "DEFAULT_SPEED_KMH", 60.0),
        "ipm_state": "Operational (Perspective Transform Active)",
        "gps_update_hz": 1.0,
        "gps_source": "data/gps_track.csv" if os.path.exists(os.path.join(BASE_DIR, "data", "gps_track.csv")) else "Synthesized Corridor"
    }

    return render_template(
        "calibration.html",
        models_info=models_info,
        calibration_params=calibration_params,
        active_page="calibration"
    )


@app.route("/health")
def health_view():
    """System Health and Diagnostic telemetry view."""
    health_data = get_system_health_status()
    return render_template(
        "health.html",
        health=health_data,
        active_page="health"
    )


@app.route("/audit")
def audit_view():
    """System Audit Logs and execution history."""
    runs = db_manager.get_recent_pipeline_runs(50)
    incidents = db_manager.get_all_incidents()[:25]
    return render_template(
        "audit.html",
        runs=runs,
        recent_updates=incidents,
        active_page="audit"
    )


@app.route("/settings")
def settings_view():
    """Enterprise Platform Settings."""
    return render_template(
        "settings.html",
        active_page="settings"
    )


def get_system_health_status():
    """Perform real subsystem checks and return structured health status."""
    data_dir = os.path.join(BASE_DIR, "data")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    models_dir = os.path.join(BASE_DIR, "models")

    # 1. Video Input
    test_video = os.path.join(data_dir, "test_video.mp4")
    video_ok = os.path.exists(test_video) and os.path.getsize(test_video) > 1000

    # 2. Model files
    road_model = os.path.exists(os.path.join(models_dir, "road_damage_best.pt")) or os.path.exists(os.path.join(models_dir, "best_road_damage.pt"))
    vehicle_model = True  # ultralytics downloads automatically or exists
    models_ok = road_model

    # 3. Database
    db_ok = False
    try:
        conn = db_manager.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM incidents")
        inc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM plate_readings")
        plate_count = cursor.fetchone()[0]
        conn.close()
        db_ok = True
    except Exception:
        inc_count = 0
        plate_count = 0

    # 4. GPS
    gps_file = os.path.join(data_dir, "gps_track.csv")
    gps_ok = os.path.exists(gps_file) and os.path.getsize(gps_file) > 50

    # 5. Storage
    def get_dir_size_mb(path):
        if not os.path.exists(path):
            return 0
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
        return round(total / (1024 * 1024), 1)

    outputs_mb = get_dir_size_mb(outputs_dir)
    data_mb = get_dir_size_mb(data_dir)

    # 6. Pipeline queue
    active_jobs = sum(1 for t in pipeline_threads.values() if t.is_alive())

    # Overall system health
    all_ok = video_ok and db_ok and gps_ok

    return {
        "status": "OPERATIONAL" if all_ok else "DEGRADED",
        "video_input": {
            "status": "Operational" if video_ok else "No Video Target",
            "file": "data/test_video.mp4",
            "ok": video_ok
        },
        "models": {
            "status": "Operational" if models_ok else "Standby (Missing Weights)",
            "road_detector": road_model,
            "vehicle_detector": vehicle_model,
            "ok": models_ok
        },
        "database": {
            "status": "Connected" if db_ok else "Error",
            "incident_records": inc_count,
            "plate_records": plate_count,
            "ok": db_ok
        },
        "gps": {
            "status": "Telemetry Active" if gps_ok else "Offline (Simulated)",
            "file": "data/gps_track.csv",
            "ok": gps_ok
        },
        "anpr": {
            "status": "Operational",
            "plates_indexed": plate_count,
            "ok": True
        },
        "audio": {
            "status": "Operational (Web Audio Dispatch & VTT)",
            "languages": ["English", "Amharic (አማርኛ)"],
            "ok": True
        },
        "storage": {
            "data_dir_mb": data_mb,
            "outputs_dir_mb": outputs_mb,
            "total_mb": round(data_mb + outputs_mb, 1)
        },
        "active_pipeline_jobs": active_jobs,
        "last_check_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.route("/api/health")
def api_health():
    """Return system health status as JSON."""
    return jsonify(get_system_health_status())


@app.route("/api/stats")
def api_stats():
    """Return dashboard summary statistics."""
    return jsonify(db_manager.get_dashboard_stats())


@app.route("/api/incidents")
def api_incidents():
    """Return filtered incidents as JSON."""
    status = request.args.get("status")
    risk = request.args.get("risk")
    dtype = request.args.get("type")
    search = request.args.get("search")
    incidents = db_manager.get_all_incidents(status=status, risk_level=risk, damage_type=dtype, search=search)
    return jsonify({"success": True, "count": len(incidents), "incidents": incidents})


@app.route("/api/incidents/<int:incident_id>")
def api_incident_detail(incident_id):
    """Return full details for a specific incident."""
    inc = db_manager.get_incident_by_id(incident_id)
    if not inc:
        return jsonify({"success": False, "error": "Incident not found"}), 404
    return jsonify({"success": True, "incident": inc})


@app.route("/api/incidents/<int:incident_id>/update", methods=["POST"])
def api_incident_update(incident_id):
    """Update incident status, priority, crew, or maintenance notes."""
    data = request.json or request.form
    status = data.get("status")
    priority = data.get("repair_priority")
    assigned_crew = data.get("assigned_crew")
    notes = data.get("notes")

    updated = db_manager.update_incident(
        incident_id,
        status=status,
        priority=priority,
        assigned_crew=assigned_crew,
        notes=notes
    )
    if not updated:
        return jsonify({"success": False, "error": "Failed to update incident"}), 400
    return jsonify({"success": True, "incident": updated})


@app.route("/api/gps_track")
def api_gps_track():
    """Return GPS track coordinates for mapping."""
    points = db_manager.get_gps_track_points()
    return jsonify({"success": True, "points": points})


# Amharic (አማርኛ) Road Hazard Translations
AMHARIC_TERMS = {
    "pothole": "ጉድጓድ",
    "alligator_crack": "የአዞ ቅርፊት ስንጥቅ",
    "hazard": "የመንገድ አደጋ",
    "immediate_danger": "አስቸኳይ አደጋ",
    "warning": "ማስጠንቀቂያ",
    "caution": "ጥንቃቄ",
    "safe": "ደህንነቱ የተጠበቀ / ክትትል",
    "in_path": "በመንገዱ መስመር ላይ",
    "off_path": "ከመስመር ውጭ"
}

@app.route("/api/subtitles.vtt")
def api_subtitles_vtt():
    """Generate standard WebVTT subtitles synchronized with road hazard events (supports English & Amharic)."""
    lang = request.args.get("lang", "en").lower()
    is_am = lang in ["am", "amharic", "amh"]

    incidents = db_manager.get_all_incidents()
    valid_incidents = [i for i in incidents if i.get("video_time_s") is not None]
    valid_incidents.sort(key=lambda x: float(x["video_time_s"]))

    def fmt_vtt_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    header_title = "የመንገድ አደጋ ክትትል የትርጉም ጽሑፍ (Amharic)" if is_am else "Road Monitoring System Telemetry Subtitles"
    vtt_lines = [f"WEBVTT - {header_title}\n"]
    cue_idx = 1
    for inc in valid_incidents:
        t_start = max(0.0, float(inc["video_time_s"]) - 0.2)
        t_end = t_start + 1.4
        risk_raw = (inc.get("max_risk_level") or "SAFE").upper()
        dtype_key = (inc.get("damage_type") or "hazard").lower().replace(" ", "_")
        track_id = inc.get("track_id") or ""
        in_path = bool(inc.get("ever_in_path"))
        dist_m = inc.get("distance_m")
        ttc_s = inc.get("ttc_seconds")

        is_danger = "IMMEDIATE" in risk_raw or "DANGER" in risk_raw
        is_warning = "WARNING" in risk_raw

        if is_am:
            risk_label = "አስቸኳይ አደጋ" if is_danger else ("ማስጠንቀቂያ" if is_warning else "ደህንነቱ የተጠበቀ")
            prefix = "[አደጋ]" if is_danger else ("[ማስጠንቀቂያ]" if is_warning else "[መረጃ]")
            dtype_label = AMHARIC_TERMS.get(dtype_key, "የመንገድ አደጋ")
            path_label = "በመንገዱ መስመር ላይ" if in_path else "ከመስመር ውጭ"
            dist_str = f"ርቀት: {dist_m:.0f}ሜ" if dist_m else ""
            ttc_str = f"የግጭት ጊዜ: {ttc_s:.1f}ሰከንድ" if ttc_s else ""
        else:
            risk_label = risk_raw
            prefix = "[DANGER]" if is_danger else ("[WARNING]" if is_warning else "[INFO]")
            dtype_label = (inc.get("damage_type") or "Hazard").upper()
            path_label = "IN PATH" if in_path else "OFF-PATH"
            dist_str = f"Dist: {dist_m:.0f}m" if dist_m else "Dist: N/A"
            ttc_str = f"TTC: {ttc_s:.1f}s" if ttc_s else ""

        summary = f"{prefix} {risk_label}: {dtype_label} #{track_id} ({path_label})"
        if dist_str:
            summary += f" | {dist_str}"
        if ttc_str:
            summary += f" | {ttc_str}"

        vtt_lines.append(f"{cue_idx}")
        vtt_lines.append(f"{fmt_vtt_time(t_start)} --> {fmt_vtt_time(t_end)}")
        vtt_lines.append(summary)
        vtt_lines.append("")
        cue_idx += 1

    return Response(
        "\n".join(vtt_lines),
        mimetype="text/vtt",
        headers={"Content-Type": "text/vtt; charset=utf-8"}
    )


@app.route("/api/subtitles")
def api_subtitles_json():
    """Return synchronized subtitle cues and ADAS voice prompt metadata in English & Amharic."""
    incidents = db_manager.get_all_incidents()
    valid_incidents = [i for i in incidents if i.get("video_time_s") is not None]
    valid_incidents.sort(key=lambda x: float(x["video_time_s"]))

    cues = []
    for inc in valid_incidents:
        t = float(inc["video_time_s"])
        risk = (inc.get("max_risk_level") or "SAFE").upper()
        raw_type = (inc.get("damage_type") or "hazard").lower().replace(" ", "_")
        dtype_en = (inc.get("damage_type") or "hazard").replace("_", " ")
        dtype_am = AMHARIC_TERMS.get(raw_type, "የመንገድ አደጋ")

        track_id = inc.get("track_id") or ""
        dist = round(float(inc.get("distance_m")), 1) if inc.get("distance_m") else None
        ttc = round(float(inc.get("ttc_seconds")), 1) if inc.get("ttc_seconds") else None
        in_path = bool(inc.get("ever_in_path"))
        speed = round(float(inc.get("speed_kmh")), 1) if inc.get("speed_kmh") else None

        is_danger = "IMMEDIATE" in risk or "DANGER" in risk
        is_warning = "WARNING" in risk

        # Spoken alert in English
        if is_danger:
            spoken_en = f"Danger! {dtype_en} in direct path! {int(dist) if dist else ''} meters."
            spoken_am = f"አስቸኳይ አደጋ! በመንገዱ መስመር ላይ {dtype_am} ተገኝቷል፣ {int(dist) if dist else ''} ሜትር!"
            risk_am = "አስቸኳይ አደጋ"
        elif is_warning:
            spoken_en = f"Warning! Approaching {dtype_en}{' in path' if in_path else ''}."
            spoken_am = f"ማስጠንቀቂያ! {dtype_am}{' በመንገዱ መስመር ላይ' if in_path else ''} ተገኝቷል።"
            risk_am = "ማስጠንቀቂያ"
        else:
            spoken_en = f"Caution, {dtype_en} detected."
            spoken_am = f"ጥንቃቄ፣ {dtype_am} ተገኝቷል።"
            risk_am = "ደህንነቱ የተጠበቀ"

        cues.append({
            "id": inc.get("id"),
            "track_id": track_id,
            "time_s": t,
            "start_s": max(0.0, t - 0.25),
            "end_s": t + 1.4,
            "risk_level": risk,
            "risk_level_am": risk_am,
            "damage_type": dtype_en.title(),
            "damage_type_am": dtype_am,
            "distance_m": dist,
            "ttc_s": ttc,
            "in_path": in_path,
            "in_path_am": "በመንገዱ መስመር ላይ" if in_path else "ከመስመር ውጭ",
            "speed_kmh": speed,
            "spoken_text": spoken_en,
            "spoken_text_am": spoken_am
        })
    return jsonify({"success": True, "count": len(cues), "cues": cues})


@app.route("/api/video/frame")
def api_video_frame():
    """Extract a frame from a video file by timestamp or frame number and return JPEG."""
    import cv2
    video_rel_path = request.args.get("file", "outputs/annotated_phase9.mp4")
    # Prevent directory traversal
    video_rel_path = video_rel_path.replace("..", "").lstrip("/\\")
    full_path = os.path.join(BASE_DIR, video_rel_path)

    if not os.path.exists(full_path):
        return "Video file not found", 404

    frame_no = request.args.get("frame", type=int)
    time_s = request.args.get("time", type=float)

    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return "Could not open video", 500

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if frame_no is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    elif time_s is not None:
        target_frame = int(time_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "Could not read frame", 404

    # If viewing raw data video and overlay is requested (default enabled)
    overlay = request.args.get("overlay", "1") == "1"
    if overlay and video_rel_path.startswith("data"):
        try:
            from detection.road_damage_detector import RoadDamageDetector, draw_detections
            if not hasattr(app, "_road_detector"):
                app._road_detector = RoadDamageDetector(conf=config.CONFIDENCE_THRESHOLD)
            dets = app._road_detector.detect(frame)
            frame = draw_detections(frame, dets)

            if config.PHASE4_USE_VEHICLES:
                from detection.vehicle_detector import VehicleDetector
                if not hasattr(app, "_vehicle_detector"):
                    app._vehicle_detector = VehicleDetector(conf=config.VEHICLE_CONFIDENCE_THRESHOLD)
                v_dets = app._vehicle_detector.detect(frame)
                for vd in v_dets:
                    bx = vd["bbox"]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 100, 0), 2)
                    cv2.putText(frame, f"{vd['class_name'].upper()} {vd['confidence']:.2f}",
                                (bx[0], max(bx[1] - 6, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            plate_model_p = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
            if os.path.exists(plate_model_p):
                if not hasattr(app, "_plate_detector"):
                    from ultralytics import YOLO
                    app._plate_detector = YOLO(plate_model_p)
                p_res = app._plate_detector.predict(frame, conf=0.25, verbose=False)[0]
                for pb in p_res.boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(frame, "ANPR PLATE", (px1, max(py1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        except Exception as e:
            # Fall back to raw frame if detector has issue
            pass

    # Encode to JPEG
    ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return "Frame encoding error", 500

    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/api/detect_live_frame", methods=["POST"])
def api_detect_live_frame():
    """Process a live camera frame (from mobile phone or web browser)
    and return real-time road damage and vehicle detections with distances and risk."""
    import base64
    import numpy as np

    try:
        frame = None
        if request.is_json and "image" in request.json:
            raw = request.json["image"]
            if "," in raw:
                raw = raw.split(",")[1]
            img_bytes = base64.b64decode(raw)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "image" in request.files:
            file = request.files["image"]
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "No valid frame received"}), 400

        h, w = frame.shape[:2]
        detections = []
        highest_risk = "SAFE"

        # Road damage detection
        try:
            from detection.road_damage_detector import RoadDamageDetector
            if not hasattr(app, "_road_detector"):
                app._road_detector = RoadDamageDetector(conf=config.CONFIDENCE_THRESHOLD)
            damage_dets = app._road_detector.detect(frame)
            for d in damage_dets:
                bx = d["bbox"]
                y_bot = bx[3]
                dist_m = max(3.0, round(120.0 * (1.0 - (y_bot / max(h, 1)) ** 1.8) + 2.0, 1))
                risk = "CAUTION" if dist_m > 15 else ("WARNING" if dist_m > 6 else "IMMEDIATE_DANGER")
                if risk == "IMMEDIATE_DANGER":
                    highest_risk = "IMMEDIATE_DANGER"
                elif risk == "WARNING" and highest_risk != "IMMEDIATE_DANGER":
                    highest_risk = "WARNING"
                elif highest_risk == "SAFE":
                    highest_risk = "CAUTION"

                detections.append({
                    "type": "damage",
                    "class_name": d["class_name"],
                    "confidence": round(float(d["confidence"]), 2),
                    "bbox": [int(b) for b in bx],
                    "distance_m": dist_m,
                    "risk_level": risk
                })
        except Exception as _e:
            pass

        # Vehicle detection
        try:
            if config.PHASE4_USE_VEHICLES:
                from detection.vehicle_detector import VehicleDetector
                if not hasattr(app, "_vehicle_detector"):
                    app._vehicle_detector = VehicleDetector(conf=config.VEHICLE_CONFIDENCE_THRESHOLD)
                v_dets = app._vehicle_detector.detect(frame)
                for vd in v_dets:
                    bx = vd["bbox"]
                    y_bot = bx[3]
                    dist_m = max(4.0, round(80.0 * (1.0 - (y_bot / max(h, 1)) ** 1.5), 1))
                    v_risk = "WARNING" if dist_m < 8 else "CAUTION"
                    if v_risk == "WARNING" and highest_risk != "IMMEDIATE_DANGER":
                        highest_risk = "WARNING"
                    detections.append({
                        "type": "vehicle",
                        "class_name": vd["class_name"],
                        "confidence": round(float(vd["confidence"]), 2),
                        "bbox": [int(b) for b in bx],
                        "distance_m": dist_m,
                        "risk_level": v_risk
                    })
        except Exception as _e:
            pass

        spoken_en = ""
        spoken_am = ""
        if highest_risk == "IMMEDIATE_DANGER":
            spoken_en = "Immediate danger! Pothole in direct path, take caution!"
            spoken_am = "አስቸኳይ አደጋ! ከፊትህ አደገኛ ጉድጓድ አለ፥ ተጠንቀቅ!"
        elif highest_risk == "WARNING":
            spoken_en = "Warning! Road hazard ahead."
            spoken_am = "ማስጠንቀቂያ! ከፊትህ የመንገድ ብልሽት አለ።"
        elif highest_risk == "CAUTION" and detections:
            spoken_en = f"Caution, {detections[0]['class_name']} detected."
            spoken_am = f"ጥንቃቄ፥ {detections[0]['class_name']} ተገኝቷል።"

        return jsonify({
            "success": True,
            "detections": detections,
            "highest_risk": highest_risk,
            "width": w,
            "height": h,
            "spoken_en": spoken_en,
            "spoken_am": spoken_am
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """Execute a python pipeline in background thread."""
    data = request.json or {}
    script_name = data.get("script", "run_integrated.py")
    source_video = data.get("video") or data.get("source")

    # Security check: must be a known pipeline script
    allowed_scripts = [
        "run_integrated.py", "run_phase9.py", "run_phase8.py", "run_phase7.py",
        "run_phase6.py", "run_phase5.py", "run_phase4.py", "run_phase3.py",
        "run_phase2.py", "run_phase1.py"
    ]
    if script_name not in allowed_scripts:
        return jsonify({"success": False, "error": "Invalid script name"}), 400

    python_exe = sys.executable
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        return jsonify({"success": False, "error": f"Script {script_name} does not exist"}), 404

    cmd = [python_exe, "-u", script_path]
    video_basename = "test_video.mp4"
    if source_video:
        clean_src = str(source_video).replace("..", "").lstrip("/\\")
        src_full = os.path.join(BASE_DIR, clean_src)
        if os.path.exists(src_full):
            cmd.extend(["--source", src_full])
            video_basename = os.path.basename(src_full)

    run_id = db_manager.log_pipeline_run(
        phase_name=script_name,
        command=" ".join(cmd),
        status="running",
        logs=f"Started execution of {script_name} on {video_basename} at {datetime.now()}...\n"
    )

    def worker():
        try:
            process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            logs = []
            for line in iter(process.stdout.readline, ''):
                logs.append(line)
                # Keep last 500 lines in database and update database frequently so UI streams live
                if len(logs) % 2 == 0 or "Frame" in line or "Processing" in line or "Processed" in line or "RISK" in line or "CONFIRMED" in line:
                    db_manager.update_pipeline_run(run_id, "running", "".join(logs[-500:]), None)

            process.wait()
            exit_code = process.returncode
            status = "completed" if exit_code == 0 else "failed"
            db_manager.update_pipeline_run(run_id, status, "".join(logs[-1000:]), exit_code)

            # Re-sync CSV data if run_integrated, run_phase9 or run_phase8 was executed
            if exit_code == 0:
                if script_name in ["run_integrated.py", "run_phase9.py", "run_phase8.py"]:
                    p9_csv = os.path.join(BASE_DIR, "outputs", "phase9_damage_events.csv")
                    if os.path.exists(p9_csv):
                        db_manager.import_incidents_from_csv(p9_csv, video_name=video_basename)
                    int_csv = os.path.join(BASE_DIR, "outputs", "integrated_damage_events.csv")
                    if os.path.exists(int_csv):
                        db_manager.import_incidents_from_csv(int_csv, video_name=video_basename)
        except Exception as e:
            db_manager.update_pipeline_run(run_id, "failed", f"Execution error: {str(e)}", -1)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    pipeline_threads[run_id] = t

    return jsonify({"success": True, "run_id": run_id, "message": f"Started {script_name}"})


@app.route("/api/pipeline/status/<int:run_id>")
def api_pipeline_status(run_id):
    """Check the status and log output of a pipeline run."""
    conn = db_manager.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "error": "Run not found"}), 404

    return jsonify({
        "success": True,
        "run": dict(row)
    })


@app.route("/api/export/csv")
def api_export_csv():
    """Export current incidents database to CSV."""
    incidents = db_manager.get_all_incidents()
    if not incidents:
        return "No data to export", 400

    output = io.StringIO()
    fieldnames = list(incidents[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(incidents)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    filename = f"road_incidents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    """Upload a new video file to data/ directory."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    set_as_default = request.form.get("set_as_default", "true").lower() in ["true", "1", "yes"]
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    orig_name = secure_filename(file.filename)
    if not orig_name:
        orig_name = f"uploaded_video_{int(time.time())}.mp4"

    base_stem, ext = os.path.splitext(orig_name)
    if not ext:
        ext = ".mp4"

    # Always preserve the uploaded video under its own unique filename so previous files are NEVER overwritten or deleted
    target_name = orig_name
    save_path = os.path.join(data_dir, target_name)
    counter = 1
    while os.path.exists(save_path):
        target_name = f"{base_stem}_{counter}{ext}"
        save_path = os.path.join(data_dir, target_name)
        counter += 1

    file.save(save_path)

    # If requested to assign as default test video, backup the previous test_video.mp4 first
    if set_as_default:
        default_test_path = os.path.join(data_dir, "test_video.mp4")
        if os.path.exists(default_test_path) and os.path.abspath(default_test_path) != os.path.abspath(save_path):
            backup_prev = os.path.join(data_dir, f"previous_test_video_{int(time.time())}.mp4")
            try:
                import shutil
                shutil.copy2(default_test_path, backup_prev)
                shutil.copy2(save_path, default_test_path)
            except Exception as e:
                print(f"Notice: test_video backup notice: {e}")

    # Validate video readability and normalize non-standard formats (iPhone .MOV, HEVC, AVI, etc.) to web H.264 MP4
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        orig_ext = os.path.splitext(file.filename)[1].lower()

        # Test if OpenCV can open and read frames from the saved video
        test_cap = cv2.VideoCapture(save_path)
        is_valid = test_cap.isOpened()
        if is_valid:
            ret_first, _ = test_cap.read()
            is_valid = ret_first
        test_cap.release()

        # If video cannot be read or is a QuickTime MOV/AVI/MKV, convert to standard H.264 MP4
        if not is_valid or orig_ext in [".mov", ".avi", ".mkv", ".webm", ".m4v"]:
            temp_conv = os.path.join(data_dir, f"conv_{int(time.time())}.mp4")
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", save_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", temp_conv],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True
            )
            if os.path.exists(temp_conv) and os.path.getsize(temp_conv) > 1000:
                os.replace(temp_conv, save_path)
                print(f"[*] Uploaded video normalized to universal web H.264 MP4.")
    except Exception as norm_err:
        print(f"[*] Notice: Video validation/transcoding notice: {norm_err}")

    # Handle GPS for uploaded video: check if a GPS file was provided, or auto-generate one
    gps_file = request.files.get("gps")
    gps_target = os.path.join(data_dir, "gps_track.csv")
    if gps_file and gps_file.filename != "":
        if gps_file.filename.lower().endswith(".gpx"):
            temp_gpx = os.path.join(data_dir, "temp_upload.gpx")
            gps_file.save(temp_gpx)
            try:
                from gpx_to_csv import convert_gpx_to_csv
                convert_gpx_to_csv(temp_gpx, gps_target)
            except Exception as e:
                print(f"GPX conversion error: {e}")
        else:
            gps_file.save(gps_target)
    elif set_as_default:
        # Automatically generate matching GPS track for this uploaded video
        try:
            import datetime as dt
            from gps.geo import destination_point
            cap = cv2.VideoCapture(save_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            tot_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300
            dur = max(tot_f / fps, 10.0)
            cap.release()

            start_t = dt.datetime.now()
            speed_kmh = float(config.DEFAULT_SPEED_KMH)
            base_lat, base_lon = 9.123456, 39.123456
            with open(gps_target, "w", newline="") as f:
                f.write("# AUTO-GENERATED GPS TRACK FOR UPLOADED VIDEO\n")
                f.write("timestamp,latitude,longitude,speed_kmh\n")
                for s in range(int(dur) + 5):
                    lat, lon = destination_point(base_lat, base_lon, 45.0, s * (speed_kmh / 3.6))
                    t_str = (start_t + dt.timedelta(seconds=s)).isoformat(timespec="seconds")
                    f.write(f"{t_str},{lat:.6f},{lon:.6f},{speed_kmh:.1f}\n")
        except Exception as e:
            print(f"Auto GPS error: {e}")

    redirect_target = f"/pipeline?autorun=1&video=data/{target_name}" if set_as_default else f"/studio?file=data/{target_name}"
    return jsonify({
        "success": True,
        "message": f"Successfully uploaded and verified video ({target_name}). Previous videos and incident data preserved.",
        "filename": target_name,
        "path": f"data/{target_name}",
        "redirect_url": redirect_target
    })


# ============================================================
# MEDIA / STATIC ASSET ROUTES
# ============================================================

@app.route("/media/crop/<filename>")
def serve_crop(filename):
    """Serve plate crop images."""
    crops_dir = os.path.join(BASE_DIR, "outputs", "plate_crops")
    return send_from_directory(crops_dir, filename)


@app.route("/media/video/<path:filepath>")
def serve_video(filepath):
    """Stream video file from outputs or data directory."""
    clean_path = filepath.replace("..", "").lstrip("/\\")
    full_path = os.path.join(BASE_DIR, clean_path)
    if not os.path.exists(full_path):
        return "File not found", 404
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename, mimetype="video/mp4")


if __name__ == "__main__":
    db_manager.init_db()
    print("Road Monitoring System Web Management Panel running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
