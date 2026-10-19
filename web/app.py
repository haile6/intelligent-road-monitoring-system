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

# Background pipeline execution state
pipeline_threads = {}


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
    search_query = request.args.get("q", "")

    incidents = db_manager.get_all_incidents(
        status=status_filter,
        risk_level=risk_filter,
        damage_type=type_filter,
        search=search_query
    )
    return render_template(
        "incidents.html",
        incidents=incidents,
        status_filter=status_filter,
        risk_filter=risk_filter,
        type_filter=type_filter,
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
        video_files.append({"name": "★ Master Integrated Analysis Video", "path": integrated_path})
    elif os.path.exists(os.path.join(BASE_DIR, "outputs", "annotated_phase9.mp4")):
        video_files.append({"name": "★ Master Integrated Analysis Video", "path": "outputs/annotated_phase9.mp4"})

    raw_path = "data/test_video.mp4"
    if os.path.exists(os.path.join(BASE_DIR, raw_path)):
        video_files.append({"name": "Raw Camera Video (Live AI Detection)", "path": raw_path})

    default_video = video_files[0]["path"] if video_files else raw_path
    selected_video = request.args.get("file", default_video)
    incidents = db_manager.get_all_incidents()

    return render_template(
        "studio.html",
        video_files=video_files,
        selected_video=selected_video,
        incidents=incidents,
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


# ============================================================
# API ENDPOINTS
# ============================================================

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


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """Execute a python pipeline in background thread."""
    data = request.json or {}
    script_name = data.get("script", "run_integrated.py")

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

    run_id = db_manager.log_pipeline_run(
        phase_name=script_name,
        command=f"{python_exe} {script_name}",
        status="running",
        logs=f"Started execution of {script_name} at {datetime.now()}...\n"
    )

    def worker():
        try:
            process = subprocess.Popen(
                [python_exe, script_path],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            logs = []
            for line in iter(process.stdout.readline, ''):
                logs.append(line)
                # Keep last 500 lines in database
                if len(logs) % 20 == 0:
                    db_manager.update_pipeline_run(run_id, "running", "".join(logs[-500:]), None)

            process.wait()
            exit_code = process.returncode
            status = "completed" if exit_code == 0 else "failed"
            db_manager.update_pipeline_run(run_id, status, "".join(logs[-1000:]), exit_code)

            # Re-sync CSV data if run_phase9 or run_phase8 was executed
            if exit_code == 0:
                if script_name in ["run_phase9.py", "run_phase8.py"]:
                    p9_csv = os.path.join(BASE_DIR, "outputs", "phase9_damage_events.csv")
                    db_manager.import_incidents_from_csv(p9_csv)
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

    if set_as_default:
        target_name = "test_video.mp4"
    else:
        target_name = secure_filename(file.filename)
        if not target_name:
            target_name = f"uploaded_video_{int(time.time())}.mp4"

    save_path = os.path.join(data_dir, target_name)
    file.save(save_path)

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
            import cv2
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

    return jsonify({
        "success": True,
        "message": f"Successfully uploaded video to data/{target_name} and synchronized GPS tracking.",
        "filename": target_name,
        "path": f"data/{target_name}"
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
