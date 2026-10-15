# Road Monitoring System & Web Management Panel

A comprehensive AI-powered road surveillance and pavement hazard management platform featuring real-time detection, tracking, distance estimation, GPS synchronization, collision risk analysis, and automatic number plate recognition (ANPR).

---

## 🚀 Quick Start: Web Management Panel

Launch the web-based management portal:

```bash
# In the project directory:
python run_web.py
```

Then open your browser to:
👉 **`http://127.0.0.1:5000`**

### Features in the Web Management Panel:
1. **Executive Dashboard (`/`)**: KPI metrics (Total Hazards, Immediate Danger, Warning, Monitored Speed), interactive Chart.js breakdown of risk severity and hazard types.
2. **Incidents & Repair Workflow (`/incidents`)**: Filter and sort road damage records, update workflow status (`Reported`, `In Review`, `Scheduled for Repair`, `In Progress`, `Repaired`, `Dismissed`), assign repair crews, and add asphalt maintenance notes.
3. **GIS Road Map (`/map`)**: Interactive OpenStreetMap (Leaflet.js) rendering the vehicle's GPS track corridor with color-coded hazard pins (Red for Immediate Danger, Orange for Warning, Green for Safe/Monitor).
4. **Video Telemetry Studio (`/studio`)**: Inspect annotated video footage with an interactive timeline and precision frame scrubber. Click any hazard keyframe to jump directly to the detection.
5. **Vehicles & License Plate OCR (`/plates`)**: Automatic Number Plate Recognition (ANPR) gallery with cropped plate photos and OCR confidence scores.
6. **AI Pipeline Runner (`/pipeline`)**: Trigger any detection phase (Phase 1 through Phase 9) directly from the browser with real-time streaming console logs.
7. **Maintenance Work Orders & Reports (`/reports`)**: Printable work-order sheets ready for field repair crews, plus one-click CSV dataset export.

---

## 🛠️ Command-Line Pipeline Execution

You can also run any individual phase via CLI:

- **Phase 1 (YOLO Road Damage Detection)**:
  ```bash
  python run_phase1.py
  ```
- **Phase 2 (Damage Tracking)**:
  ```bash
  python run_phase2.py
  ```
- **Phase 3 (Multi-Vehicle Detection & Tracking)**:
  ```bash
  python run_phase3.py
  ```
- **Phase 4 (Vehicle Path Corridor Analysis)**:
  ```bash
  python run_phase4.py
  ```
- **Phase 5 (Perspective Distance Estimation)**:
  ```bash
  python run_phase5.py
  ```
- **Phase 6 (GPS Synchronization & Geotagging)**:
  ```bash
  python run_phase6.py
  ```
- **Phase 7 (GPS Speed Provider)**:
  ```bash
  python run_phase7.py
  ```
- **Phase 8 (Time-To-Collision / TTC)**:
  ```bash
  python run_phase8.py
  ```
- **Phase 9 (Integrated Risk Analysis & Collision Avoidance)**:
  ```bash
  python run_phase9.py
  ```

---

## 📁 Project Structure

```text
road_monitoring_system/
├── run_web.py               # Main launcher for Web Management Panel
├── config.py                # Global parameters (FOV, tilt, risk thresholds)
├── database/
│   ├── db_manager.py        # SQLite persistence and auto-synchronization
│   └── road_monitoring.db  # SQLite database
├── web/
│   ├── app.py               # Flask backend & REST APIs
│   ├── templates/           # Modern responsive HTML5 templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── incidents.html
│   │   ├── map.html
│   │   ├── studio.html
│   │   ├── plates.html
│   │   ├── pipeline.html
│   │   └── reports.html
│   └── static/
│       └── css/custom.css   # Modern dashboard styles
├── detection/               # YOLO detectors & multi-object trackers
├── gps/                     # GPS synchronization & speed calculation
├── risk/                    # Distance estimation, TTC, and risk classification
├── data/                    # Input video (test_video.mp4) & GPS CSV
└── outputs/                 # Annotated videos, CSV logs & plate crops
```
