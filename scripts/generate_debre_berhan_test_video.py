r"""
Generate high-fidelity 40-second Debre Berhan test video and synchronized GPS telemetry.
Corridor: Debre Berhan University (DBU) Main Gate -> Kebele 08 -> Stadium / Ankober Junction.
Coordinates: ~ 9.6865° N, 39.5290° E to 9.6725° N, 39.5410° E (40 seconds duration).
"""
import os
import cv2
import csv
import math
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# 1. Generate 40-Second Synchronized Debre Berhan GPS Track CSV
def generate_debre_berhan_gps(output_csv_path, duration_seconds=40, fps=1.0):
    """
    Creates realistic 1Hz GPS track along Debre Berhan arterial highway.
    Start: DBU Main Gate (9.686500, 39.529000)
    Mid:   Kebele 08 Commercial Arterial (9.678200, 39.536200)
    End:   Stadium / Ankober Junction (9.672500, 39.541000)
    """
    start_lat, start_lon = 9.686500, 39.529000
    mid_lat, mid_lon = 9.678200, 39.536200
    end_lat, end_lon = 9.672500, 39.541000

    start_time = datetime(2026, 11, 16, 10, 15, 0)
    num_points = int(duration_seconds * fps)

    records = []
    for i in range(num_points):
        alpha = i / max(1, num_points - 1)

        # 2-segment smooth trajectory: DBU -> Kebele 08 -> Stadium
        if alpha < 0.60:
            sub_a = alpha / 0.60
            lat = start_lat + (mid_lat - start_lat) * sub_a
            lon = start_lon + (mid_lon - start_lon) * sub_a
        else:
            sub_a = (alpha - 0.60) / 0.40
            lat = mid_lat + (end_lat - mid_lat) * sub_a
            lon = mid_lon + (end_lon - mid_lon) * sub_a

        # Subtle road meander
        lat += 0.00007 * math.sin(alpha * math.pi * 4)
        lon += 0.00005 * math.cos(alpha * math.pi * 3)

        # Realistic velocity profile with acceleration, cruising, and deceleration near hazards
        if 0.20 <= alpha <= 0.32:  # Approaching Hazard 1 (Off-path)
            speed = 46.0 + 2.0 * math.sin(i * 0.5)
        elif 0.45 <= alpha <= 0.58:  # Caution Crack area
            speed = 36.5 + 2.5 * math.cos(i * 0.4)
        elif 0.70 <= alpha <= 0.88:  # Approaching Severe Cavity (Deceleration to 26 km/h)
            speed = 26.0 + 5.0 * math.sin((alpha - 0.70) / 0.18 * math.pi)
        else:
            speed = 42.0 + 3.0 * math.sin(i * 0.3)

        cur_time = start_time + timedelta(seconds=i / fps)
        records.append({
            "timestamp": cur_time.isoformat(),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed_kmh": round(speed, 1)
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("# RoadEye Official Debre Berhan 40-Second Arterial GPS Track\n")
        writer = csv.DictWriter(f, fieldnames=["timestamp", "latitude", "longitude", "speed_kmh"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Generated 40-Second Debre Berhan GPS Track: {output_csv_path} ({len(records)} points)")
    return records


# 2. Generate 40-Second Driving Video with Multi-Hazard Encounters & HUD
def generate_debre_berhan_video(gps_records, output_video_path, duration_seconds=40, fps=25.0):
    """
    Renders realistic 40-second driving video with 3 distinct road distress events.
    """
    width, height = 1280, 720
    total_frames = int(duration_seconds * fps)

    source_video = os.path.join(DATA_DIR, "ethiopian_road_driving_test.mp4")
    if not os.path.exists(source_video):
        source_video = os.path.join(DATA_DIR, "test_video.mp4")

    cap = None
    if os.path.exists(source_video):
        cap = cv2.VideoCapture(source_video)
        print(f"[INFO] Sourcing video texture from: {source_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    horizon_y = int(height * 0.48)
    print(f"Rendering {total_frames} frames (40.0s) of Debre Berhan corridor footage...")

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        gps_idx = min(int(t_sec), len(gps_records) - 1)
        gps = gps_records[gps_idx]
        lat = gps["latitude"]
        lon = gps["longitude"]
        speed = gps["speed_kmh"]

        # Read base frame
        frame = None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

        if frame is None or frame.shape[0] != height or frame.shape[1] != width:
            if frame is not None:
                frame = cv2.resize(frame, (width, height))
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(horizon_y):
                    ratio = y / horizon_y
                    frame[y, :] = (int(205 - 40 * ratio), int(155 - 30 * ratio), int(95 - 20 * ratio))
                for y in range(horizon_y, height):
                    gray_val = int(50 + 22 * (y - horizon_y) / (height - horizon_y))
                    frame[y, :] = (gray_val, gray_val, gray_val)

        # -------------------------------------------------------------
        # HAZARD ENCOUNTER 1: Off-Path Warning Pothole (t = 8s - 14s, frames 200 - 350)
        # -------------------------------------------------------------
        if 200 <= f_idx <= 350:
            prog = (f_idx - 200) / 150.0
            p_y = int(horizon_y + (height - horizon_y - 80) * (prog ** 1.9))
            p_w = int(20 + 110 * (prog ** 1.5))
            p_h = int(10 + 50 * (prog ** 1.5))
            # Offset to the right shoulder (off-path)
            p_x = int(width * 0.68 + 80 * (prog ** 1.3))
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 15, 0, 360, (22, 25, 30), -1)
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 15, 0, 360, (55, 60, 75), 2)
            cv2.ellipse(frame, (p_x - 10, p_y + 5), (int(p_w * 0.5), int(p_h * 0.4)), 15, 0, 360, (12, 14, 18), -1)

        # -------------------------------------------------------------
        # HAZARD ENCOUNTER 2: Surface Alligator Crack (t = 19s - 25s, frames 475 - 625)
        # -------------------------------------------------------------
        if 475 <= f_idx <= 625:
            prog = (f_idx - 475) / 150.0
            c_y = int(horizon_y + (height - horizon_y - 100) * (prog ** 1.8))
            c_w = int(40 + 220 * (prog ** 1.5))
            c_x = int(width * 0.45)
            # Draw realistic crack web
            pts = np.array([
                [c_x - c_w // 2, c_y],
                [c_x - c_w // 4, c_y + 12],
                [c_x, c_y - 8],
                [c_x + c_w // 4, c_y + 15],
                [c_x + c_w // 2, c_y + 4]
            ], np.int32)
            cv2.polylines(frame, [pts], False, (18, 20, 26), max(2, int(4 * prog)))
            cv2.line(frame, (c_x - 15, c_y), (c_x - 5, c_y + 25), (20, 22, 28), max(1, int(3 * prog)))
            cv2.line(frame, (c_x + 30, c_y + 5), (c_x + 50, c_y + 35), (20, 22, 28), max(1, int(3 * prog)))

        # -------------------------------------------------------------
        # HAZARD ENCOUNTER 3: Severe In-Path Deep Cavity (t = 28s - 36s, frames 700 - 900)
        # -------------------------------------------------------------
        if 700 <= f_idx <= 900:
            prog = (f_idx - 700) / 200.0
            p_y = int(horizon_y + (height - horizon_y - 50) * (prog ** 1.8))
            p_w = int(30 + 160 * (prog ** 1.6))
            p_h = int(15 + 75 * (prog ** 1.6))
            # Directly in the vehicle tire trajectory (in-path)
            p_x = int(width * 0.49 + 25 * (prog ** 1.2))
            
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 0, 0, 360, (18, 20, 25), -1)
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 0, 0, 360, (50, 55, 68), 2)
            # Deep inner shadow cavity
            cv2.ellipse(frame, (p_x - int(p_w * 0.2), p_y + int(p_h * 0.15)), 
                        (int(p_w * 0.65), int(p_h * 0.55)), 0, 0, 360, (8, 10, 14), -1)

        # -------------------------------------------------------------
        # Top Modern Telemetry HUD Banner
        # -------------------------------------------------------------
        hud_h = 58
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), (11, 15, 25), -1)
        cv2.line(overlay, (0, hud_h), (width, hud_h), (56, 189, 248), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Brand & Route Info
        cv2.putText(frame, "RoadEye(TM) DEBRE BERHAN CORRIDOR (40s)", (20, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        loc_str = f"DBU Gate -> Kebele 08 -> Stadium | GPS: {lat:.6f} N, {lon:.6f} E"
        cv2.putText(frame, loc_str, (20, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 215, 230), 1)

        # Right Telemetry HUD
        spd_str = f"SPEED: {speed:.1f} km/h"
        time_str = f"T: {t_sec:04.1f}s / 40.0s | FR: {f_idx:04d}"
        cv2.putText(frame, spd_str, (width - 270, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2)
        cv2.putText(frame, time_str, (width - 270, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1)

        out.write(frame)

    if cap:
        cap.release()
    out.release()
    print(f"[OK] 40-Second Debre Berhan Video Successfully Generated: {output_video_path}")


def main():
    gps_path = os.path.join(DATA_DIR, "debre_berhan_gps_track.csv")
    main_gps_path = os.path.join(DATA_DIR, "gps_track.csv")
    video_path = os.path.join(DATA_DIR, "debre_berhan_test_video.mp4")

    print("==================================================")
    print(" GENERATING 40-SECOND DEBRE BERHAN VIDEO & GPS    ")
    print("==================================================")
    
    # 1. Generate 40-Second GPS track
    gps_records = generate_debre_berhan_gps(gps_path, duration_seconds=40)
    generate_debre_berhan_gps(main_gps_path, duration_seconds=40)

    # 2. Generate 40-Second Video (1000 frames @ 25 FPS)
    generate_debre_berhan_video(gps_records, video_path, duration_seconds=40, fps=25.0)

    print("\n[COMPLETE] 40-Second Debre Berhan video and GPS track are 100% ready!")


if __name__ == "__main__":
    main()
