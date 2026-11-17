r"""
Generate high-fidelity Debre Berhan test video and synchronized GPS telemetry.
Location: Debre Berhan University (DBU) Main Gate -> Kebele 08 Commercial Arterial Highway.
Coordinates: ~ 9.6865° N, 39.5290° E to 9.6782° N, 39.5362° E.
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

# 1. Generate Synchronized Debre Berhan GPS Track CSV
def generate_debre_berhan_gps(output_csv_path, duration_seconds=30, fps=1.0):
    """
    Creates realistic 1Hz GPS track along Debre Berhan arterial corridor.
    Start: DBU Main Gate (9.686500, 39.529000)
    End:   Kebele 08 Commercial Center (9.678200, 39.536200)
    """
    start_lat, start_lon = 9.686500, 39.529000
    end_lat, end_lon = 9.678200, 39.536200
    
    start_time = datetime(2026, 11, 16, 10, 15, 0)
    num_points = int(duration_seconds * fps)
    
    records = []
    for i in range(num_points):
        alpha = i / max(1, num_points - 1)
        # S-curve smooth driving interpolation
        t_smooth = 3 * (alpha ** 2) - 2 * (alpha ** 3)
        lat = start_lat + (end_lat - start_lat) * t_smooth
        lon = start_lon + (end_lon - start_lon) * t_smooth
        
        # Add slight natural road curvature
        lat += 0.00008 * math.sin(alpha * math.pi * 3)
        lon += 0.00006 * math.cos(alpha * math.pi * 2)
        
        # Realistic urban speed: 38 - 48 km/h, slight deceleration around pothole at alpha ~ 0.6
        if 0.55 <= alpha <= 0.70:
            speed = 28.5 + 4.0 * math.sin((alpha - 0.55) / 0.15 * math.pi)
        else:
            speed = 42.0 + 3.5 * math.sin(i * 0.4)
            
        cur_time = start_time + timedelta(seconds=i / fps)
        time_str = cur_time.isoformat()
        records.append({
            "timestamp": time_str,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed_kmh": round(speed, 1)
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("# RoadEye Official Debre Berhan Arterial GPS Track\n")
        writer = csv.DictWriter(f, fieldnames=["timestamp", "latitude", "longitude", "speed_kmh"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Generated Debre Berhan GPS Track: {output_csv_path} ({len(records)} points)")
    return records


# 2. Generate Video with Road Scene & Debre Berhan Telemetry HUD
def generate_debre_berhan_video(gps_records, output_video_path, duration_seconds=20, fps=25.0):
    """
    Renders realistic driving video with authentic Debre Berhan coordinates and road defects.
    """
    width, height = 1280, 720
    total_frames = int(duration_seconds * fps)
    
    # Try using an existing test video as road texture base if available
    source_video = os.path.join(DATA_DIR, "ethiopian_road_driving_test.mp4")
    if not os.path.exists(source_video):
        source_video = os.path.join(DATA_DIR, "test_video.mp4")
    
    cap = None
    if os.path.exists(source_video):
        cap = cv2.VideoCapture(source_video)
        print(f"[INFO] Using base video frames from: {source_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Horizon line
    horizon_y = int(height * 0.48)
    
    print(f"Rendering {total_frames} frames of Debre Berhan driving video...")
    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        gps_idx = min(int(t_sec), len(gps_records) - 1)
        gps = gps_records[gps_idx]
        lat = gps["latitude"]
        lon = gps["longitude"]
        speed = gps["speed_kmh"]

        # Read base frame or generate synthetic asphalt road
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
                # Procedural asphalt road
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # Sky gradient
                for y in range(horizon_y):
                    ratio = y / horizon_y
                    frame[y, :] = (int(200 - 40 * ratio), int(150 - 30 * ratio), int(90 - 20 * ratio))
                # Road texture
                for y in range(horizon_y, height):
                    gray_val = int(50 + 20 * (y - horizon_y) / (height - horizon_y))
                    frame[y, :] = (gray_val, gray_val, gray_val)

        # Draw realistic Pothole Defect in frame (Hazard approaching)
        # Hazard 1: Severe In-Path Cavity (frames 150 to 320)
        if 120 <= f_idx <= 340:
            p_prog = (f_idx - 120) / 220.0  # 0.0 (far) to 1.0 (near)
            # Perspective scaling
            p_y = int(horizon_y + (height - horizon_y - 60) * (p_prog ** 1.8))
            p_w = int(25 + 140 * (p_prog ** 1.6))
            p_h = int(12 + 65 * (p_prog ** 1.6))
            p_x = int(width * 0.48 + 30 * (p_prog ** 1.2))
            
            # Draw realistic dark asphalt pothole
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 0, 0, 360, (20, 22, 28), -1)
            cv2.ellipse(frame, (p_x, p_y), (p_w, p_h), 0, 0, 360, (45, 50, 60), 2)
            # Inner deep shadow
            cv2.ellipse(frame, (p_x - int(p_w * 0.15), p_y + int(p_h * 0.1)), 
                        (int(p_w * 0.6), int(p_h * 0.5)), 0, 0, 360, (10, 12, 16), -1)

        # Draw Top HUD Banner
        hud_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), (11, 15, 25), -1)
        cv2.line(overlay, (0, hud_h), (width, hud_h), (56, 189, 248), 1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Left HUD: Brand & Location
        cv2.putText(frame, "RoadEye(TM) DEBRE BERHAN CORRIDOR", (20, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        loc_str = f"DBU Main Gate -> Kebele 08 Arterial | GPS: {lat:.6f} N, {lon:.6f} E"
        cv2.putText(frame, loc_str, (20, 47), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 225), 1)

        # Right HUD: Telemetry (Speed, Frame, Time)
        spd_str = f"SPEED: {speed:.1f} km/h"
        time_str = f"T: {t_sec:04.1f}s | FRAME: {f_idx:04d}"
        cv2.putText(frame, spd_str, (width - 240, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2)
        cv2.putText(frame, time_str, (width - 240, 47), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

        out.write(frame)

    if cap:
        cap.release()
    out.release()
    print(f"[OK] Debre Berhan Video Successfully Generated: {output_video_path}")


def main():
    gps_path = os.path.join(DATA_DIR, "debre_berhan_gps_track.csv")
    main_gps_path = os.path.join(DATA_DIR, "gps_track.csv")
    video_path = os.path.join(DATA_DIR, "debre_berhan_test_video.mp4")

    print("==================================================")
    print(" GENERATING DEBRE BERHAN TEST VIDEO & GPS TRACK   ")
    print("==================================================")
    
    # 1. Generate Debre Berhan GPS
    gps_records = generate_debre_berhan_gps(gps_path, duration_seconds=20)
    
    # Also update the primary data/gps_track.csv so the whole app syncs with Debre Berhan!
    generate_debre_berhan_gps(main_gps_path, duration_seconds=20)

    # 2. Generate Debre Berhan Video
    generate_debre_berhan_video(gps_records, video_path, duration_seconds=20, fps=25.0)

    print("\n[COMPLETE] Debre Berhan test video and GPS track are ready for testing and demo!")

if __name__ == "__main__":
    main()
