r"""
Makes a SIMULATED GPS file so you can test Phase 6 without a real GPS recording.
The positions are INVENTED (a car driving in a straight line). Do not use it as real data.

Examples:
  python make_sample_gps.py --video data\test_video.mp4
  python make_sample_gps.py --seconds 120 --speed 40 --start "2026-10-08T10:30:00"
"""
import argparse
import datetime as dt
import os

import config
from gps.geo import destination_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="read the duration from this video")
    ap.add_argument("--seconds", type=float, help="duration if no video is given")
    ap.add_argument("--start", default="2026-10-08T10:30:00")
    ap.add_argument("--speed", type=float, default=35.0, help="km/h")
    ap.add_argument("--heading", type=float, default=45.0, help="degrees, 0 = north, 90 = east")
    ap.add_argument("--lat", type=float, default=9.123456)
    ap.add_argument("--lon", type=float, default=39.123456)
    ap.add_argument("--out", default=config.GPS_PATH)
    a = ap.parse_args()

    if a.video:
        import cv2
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
    elif a.seconds:
        duration = a.seconds
    else:
        raise SystemExit("Give --video or --seconds")

    start = dt.datetime.fromisoformat(a.start)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write("# SIMULATED GPS TRACK - invented positions, not real data\n")
        f.write("timestamp,latitude,longitude,speed_kmh\n")
        for s in range(int(duration) + 3):
            lat, lon = destination_point(a.lat, a.lon, a.heading, s * a.speed / 3.6)
            t = (start + dt.timedelta(seconds=s)).isoformat(timespec="seconds")
            f.write(f"{t},{lat:.6f},{lon:.6f},{a.speed:.1f}\n")
    print(f"Saved {a.out}  ({duration:.0f} s, {a.speed} km/h, SIMULATED)")


if __name__ == "__main__":
    main()
