r"""
Converts a GPX file (exported by a phone GPS app) to the CSV format used by this project.

  python gpx_to_csv.py my_track.gpx
  python gpx_to_csv.py my_track.gpx --tz-offset-hours 3 --out data\gps_track.csv

IMPORTANT - time zones: GPX files normally store UTC time. Ethiopia is UTC+3.
If you note the video start time in LOCAL time, convert with --tz-offset-hours 3.
Speed: used from the file if it has one (metres/second), otherwise computed from the positions.
"""
import argparse
import datetime as dt
import xml.etree.ElementTree as ET

import config
from gps.geo import haversine_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpx")
    ap.add_argument("--out", default=config.GPS_PATH)
    ap.add_argument("--tz-offset-hours", type=float, default=0.0)
    a = ap.parse_args()

    rows = []
    for p in ET.parse(a.gpx).getroot().iter():
        if not p.tag.endswith("trkpt"):
            continue
        time_text, speed_ms = None, None
        for c in p.iter():
            if c.tag.endswith("}time") or c.tag == "time":
                time_text = c.text
            elif c.tag.lower().endswith("speed") and c.text:
                try:
                    speed_ms = float(c.text)
                except ValueError:
                    pass
        if not time_text:
            continue
        t = dt.datetime.fromisoformat(time_text.strip().replace("Z", "+00:00"))
        if t.tzinfo is not None:
            t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
        t += dt.timedelta(hours=a.tz_offset_hours)
        rows.append([t, float(p.get("lat")), float(p.get("lon")),
                     None if speed_ms is None else speed_ms * 3.6])

    if len(rows) < 2:
        raise SystemExit("No usable track points (with a time) found in the GPX file.")
    rows.sort(key=lambda r: r[0])

    if any(r[3] is None for r in rows):                  # compute speed from the positions
        speeds = [None]
        for prev, cur in zip(rows, rows[1:]):
            dt_s = (cur[0] - prev[0]).total_seconds()
            speeds.append(haversine_m(prev[1], prev[2], cur[1], cur[2]) / dt_s * 3.6
                          if dt_s > 0 else None)
        speeds[0] = speeds[1]
        for i, r in enumerate(rows):                     # 3-point average to reduce jitter
            window = [s for s in speeds[max(0, i - 1):i + 2] if s is not None]
            r[3] = sum(window) / len(window) if window else None
        print("Speed was computed from the positions (the GPX file had no speed).")

    with open(a.out, "w", newline="") as f:
        f.write("timestamp,latitude,longitude,speed_kmh\n")
        for t, lat, lon, spd in rows:
            f.write(f"{t.isoformat(timespec='seconds')},{lat:.6f},{lon:.6f},"
                    f"{'' if spd is None else f'{spd:.1f}'}\n")
    print(f"Saved {a.out}: {len(rows)} points, {rows[0][0]} to {rows[-1][0]}")


if __name__ == "__main__":
    main()
