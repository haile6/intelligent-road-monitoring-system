"""
Phase 6: reads the GPS CSV file.

Required columns: timestamp, latitude, longitude
Optional column:  speed_kmh   (if missing, speed is unknown - Phase 7 deals with that)
Lines starting with # are ignored (comments).
"""
import csv
import datetime as dt


def parse_time(text):
    """Reads '2026-10-08T10:30:00' (also with Z or +03:00). Time zones are converted to UTC;
    text without a time zone is used as it is."""
    t = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t


def load_gps_csv(path):
    """Returns a list of dictionaries sorted by time:
         {"time": datetime, "latitude": float, "longitude": float, "speed_kmh": float or None}"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise SystemExit(f"GPS file is empty: {path}")

    reader = csv.DictReader(lines)
    reader.fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
    missing = [c for c in ("timestamp", "latitude", "longitude") if c not in reader.fieldnames]
    if missing:
        raise SystemExit(f"GPS file is missing the column(s): {', '.join(missing)}\n"
                         f"Found columns: {reader.fieldnames}\n"
                         "Expected: timestamp,latitude,longitude,speed_kmh")

    records, bad = [], []
    for n, row in enumerate(reader, start=2):
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("latitude/longitude out of range")
            speed_text = (row.get("speed_kmh") or "").strip()
            speed = float(speed_text) if speed_text else None
            if speed is not None and speed < 0:
                raise ValueError("negative speed")
            records.append({"time": parse_time(row["timestamp"]), "latitude": lat,
                            "longitude": lon, "speed_kmh": speed})
        except (ValueError, TypeError, KeyError) as e:
            bad.append(f"line {n}: {e}")

    if bad:
        print(f"WARNING: {len(bad)} bad GPS row(s) skipped, for example: {bad[:3]}")

    records.sort(key=lambda r: r["time"])
    unique = []
    for r in records:                       # remove repeated timestamps
        if not unique or r["time"] != unique[-1]["time"]:
            unique.append(r)
    if len(unique) < 2:
        raise SystemExit("The GPS file needs at least 2 valid rows.")
    return unique


def describe(records):
    """One-line summary printed at program start."""
    t0, t1 = records[0]["time"], records[-1]["time"]
    speeds = [r["speed_kmh"] for r in records if r["speed_kmh"] is not None]
    speed_text = (f"speed {min(speeds):.0f}-{max(speeds):.0f} km/h" if speeds
                  else "NO speed column (speed unknown)")
    return (f"{len(records)} GPS records, {t0.isoformat(' ', 'seconds')} to "
            f"{t1.isoformat(' ', 'seconds')} ({(t1 - t0).total_seconds():.0f} s), {speed_text}")
