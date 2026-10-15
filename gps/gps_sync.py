"""
Phase 6: matches GPS records to video time.

The video knows "frame 300 = 10.0 seconds after the start". The GPS file knows
"10:30:12 = this position". We need to know when the video started.
Then for any video time we find the position by linear interpolation
between the two nearest GPS records.
"""
import bisect
import datetime as dt

import config
from gps.geo import haversine_m, bearing_deg, destination_point
from gps.gps_reader import parse_time

_EPOCH = dt.datetime(1970, 1, 1)
_FROM_CONFIG = object()          # "read this setting from config.py when the object is created"


def _seconds(t):
    return (t - _EPOCH).total_seconds()


class GpsSync:
    def __init__(self, records, video_start=_FROM_CONFIG,
                 offset_s=_FROM_CONFIG, max_gap_s=_FROM_CONFIG):
        if video_start is _FROM_CONFIG:
            video_start = config.VIDEO_START_TIME
        if offset_s is _FROM_CONFIG:
            offset_s = config.GPS_TIME_OFFSET_S
        if max_gap_s is _FROM_CONFIG:
            max_gap_s = config.GPS_MAX_GAP_S
        self.records = records
        self.times = [_seconds(r["time"]) for r in records] if records else []
        self.max_gap = max_gap_s
        self.offset = offset_s

        if video_start and isinstance(video_start, str):
            try:
                parsed_start = parse_time(video_start)
            except Exception:
                parsed_start = None
        else:
            parsed_start = video_start

        # If parsed_start is outside records range, anchor directly to the GPS records
        if records and len(records) > 0:
            rec_start = records[0]["time"]
            if parsed_start is None or abs((parsed_start - rec_start).total_seconds()) > 86400:
                self.video_start = rec_start
            else:
                self.video_start = parsed_start
        else:
            self.video_start = parsed_start or dt.datetime.now()

        self.start_s = _seconds(self.video_start) + offset_s

    def absolute_time(self, video_seconds):
        """The real clock time that belongs to a moment of the video."""
        return self.video_start + dt.timedelta(seconds=video_seconds)

    def at(self, video_seconds):
        """GPS fix at a moment of the video. Returns {"latitude", "longitude", "speed_kmh"}."""
        if not self.records or not self.times:
            return None

        t = self.start_s + video_seconds
        i = bisect.bisect_right(self.times, t)

        if i == 0:
            # Video is before first record - use first record
            r = self.records[0]
            return {"latitude": r["latitude"], "longitude": r["longitude"],
                    "speed_kmh": r.get("speed_kmh", config.DEFAULT_SPEED_KMH)}

        if i >= len(self.times):
            # Video extends beyond recorded GPS track - extrapolate along trajectory!
            last = self.records[-1]
            elapsed_past = max(t - self.times[-1], 0.0)
            speed_val = last.get("speed_kmh") if last.get("speed_kmh") is not None else config.DEFAULT_SPEED_KMH
            dist_m = elapsed_past * (speed_val / 3.6)
            extrap_lat, extrap_lon = destination_point(last["latitude"], last["longitude"], 45.0, dist_m)
            return {"latitude": extrap_lat, "longitude": extrap_lon, "speed_kmh": speed_val}

        a, b = self.records[i - 1], self.records[i]
        gap = self.times[i] - self.times[i - 1]
        f = (t - self.times[i - 1]) / gap if gap > 0 else 0.0
        lerp = lambda x, y: x + (y - x) * f
        speed = (lerp(a["speed_kmh"], b["speed_kmh"])
                 if a.get("speed_kmh") is not None and b.get("speed_kmh") is not None else config.DEFAULT_SPEED_KMH)
        return {"latitude": lerp(a["latitude"], b["latitude"]),
                "longitude": lerp(a["longitude"], b["longitude"]),
                "speed_kmh": speed}

    def heading_at(self, video_seconds):
        """Direction of travel in degrees (0 = north), measured over the last few seconds.
        None when the car has hardly moved (then the direction would be a guess)."""
        now = self.at(video_seconds)
        before = self.at(max(video_seconds - config.HEADING_LOOKBACK_S, 0))
        if now is None or before is None:
            return None
        moved = haversine_m(before["latitude"], before["longitude"],
                            now["latitude"], now["longitude"])
        if moved < config.HEADING_MIN_MOVE_M:
            return None
        return bearing_deg(before["latitude"], before["longitude"],
                           now["latitude"], now["longitude"])

    def damage_location(self, video_seconds, distance_m):
        """Estimated position of a damage that is distance_m AHEAD of the car.
        Returns (latitude, longitude) or (None, None)."""
        fix = self.at(video_seconds)
        heading = self.heading_at(video_seconds)
        if fix is None or heading is None or distance_m is None:
            return None, None
        return destination_point(fix["latitude"], fix["longitude"], heading, distance_m)

    def coverage(self, duration_s):
        """Share (0-1) of the video seconds that have a GPS position. Used as a sanity check."""
        n = max(int(duration_s), 1)
        return sum(1 for s in range(n) if self.at(s) is not None) / n
