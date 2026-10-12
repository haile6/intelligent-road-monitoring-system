"""
Phase 7: the speed of the camera vehicle.

Order of preference:
  1. "gps"                     speed written in the GPS file (km/h)
  2. "computed_from_positions" speed calculated from two GPS positions a few seconds apart
  3. "default"                 DEFAULT_SPEED_KMH from config.py  <-- a TEST value, not a measurement

The result always says where the number came from, so a guessed speed is never
mistaken for a measured one.

Conversion used everywhere:   speed_mps = speed_kmh / 3.6      (36 km/h = 10 m/s)
"""
import config
from gps.geo import haversine_m


def kmh_to_mps(speed_kmh):
    return speed_kmh / 3.6


class SpeedProvider:
    def __init__(self, gps_sync=None):
        self.gps = gps_sync            # may be None when there is no GPS file at all

    def _result(self, speed_kmh, source):
        return {
            "speed_kmh": speed_kmh,
            "speed_mps": kmh_to_mps(speed_kmh),
            "source": source,
            "is_default": source == "default",
            "is_stopped": speed_kmh < config.MIN_MOVING_SPEED_KMH,
        }

    def get(self, video_seconds):
        """Speed at a moment of the video. Always returns a dictionary:
           speed_kmh, speed_mps, source, is_default, is_stopped"""
        if not config.FORCE_DEFAULT_SPEED and self.gps is not None:
            fix = self.gps.at(video_seconds)
            if fix is not None:
                # 1. speed from the GPS file
                s = fix["speed_kmh"]
                if s is not None and 0 <= s <= config.MAX_PLAUSIBLE_SPEED_KMH:
                    return self._result(s, "gps")

                # 2. speed from positions (needs a position a little earlier)
                if config.SPEED_FROM_POSITIONS:
                    earlier_t = max(video_seconds - config.SPEED_WINDOW_S, 0.0)
                    elapsed = video_seconds - earlier_t
                    earlier = self.gps.at(earlier_t)
                    if earlier is not None and elapsed >= 0.5:
                        metres = haversine_m(earlier["latitude"], earlier["longitude"],
                                             fix["latitude"], fix["longitude"])
                        s = metres / elapsed * 3.6
                        if s <= config.MAX_PLAUSIBLE_SPEED_KMH:
                            return self._result(s, "computed_from_positions")

        # 3. last resort: the test value
        return self._result(float(config.DEFAULT_SPEED_KMH), "default")
