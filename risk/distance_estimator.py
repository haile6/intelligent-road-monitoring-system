"""
Phase 5: estimated distance from the camera vehicle to a road-damage detection.

IMPORTANT
  A YOLO box does not contain a distance in metres. This module ESTIMATES it with
  a simple perspective model:

     * the road in front of the car is flat,
     * the camera is at a known height above the road and pitched down by a known angle,
     * the bottom edge of the damage box is where the damage touches the road.

  For a pixel row y in the picture:
        angle below the optical axis   a   = atan((y - centre_row) / focal_length)
        angle below the horizontal     phi = tilt + a
        distance along the road        d   = camera_height / tan(phi)

  The result is only "Estimated distance". It is less accurate for far damage,
  uphill/downhill roads, bumpy driving or a wrong camera height/tilt.

  The class layout allows other methods later (calibrated monocular depth, stereo
  camera, depth camera, visual odometry): write a new subclass of DistanceEstimator
  and select it with DISTANCE_METHOD in config.py.
"""
import math

import config


class DistanceEstimator:
    """Base class. Every method must return a dictionary from estimate()."""
    method = "base"

    def estimate(self, bbox, frame=None):
        """Returns {"distance_m": float or None, "method": str}.
        None means 'cannot be estimated' (for example: above the horizon, or too far)."""
        raise NotImplementedError


class PerspectiveDistanceEstimator(DistanceEstimator):
    method = "perspective (flat road)"

    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.cy = frame_height / 2.0                         # centre row of the picture
        self.camera_height = config.CAMERA_HEIGHT_M
        self.tilt = math.radians(config.CAMERA_TILT_DEG)

        if config.FOCAL_LENGTH_PX:
            self.focal = float(config.FOCAL_LENGTH_PX)
        else:
            half_fov = math.radians(config.CAMERA_HORIZONTAL_FOV_DEG) / 2
            self.focal = (frame_width / 2.0) / math.tan(half_fov)

        # The picture row where the horizon appears (rays parallel to the road)
        self.horizon_row = self.cy - self.focal * math.tan(self.tilt)

    def distance_at_row(self, row):
        """Distance in metres to the road point seen at this picture row (or None)."""
        angle_in_image = math.atan((row - self.cy) / self.focal)
        phi = self.tilt + angle_in_image                     # angle below the horizontal
        if phi < math.radians(0.5):                          # at or above the horizon
            return None
        distance = self.camera_height / math.tan(phi)
        if distance > config.DISTANCE_MAX_M:
            return None
        return distance

    def estimate(self, bbox, frame=None):
        bottom_row = bbox[3]                                 # where the damage touches the road
        return {"distance_m": self.distance_at_row(bottom_row), "method": self.method}


def make_estimator(frame_width, frame_height):
    if config.DISTANCE_METHOD == "perspective":
        return PerspectiveDistanceEstimator(frame_width, frame_height)
    raise SystemExit(f"DISTANCE_METHOD '{config.DISTANCE_METHOD}' is not implemented yet. "
                     "Use 'perspective'.")
