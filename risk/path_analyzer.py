"""
Phase 4: is the road damage in the path of OUR vehicle?

IMPORTANT - this is an APPROXIMATION.
We assume the camera looks straight along the road and that the car will drive through
a fixed trapezoid in the lower-centre of the picture. It does not know about curves,
lane markings, or where your car really is on the road.
It should later be improved with lane detection, camera calibration and
vehicle localisation.
"""
import cv2
import numpy as np

import config


class PathAnalyzer:
    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height

        cx = config.PATH_CENTER_X_RATIO * self.w
        bottom_half = config.PATH_WIDTH_RATIO * self.w / 2
        top_half = config.PATH_TOP_WIDTH_RATIO * self.w / 2
        top_y = int(config.PATH_TOP_Y_RATIO * self.h)

        # corners: bottom-left, bottom-right, top-right, top-left
        self.polygon = np.array([
            [cx - bottom_half, self.h - 1],
            [cx + bottom_half, self.h - 1],
            [cx + top_half, top_y],
            [cx - top_half, top_y],
        ], dtype=np.int32)

        # A picture-sized mask: 1 inside the path, 0 outside. Made once, reused every frame.
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.polygon], 1)

    def analyze(self, bbox):
        """bbox = (x1, y1, x2, y2). Returns:
             in_path        True / False
             overlap_ratio  share of the box that lies inside the path (0.0 - 1.0)
        """
        x1 = max(0, min(self.w - 1, int(bbox[0])))
        y1 = max(0, min(self.h - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(self.w, int(bbox[2])))
        y2 = max(y1 + 1, min(self.h, int(bbox[3])))

        area = (x2 - x1) * (y2 - y1)
        inside = int(self.mask[y1:y2, x1:x2].sum())
        overlap = inside / area

        # Where the damage touches the road: bottom-centre of its box
        bx = (x1 + x2) // 2
        by = min(y2 - 1, self.h - 1)
        bottom_centre_inside = bool(self.mask[by, bx])

        in_path = overlap >= config.PATH_MIN_OVERLAP_RATIO or bottom_centre_inside
        return {"in_path": in_path, "overlap_ratio": overlap}
