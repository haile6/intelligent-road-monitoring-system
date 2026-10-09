"""
Trackers (Phase 2 + Phase 3).

Problem: one pothole (or one car) is visible in many video frames, so YOLO "finds" it
again and again. A tracker gives every physical object ONE track ID, and only confirms
it after it has been seen in several frames.

How it works (simple version):
  1. For each new frame we compare every new detection with every known track.
  2. If a detection overlaps a track enough (IoU), it is the same object.
  3. Detections that match nothing start a new track.
  4. Tracks that are not seen for a while are removed from the active list.

This is an approximation. It can later be replaced by ByteTrack or DeepSORT.
"""
import config


def iou(a, b):
    """Overlap between two boxes (x1, y1, x2, y2): 0 = no overlap, 1 = identical."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id, det, frame_no, timestamp):
        self.id = track_id
        self.class_id = det["class_id"]
        self.class_name = det["class_name"]
        self.bbox = det["bbox"]
        self.confidence = det["confidence"]        # latest confidence
        self.best_confidence = det["confidence"]   # highest confidence so far
        self.first_frame = self.last_frame = frame_no
        self.first_time = self.last_time = self.confirmed_time = timestamp
        self.hits = 1                # number of frames this object was detected in
        self.missed = 0              # frames in a row where it was not detected
        self.confirmed = False
        self.confirmed_frame = None

    def update(self, det, frame_no, timestamp=None, adopt_class=False):
        # Vehicles: YOLO may call the same vehicle "car" then "truck".
        # We keep the class that had the highest confidence.
        if adopt_class and det["confidence"] > self.best_confidence:
            self.class_id = det["class_id"]
            self.class_name = det["class_name"]
        self.bbox = det["bbox"]
        self.confidence = det["confidence"]
        self.best_confidence = max(self.best_confidence, det["confidence"])
        self.last_frame = frame_no
        if timestamp is not None:
            self.last_time = timestamp
        self.hits += 1
        self.missed = 0


class DamageTracker:
    def __init__(self, iou_threshold=config.TRACK_IOU_THRESHOLD,
                 max_missed=config.TRACK_MAX_MISSED_FRAMES,
                 min_confirm=config.MIN_CONFIRMATION_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_confirm = min_confirm
        self.match_class = True    # True: only match boxes of the same class (road damage)
        self.active = []           # tracks still being followed
        self.all_tracks = []       # every track ever created (for the final report)
        self._next_id = 1

    def update(self, detections, frame_no, timestamp):
        """Call once per frame. Returns (visible_tracks, newly_confirmed_tracks)."""
        # 1. Score every (track, detection) pair
        pairs = []
        for ti, t in enumerate(self.active):
            for di, d in enumerate(detections):
                if self.match_class and t.class_id != d["class_id"]:
                    continue
                score = iou(t.bbox, d["bbox"])
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)               # best overlaps first

        # 2. Match greedily: each track and each detection is used at most once
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.active[ti].update(detections[di], frame_no, timestamp,
                                   adopt_class=not self.match_class)
            used_t.add(ti)
            used_d.add(di)

        # 3. Tracks that were not matched: count the miss
        for ti, t in enumerate(self.active):
            if ti not in used_t:
                t.missed += 1

        # 4. Detections that were not matched: start new tracks
        for di, d in enumerate(detections):
            if di not in used_d:
                t = Track(self._next_id, d, frame_no, timestamp)
                self._next_id += 1
                self.active.append(t)
                self.all_tracks.append(t)

        # 5. Confirm tracks that were seen often enough
        newly_confirmed = []
        for t in self.active:
            if not t.confirmed and t.hits >= self.min_confirm:
                t.confirmed = True
                t.confirmed_frame = frame_no
                t.confirmed_time = timestamp
                newly_confirmed.append(t)

        # 6. Remove tracks that have been missing for too long
        self.active = [t for t in self.active if t.missed <= self.max_missed]

        visible = [t for t in self.active if t.missed == 0]
        return visible, newly_confirmed

    def confirmed_tracks(self):
        return [t for t in self.all_tracks if t.confirmed]


class VehicleTracker(DamageTracker):
    """Same tracker, but for OTHER vehicles: boxes match regardless of class,
    and the track keeps its most confident class."""
    def __init__(self):
        super().__init__(iou_threshold=config.VEHICLE_TRACK_IOU_THRESHOLD,
                         max_missed=config.VEHICLE_TRACK_MAX_MISSED_FRAMES,
                         min_confirm=config.VEHICLE_MIN_CONFIRMATION_FRAMES)
        self.match_class = False
