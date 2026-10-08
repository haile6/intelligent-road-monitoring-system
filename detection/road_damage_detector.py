"""
Phase 1: runs YOUR existing YOLO model on single frames.
The model has only two classes: 0 = alligator_crack, 1 = pothole.
"""
import cv2
from ultralytics import YOLO

import config


class RoadDamageDetector:
    def __init__(self, model_path=config.MODEL_PATH,
                 conf=config.CONFIDENCE_THRESHOLD, imgsz=config.IMAGE_SIZE):
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self._check_classes()

    def _check_classes(self):
        """Stop immediately if the model does not have exactly our two classes."""
        names = {int(k): v for k, v in self.model.names.items()}
        print("Model classes:")
        for k in sorted(names):
            print(f"  {k}: {names[k]}")
        if names != config.EXPECTED_CLASSES:
            raise SystemExit(
                "\nERROR: this model does not have the expected classes.\n"
                f"  Expected: {config.EXPECTED_CLASSES}\n"
                f"  Found:    {names}\n"
                "Check MODEL_PATH in config.py."
            )

    def detect(self, frame):
        """Returns a list of detections found in one video frame.
        Each detection is a dictionary:
            class_id, class_name, confidence, bbox = (x1, y1, x2, y2)
        """
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        detections = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            class_id = int(cls)
            detections.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": float(conf),
                "bbox": tuple(int(v) for v in box),
            })
        return detections


def draw_detections(frame, detections):
    """Draws a box and a label such as 'POTHOLE 0.91' for every detection."""
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = config.CLASS_COLORS.get(d["class_name"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

        label = f'{d["class_name"].upper()} {d["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
