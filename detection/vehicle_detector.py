"""
Phase 3: detects OTHER vehicles (car, truck, bus, motorcycle) with a pretrained COCO model.
This is a SEPARATE model. The road-damage model is not changed or retrained.
"""
from ultralytics import YOLO

import config


def is_own_vehicle(bbox, frame_width, frame_height):
    """Heuristic: the bonnet/hood of the camera vehicle appears as a very wide box
    touching the bottom edge. We do not want to count our own vehicle."""
    x1, y1, x2, y2 = bbox
    touches_bottom = y2 >= frame_height * (1 - config.OWN_VEHICLE_BOTTOM_MARGIN)
    very_wide = (x2 - x1) >= frame_width * config.OWN_VEHICLE_MIN_WIDTH_RATIO
    return touches_bottom and very_wide


class VehicleDetector:
    def __init__(self, model_path=config.VEHICLE_MODEL_PATH,
                 conf=config.VEHICLE_CONFIDENCE_THRESHOLD, imgsz=config.VEHICLE_IMAGE_SIZE):
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self._check_classes()

    def _check_classes(self):
        names = {int(k): v for k, v in self.model.names.items()}
        print("Vehicle model classes used:")
        for class_id, expected in config.VEHICLE_CLASSES.items():
            print(f"  {class_id}: {names.get(class_id)}")
            if names.get(class_id) != expected:
                raise SystemExit(
                    f"\nERROR: vehicle model class {class_id} is '{names.get(class_id)}', "
                    f"expected '{expected}'.\nUse a COCO-trained model (for example yolo11s.pt)."
                )

    def detect(self, frame):
        """Returns the same kind of dictionaries as the road-damage detector."""
        height, width = frame.shape[:2]
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                    classes=list(config.VEHICLE_CLASSES), verbose=False)[0]
        detections = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            bbox = tuple(int(v) for v in box)
            if config.IGNORE_OWN_VEHICLE and is_own_vehicle(bbox, width, height):
                continue
            class_id = int(cls)
            detections.append({
                "class_id": class_id,
                "class_name": config.VEHICLE_CLASSES[class_id],
                "confidence": float(conf),
                "bbox": bbox,
            })
        return detections
