"""
PHASE 1: run the existing YOLO model on a video and save an annotated copy.
No tracking, GPS, vehicles or dashboard yet.

Run:   python run_phase1.py
"""
import os
import cv2

import config
from detection.road_damage_detector import RoadDamageDetector, draw_detections


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(f"Model not found: {config.MODEL_PATH}\nFix MODEL_PATH in config.py")
    if not os.path.exists(config.VIDEO_PATH):
        raise SystemExit(f"Video not found: {config.VIDEO_PATH}\nFix VIDEO_PATH in config.py")

    detector = RoadDamageDetector()          # also prints and checks the class list

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open the video. Try another MP4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {width}x{height}, {fps:.1f} fps, {total} frames")

    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PATH), exist_ok=True)
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts = {name: 0 for name in config.EXPECTED_CLASSES.values()}
    frames_with_damage = 0
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        detections = detector.detect(frame)
        if detections:
            frames_with_damage += 1
        for d in detections:
            counts[d["class_name"]] += 1       # detections per frame, NOT unique potholes

        writer.write(draw_detections(frame, detections))

        if config.SHOW_PREVIEW:
            cv2.imshow("Phase 1 - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_no % 50 == 0:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nDONE")
    print(f"  Frames processed:        {frame_no}")
    print(f"  Frames with detections:  {frames_with_damage}")
    for name, n in counts.items():
        print(f"  {name} detections:      {n}   (counted per frame)")
    print(f"  Saved: {config.OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
