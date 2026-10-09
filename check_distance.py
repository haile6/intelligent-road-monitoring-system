r"""
Calibration helper: click on the road in a video frame and see the estimated distance.

Use it with a real, measured distance:
  1. On a straight, flat road mark points with chalk or cones at 5 m, 10 m, 20 m, 30 m.
  2. Film the road from your car (camera mounted exactly as it will be used).
  3. Run:   python check_distance.py --video data\calibration.mp4 --frame 30
  4. Click on the point where each mark touches the road, compare with the true distance.

If the estimates are TOO LARGE : increase CAMERA_TILT_DEG a little (or check CAMERA_HEIGHT_M).
If the estimates are TOO SMALL : decrease CAMERA_TILT_DEG a little.
The dashed line is the horizon the program assumes - it should sit on the real horizon.
Press q to quit, c to clear the clicks.
"""
import argparse
import cv2

import config
from risk.distance_estimator import make_estimator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=config.VIDEO_PATH)
    ap.add_argument("--image", help="use a picture instead of a video")
    ap.add_argument("--frame", type=int, default=1, help="frame number to show")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(args.frame - 1, 0))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Could not read that frame.")
    if frame is None:
        raise SystemExit("Could not open the picture.")

    h, w = frame.shape[:2]
    est = make_estimator(w, h)
    print(f"Picture {w}x{h} | focal length {est.focal:.0f} px | horizon at row {est.horizon_row:.0f}")
    print("Click on the road. Press q to quit, c to clear.")

    base = frame.copy()
    cv2.line(base, (0, int(est.horizon_row)), (w, int(est.horizon_row)), (255, 255, 0), 1)
    cv2.putText(base, "assumed horizon", (10, max(int(est.horizon_row) - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    canvas = {"img": base.copy()}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        d = est.distance_at_row(y)
        text = f"~{d:.1f} m" if d is not None else "unknown (above horizon / too far)"
        print(f"  row {y}: estimated distance {text}")
        cv2.circle(canvas["img"], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(canvas["img"], text, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.namedWindow("check distance")
    cv2.setMouseCallback("check distance", on_click)
    while True:
        cv2.imshow("check distance", canvas["img"])
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas["img"] = base.copy()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
