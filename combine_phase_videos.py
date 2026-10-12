
import os
import cv2

# Existing phase output videos
PHASE_VIDEOS = [
    ("Phase 1 - Road Damage Detection",
     r"outputs\annotated_test.mp4"),
    ("Phase 2 - Damage Tracking",
     r"outputs\phase2_tracked.mp4"),
    ("Phase 3 - Vehicle Detection and Tracking",
     r"outputs\annotated_phase3.mp4"),
    ("Phase 4 - Path Risk Estimation",
     r"outputs\annotated_phase4.mp4"),
    ("Phase 5 - Distance Estimation",
     r"outputs\annotated_phase5.mp4"),
    ("Phase 6 - GPS Synchronization",
     r"outputs\annotated_phase6.mp4"),
    ("Phase 7 - License Plate Recognition",
     r"outputs\phase7\annotated_phase7.mp4"),
]

OUTPUT_DIR = "outputs"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR, "final_integrated_video.mp4"
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    available = []

    print("Checking phase videos...\n")

    for phase_name, video_path in PHASE_VIDEOS:
        if os.path.isfile(video_path):
            available.append((phase_name, video_path))
            print(f"[FOUND] {phase_name}: {video_path}")
        else:
            print(f"[MISSING] {phase_name}: {video_path}")

    if not available:
        raise SystemExit(
            "No phase videos were found. Check the paths."
        )

    # Use the first available video's properties.
    first_cap = cv2.VideoCapture(available[0][1])

    if not first_cap.isOpened():
        raise SystemExit("Cannot open the first video.")

    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise SystemExit(
            f"Cannot create output video: {OUTPUT_VIDEO}"
        )

    try:
        for phase_name, video_path in available:
            print(f"\nAdding: {phase_name}")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"WARNING: Cannot open {video_path}")
                continue

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                # Resize phase outputs to the same dimensions.
                if (
                    frame.shape[1] != width
                    or frame.shape[0] != height
                ):
                    frame = cv2.resize(
                        frame, (width, height)
                    )

                writer.write(frame)

            cap.release()

    finally:
        writer.release()

    print("\nVIDEO COMBINATION COMPLETE")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Phases included: {len(available)}")
    print(
        "Note: the videos are concatenated in sequence, "
        "not merged frame-by-frame."
    )


if __name__ == "__main__":
    main()
