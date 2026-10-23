"""
RoadEye Custom Road Damage Model Trainer & Transfer Learning Utility.
Allows municipal operators and researchers to fine-tune or retrain the 
road crack and pothole detection model using custom labeled datasets.
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_YAML = os.path.join(BASE_DIR, "data", "custom_dataset", "dataset.yaml")
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "road_damage_best.pt")
OUTPUT_MODEL_DEST = os.path.join(BASE_DIR, "models", "road_damage_best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-tune Road Damage & Crack Model")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML,
                        help="Path to dataset.yaml describing train/val paths and class names")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Base model weights to start training from (default: models/road_damage_best.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, decrease to 8 or 4 if GPU runs out of VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image resolution in pixels (default: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to run on (e.g., '0' for CUDA GPU, 'cpu' for CPU)")
    parser.add_argument("--name", type=str, default="road_damage_train",
                        help="Experiment name")
    parser.add_argument("--auto-deploy", action="store_true", default=True,
                        help="Automatically backup previous model and copy best.pt to models/road_damage_best.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("      ROADEYE: CUSTOM ROAD DAMAGE MODEL TRAINING ENGINE")
    print("=" * 70)
    print(f"[*] Dataset Config:    {args.data}")
    print(f"[*] Initial Weights:   {args.weights}")
    print(f"[*] Training Epochs:   {args.epochs}")
    print(f"[*] Batch Size:        {args.batch}")
    print(f"[*] Image Resolution:  {args.imgsz}x{args.imgsz}")
    print(f"[*] Compute Target:    {args.device if args.device else 'Auto-detect (CUDA if available, else CPU)'}")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"\n[!] Error: Dataset configuration file not found at: {args.data}")
        print("[*] Please create your dataset folder and verify dataset.yaml.")
        print("[*] Refer to data/custom_dataset/README.md for dataset structure instructions.")
        sys.exit(1)

    # 1. Load Pretrained YOLO Weights (Transfer Learning)
    if os.path.exists(args.weights):
        print(f"[*] Loading pretrained weights from {args.weights} for transfer learning...")
        model = YOLO(args.weights)
    else:
        print(f"[*] Pretrained weights not found at {args.weights}, starting from standard YOLOv8s base...")
        model = YOLO("yolov8s.pt")

    # 2. Execute Training
    print("\n[*] Starting neural network training on your custom road dataset...\n")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        name=args.name,
        patience=15,          # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )

    # 3. Validation & Metrics
    print("\n" + "=" * 70)
    print("[*] Evaluating validation metrics on test partition...")
    metrics = model.val()
    map50 = getattr(metrics.box, "map50", 0.0)
    map50_95 = getattr(metrics.box, "map", 0.0)
    print(f"  -> Validation mAP@0.50:       {map50:.4f}")
    print(f"  -> Validation mAP@0.50:0.95:  {map50_95:.4f}")
    print("=" * 70)

    # 4. Locate best trained weights
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_pt) and args.auto_deploy:
            # Backup old model
            if os.path.exists(OUTPUT_MODEL_DEST):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(BASE_DIR, "models", f"road_damage_best_backup_{timestamp}.pt")
                shutil.copy2(OUTPUT_MODEL_DEST, backup_path)
                print(f"[*] Created backup of previous model: {backup_path}")

            # Overwrite with new best model
            shutil.copy2(best_pt, OUTPUT_MODEL_DEST)
            print(f"[*] Successfully deployed trained model to: {OUTPUT_MODEL_DEST}")
            print("[*] The system will now use your newly trained model across all pipelines and web studio!")

    print("\n[✓] Training complete! Your custom dataset has been integrated.")


if __name__ == "__main__":
    main()
