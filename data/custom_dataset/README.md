# Custom Road Damage Dataset Directory

Place your custom road crack and pothole training data in this folder to merge with and fine-tune the detection model.

## Folder Structure

```text
data/custom_dataset/
├── dataset.yaml            # Dataset configuration with class mapping
├── images/
│   ├── train/              # Training photos (.jpg, .jpeg, .png)
│   └── val/                # Validation photos (.jpg, .jpeg, .png)
└── labels/
    ├── train/              # YOLO format annotation files (.txt)
    └── val/                # YOLO format annotation files (.txt)
```

## Annotation File Format (Standard YOLO)
Each image must have a corresponding `.txt` file with the exact same name in the `labels/` directory (e.g., `image001.jpg` has `image001.txt`).

Each line inside the `.txt` file represents one bounding box:
```text
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates must be normalized between `0.0` and `1.0`.

### Class IDs:
- `0`: `alligator_crack`
- `1`: `pothole`

*(If you export your labeled dataset from **Roboflow**, **CVAT**, or **LabelImg**, choose **YOLOv8 PyTorch** export format and copy the folders here).*

## How to Train & Merge with the Existing Model
Run the automated training script:
```bash
python train_custom_damage.py --epochs 50 --batch 16
```
This will:
1. Load the existing pretrained model `models/road_damage_best.pt` (transfer learning).
2. Train and adapt the weights to your custom road data.
3. Automatically back up the old weights and deploy the new `best.pt` into `models/road_damage_best.pt`.
