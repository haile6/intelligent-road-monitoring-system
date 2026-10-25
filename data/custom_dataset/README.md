# Debre Berhan Road Damage Dataset (Potholes & Alligator Cracks)

## 📌 Dataset Overview

This dataset was **manually collected on-site in Debre Berhan City (ደብረ ብርሃን), Ethiopia** to represent real-world asphalt pavement distress and hazardous driving conditions. It specifically targets severe road defects common in urban and regional Ethiopian roadways.

* **Collection Location**: Debre Berhan City, Amhara Region, Ethiopia
* **Collection Method**: Manually captured field road imagery using mobile cameras and dashcam setups under natural lighting, varied weather, and realistic roadway textures.
* **Annotation Platform**: Manually labeled and reviewed using **CVAT (Computer Vision Annotation Tool)**.
* **Export Format**: Standard YOLOv8 / YOLO11 PyTorch bounding box format.

---

## 🏷️ Class Definitions

| Class ID | Class Name | Description | Visual Characteristics |
| :--- | :--- | :--- | :--- |
| `0` | **`alligator_crack`** | Structural fatigue cracking in asphalt pavement. | Interconnected pattern resembling alligator skin or reptile scales. |
| `1` | **`pothole`** | Bowl-shaped hole or depression in road surface. | Noticeable edge cavity, asphalt loss, varying depth and shadow. |

---

## 📊 Dataset Statistics & Splits

The dataset comprises **725 annotated images** with corresponding YOLO text annotation files:

| Split | Images | Annotations | Percentage | Directory Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 507 | 507 | 70% | `images/train/`, `labels/train/` |
| **Validation** | 109 | 109 | 15% | `images/val/`, `labels/val/` |
| **Test** | 109 | 109 | 15% | `images/test/`, `labels/test/` |
| **Total** | **725** | **725** | **100%** | |

---

## 📂 Directory Layout

```text
data/custom_dataset/
├── dataset.yaml            # Ultralytics YOLO configuration file
├── README.md               # Dataset documentation (this file)
├── classes.txt             # Class name mapping (0: alligator_crack, 1: pothole)
├── split.csv               # Image-to-split partition registry
├── images/
│   ├── train/              # 507 training images (.jpg)
│   ├── val/                # 109 validation images (.jpg)
│   └── test/               # 109 evaluation test images (.jpg)
└── labels/
    ├── train/              # 507 YOLO annotation files (.txt)
    ├── val/                # 109 YOLO annotation files (.txt)
    └── test/               # 109 YOLO annotation files (.txt)
```

---

## 📝 Annotation Format

Each image has a paired `.txt` annotation file with the same base name (e.g., `crack1__20260925_175646.jpg` has `crack1__20260925_175646.txt`).

Each line denotes a single bounding box in normalized coordinates:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Where all coordinate values are normalized between `0.0` and `1.0`.

---

## 🚀 Usage with Ultralytics YOLO

To train or fine-tune YOLO models with this dataset, reference `dataset.yaml`:

```python
from ultralytics import YOLO

# Load model (pretrained or base YOLO)
model = YOLO("models/road_damage_best.pt")

# Train on the Debre Berhan dataset
results = model.train(
    data="data/custom_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="debre_berhan_damage_model"
)
```
