"""
Phase 11: Road Damage Severity Assessment.
Estimates the physical surface area and depth/severity rating of potholes
and alligator cracks from camera perspective distance and bounding box pixels.
"""
import math
import config


def estimate_damage_severity(bbox, distance_m, damage_type="pothole", frame_width=1280):
    """
    Computes physical metric dimensions and assigns a severity rating.
    
    Returns:
        dict: {
            "width_m": float,
            "length_m": float,
            "area_m2": float,
            "severity_level": "Low" | "Medium" | "High" | "Severe",
            "score": float (0-100)
        }
    """
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    if distance_m is None or distance_m <= 0:
        # Fallback based on pixel size relative to frame
        area_ratio = (box_w * box_h) / (frame_width * frame_width * 0.5)
        if area_ratio > 0.08:
            return {"width_m": None, "length_m": None, "area_m2": None, "severity_level": "High", "score": 75}
        elif area_ratio > 0.03:
            return {"width_m": None, "length_m": None, "area_m2": None, "severity_level": "Medium", "score": 50}
        else:
            return {"width_m": None, "length_m": None, "area_m2": None, "severity_level": "Low", "score": 25}

    # Perspective physical width: W_m = (W_px * distance_m) / focal_length_px
    # Approximate focal length from Horizontal FOV (70 deg)
    fov_rad = math.radians(getattr(config, "CAMERA_HORIZONTAL_FOV_DEG", 70.0))
    focal_px = frame_width / (2.0 * math.tan(fov_rad / 2.0))

    physical_w_m = (box_w * distance_m) / focal_px
    # Ground plane projection length is compressed by tilt angle
    tilt_rad = math.radians(getattr(config, "CAMERA_TILT_DEG", 5.0) + 10.0)
    physical_l_m = (box_h * distance_m) / (focal_px * max(0.2, math.sin(tilt_rad)))

    # Surface Area in square metres
    area_m2 = physical_w_m * physical_l_m

    # Type weighting: potholes cause more structural vehicle impact
    type_mult = 1.3 if damage_type.lower() == "pothole" else 1.0
    effective_score = min(100.0, (area_m2 * 120.0) * type_mult)

    if effective_score >= 70 or area_m2 >= 0.6:
        severity = "Severe"
    elif effective_score >= 45 or area_m2 >= 0.25:
        severity = "High"
    elif effective_score >= 20 or area_m2 >= 0.08:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "width_m": round(physical_w_m, 2),
        "length_m": round(physical_l_m, 2),
        "area_m2": round(area_m2, 3),
        "severity_level": severity,
        "score": round(effective_score, 1)
    }
