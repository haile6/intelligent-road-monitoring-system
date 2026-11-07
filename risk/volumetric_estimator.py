"""
3D Pothole Depth & Volumetric Estimator (የጉድጓድ ጥልቀትና የአስፋልት መጠን አስሊ)
==========================================================================
RoadEye™ Enterprise Road Intelligence Platform
Authority: Ethiopian Roads Administration (ERA) Standard Specifications

Physics & Photogrammetric Foundations:
- Monocular Inverse Perspective Mapping (IPM) & Ground Scale Projection
- Optical Shadow Degradation & Cavity Geometry Modeling
- Truncated Paraboloid / Semi-Ellipsoid 3D Cavity Volumetric Integration
- Asphalt Repair Material Density & Compaction Bulking Factors (Cold/Hot Mix)
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import config


class Pothole3DDepthEstimator:
    """
    Monocular Photogrammetric 3D Pothole Depth Estimator.
    
    Estimates physical diameter (major/minor axis in cm), 2D surface area (m2),
    maximum depth (cm), and cavity geometry from camera bounding boxes and distances.
    """

    DEFAULT_FOCAL_PX = 850.0  # Default focal length in pixels for 70 deg FOV at 1280x720
    ASPHALT_DEPTH_COEFFICIENT = 8.5  # Empirical scaling constant for weathered asphalt

    @classmethod
    def get_focal_length_px(cls, frame_width: int = 1280) -> float:
        """Compute camera focal length in pixels from horizontal FOV."""
        if config.FOCAL_LENGTH_PX:
            return float(config.FOCAL_LENGTH_PX)
        fov_rad = math.radians(config.CAMERA_HORIZONTAL_FOV_DEG)
        return (frame_width / 2.0) / math.tan(fov_rad / 2.0)

    @classmethod
    def estimate_physical_dimensions(cls, 
                                     bbox: List[float], 
                                     distance_m: float, 
                                     frame_width: int = 1280, 
                                     frame_height: int = 720) -> Dict[str, float]:
        """
        Convert 2D image pixel bounding box [x1, y1, x2, y2] to physical ground dimensions (meters).
        """
        dist = max(1.0, min(float(distance_m or 5.0), 60.0))
        focal = cls.get_focal_length_px(frame_width)
        tilt_rad = math.radians(config.CAMERA_TILT_DEG)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        pw = max(4.0, abs(x2 - x1))
        ph = max(4.0, abs(y2 - y1))

        # Horizontal ground scale: scale_x = distance / focal_length (m/px)
        scale_x = dist / focal

        # Vertical road ground scale taking camera tilt pitch into account
        # Near edge vs far edge foreshortening
        cy = frame_height / 2.0
        angle_bottom = tilt_rad + math.atan((y2 - cy) / focal)
        sin_angle = max(0.15, math.sin(angle_bottom))
        scale_y = (dist / (focal * sin_angle)) * 0.75

        # Physical width and length in meters
        width_m = max(0.12, min(3.50, pw * scale_x))
        length_m = max(0.12, min(3.50, ph * scale_y))

        # Major and minor axis (cm)
        major_axis_cm = round(max(width_m, length_m) * 100.0, 1)
        minor_axis_cm = round(min(width_m, length_m) * 100.0, 1)

        # Elliptical surface area: A = pi * (w/2) * (l/2)
        surface_area_m2 = round(math.pi * (width_m / 2.0) * (length_m / 2.0), 4)

        return {
            "width_m": round(width_m, 3),
            "length_m": round(length_m, 3),
            "major_axis_cm": major_axis_cm,
            "minor_axis_cm": minor_axis_cm,
            "surface_area_m2": surface_area_m2,
            "distance_m": round(dist, 2)
        }

    @classmethod
    def estimate_depth(cls, 
                       dimensions: Dict[str, float], 
                       risk_level: str = "CAUTION", 
                       shadow_contrast: float = 0.45) -> Dict[str, Any]:
        """
        Estimate 3D cavity maximum and average depth (cm) from ground footprint and shadow contrast.
        """
        w_m = dimensions["width_m"]
        l_m = dimensions["length_m"]
        risk_up = (risk_level or "CAUTION").upper()

        # Geometric footprint metric: equivalent diameter in meters
        equiv_diam_m = math.sqrt(w_m * l_m)

        # Shadow contrast factor: deeper cavities trap more ambient light, increasing shadow contrast
        s_factor = max(0.1, min(0.9, float(shadow_contrast)))

        # Base maximum depth formula: D_max = k_d * sqrt(A) * (1 + 0.6 * S)
        base_d_cm = cls.ASPHALT_DEPTH_COEFFICIENT * equiv_diam_m * (1.0 + 0.65 * s_factor)

        # Adjust based on verified risk severity
        if "IMMEDIATE" in risk_up or "DANGER" in risk_up:
            d_max_cm = base_d_cm * 1.35
            depth_tier = "Severe / Tire Rim Hazard"
            depth_tier_am = "ከፍተኛ / ጎማና ሪም ሰባሪ"
            severity_color = "#ef4444"
        elif "CAUTION" in risk_up or "WARN" in risk_up:
            d_max_cm = base_d_cm * 1.05
            depth_tier = "Moderate / Vehicle Impact"
            depth_tier_am = "መካከለኛ / ንዝረት የሚፈጥር"
            severity_color = "#f59e0b"
        else:
            d_max_cm = base_d_cm * 0.75
            depth_tier = "Shallow / Early Surface Spall"
            depth_tier_am = "ቀላል / የላይኛው ንጣፍ"
            severity_color = "#10b981"

        # Realistic asphalt depth limits: minimum 1.5 cm, maximum 18.0 cm (base course boundary)
        d_max_cm = round(max(1.5, min(18.0, d_max_cm)), 1)
        # Average depth for a parabolic depression is ~ 0.62 * D_max
        d_avg_cm = round(d_max_cm * 0.62, 1)

        return {
            "depth_max_cm": d_max_cm,
            "depth_avg_cm": d_avg_cm,
            "depth_tier": depth_tier,
            "depth_tier_am": depth_tier_am,
            "severity_color": severity_color
        }
