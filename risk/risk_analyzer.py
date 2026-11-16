"""
Phase 9: Dual-Layer Risk Analysis for Road Damage Events.

Evaluates both:
  1. Driver Kinematic Exposure (Collision TTC, Speed & Corridor Trajectory)
  2. Pavement Structural Severity (Cavity Depth, Surface Area, Damage Type)

Levels (lowest -> highest):  SAFE / MONITOR,  CAUTION,  WARNING,  IMMEDIATE DANGER
"""
import config


def _result(level, reasons, ttc_s=None, asset_risk_level="SAFE", unified_score=0.0, structural_severity=None):
    return {
        "level": level,
        "rank": config.RISK_LEVELS.index(level) if level in config.RISK_LEVELS else 0,
        "label": config.RISK_DISPLAY.get(level, level),
        "reasons": reasons,
        "ttc_s": ttc_s,
        "asset_risk_level": asset_risk_level,
        "unified_score": round(unified_score, 1),
        "structural_severity": structural_severity or "Unknown"
    }


def assess_risk(track, speed):
    """
    Evaluates unified multi-factor collision and municipal asset risk.
    track needs: confirmed, in_path, ttc_s, ttc_status, distance_m, mean_confidence, class_name
                 (optional: severity, severity_level, depth_max_cm, area_m2)
    speed needs: speed_kmh, is_stopped, is_default
    """
    sev_level = getattr(track, "severity", None) or getattr(track, "severity_level", None)
    area_m2 = getattr(track, "area_m2", None)
    depth_cm = getattr(track, "depth_max_cm", None)

    # 1. Structural score mapping
    if sev_level == "Severe" or (depth_cm and depth_cm >= 8.0) or (area_m2 and area_m2 >= 0.50):
        struct_score = 90.0
        inferred_sev = "Severe"
    elif sev_level == "High" or (depth_cm and depth_cm >= 5.0) or (area_m2 and area_m2 >= 0.25):
        struct_score = 70.0
        inferred_sev = "High"
    elif sev_level == "Medium" or (depth_cm and depth_cm >= 2.5) or (area_m2 and area_m2 >= 0.08):
        struct_score = 45.0
        inferred_sev = "Medium"
    elif sev_level == "Low":
        struct_score = 20.0
        inferred_sev = "Low"
    else:
        struct_score = 50.0
        inferred_sev = "Moderate"

    if not track.confirmed:
        return _result("SAFE", ["not confirmed yet"], structural_severity=inferred_sev)

    # 2. Out-of-path / roadside hazards
    if not track.in_path:
        reasons = ["not in our path"]
        if struct_score >= 80.0:
            asset_level = "WARNING"
            reasons.append("off-path severe pavement distress: prioritized for municipal maintenance")
            score = 0.55 * struct_score + 0.45 * 10.0
        elif struct_score >= 60.0:
            asset_level = "CAUTION"
            reasons.append("off-path moderate distress: logged in municipal asset ledger")
            score = 0.55 * struct_score + 0.45 * 10.0
        else:
            asset_level = "SAFE"
            score = 0.55 * struct_score
        return _result("SAFE", reasons, ttc_s=track.ttc_s, asset_risk_level=asset_level, unified_score=score, structural_severity=inferred_sev)

    if track.ttc_status == "stopped" or speed["is_stopped"]:
        return _result("SAFE", ["vehicle stopped"], structural_severity=inferred_sev)
    if track.distance_m is None or track.ttc_s is None:
        return _result("SAFE", ["distance unknown"], structural_severity=inferred_sev)
    if track.distance_m > config.RISK_MAX_DISTANCE_M:
        return _result("SAFE", [f"too far to judge (~{track.distance_m:.0f} m)"], structural_severity=inferred_sev)

    reasons = []

    # 3. Kinematic level from time-to-collision (TTC) with high-speed scaling
    factor = 1.0
    if speed["speed_kmh"] >= config.RISK_HIGH_SPEED_KMH:
        factor = config.RISK_HIGH_SPEED_FACTOR
        if factor != 1.0:
            reasons.append(f"high speed ({speed['speed_kmh']:.0f} km/h): TTC limits x{factor:g}")
    ttc = track.ttc_s
    if ttc < config.RISK_TTC_DANGER_S * factor:
        rank = 3
    elif ttc < config.RISK_TTC_WARNING_S * factor:
        rank = 2
    elif ttc <= config.RISK_TTC_CAUTION_S * factor:
        rank = 1
    else:
        rank = 0
    reasons.insert(0, f"TTC {ttc:.1f} s")

    # 4. Damage type adjustment
    adjust = config.RISK_TYPE_ADJUSTMENT.get(track.class_name, 0)
    if adjust and rank >= 1:
        new_rank = max(1, min(3, rank + adjust))
        if new_rank != rank:
            reasons.append(f"{track.class_name} adjustment {adjust:+d}")
            rank = new_rank

    # 5. Dual-Layer Multi-Factor Physical Damage Adjustment:
    # A minor superficial crack or shallow distress in path should not trigger false IMMEDIATE DANGER
    if rank == 3 and inferred_sev == "Low":
        rank = 2
        reasons.append("minor surface damage: downgraded from IMMEDIATE DANGER")
    elif inferred_sev == "Low":
        if rank == 2 and track.class_name != "pothole":
            rank = 1
            reasons.append("minor superficial damage: softened to CAUTION")
        else:
            reasons.append("minor damage: low structural distress")
    # Conversely, a deep severe cavity (>=8cm) in path escalates WARNING to IMMEDIATE DANGER
    elif rank == 2 and inferred_sev == "Severe" and track.class_name == "pothole":
        rank = 3
        reasons.append("deep severe cavity: escalated to IMMEDIATE DANGER")

    # 6. Confidence filter
    if track.mean_confidence < config.RISK_MIN_CONFIDENCE and rank > 1:
        reasons.append(f"low confidence ({track.mean_confidence:.2f}): limited to CAUTION")
        rank = 1

    if speed["is_default"]:
        reasons.append("speed is a DEFAULT test value")

    kinematic_score = (rank / 3.0) * 100.0
    unified_score = (0.55 * struct_score) + (0.45 * kinematic_score)
    final_level = config.RISK_LEVELS[rank]

    return _result(final_level, reasons, ttc_s=ttc, asset_risk_level=final_level, unified_score=unified_score, structural_severity=inferred_sev)


def scene_risk(tracks):
    """
    The risk the driver faces right now = highest risk among damage in our path.
    Returns (level, rank, track_or_None). Ties are decided by the shorter time-to-impact.
    """
    best = None
    for t in tracks:
        key = (t.risk_rank, -(t.ttc_s if t.ttc_s is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, t)
    if best is None or best[1].risk_rank == 0:
        return "SAFE", 0, None
    return best[1].risk_level, best[1].risk_rank, best[1]
    for t in tracks:
        key = (t.risk_rank, -(t.ttc_s if t.ttc_s is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, t)
    if best is None or best[1].risk_rank == 0:
        return "SAFE", 0, None
    return best[1].risk_level, best[1].risk_rank, best[1]
