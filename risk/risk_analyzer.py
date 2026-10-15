"""
Phase 9: risk level of a road-damage event.

Levels (lowest -> highest):  SAFE / MONITOR,  CAUTION,  WARNING,  IMMEDIATE DANGER

How the level is decided
  1. Only CONFIRMED damage that is IN OUR PATH can be a risk. Anything else is SAFE.
     (A pothole at the roadside is not a reason to warn the driver.)
  2. No risk if the car is stopped, if the distance is unknown, or if the damage is too far away.
  3. The basic level comes from the time-to-impact (TTC):
        TTC > 5 s -> SAFE      3-5 s -> CAUTION      1-3 s -> WARNING      < 1 s -> IMMEDIATE DANGER
     At high speed the TTC limits are multiplied (the driver needs more time to react).
  4. Damage type can move the level by one step (RISK_TYPE_ADJUSTMENT).
  5. Low average detection confidence limits the level to CAUTION.

Every result lists its reasons, so you can always see WHY a level was chosen.
These are prototype rules, not validated safety limits.
"""
import config


def _result(level, reasons, ttc_s=None):
    return {"level": level,
            "rank": config.RISK_LEVELS.index(level),
            "label": config.RISK_DISPLAY[level],
            "reasons": reasons,
            "ttc_s": ttc_s}


def assess_risk(track, speed):
    """track needs: confirmed, in_path, ttc_s, ttc_status, distance_m, mean_confidence, class_name
       speed needs: speed_kmh, is_stopped, is_default"""
    if not track.confirmed:
        return _result("SAFE", ["not confirmed yet"])
    if not track.in_path:
        return _result("SAFE", ["not in our path"])
    if track.ttc_status == "stopped" or speed["is_stopped"]:
        return _result("SAFE", ["vehicle stopped"])
    if track.distance_m is None or track.ttc_s is None:
        return _result("SAFE", ["distance unknown"])
    if track.distance_m > config.RISK_MAX_DISTANCE_M:
        return _result("SAFE", [f"too far to judge (~{track.distance_m:.0f} m)"])

    reasons = []

    # ---- 3. level from time-to-impact (limits grow at high speed) ----
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

    # ---- 4. damage type ----
    adjust = config.RISK_TYPE_ADJUSTMENT.get(track.class_name, 0)
    if adjust and rank >= 1:
        new_rank = max(1, min(3, rank + adjust))
        if new_rank != rank:
            reasons.append(f"{track.class_name} adjustment {adjust:+d}")
            rank = new_rank

    # ---- 5. confidence ----
    if track.mean_confidence < config.RISK_MIN_CONFIDENCE and rank > 1:
        reasons.append(f"low confidence ({track.mean_confidence:.2f}): limited to CAUTION")
        rank = 1

    if speed["is_default"]:
        reasons.append("speed is a DEFAULT test value")

    return _result(config.RISK_LEVELS[rank], reasons, ttc)


def scene_risk(tracks):
    """The risk the driver faces right now = the highest risk among the damage in our path.
    Returns (level, rank, track_or_None). Ties are decided by the shorter time-to-impact."""
    best = None
    for t in tracks:
        key = (t.risk_rank, -(t.ttc_s if t.ttc_s is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, t)
    if best is None or best[1].risk_rank == 0:
        return "SAFE", 0, None
    return best[1].risk_level, best[1].risk_rank, best[1]
