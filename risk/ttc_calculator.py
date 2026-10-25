"""
Phase 8: time-to-impact (TTC).

    TTC = distance / speed

Example:  distance = 20 m,  speed = 36 km/h
          36 / 3.6 = 10 m/s      ->      TTC = 20 / 10 = 2 seconds

Safety rules:
  * speed zero (or below MIN_MOVING_SPEED_KMH)  -> TTC is "not available" (no division by zero)
  * distance unknown                            -> TTC is "not available"

TTC is only as good as its inputs: the distance is an ESTIMATE (Phase 5) and the speed may be
a DEFAULT test value (Phase 7). It also assumes the car keeps its speed and the damage is fixed.
"""
import config


def calculate_ttc(distance_m, speed_mps):
    """Returns {"ttc_s": number or None, "status": "ok" | "stopped" | "no_distance"}."""
    if distance_m is None or distance_m < 0:
        return {"ttc_s": None, "status": "no_distance"}
    if speed_mps is None or speed_mps < config.MIN_MOVING_SPEED_KMH / 3.6:
        return {"ttc_s": None, "status": "stopped"}          # also protects against dividing by zero
    effective_distance = max(distance_m - config.CAMERA_TO_FRONT_M, 0.0)
    return {"ttc_s": effective_distance / speed_mps, "status": "ok"}


def format_ttc(ttc_s, status="ok"):
    """Text for the screen: '1.8 s', '>60 s', or 'N/A (...)'."""
    if ttc_s is None:
        return "N/A (vehicle stopped)" if status == "stopped" else "N/A (distance unknown)"
    if ttc_s > config.TTC_FAR_S:
        return f">{config.TTC_FAR_S:.0f} s"
    return f"{ttc_s:.1f} s"


def calculate_lead_vehicle_ttc(distance_m, ego_speed_mps, lead_speed_mps=None, closing_rate_mps=None):
    """
    Computes Forward Collision Warning (FCW) & Relative Speed TTC.
    Handles the case where:
      - The front car's speed is weak/slow (v_lead)
      - The back/ego car's speed is high (v_ego)
      - Relative closing velocity: v_rel = v_ego - v_lead

    Safety Thresholds:
      - TTC < 1.0s: IMMEDIATE DANGER (High-speed impact imminent!)
      - TTC 1.0s - 2.0s: WARNING (Brake now!)
      - TTC 2.0s - 3.5s: CAUTION (Slow lead vehicle ahead)
      - Closing <= 0 or TTC > 3.5s: SAFE
    """
    if distance_m is None:
        return {
            "ttc_s": None,
            "closing_speed_kmh": 0.0,
            "lead_speed_kmh": 0.0,
            "risk_level": "SAFE",
            "reason": "Distance unknown",
            "status": "no_distance"
        }

    # If closing rate is directly provided (from tracking delta-d / delta-t):
    if closing_rate_mps is not None:
        v_rel = closing_rate_mps
        v_lead = max(ego_speed_mps - v_rel, 0.0) if ego_speed_mps is not None else 0.0
    elif lead_speed_mps is not None and ego_speed_mps is not None:
        v_lead = max(lead_speed_mps, 0.0)
        v_rel = max(ego_speed_mps - v_lead, 0.0)
    else:
        # Default assumption: lead car is moving slowly or stopped relative to ego
        v_rel = ego_speed_mps if ego_speed_mps is not None else 0.0
        v_lead = 0.0

    closing_speed_kmh = max(v_rel * 3.6, 0.0)
    lead_speed_kmh = max(v_lead * 3.6, 0.0)
    ego_speed_kmh = (ego_speed_mps or 0.0) * 3.6

    # If our car is slower than front car or not closing in:
    if v_rel <= 0.3:  # less than ~1 km/h closing
        return {
            "ttc_s": None,
            "closing_speed_kmh": round(closing_speed_kmh, 1),
            "lead_speed_kmh": round(lead_speed_kmh, 1),
            "risk_level": "SAFE",
            "reason": "Vehicle maintaining distance or pulling away",
            "status": "separating"
        }

    effective_distance = max(distance_m - config.CAMERA_TO_FRONT_M, 0.0)
    ttc_s = effective_distance / v_rel if v_rel > 0 else 999.0

    # Risk evaluation based on closing speed and time-to-collision
    if ttc_s < 1.0 or (distance_m < 8.0 and closing_speed_kmh > 20):
        level = "IMMEDIATE_DANGER"
        reason = f"CRITICAL BRAKE! Front car slow ({lead_speed_kmh:.0f} km/h), Closing fast (+{closing_speed_kmh:.0f} km/h, TTC {ttc_s:.1f}s)"
    elif ttc_s < 2.0 or (distance_m < 15.0 and closing_speed_kmh > 25):
        level = "WARNING"
        reason = f"BRAKE WARNING: High closing speed (+{closing_speed_kmh:.0f} km/h, TTC {ttc_s:.1f}s)"
    elif ttc_s < 3.5 or (distance_m < 25.0 and closing_speed_kmh > 15):
        level = "CAUTION"
        reason = f"CAUTION: Front car speed weak ({lead_speed_kmh:.0f} km/h vs {ego_speed_kmh:.0f} km/h)"
    else:
        level = "SAFE"
        reason = "Following lead vehicle with safe headway"

    return {
        "ttc_s": round(ttc_s, 2),
        "closing_speed_kmh": round(closing_speed_kmh, 1),
        "lead_speed_kmh": round(lead_speed_kmh, 1),
        "risk_level": level,
        "reason": reason,
        "status": "closing"
    }

