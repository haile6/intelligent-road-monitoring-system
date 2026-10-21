"""
Priority Weight Score for Road Damage Incidents
================================================
Computes a numeric score (0–100) for each detected road damage incident.
This weight is designed for use by responsible bodies such as:
  - Ethiopian Roads Authority (ERA)
  - Addis Ababa City Roads Authority (AACRA)
  - Regional road maintenance departments

A higher weight = fix this FIRST.

Weight Formula
--------------
The score is composed of five factors, each contributing a sub-score:

  1. RISK LEVEL          (0–40 pts)  — Most important: how dangerous is it RIGHT NOW?
  2. DAMAGE SEVERITY     (0–20 pts)  — How bad is the physical damage (size, depth)?
  3. TRAFFIC SPEED       (0–20 pts)  — High-speed roads = more dangerous = higher priority
  4. DETECTION CONF.     (0–10 pts)  — How certain are we about the detection?
  5. DAMAGE TYPE         (0–10 pts)  — Potholes are more dangerous than cracks

Final Score (0–100) -> Priority Label:
  80-100  ->  CRITICAL      (fix within 24 hours)
  60-79   ->  HIGH          (fix within 1 week)
  40-59   ->  MEDIUM        (fix within 1 month)
  20-39   ->  LOW           (schedule for next maintenance cycle)
   0-19   ->  MONITOR       (log and monitor only)
"""


# ---------------------------------------------------------------------------
# Sub-score tables
# ---------------------------------------------------------------------------

RISK_LEVEL_SCORE = {
    "IMMEDIATE_DANGER": 40,
    "WARNING":          28,
    "CAUTION":          14,
    "SAFE":              0,
    "SAFE / MONITOR":    0,
}

SEVERITY_SCORE = {
    "Severe": 20,
    "High":   14,
    "Medium":  8,
    "Low":     3,
    "":        0,
}

DAMAGE_TYPE_SCORE = {
    "pothole":         10,
    "alligator_crack":  6,
}

# Priority label thresholds (score >= threshold -> label)
PRIORITY_THRESHOLDS = [
    (80, "Critical"),
    (60, "High"),
    (40, "Medium"),
    (20, "Low"),
    (0,  "Monitor"),
]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_priority_weight(risk_level, severity, speed_kmh, confidence, damage_type):
    """
    Compute priority weight score (0-100) for a single road damage incident.

    Parameters
    ----------
    risk_level   : str   -- one of IMMEDIATE_DANGER / WARNING / CAUTION / SAFE
    severity     : str   -- one of Severe / High / Medium / Low
    speed_kmh    : float -- vehicle speed at detection time (km/h); None if unknown
    confidence   : float -- model detection confidence 0.0-1.0; None if unknown
    damage_type  : str   -- 'pothole' or 'alligator_crack'

    Returns
    -------
    dict with keys:
        weight          int   0-100
        priority_label  str   'Critical' / 'High' / 'Medium' / 'Low' / 'Monitor'
        breakdown       dict  sub-scores for each factor
        explanation     str   human-readable summary for reports
    """
    # 1. Risk level score (0-40)
    risk_key = (risk_level or "SAFE").strip().upper().replace(" ", "_")
    if "IMMEDIATE" in risk_key or "DANGER" in risk_key:
        risk_key = "IMMEDIATE_DANGER"
    elif "WARNING" in risk_key:
        risk_key = "WARNING"
    elif "CAUTION" in risk_key:
        risk_key = "CAUTION"
    else:
        risk_key = "SAFE"
    s_risk = RISK_LEVEL_SCORE.get(risk_key, 0)

    # 2. Severity score (0-20)
    s_severity = SEVERITY_SCORE.get((severity or "").strip().capitalize(), 0)

    # 3. Speed score (0-20)
    #    Scale: 0 km/h -> 0 pts, >= 120 km/h -> 20 pts (linear capped)
    if speed_kmh is not None and speed_kmh > 0:
        s_speed = round(min(speed_kmh / 120.0, 1.0) * 20)
    else:
        s_speed = 5   # unknown speed -> moderate contribution

    # 4. Confidence score (0-10)
    #    High confidence = we are sure of the damage = full score
    if confidence is not None:
        s_conf = round(float(confidence) * 10)
    else:
        s_conf = 5   # unknown confidence -> moderate

    # 5. Damage type score (0-10)
    s_type = DAMAGE_TYPE_SCORE.get((damage_type or "").strip().lower(), 4)

    # Total weight (0-100)
    weight = s_risk + s_severity + s_speed + s_conf + s_type
    weight = max(0, min(100, weight))

    # Priority label
    priority_label = "Monitor"
    for threshold, label in PRIORITY_THRESHOLDS:
        if weight >= threshold:
            priority_label = label
            break

    # Human-readable explanation for official reports
    speed_str = "{:.0f} km/h".format(speed_kmh) if speed_kmh is not None else "unknown"
    conf_str  = "{:.0f}%".format(confidence * 100) if confidence is not None else "unknown"
    explanation = (
        "Priority weight {}/100 -> {}. "
        "Risk={} (+{}pts), Severity={} (+{}pts), "
        "Speed={} (+{}pts), Confidence={} (+{}pts), "
        "Type={} (+{}pts).".format(
            weight, priority_label,
            risk_key, s_risk,
            severity or "N/A", s_severity,
            speed_str, s_speed,
            conf_str, s_conf,
            damage_type or "N/A", s_type,
        )
    )

    return {
        "weight":         weight,
        "priority_label": priority_label,
        "breakdown": {
            "risk_level":  s_risk,
            "severity":    s_severity,
            "speed":       s_speed,
            "confidence":  s_conf,
            "damage_type": s_type,
        },
        "explanation": explanation,
    }


def compute_location_priority(incidents):
    """
    Group incidents by GPS location cluster and compute an aggregated
    location-level priority score. Useful for the responsible body to
    identify the worst road *segments* rather than individual incidents.

    incidents: list of dicts, each should have keys:
        latitude, longitude, priority_weight  (all may be None)

    Returns list of cluster dicts sorted by descending aggregate_weight:
        {
            'lat': float, 'lon': float,
            'incident_count': int,
            'max_weight': int,
            'aggregate_weight': int,   <- use this for ranking road segments
            'priority_label': str,
        }
    """
    from collections import defaultdict

    # Simple grid clustering: round lat/lon to 4 decimal places (~11 m grid)
    clusters = defaultdict(list)
    for inc in incidents:
        lat = inc.get("latitude") or inc.get("damage_latitude")
        lon = inc.get("longitude") or inc.get("damage_longitude")
        w   = inc.get("priority_weight") or 0
        if lat and lon:
            key = (round(float(lat), 4), round(float(lon), 4))
            clusters[key].append(w)

    result = []
    for (lat, lon), weights in clusters.items():
        max_w  = max(weights)
        # Aggregate: max score + repeat bonus for multiple detections (up to +20)
        repeat_bonus = min(len(weights) * 3, 20)
        agg_w = min(max_w + repeat_bonus, 100)
        label = "Monitor"
        for threshold, lbl in PRIORITY_THRESHOLDS:
            if agg_w >= threshold:
                label = lbl
                break
        result.append({
            "lat":              lat,
            "lon":              lon,
            "incident_count":   len(weights),
            "max_weight":       max_w,
            "aggregate_weight": agg_w,
            "priority_label":   label,
        })

    return sorted(result, key=lambda x: x["aggregate_weight"], reverse=True)
