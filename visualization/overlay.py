"""Drawing helpers (Phase 2 + Phase 3)."""
import cv2
import config


def _label(frame, x1, y1, text, color, scale=0.6):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y1 - th - base - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def draw_tracks(frame, tracks, path_mode=False, show_distance=False, show_ttc=False,
                show_risk=False):
    """Road damage. Confirmed = coloured box + ID.
    Not yet confirmed = thin grey box with a 'checking 2/3' counter.
    path_mode (Phase 4): damage in our path = thick box + 'IN PATH', otherwise a thin box."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.confirmed:
            color = config.CLASS_COLORS.get(t.class_name, (0, 255, 0))
            thickness = config.BOX_THICKNESS
            text = f"ID {t.id} {t.class_name.upper()} {t.confidence:.2f}"
            if path_mode:
                thickness = config.BOX_THICKNESS + 2 if t.in_path else 2
                if t.in_path:
                    text += "  IN PATH"
            if show_distance and t.distance_m is not None:
                text += f"  ~{t.distance_m:.0f} m"          # estimated distance
            if show_ttc and t.in_path and t.ttc_s is not None:
                text += f"  TTC {t.ttc_s:.1f}s"             # time to impact (damage in our path only)
            if show_risk and t.in_path and t.risk_rank >= 1:
                text += f"  {config.RISK_DISPLAY[t.risk_level]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            _label(frame, x1, y1, text, color)
        else:
            color = (170, 170, 170)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            _label(frame, x1, y1, f"checking {t.hits}/{config.MIN_CONFIRMATION_FRAMES}", color)
    return frame


def draw_vehicles(frame, tracks):
    """Other vehicles. Displays tracking ID, estimated distance, relative closing speed, collision warning, and ANPR license plate."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.confirmed:
            risk = getattr(t, "risk_level", "SAFE")
            if risk == "IMMEDIATE_DANGER":
                color = (0, 0, 255)  # Red
                thickness = 3
            elif risk == "WARNING":
                color = (0, 140, 255)  # Orange
                thickness = 2
            elif risk == "CAUTION":
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:
                color = config.VEHICLE_COLORS.get(t.class_name, (255, 255, 255))
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label_text = f"{t.class_name.upper()} ID {t.id}"
            if getattr(t, "distance_m", None) is not None:
                label_text += f" ~{t.distance_m:.0f}m"
            if getattr(t, "closing_speed_kmh", 0) > 3:
                label_text += f" +{t.closing_speed_kmh:.0f}km/h"
            if getattr(t, "ttc_s", None) is not None and t.ttc_s < 5.0:
                label_text += f" TTC {t.ttc_s:.1f}s"
            if risk in ["WARNING", "IMMEDIATE_DANGER"]:
                label_text += " [SLOW VEHICLE!]"

            _label(frame, x1, max(y1, 20), label_text, color, 0.50)

            # Draw Plate Bounding Box if detected on vehicle
            plate_box = getattr(t, "plate_bbox", None)
            if plate_box:
                px1, py1, px2, py2 = plate_box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

            # Draw License Plate Tag Badge
            plate_str = getattr(t, "plate_text", None)
            if plate_str and plate_str != "UNREADABLE / LOW CONF":
                tag_str = f"PLATE: {plate_str}"
                (tw, th), base = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                badge_y = min(frame.shape[0] - 8, y2 + th + base + 8)
                cv2.rectangle(frame, (x1, badge_y - th - 6), (x1 + tw + 10, badge_y + 4), (15, 15, 15), -1)
                cv2.rectangle(frame, (x1, badge_y - th - 6), (x1 + tw + 10, badge_y + 4), (0, 255, 255), 1)
                cv2.putText(frame, tag_str, (x1 + 5, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 1)
    return frame


def draw_path(frame, polygon, danger, color=None):
    """Draws our projected path as a see-through shape: green = clear, red = damage in path.
    Phase 9: pass color to show the risk level instead."""
    if color is None:
        color = (0, 0, 255) if danger else (0, 200, 0)
    layer = frame.copy()
    cv2.fillPoly(layer, [polygon], color)
    cv2.addWeighted(layer, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)
    return frame


def draw_horizon(frame, row):
    """Dashed line showing the horizon the distance estimator assumes."""
    y = int(row)
    if 0 <= y < frame.shape[0]:
        for x in range(0, frame.shape[1], 24):
            cv2.line(frame, (x, y), (x + 12, y), (255, 255, 0), 1)
        cv2.putText(frame, "assumed horizon", (frame.shape[1] - 190, max(y - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame


def draw_banner(frame, lines):
    """Small text panel in the top-left corner (width adjusts to the text).
    A line is either a text, or a pair (text, colour) - for example to highlight a test value."""
    items = [(ln, (255, 255, 255)) if isinstance(ln, str) else ln for ln in lines]
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for t, _ in items]
    cv2.rectangle(frame, (0, 0), (max(widths) + 20, 12 + 26 * len(items)), (0, 0, 0), -1)
    for i, (text, color) in enumerate(items):
        cv2.putText(frame, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame
