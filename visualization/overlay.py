"""Drawing helpers (Phase 2 + Phase 3)."""
import cv2
import config


def _label(frame, x1, y1, text, color, scale=0.6):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y1 - th - base - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def draw_tracks(frame, tracks, path_mode=False, show_distance=False):
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            _label(frame, x1, y1, text, color)
        else:
            color = (170, 170, 170)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            _label(frame, x1, y1, f"checking {t.hits}/{config.MIN_CONFIRMATION_FRAMES}", color)
    return frame


def draw_vehicles(frame, tracks):
    """Other vehicles. Only confirmed vehicles are labelled, e.g. 'CAR ID 3'."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.confirmed:
            color = config.VEHICLE_COLORS.get(t.class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _label(frame, x1, y2 + 22, f"{t.class_name.upper()} ID {t.id}", color, 0.55)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 1)
    return frame


def draw_path(frame, polygon, danger):
    """Draws our projected path as a see-through shape: green = clear, red = damage in path."""
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
    """Small text panel in the top-left corner (width adjusts to the text)."""
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for t in lines]
    cv2.rectangle(frame, (0, 0), (max(widths) + 20, 12 + 26 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
    return frame
