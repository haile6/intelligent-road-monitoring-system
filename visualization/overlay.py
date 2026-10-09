"""Drawing helpers (Phase 2 + Phase 3)."""
import cv2
import config


def _label(frame, x1, y1, text, color, scale=0.6):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y1 - th - base - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def draw_tracks(frame, tracks):
    """Road damage. Confirmed = thick coloured box + ID.
    Not yet confirmed = thin grey box with a 'checking 2/3' counter."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.confirmed:
            color = config.CLASS_COLORS.get(t.class_name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
            _label(frame, x1, y1, f"ID {t.id} {t.class_name.upper()} {t.confidence:.2f}", color)
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


def draw_banner(frame, lines):
    """Small text panel in the top-left corner (width adjusts to the text)."""
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for t in lines]
    cv2.rectangle(frame, (0, 0), (max(widths) + 20, 12 + 26 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
    return frame
