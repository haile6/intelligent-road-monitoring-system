"""Drawing helpers. Phase 2: draw tracked road damage."""
import cv2
import config


def draw_tracks(frame, tracks):
    """Confirmed damage = thick coloured box + ID.
    Not yet confirmed = thin grey box with a 'checking 2/3' counter."""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        if t.confirmed:
            color = config.CLASS_COLORS.get(t.class_name, (0, 255, 0))
            thickness = config.BOX_THICKNESS
            label = f"ID {t.id} {t.class_name.upper()} {t.confidence:.2f}"
        else:
            color = (170, 170, 170)
            thickness = 1
            label = f"checking {t.hits}/{config.MIN_CONFIRMATION_FRAMES}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_banner(frame, lines):
    """Draws a small text panel in the top-left corner."""
    h = 12 + 26 * len(lines)
    cv2.rectangle(frame, (0, 0), (430, h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
    return frame
