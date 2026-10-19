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


def draw_subtitle_bar(frame, text, risk_level="SAFE"):
    """
    Draws an automotive ADAS subtitle banner at the bottom center of the frame.
    Supports English as well as Amharic (አማርኛ) Unicode script via system font.
    risk_level: 'SAFE', 'WARNING', or 'IMMEDIATE_DANGER'
    """
    if not text:
        return frame

    h, w = frame.shape[:2]
    risk_colors = {
        "SAFE": (0, 200, 0),              # Green
        "WARNING": (0, 140, 255),          # Amber
        "IMMEDIATE_DANGER": (0, 0, 255),   # Red
        "CAUTION": (0, 255, 255)           # Yellow
    }
    color = risk_colors.get(risk_level.upper(), (0, 200, 0))

    is_unicode = any(ord(c) > 127 for c in text)
    if is_unicode:
        try:
            import os
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            font_path = "C:/Windows/Fonts/ebrima.ttf"
            font_size = 22
            font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()

            # Measure text using PIL
            dummy_img = Image.new("RGB", (1, 1))
            d_ctx = ImageDraw.Draw(dummy_img)
            bbox = d_ctx.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            box_w = min(tw + 44, w - 40)
            box_h = th + 28
            box_x = (w - box_w) // 2
            box_y = h - box_h - 22

            # Dark HUD glass background
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

            # Glowing risk-colored border & indicator pill
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
            cv2.rectangle(frame, (box_x, box_y), (box_x + 8, box_y + box_h), color, -1)

            # Draw Amharic / Unicode text onto frame via PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            draw.text((box_x + 22, box_y + 10), text, font=font, fill=(255, 255, 255))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # Standard ASCII rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    box_w = min(tw + 40, w - 40)
    box_h = th + base + 24
    box_x = (w - box_w) // 2
    box_y = h - box_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
    cv2.rectangle(frame, (box_x, box_y), (box_x + 8, box_y + box_h), color, -1)

    text_x = box_x + 20
    text_y = box_y + th + ((box_h - th) // 2)
    cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


