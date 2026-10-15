"""
Phase 10: Driver Warning with Audio Chime / Beep.
Plays auditory collision hazard warnings in a non-blocking background thread
when an in-path hazard reaches WARNING or IMMEDIATE DANGER.
"""
import threading
import time

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Sound rate limiter (do not beep more than once every N seconds)
_last_beep_time = 0
_beep_lock = threading.Lock()


def play_warning_sound(risk_level):
    """Play risk-appropriate chime in background thread."""
    global _last_beep_time
    now = time.time()
    
    # Cooldown of 1.2s to prevent audio clutter
    with _beep_lock:
        if now - _last_beep_time < 1.2:
            return
        _last_beep_time = now

    def _worker():
        if not HAS_WINSOUND:
            print("\a", end="", flush=True)  # ASCII bell fallback
            return
        try:
            if "DANGER" in risk_level.upper() or "IMMEDIATE" in risk_level.upper():
                # Urgent double high-frequency beep for Immediate Danger
                winsound.Beep(1800, 180)
                time.sleep(0.05)
                winsound.Beep(1800, 220)
            elif "WARNING" in risk_level.upper():
                # Single medium-frequency tone for Warning
                winsound.Beep(1200, 250)
            elif "CAUTION" in risk_level.upper():
                winsound.Beep(850, 180)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
