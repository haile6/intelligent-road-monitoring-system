"""
Phase 10 & Audio Integration: Driver Warning with Voice Alerts & ADAS Chimes.
1. Live Warning: Plays non-blocking auditory collision hazard beeps through system speaker.
2. Timeline Audio Track Generator: Synthesizes synchronized ADAS chimes and spoken voice alerts
   and merges them directly into the output video as an AAC audio track.
"""
import os
import sys
import threading
import time
import wave
import numpy as np

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Sound rate limiter (do not beep more than once every N seconds in live speaker)
_last_beep_time = 0
_beep_lock = threading.Lock()

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_clips")


def play_warning_sound(risk_level):
    """Play risk-appropriate chime in background thread (live speaker)."""
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
                winsound.Beep(1800, 180)
                time.sleep(0.05)
                winsound.Beep(1800, 220)
            elif "WARNING" in risk_level.upper():
                winsound.Beep(1200, 250)
            elif "CAUTION" in risk_level.upper():
                winsound.Beep(850, 180)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


class ADASAudioTrackGenerator:
    """
    Synthesizes a synchronized ADAS warning audio track with vehicle alert chimes
    and spoken voice announcements across the video timeline, then muxes it into MP4.
    """
    def __init__(self, duration_s, sample_rate=22050):
        self.duration_s = max(1.0, float(duration_s))
        self.sample_rate = sample_rate
        total_samples = int((self.duration_s + 2.0) * self.sample_rate)
        self.track = np.zeros(total_samples, dtype=np.float32)
        self.last_voice_time = -999.0
        self.last_tone_time = -999.0
        self.events_logged = 0

    def add_tone(self, t_start, freq=1200, duration=0.2, volume=0.55):
        """Generate smooth sine wave chime with attack/decay envelope."""
        if t_start < 0 or t_start >= self.duration_s + 1.0:
            return
        idx_start = int(t_start * self.sample_rate)
        n_samples = int(duration * self.sample_rate)
        if idx_start + n_samples > len(self.track):
            n_samples = len(self.track) - idx_start

        if n_samples <= 0:
            return

        t = np.linspace(0, duration, n_samples, endpoint=False)
        sine = np.sin(2 * np.pi * freq * t)
        # Apply smooth cosine attack/decay envelope to prevent audio pops
        env = np.ones(n_samples, dtype=np.float32)
        fade_len = min(int(0.02 * self.sample_rate), n_samples // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            env[:fade_len] = fade_in
            env[-fade_len:] = fade_out

        tone = (sine * env * volume).astype(np.float32)
        self.track[idx_start:idx_start + n_samples] += tone

    def add_voice(self, t_start, clip_name, volume=0.90):
        """Load and mix a pre-synthesized spoken voice alert WAV clip."""
        # Enforce minimum 1.8s cooldown between spoken voice messages to prevent audio overlapping
        if t_start - self.last_voice_time < 1.8:
            return False
        
        clip_path = os.path.join(VOICE_DIR, f"{clip_name}.wav")
        if not os.path.exists(clip_path):
            return False

        try:
            with wave.open(clip_path, "rb") as wf:
                nchannels = wf.getnchannels()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                raw_bytes = wf.readframes(nframes)

            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if nchannels > 1:
                audio_data = audio_data.reshape(-1, nchannels)[:, 0]

            # Resample if sample rates differ
            if framerate != self.sample_rate:
                from scipy import signal
                num_target = int(len(audio_data) * self.sample_rate / framerate)
                audio_data = signal.resample(audio_data, num_target)

            idx_start = int(t_start * self.sample_rate)
            n_samples = len(audio_data)
            if idx_start + n_samples > len(self.track):
                n_samples = len(self.track) - idx_start
                audio_data = audio_data[:n_samples]

            if n_samples > 0:
                self.track[idx_start:idx_start + n_samples] += (audio_data * volume)
                self.last_voice_time = t_start
                return True
        except Exception as e:
            return False
        return False

    def add_event(self, timestamp, event_type):
        """Add appropriate auditory chimes and voice announcements for ADAS events."""
        t = max(0.0, float(timestamp))
        event = event_type.upper()

        if "IMMEDIATE" in event or "DANGER" in event:
            # Urgent high-pitch double tone + urgent emergency voice
            self.add_tone(t, 1800, 0.12, volume=0.7)
            self.add_tone(t + 0.15, 2200, 0.18, volume=0.75)
            self.add_voice(t + 0.35, "immediate_danger", volume=0.95)
            self.events_logged += 1

        elif "LEAD" in event or "FCW" in event:
            # Rapid closing collision chime + lead vehicle voice
            self.add_tone(t, 1500, 0.12, volume=0.65)
            self.add_tone(t + 0.15, 1500, 0.15, volume=0.65)
            self.add_voice(t + 0.35, "lead_vehicle", volume=0.90)
            self.events_logged += 1

        elif "WARNING" in event:
            # Warning tone + pothole warning voice
            self.add_tone(t, 1200, 0.20, volume=0.6)
            self.add_voice(t + 0.25, "pothole_warning", volume=0.90)
            self.events_logged += 1

        elif "CAUTION" in event:
            # Soft caution chime + road hazard voice
            self.add_tone(t, 850, 0.18, volume=0.5)
            self.add_voice(t + 0.22, "road_hazard", volume=0.85)
            self.events_logged += 1

        elif "ANPR" in event or "PLATE" in event:
            # Modern quick scan blip
            self.add_tone(t, 1400, 0.08, volume=0.45)

    def save_wav(self, output_wav_path):
        """Export normalized 16-bit PCM WAV audio file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
        # Peak normalization to prevent clipping distortion
        peak = np.max(np.abs(self.track))
        if peak > 0.95:
            norm_track = (self.track / peak) * 0.95
        else:
            norm_track = self.track

        pcm16 = (norm_track * 32767.0).astype(np.int16)
        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())

        return output_wav_path

    def merge_into_video(self, video_path, output_path=None):
        """
        Merge the synthesized audio track into the video container using ffmpeg.
        Outputs web-standard H.264 video with AAC stereo audio track and +faststart.
        """
        import subprocess
        import imageio_ffmpeg

        if not os.path.exists(video_path):
            return False

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_wav = video_path.replace(".mp4", "_temp_audio.wav")
        self.save_wav(temp_wav)

        target_out = output_path if output_path else video_path.replace(".mp4", "_with_audio.mp4")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-i", temp_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            target_out
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(target_out) and os.path.getsize(target_out) > 1000:
                if output_path is None:
                    # In-place replacement
                    os.replace(target_out, video_path)
                return True
        except Exception as e:
            print(f"[*] Audio-video muxing notice: {e}")
            return False
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

        return False
