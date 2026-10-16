"""
Generate pre-rendered audio voice alerts for ADAS warning system using Windows SAPI Speech.
"""
import os
import subprocess

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_clips")
os.makedirs(VOICE_DIR, exist_ok=True)

CLIPS = {
    "pothole_warning": "Warning! Pothole in driving path.",
    "road_hazard": "Caution! Road hazard detected ahead.",
    "immediate_danger": "Emergency! Collision warning, brake now.",
    "lead_vehicle": "Caution! Lead vehicle slowing down.",
    "anpr_scan": "Vehicle plate recognized."
}

def generate_all():
    print("[*] Generating ADAS voice alert clips...")
    for name, text in CLIPS.items():
        wav_path = os.path.join(VOICE_DIR, f"{name}.wav")
        vbs_path = os.path.join(VOICE_DIR, f"temp_{name}.vbs")
        
        # Clean path with forward slashes for VBScript
        clean_wav = wav_path.replace("\\", "/")
        vbs_content = f'''Set voice = CreateObject("SAPI.SpVoice")
Set stream = CreateObject("SAPI.SpFileStream")
stream.Open "{clean_wav}", 3
Set voice.AudioOutputStream = stream
voice.Speak "{text}"
stream.Close
'''
        with open(vbs_path, "w", encoding="utf-8") as f:
            f.write(vbs_content)

        try:
            subprocess.run(["cscript", "//nologo", vbs_path], check=True)
            if os.path.exists(wav_path):
                print(f"  + Generated {name}.wav ({os.path.getsize(wav_path)} bytes)")
        except Exception as e:
            print(f"  - Error generating {name}.wav: {e}")
        finally:
            if os.path.exists(vbs_path):
                os.remove(vbs_path)

if __name__ == "__main__":
    generate_all()
