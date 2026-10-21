r"""
Run the Road Monitoring System Web Management Panel.

Usage:
    python run_web.py
    python run_web.py --port 8080 --host 0.0.0.0
"""
import os
import sys
import argparse

# Add current folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import db_manager
from web.app import app

def main():
    parser = argparse.ArgumentParser(description="Road Monitoring System Web Management Panel")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0 for mobile and local access)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 65)
    print("      ROAD MONITORING SYSTEM - WEB MANAGEMENT PANEL")
    print("=" * 65)
    print(f"[*] Initializing SQLite Database...")
    db_manager.init_db()

    # Detect computer's local Wi-Fi / LAN IP address
    import socket
    def get_lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    lan_ip = get_lan_ip()

    # Check if port is in use and auto-select next available port
    def is_port_in_use(port, host):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    target_port = args.port
    if is_port_in_use(target_port, args.host):
        print(f"[!] Notice: Port {target_port} is already in use by another running app.")
        for p in [8080, 8888, 9090, 8000, 7070]:
            if not is_port_in_use(p, args.host):
                target_port = p
                break
        print(f"[+] Automatically assigned available port: {target_port}")

    stats = db_manager.get_dashboard_stats()
    print(f"[*] Loaded Incidents: {stats['total_incidents']} (Immediate Danger: {stats['immediate_danger']}, Warning: {stats['warning_count']})")
    print(f"[*] Loaded Plate Readings: {stats['total_plates']}")
    print(f"[*] Desktop Browser URL: http://127.0.0.1:{target_port}")
    if lan_ip != "127.0.0.1":
        print(f"[*] Mobile Phone / LAN:  http://{lan_ip}:{target_port}  (Open on your phone)")
    print("=" * 65)
    print("Press CTRL+C to stop the server.\n")

    # Automatically launch Google Chrome / default browser to the Studio
    import threading
    import webbrowser
    def auto_open_browser():
        time.sleep(1.2)
        studio_url = f"http://127.0.0.1:{target_port}/studio"
        try:
            for path in [
                "C:/Program Files/Google/Chrome/Application/chrome.exe",
                "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe")
            ]:
                if os.path.exists(path):
                    webbrowser.get(f'"{path}" %s').open(studio_url)
                    return
        except Exception:
            pass
        webbrowser.open(studio_url)

    threading.Thread(target=auto_open_browser, daemon=True).start()

    app.run(host=args.host, port=target_port, debug=args.debug)

if __name__ == "__main__":
    main()
