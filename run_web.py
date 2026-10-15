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
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 65)
    print("      ROAD MONITORING SYSTEM - WEB MANAGEMENT PANEL")
    print("=" * 65)
    print(f"[*] Initializing SQLite Database...")
    db_manager.init_db()

    # Check if port is in use and auto-select next available port
    import socket
    def is_port_in_use(port, host):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0

    target_port = args.port
    if is_port_in_use(target_port, args.host):
        print(f"[!] Notice: Port {target_port} is already in use by another running app.")
        for p in [5050, 5001, 8080, 8000, 8888]:
            if not is_port_in_use(p, args.host):
                target_port = p
                break
        print(f"[+] Automatically assigned available port: {target_port}")

    stats = db_manager.get_dashboard_stats()
    print(f"[*] Loaded Incidents: {stats['total_incidents']} (Immediate Danger: {stats['immediate_danger']}, Warning: {stats['warning_count']})")
    print(f"[*] Loaded Plate Readings: {stats['total_plates']}")
    print(f"[*] Web Panel URL: http://{args.host}:{target_port}")
    print("=" * 65)
    print("Press CTRL+C to stop the server.\n")

    app.run(host=args.host, port=target_port, debug=args.debug)

if __name__ == "__main__":
    main()
