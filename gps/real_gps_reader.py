"""
Phase 17: Real GPS Receiver Interface (NMEA 0183 / Serial USB).
Connects to hardware GPS receivers (e.g. u-blox, GlobalSat, Garmin)
over a COM port and parses live latitude, longitude, and speed in real time.
"""
import time
import threading

class RealGpsReceiver:
    def __init__(self, port="COM3", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.latest_fix = {
            "latitude": None,
            "longitude": None,
            "speed_kmh": 0.0,
            "timestamp": None,
            "fix_quality": 0,
            "satellites": 0
        }
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _parse_nmea(self, line):
        try:
            parts = line.strip().split(",")
            # $GPRMC: Recommended Minimum Specific GPS Data
            if parts[0] in ["$GPRMC", "$GNRMC"] and len(parts) >= 8:
                status = parts[2]
                if status == "A":  # Active / Valid
                    raw_lat = parts[3]
                    lat_dir = parts[4]
                    raw_lon = parts[5]
                    lon_dir = parts[6]
                    speed_knots = float(parts[7]) if parts[7] else 0.0

                    lat = float(raw_lat[:2]) + float(raw_lat[2:]) / 60.0
                    if lat_dir == "S": lat = -lat

                    lon = float(raw_lon[:3]) + float(raw_lon[3:]) / 60.0
                    if lon_dir == "W": lon = -lon

                    self.latest_fix["latitude"] = lat
                    self.latest_fix["longitude"] = lon
                    self.latest_fix["speed_kmh"] = speed_knots * 1.852
                    self.latest_fix["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    def _read_loop(self):
        try:
            import serial
            with serial.Serial(self.port, self.baudrate, timeout=1.0) as ser:
                while self.running:
                    line = ser.readline().decode("ascii", errors="replace")
                    if line:
                        self._parse_nmea(line)
        except Exception as e:
            print(f"[!] Real GPS Serial Error ({self.port}): {e}")

    def get_fix(self):
        return self.latest_fix.copy()
