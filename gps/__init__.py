"""
GPS telemetry, synchronization, and geospatial mapping package.
"""
from gps.gps_reader import load_gps_csv, describe
from gps.gps_sync import GpsSync
from gps.speed_provider import SpeedProvider
from gps.folium_map import generate_folium_map
from gps.geo import destination_point, haversine_distance

__all__ = [
    "load_gps_csv",
    "describe",
    "GpsSync",
    "SpeedProvider",
    "generate_folium_map",
    "destination_point",
    "haversine_distance",
]
