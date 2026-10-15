"""
Phase 14: Standalone Interactive GPS Road-Damage Map using Folium.
Generates an interactive HTML map with vehicle trajectory, risk-colored pins,
and detailed inspection popups.
"""
import os
import folium
from folium.plugins import HeatMap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_folium_map(incidents, gps_points, output_html=None):
    """
    Creates a rich standalone Folium map from incident records and GPS track.
    Saves to outputs/road_damage_map.html by default.
    """
    if output_html is None:
        output_html = os.path.join(BASE_DIR, "outputs", "road_damage_map.html")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # Determine center
    center_lat, center_lon = 9.1235, 39.1235
    if gps_points:
        center_lat = gps_points[0]["lat"]
        center_lon = gps_points[0]["lon"]
    elif incidents and incidents[0].get("damage_latitude"):
        center_lat = incidents[0]["damage_latitude"]
        center_lon = incidents[0]["damage_longitude"]

    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles="OpenStreetMap"
    )

    # 1. Plot GPS trajectory polyline
    if gps_points:
        coords = [[p["lat"], p["lon"]] for p in gps_points]
        folium.PolyLine(
            coords,
            color="#2563eb",
            weight=5,
            opacity=0.8,
            dash_array="8, 6",
            tooltip="Vehicle GPS Driving Path"
        ).add_to(m)

    # 2. Add Heatmap layer
    heat_data = []
    for inc in incidents:
        lat = inc.get("damage_latitude") or inc.get("latitude")
        lon = inc.get("damage_longitude") or inc.get("longitude")
        if lat and lon:
            weight = 1.0
            risk = str(inc.get("max_risk_level", "")).upper()
            if "IMMEDIATE" in risk or "DANGER" in risk:
                weight = 3.0
            elif "WARNING" in risk:
                weight = 2.0
            heat_data.append([lat, lon, weight])

    if heat_data:
        HeatMap(heat_data, radius=18, blur=12, name="Hazard Density Heatmap").add_to(m)

    # 3. Add Incident Markers
    for inc in incidents:
        lat = inc.get("damage_latitude") or inc.get("latitude")
        lon = inc.get("damage_longitude") or inc.get("longitude")
        if not lat or not lon:
            continue

        risk = str(inc.get("max_risk_level", "")).upper()
        if "IMMEDIATE" in risk or "DANGER" in risk:
            color = "red"
            icon = "exclamation-triangle"
        elif "WARNING" in risk:
            color = "orange"
            icon = "bell"
        elif "CAUTION" in risk:
            color = "yellow"
            icon = "info-sign"
        else:
            color = "green"
            icon = "ok-sign"

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 6px 0; color: #1e293b; text-transform: capitalize;">
                {inc.get('damage_type', 'Damage')} #{inc.get('track_id', '')}
            </h4>
            <div style="font-size: 12px; line-height: 1.6; color: #475569;">
                <b>Risk:</b> <span style="color: {color}; font-weight: bold;">{inc.get('max_risk_level', 'SAFE')}</span><br>
                <b>Severity:</b> {inc.get('severity') or 'Medium'}<br>
                <b>Distance:</b> {inc.get('distance_m', '--')} m<br>
                <b>Time to Collision:</b> {inc.get('ttc_seconds') or '--'} s<br>
                <b>Speed:</b> {inc.get('speed_kmh', '--')} km/h<br>
                <b>Status:</b> {inc.get('status', 'Reported')}<br>
                <div style="margin-top: 4px; font-size: 11px; color: #94a3b8;">
                    Coords: {lat:.6f}, {lon:.6f}
                </div>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=9 if color in ["red", "orange"] else 7,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{inc.get('damage_type')} #{inc.get('track_id')} - {inc.get('max_risk_level')}"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"[*] Folium interactive map saved: {output_html}")
    return output_html
