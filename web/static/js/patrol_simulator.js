/**
 * RoadEye™ Virtual Patrol Playback & Radar Proximity Simulator
 * -------------------------------------------------------------
 * Powers real-time interactive vehicle patrol animation along GPS trajectories,
 * sub-second geodesic interpolation, live speedometer HUD, compass bearing calculation,
 * and proximity radar ping alerts when approaching detected pavement hazards.
 */

class PatrolSimulator {
    constructor(map, options = {}) {
        this.map = map;
        this.options = Object.assign({
            speedMultiplier: 1.0,
            autoPan: true,
            proximityThresholdMeters: 45.0,
            frameRateFps: 40,
            onTelemetryUpdate: null,
            onHazardDetected: null
        }, options);

        this.rawWaypoints = [];
        this.interpolatedPath = [];
        this.hazards = [];
        this.currentIndex = 0;
        this.isPlaying = false;
        this.timer = null;
        this.vehicleMarker = null;
        this.traveledPolyline = null;
        this.totalDistanceM = 0;
        this.cumulativeDistances = [];
        this.detectedHazardIds = new Set();
        this.activePingMarkers = new Map();

        this._initVehicleIcon();
    }

    /**
     * Build custom Leaflet vector vehicle marker with dynamic 360° rotation & headlight beams.
     */
    _initVehicleIcon() {
        this.vehicleIcon = L.divIcon({
            className: 'patrol-vehicle-container',
            html: `
                <div class="patrol-vehicle-wrapper" id="patrolVehicleWrapper">
                    <div class="patrol-headlight-beam"></div>
                    <div class="patrol-car-body">
                        <div class="patrol-roof-light"></div>
                        <i class="bi bi-car-front-fill patrol-car-svg"></i>
                    </div>
                    <div class="patrol-radar-pulse"></div>
                </div>
            `,
            iconSize: [44, 44],
            iconAnchor: [22, 22]
        });
    }

    /**
     * Load GPS points and hazard incident markers into the simulation engine.
     */
    loadTrajectory(gpsPoints, hazardIncidents = []) {
        if (!gpsPoints || gpsPoints.length < 2) {
            console.warn("[PatrolSimulator] Insufficient GPS points for trajectory.");
            return;
        }

        this.rawWaypoints = gpsPoints.map(p => ({
            lat: parseFloat(p.latitude || p.lat),
            lon: parseFloat(p.longitude || p.lon),
            speed: parseFloat(p.speed_kmh || p.speed || 36.0),
            timestamp: p.timestamp || ''
        })).filter(p => !isNaN(p.lat) && !isNaN(p.lon));

        this.hazards = hazardIncidents.map(h => ({
            id: h.id,
            track_id: h.track_id || h.id,
            lat: parseFloat(h.damage_latitude || h.latitude || 0),
            lon: parseFloat(h.damage_longitude || h.longitude || 0),
            type: h.damage_type || 'pothole',
            risk: (h.max_risk_level || 'medium').toLowerCase(),
            distance_m: parseFloat(h.distance_m || 0),
            ttc_seconds: parseFloat(h.ttc_seconds || 0),
            markerRef: null
        })).filter(h => h.lat !== 0 && h.lon !== 0);

        this._interpolateTrajectory();
        this.reset();
    }

    /**
     * Interpolate raw waypoints into smooth sub-meter simulation steps.
     */
    _interpolateTrajectory() {
        this.interpolatedPath = [];
        this.cumulativeDistances = [0];
        let runningDist = 0;

        for (let i = 0; i < this.rawWaypoints.length - 1; i++) {
            const p1 = this.rawWaypoints[i];
            const p2 = this.rawWaypoints[i + 1];

            const segmentDist = this._haversine(p1.lat, p1.lon, p2.lat, p2.lon);
            const bearing = this._calculateBearing(p1.lat, p1.lon, p2.lat, p2.lon);

            // Interpolate roughly every 1.5 to 2.5 meters for fluid 40fps playback
            const steps = Math.max(8, Math.floor(segmentDist / 2.0));

            for (let s = 0; s < steps; s++) {
                const ratio = s / steps;
                const curLat = p1.lat + (p2.lat - p1.lat) * ratio;
                const curLon = p1.lon + (p2.lon - p1.lon) * ratio;
                const curSpeed = p1.speed + (p2.speed - p1.speed) * ratio;

                const stepDist = (segmentDist / steps);
                runningDist += stepDist;

                this.interpolatedPath.push({
                    lat: curLat,
                    lon: curLon,
                    bearing: bearing,
                    speed: Math.max(15, curSpeed),
                    distanceM: runningDist
                });
                this.cumulativeDistances.push(runningDist);
            }
        }

        // Push final point
        const lastPt = this.rawWaypoints[this.rawWaypoints.length - 1];
        const lastBearing = this.interpolatedPath.length > 0 ? this.interpolatedPath[this.interpolatedPath.length - 1].bearing : 0;
        this.interpolatedPath.push({
            lat: lastPt.lat,
            lon: lastPt.lon,
            bearing: lastBearing,
            speed: lastPt.speed || 30.0,
            distanceM: runningDist
        });

        this.totalDistanceM = runningDist;
    }

    /**
     * Start or resume simulation playback.
     */
    play() {
        if (this.interpolatedPath.length === 0) return;
        if (this.currentIndex >= this.interpolatedPath.length - 1) {
            this.currentIndex = 0;
            this.detectedHazardIds.clear();
        }

        this.isPlaying = true;
        this._updatePlayButtonUI(true);

        const intervalMs = Math.max(12, Math.floor(1000 / (this.options.frameRateFps * this.options.speedMultiplier)));
        clearInterval(this.timer);
        this.timer = setInterval(() => this._tick(), intervalMs);
    }

    /**
     * Pause simulation playback.
     */
    pause() {
        this.isPlaying = false;
        clearInterval(this.timer);
        this._updatePlayButtonUI(false);
    }

    /**
     * Toggle Play / Pause.
     */
    togglePlay() {
        if (this.isPlaying) this.pause();
        else this.play();
    }

    /**
     * Adjust playback speed multiplier (1x, 2x, 4x, 8x).
     */
    setSpeed(multiplier) {
        this.options.speedMultiplier = parseFloat(multiplier) || 1.0;
        if (this.isPlaying) {
            this.play(); // Restart interval with new speed
        }
    }

    /**
     * Seek to a specific normalized progress (0.0 to 1.0).
     */
    seek(progress) {
        if (this.interpolatedPath.length === 0) return;
        const clamped = Math.max(0, Math.min(1.0, progress));
        this.currentIndex = Math.floor(clamped * (this.interpolatedPath.length - 1));
        this._updateStep(this.interpolatedPath[this.currentIndex]);
    }

    /**
     * Reset simulation back to initial waypoint.
     */
    reset() {
        this.pause();
        this.currentIndex = 0;
        this.detectedHazardIds.clear();

        if (this.interpolatedPath.length > 0) {
            const firstPt = this.interpolatedPath[0];

            if (!this.vehicleMarker) {
                this.vehicleMarker = L.marker([firstPt.lat, firstPt.lon], {
                    icon: this.vehicleIcon,
                    zIndexOffset: 1000
                }).addTo(this.map);
            } else {
                this.vehicleMarker.setLatLng([firstPt.lat, firstPt.lon]);
            }

            if (!this.traveledPolyline) {
                this.traveledPolyline = L.polyline([], {
                    color: '#10B981',
                    weight: 4.5,
                    opacity: 0.90,
                    dashArray: null
                }).addTo(this.map);
            } else {
                this.traveledPolyline.setLatLngs([]);
            }

            this._updateStep(firstPt);
        }
    }

    /**
     * Core animation frame tick.
     */
    _tick() {
        if (this.currentIndex >= this.interpolatedPath.length - 1) {
            this.pause();
            return;
        }

        this.currentIndex++;
        const currentPt = this.interpolatedPath[this.currentIndex];
        this._updateStep(currentPt);
        this._checkProximityHazards(currentPt);
    }

    /**
     * Update vehicle marker position, heading rotation, traveled trail, and telemetry HUD.
     */
    _updateStep(pt) {
        if (!pt) return;

        // 1. Move vehicle marker
        if (this.vehicleMarker) {
            this.vehicleMarker.setLatLng([pt.lat, pt.lon]);

            // Rotate car wrapper element matching bearing
            const wrapper = document.getElementById('patrolVehicleWrapper');
            if (wrapper) {
                wrapper.style.transform = `rotate(${Math.round(pt.bearing)}deg)`;
            }
        }

        // 2. Update traveled green polyline trail
        if (this.traveledPolyline) {
            const traveledCoords = this.interpolatedPath.slice(0, this.currentIndex + 1).map(p => [p.lat, p.lon]);
            this.traveledPolyline.setLatLngs(traveledCoords);
        }

        // 3. Auto-pan camera if enabled
        if (this.options.autoPan && this.map) {
            if (this.currentIndex % 15 === 0) {
                this.map.panTo([pt.lat, pt.lon], { animate: true, duration: 0.25 });
            }
        }

        // 4. Compute telemetry
        const progress = this.interpolatedPath.length > 1 ? (this.currentIndex / (this.interpolatedPath.length - 1)) : 0;
        const telemetry = {
            lat: pt.lat,
            lon: pt.lon,
            speedKmh: Math.round(pt.speed),
            bearing: Math.round(pt.bearing),
            cardinal: this._bearingToCardinal(pt.bearing),
            distanceTraveledM: Math.round(pt.distanceM),
            totalDistanceM: Math.round(this.totalDistanceM),
            progress: progress,
            hazardCount: this.detectedHazardIds.size,
            isPlaying: this.isPlaying
        };

        if (typeof this.options.onTelemetryUpdate === 'function') {
            this.options.onTelemetryUpdate(telemetry);
        }
    }

    /**
     * Check for hazards within proximity threshold and trigger radar pulse.
     */
    _checkProximityHazards(currentPt) {
        if (!this.hazards || this.hazards.length === 0) return;

        this.hazards.forEach(hazard => {
            const dist = this._haversine(currentPt.lat, currentPt.lon, hazard.lat, hazard.lon);

            if (dist <= this.options.proximityThresholdMeters) {
                if (!this.detectedHazardIds.has(hazard.id)) {
                    this.detectedHazardIds.add(hazard.id);
                    this._triggerRadarPing(hazard, dist);
                }
            }
        });
    }

    /**
     * Trigger visual radar ping ripple animation on map at hazard coordinate.
     */
    _triggerRadarPing(hazard, distanceM) {
        const pingColor = hazard.risk === 'high' ? '#EF4444' : (hazard.risk === 'medium' ? '#F59E0B' : '#10B981');
        
        // Create transient radar ripple marker
        const rippleIcon = L.divIcon({
            className: 'patrol-radar-ping-icon',
            html: `<div class="radar-ping-ripple" style="border-color: ${pingColor}; background: ${pingColor}33;"></div>`,
            iconSize: [60, 60],
            iconAnchor: [30, 30]
        });

        const rippleMarker = L.marker([hazard.lat, hazard.lon], { icon: rippleIcon, interactive: false }).addTo(this.map);

        // Remove ripple after 3 seconds
        setTimeout(() => {
            this.map.removeLayer(rippleMarker);
        }, 3200);

        // Notify HUD callback with exact defect telemetry
        if (typeof this.options.onHazardDetected === 'function') {
            this.options.onHazardDetected({
                hazard: hazard,
                trackId: hazard.track_id || hazard.id,
                damageType: hazard.type || 'pothole',
                risk: hazard.risk || 'medium',
                distanceM: hazard.distance_m > 0 ? hazard.distance_m : Number(distanceM.toFixed(1)),
                ttcSeconds: hazard.ttc_seconds > 0 ? hazard.ttc_seconds : Number((distanceM / 8.33).toFixed(2)),
                lat: hazard.lat,
                lon: hazard.lon,
                totalDetected: this.detectedHazardIds.size
            });
        }
    }

    _updatePlayButtonUI(isPlaying) {
        const playBtn = document.getElementById('patrolBtnPlay');
        if (playBtn) {
            playBtn.innerHTML = isPlaying ? '<i class="bi bi-pause-fill"></i>' : '<i class="bi bi-play-fill"></i>';
            playBtn.classList.toggle('btn-warning', isPlaying);
            playBtn.classList.toggle('btn-primary', !isPlaying);
            playBtn.title = isPlaying ? 'Pause Simulation' : 'Play Simulation';
        }
    }

    _calculateBearing(lat1, lon1, lat2, lon2) {
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const y = Math.sin(dLon) * Math.cos(lat2 * Math.PI / 180);
        const x = Math.cos(lat1 * Math.PI / 180) * Math.sin(lat2 * Math.PI / 180) -
                  Math.sin(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.cos(dLon);
        const brng = Math.atan2(y, x) * 180 / Math.PI;
        return (brng + 360) % 360;
    }

    _haversine(lat1, lon1, lat2, lon2) {
        const R = 6371000; // Radius in meters
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) ** 2 +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon / 2) ** 2;
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    _bearingToCardinal(deg) {
        const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const idx = Math.round(deg / 45) % 8;
        return cardinals[idx];
    }
}

window.PatrolSimulator = PatrolSimulator;

