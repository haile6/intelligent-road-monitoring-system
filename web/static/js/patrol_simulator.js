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
            audioAlerts: true,
            proximityThresholdMeters: 100.0,
            frameRateFps: 40,
            onTelemetryUpdate: null,
            onHazardDetected: null
        }, options);
        this.currentPt = null;
        this.audioCtx = null;

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
        this.radarRangeCircle = null;

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
            timestamp: p.timestamp || '',
            landmark: p.landmark || p.road_name || ''
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

        const segmentBearings = [];
        const segmentDists = [];
        for (let i = 0; i < this.rawWaypoints.length - 1; i++) {
            const p1 = this.rawWaypoints[i];
            const p2 = this.rawWaypoints[i + 1];
            segmentDists.push(this._haversine(p1.lat, p1.lon, p2.lat, p2.lon));
            segmentBearings.push(this._calculateBearing(p1.lat, p1.lon, p2.lat, p2.lon));
        }

        for (let i = 0; i < this.rawWaypoints.length - 1; i++) {
            const p1 = this.rawWaypoints[i];
            const p2 = this.rawWaypoints[i + 1];

            const segmentDist = segmentDists[i];
            const bearing = segmentBearings[i];
            const nextBearing = (i + 1 < segmentBearings.length) ? segmentBearings[i + 1] : bearing;

            // Turn angle delta between consecutive segments
            let turnAngle = Math.abs(nextBearing - bearing);
            if (turnAngle > 180) turnAngle = 360 - turnAngle;

            // Interpolate roughly every 1.5 to 2.5 meters for fluid 40fps playback
            const steps = Math.max(8, Math.floor(segmentDist / 2.0));

            for (let s = 0; s < steps; s++) {
                const ratio = s / steps;
                const curLat = p1.lat + (p2.lat - p1.lat) * ratio;
                const curLon = p1.lon + (p2.lon - p1.lon) * ratio;

                // Smooth bearing interpolation entering bend
                let curBearing = bearing;
                if (turnAngle > 5 && ratio > 0.60) {
                    const turnProgress = (ratio - 0.60) / 0.40;
                    curBearing = this._interpolateBearing(bearing, nextBearing, turnProgress);
                }

                // Realistic cornering physics: vehicle decelerates during turns, accelerates on straights
                let corneringDecel = 1.0;
                if (turnAngle > 15) {
                    const turnSeverity = Math.min(1.0, turnAngle / 90.0);
                    const decelCurve = Math.sin(ratio * Math.PI) * 0.35 * turnSeverity;
                    corneringDecel = Math.max(0.65, 1.0 - decelCurve);
                }

                // Base speed interpolated between waypoints
                const baseSpeed = p1.speed + (p2.speed - p1.speed) * ratio;
                // Subtle micro-variations (+/- 0.4 km/h) for authentic live speedometer readout
                const microNoise = Math.sin(s * 1.3) * 0.4;
                const dynamicSpeed = Math.max(12, (baseSpeed * corneringDecel) + microNoise);

                const stepDist = (segmentDist / steps);
                runningDist += stepDist;

                const curLandmark = (ratio < 0.5) ? (p1.landmark || '') : (p2.landmark || p1.landmark || '');
                this.interpolatedPath.push({
                    lat: curLat,
                    lon: curLon,
                    bearing: curBearing,
                    speed: dynamicSpeed,
                    distanceM: runningDist,
                    landmark: curLandmark
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
            distanceM: runningDist,
            landmark: lastPt.landmark || ''
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
     * Adjust radar proximity detection radius horizon (e.g. 50m, 100m, 150m).
     */
    setRadarRange(meters) {
        this.options.proximityThresholdMeters = parseFloat(meters) || 100.0;
        if (this.radarRangeCircle) {
            this.radarRangeCircle.setRadius(this.options.proximityThresholdMeters);
        }
        console.log(`[PatrolSimulator] Radar proximity threshold updated to: ${this.options.proximityThresholdMeters}m`);
    }

    /**
     * Adjust playback speed multiplier (1x, 2x, 4x, 8x).
     */
    /**
     * Adjust playback speed multiplier (0.5x, 1x, 2x, 4x, 8x).
     */
    setSpeed(multiplier) {
        this.options.speedMultiplier = Math.max(0.25, Math.min(16.0, parseFloat(multiplier) || 1.0));
        if (this.isPlaying) {
            this.play(); // Restart interval with new frame timing
        }
    }

    /**
     * Seek to a specific normalized progress (0.0 to 1.0) with bidirectional hazard state re-evaluation.
     */
    seek(progress) {
        if (this.interpolatedPath.length === 0) return;
        const clamped = Math.max(0, Math.min(1.0, progress));
        this.currentIndex = Math.floor(clamped * (this.interpolatedPath.length - 1));

        // Bidirectional hazard sync: reset later hazard pings when scrubbing backward
        this.detectedHazardIds.clear();
        this.activePingMarkers.forEach(m => {
            if (this.map && this.map.hasLayer(m)) this.map.removeLayer(m);
        });
        this.activePingMarkers.clear();

        for (let i = 0; i <= this.currentIndex; i++) {
            const stepPt = this.interpolatedPath[i];
            this.hazards.forEach(hazard => {
                const dist = this._haversine(stepPt.lat, stepPt.lon, hazard.lat, hazard.lon);
                if (dist <= this.options.proximityThresholdMeters) {
                    this.detectedHazardIds.add(hazard.id);
                }
            });
        }

        this._updateStep(this.interpolatedPath[this.currentIndex]);
    }

    /**
     * Relative scrub forward or backward by delta progress (e.g. +0.02 or -0.02).
     */
    stepRelative(deltaProgress) {
        if (this.interpolatedPath.length === 0) return;
        const curProgress = (this.interpolatedPath.length > 1) ? (this.currentIndex / (this.interpolatedPath.length - 1)) : 0;
        this.seek(curProgress + deltaProgress);
    }

    /**
     * Query waypoint metadata at arbitrary progress ratio for hover scrubber tooltips.
     */
    getWaypointMetaAtProgress(progress) {
        if (this.interpolatedPath.length === 0) return null;
        const clamped = Math.max(0, Math.min(1.0, progress));
        const idx = Math.floor(clamped * (this.interpolatedPath.length - 1));
        const pt = this.interpolatedPath[idx];
        const totalSec = Math.max(1, Math.round((this.totalDistanceM / 1000) / (35 / 3600)));
        const curSec = Math.round(totalSec * clamped);
        const curMin = String(Math.floor(curSec / 60)).padStart(2, '0');
        const curRemSec = String(curSec % 60).padStart(2, '0');

        return {
            distanceKm: (pt.distanceM / 1000).toFixed(2),
            totalKm: (this.totalDistanceM / 1000).toFixed(2),
            timeFormatted: `${curMin}:${curRemSec}`,
            progressPct: Math.round(clamped * 100),
            bearing: Math.round(pt.bearing),
            speedKmh: Math.round(pt.speed)
        };
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

            if (!this.radarRangeCircle) {
                this.radarRangeCircle = L.circle([firstPt.lat, firstPt.lon], {
                    radius: this.options.proximityThresholdMeters,
                    color: '#38BDF8',
                    fillColor: '#38BDF8',
                    fillOpacity: 0.05,
                    weight: 1.5,
                    dashArray: '4, 6',
                    interactive: false
                }).addTo(this.map);
            } else {
                this.radarRangeCircle.setLatLng([firstPt.lat, firstPt.lon]);
                this.radarRangeCircle.setRadius(this.options.proximityThresholdMeters);
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
        this.currentPt = pt;

        // 1. Move vehicle marker
        if (this.vehicleMarker) {
            this.vehicleMarker.setLatLng([pt.lat, pt.lon]);

            // Rotate car wrapper element matching bearing
            const wrapper = document.getElementById('patrolVehicleWrapper');
            if (wrapper) {
                wrapper.style.transform = `rotate(${Math.round(pt.bearing)}deg)`;
            }
        }

        // 1b. Move 100m scanning radar range circle
        if (this.radarRangeCircle) {
            this.radarRangeCircle.setLatLng([pt.lat, pt.lon]);
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

        // 4. Compute comprehensive live telemetry (Speed, Heading, Dual Odometer, ETA)
        const progress = this.interpolatedPath.length > 1 ? (this.currentIndex / (this.interpolatedPath.length - 1)) : 0;
        const currentSpeedKmh = Math.max(0, Math.round(pt.speed));
        const currentSpeedMps = currentSpeedKmh / 3.6;
        const distanceTraveledM = Math.round(pt.distanceM);
        const remainingDistanceM = Math.max(0, Math.round(this.totalDistanceM - pt.distanceM));
        const etaSeconds = currentSpeedMps > 0.5 ? Math.round(remainingDistanceM / currentSpeedMps) : 0;

        const telemetry = {
            lat: pt.lat,
            lon: pt.lon,
            speedKmh: currentSpeedKmh,
            speedMps: currentSpeedMps.toFixed(1),
            bearing: Math.round(pt.bearing),
            cardinal: this._bearingToCardinal(pt.bearing, 8),
            cardinal16: this._bearingToCardinal(pt.bearing, 16),
            landmark: pt.landmark || '',
            roadName: pt.landmark || 'Active Inspection Corridor',
            distanceTraveledM: distanceTraveledM,
            remainingDistanceM: remainingDistanceM,
            totalDistanceM: Math.round(this.totalDistanceM),
            etaSeconds: etaSeconds,
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
        const isDanger = hazard.risk.includes('danger') || hazard.risk.includes('immediate') || hazard.risk === 'high';
        const isCaution = hazard.risk.includes('caution') || hazard.risk === 'medium';
        const pingColor = isDanger ? '#EF4444' : (isCaution ? '#F59E0B' : '#10B981');
        const defectName = (hazard.type || 'Pothole').toUpperCase();
        const distFormatted = (hazard.distance_m > 0 ? hazard.distance_m : distanceM).toFixed(1);

        // 1. Create transient multi-ring concentric sonar burst marker
        const sonarIcon = L.divIcon({
            className: 'patrol-sonar-burst-icon',
            html: `
                <div class="concentric-sonar-container">
                    <div class="sonar-core-pulse" style="background: ${pingColor}; box-shadow: 0 0 16px ${pingColor};"></div>
                    <div class="sonar-ring sonar-ring-1" style="border-color: ${pingColor};"></div>
                    <div class="sonar-ring sonar-ring-2" style="border-color: ${pingColor};"></div>
                    <div class="sonar-ring sonar-ring-3" style="border-color: ${pingColor};"></div>
                    <div class="sonar-distance-pill" style="border-color: ${pingColor}; color: ${pingColor};">
                        ⚠️ ${defectName} #${hazard.track_id || hazard.id} &bull; ${distFormatted}m
                    </div>
                </div>
            `,
            iconSize: [100, 100],
            iconAnchor: [50, 50]
        });

        const sonarMarker = L.marker([hazard.lat, hazard.lon], { icon: sonarIcon, interactive: false }).addTo(this.map);

        // 2. Draw temporary forward radar acquisition laser beam from vehicle to defect
        if (this.currentPt) {
            const laserBeam = L.polyline([[this.currentPt.lat, this.currentPt.lon], [hazard.lat, hazard.lon]], {
                color: pingColor,
                weight: 2.2,
                dashArray: '5, 5',
                opacity: 0.85
            }).addTo(this.map);

            setTimeout(() => {
                if (this.map) this.map.removeLayer(laserBeam);
            }, 1800);
        }

        // 3. Play Web Audio synthesized proximity sonar beep
        this._playSonarBeep(hazard.risk);

        // Remove sonar rings after 3.2 seconds
        setTimeout(() => {
            if (this.map) this.map.removeLayer(sonarMarker);
        }, 3200);

        // 4. Notify HUD callback with exact defect telemetry
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

    /**
     * Synthesize clean, subtle acoustic proximity radar sonar ping via Web Audio API.
     */
    _playSonarBeep(risk = 'medium') {
        if (!this.options.audioAlerts) return;
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            if (!this.audioCtx) this.audioCtx = new AudioCtx();
            if (this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }

            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();

            const isDanger = (risk || '').includes('danger') || (risk || '').includes('immediate');
            const freq = isDanger ? 920 : 660;

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(freq * 1.45, this.audioCtx.currentTime + 0.12);

            gain.gain.setValueAtTime(0.09, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.25);

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            osc.start();
            osc.stop(this.audioCtx.currentTime + 0.26);
        } catch (e) {
            // Audio policy / mute fallback
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

    _interpolateBearing(b1, b2, t) {
        let diff = (b2 - b1) % 360;
        if (diff > 180) diff -= 360;
        if (diff < -180) diff += 360;
        return (b1 + diff * t + 360) % 360;
    }

    _bearingToCardinal(deg, precision = 8) {
        if (precision === 16) {
            const cardinals16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
            const idx16 = Math.round(deg / 22.5) % 16;
            return cardinals16[idx16];
        }
        const cardinals8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const idx8 = Math.round(deg / 45) % 8;
        return cardinals8[idx8];
    }
}

window.PatrolSimulator = PatrolSimulator;

