import numpy as np

from skyfield.api import EarthSatellite, load, wgs84


# ==========================================================
# TIME GRID (UTC → TT internally)
# ==========================================================

from datetime import datetime

def generate_time_array(
    start_utc=None,
    duration_s=600,
    step_s=1
):
    """
    Generate time array.

    Parameters
    ----------
    start_utc : tuple (Y, M, D, h, m, s) or None
    """

    ts = load.timescale()

    # ----------------------------
    # DEFAULT: now
    # ----------------------------
    if start_utc is None:
        now = datetime.utcnow()
        start_utc = (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second
        )

    year, month, day, hour, minute, second = start_utc

    seconds = np.arange(0, duration_s, step_s)

    times = ts.utc(
        year,
        month,
        day,
        hour,
        minute,
        second + seconds
    )

    return times
# ==========================================================
# SATELLITE (SGP4)
# ==========================================================

def create_satellite(name, line1, line2):
    """
    Create satellite object from TLE.
    """
    return EarthSatellite(line1, line2, name)


# ==========================================================
# GROUND STATION (WGS84 → ECEF)
# ==========================================================

def create_ground_station(lat_deg, lon_deg, elevation_m=0.0):
    """
    Ground station in WGS84.

    Returns
    -------
    gs : Skyfield geographic position
    """
    return wgs84.latlon(
        latitude_degrees=lat_deg,
        longitude_degrees=lon_deg,
        elevation_m=elevation_m
    )


# ==========================================================
# SATELLITE POSITION (ECI / GCRS)
# ==========================================================

def satellite_position_eci(satellite, times):
    """
    Satellite position in ECI frame (GCRS).

    Returns
    -------
    r_sat : ndarray shape (3, N) [m]
    """

    geocentric = satellite.at(times)

    # position in km → convert to meters
    r = geocentric.position.km * 1e3

    return r


# ==========================================================
# GROUND STATION POSITION (ECI)
# ==========================================================

def ground_station_position_eci(ground_station, times):
    """
    Ground station position in ECI frame.

    Includes Earth rotation.

    Returns
    -------
    r_gs : ndarray shape (3, N) [m]
    """

    geocentric = ground_station.at(times)

    r = geocentric.position.km * 1e3

    return r


# ==========================================================
# RELATIVE GEOMETRY (RAW VECTORS)
# ==========================================================

def relative_vectors(r_sat, r_gs):
    """
    Line-of-sight vectors.

    Returns
    -------
    los : ndarray shape (3, N)
    """

    return r_sat - r_gs


# ==========================================================
# DISTANCE
# ==========================================================

def compute_distance(los):
    """
    Euclidean distance.

    Returns
    -------
    R : ndarray (N,) [m]
    """

    return np.linalg.norm(los, axis=0)


# ==========================================================
# ELEVATION ANGLE (PHYSICAL)
# ==========================================================

def compute_elevation(r_sat, r_gs):
    """
    Compute elevation angle from geometry.

    elevation = arcsin( (LOS · zenith) / |LOS| )

    where:
    zenith = r_gs / |r_gs|
    """

    los = r_sat - r_gs

    R = np.linalg.norm(los, axis=0)

    # unit zenith vector
    zenith = r_gs / np.linalg.norm(r_gs, axis=0)

    # projection
    cos_z = np.sum(los * zenith, axis=0) / R

    # numerical stability
    cos_z = np.clip(cos_z, -1.0, 1.0)

    elevation = np.arcsin(cos_z)

    return elevation


# ==========================================================
# FULL ORBIT PIPELINE (NO APPROXIMATION)
# ==========================================================

def propagate_orbit_full(satellite, ground_station, times):
    """
    Full physical propagation pipeline.

    Returns
    -------
    dict:
        r_sat
        r_gs
        los
        distance
        elevation
    """

    # ----------------------------
    # Positions
    # ----------------------------
    r_sat = satellite_position_eci(satellite, times)
    r_gs = ground_station_position_eci(ground_station, times)

    # ----------------------------
    # LOS
    # ----------------------------
    los = relative_vectors(r_sat, r_gs)

    # ----------------------------
    # Distance
    # ----------------------------
    R = compute_distance(los)

    # ----------------------------
    # Elevation
    # ----------------------------
    elevation = compute_elevation(r_sat, r_gs)

    return {
        "r_sat": r_sat,
        "r_gs": r_gs,
        "los": los,
        "distance": R,
        "elevation": elevation
    }
