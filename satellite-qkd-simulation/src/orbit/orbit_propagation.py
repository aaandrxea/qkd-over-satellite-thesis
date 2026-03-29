import numpy as np

from skyfield.api import EarthSatellite, load, wgs84


# ================================
# SATELLITE CREATION
# ================================

def create_satellite(name, line1, line2):
    """
    Crea oggetto satellite da TLE
    """
    return EarthSatellite(line1, line2, name)


# ================================
# TIME GRID
# ================================

def generate_time_array(duration_s=600, step_s=1):
    """
    Genera array temporale
    """
    ts = load.timescale()

    t = ts.utc(
        2024, 1, 1,
        0, 0,
        np.arange(0, duration_s, step_s)
    )

    return t


# ================================
# GROUND STATION
# ================================

def create_ground_station(lat, lon, elevation_m=0):
    """
    Stazione a terra
    """
    return wgs84.latlon(latitude_degrees=lat,
                        longitude_degrees=lon,
                        elevation_m=elevation_m)


# ================================
# PROPAGATION
# ================================

def propagate_satellite(satellite, ground_station, times):
    """
    Calcola posizione relativa satellite-ground
    """

    difference = satellite - ground_station
    topocentric = difference.at(times)

    alt, az, distance = topocentric.altaz()

    elevation = alt.degrees
    distance_m = distance.m

    return elevation, distance_m
