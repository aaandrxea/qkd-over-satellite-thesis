import numpy as np
import os 
# ORBIT
from src.orbit.tle_loader import load_tle
from src.orbit.orbit_propagation import (
    create_satellite,
    generate_time_array,
    create_ground_station,
    propagate_satellite
)

# GEOMETRY
from src.geometry.link_geometry import compute_link_geometry

# PIPELINE QKD
from src.simulation.run_simulation import run_simulation


def run_orbit_simulation(
    tle_path,
    ground_lat,
    ground_lon,
    ground_alt,
    duration_s,
    step_s,
    wavelength,
    tx_diameter,
    rx_diameter,
    mu,
    nu,
    p_dark,
    e_opt
):
    """
    Simulazione completa con orbita reale
    """

    # ============================
    # LOAD TLE
    # ============================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    tle_full_path = os.path.join(BASE_DIR, tle_path)

    name, l1, l2 = load_tle(tle_full_path)

    satellite = create_satellite(name, l1, l2)
    # ============================
    # TIME
    # ============================
    times = generate_time_array(duration_s, step_s)

    # ============================
    # GROUND STATION
    # ============================
    ground = create_ground_station(
        ground_lat,
        ground_lon,
        ground_alt
    )

    # ============================
    # ORBIT PROPAGATION
    # ============================
    elevation_deg, distance_m = propagate_satellite(
        satellite,
        ground,
        times
    )

    # ============================
    # GEOMETRY
    # ============================
    geom = compute_link_geometry(
        elevation_deg,
        distance_m,
        min_elevation=0.0
    )

    R = geom["R"]
    elevation = geom["elevation"]

    # Se nessuna visibilità
    if len(R) == 0:
        raise ValueError("Satellite never visible in this time window")

    # ============================
    # QKD PIPELINE
    # ============================
    results = run_simulation(
        R=R,
        elevation=elevation,
        wavelength=wavelength,
        tx_diameter=tx_diameter,
        rx_diameter=rx_diameter,
        mu=mu,
        nu=nu,
        p_dark=p_dark,
        e_opt=e_opt
    )

    return {
        "time_mask": geom["mask"],
        "distance": R,
        "elevation": elevation,
        "qber": results["qber"],
        "skr": results["keyrate"]["skr"],
        "skr_decoy": results["decoy"]["skr_decoy"]
    }
