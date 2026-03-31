import numpy as np
from src.utils.quick_plot import plot_results
from src.utils.io import load_yaml
from datetime import timedelta
# ORBIT
from src.orbit.tle_loader import load_tle
from src.orbit.orbit_propagation import (
    create_satellite,
    create_ground_station,
    generate_time_array,
    propagate_orbit_full
)

# GEOMETRY
from src.geometry.link_geometry import visibility_mask

# CHANNEL
from src.channel.link_budget import compute_link_budget

# DETECTION
from src.detection.detector import compute_detection

# QKD
from src.qkd.key_rate import compute_key_rate
def find_pass_window(satellite, gs):

    times = generate_time_array(
        duration_s=86400,
        step_s=5
    )

    orbit = propagate_orbit_full(satellite, gs, times)

    elevation = orbit["elevation"]
    elevation_deg = np.rad2deg(elevation)

    peak_idx = np.argmax(elevation_deg)

    peak_time = times[peak_idx]

    return peak_time
# ==========================================================
# MAIN SIMULATION
# ==========================================================

def run_orbit_simulation(
    step_s=0.1,
    duration_s=86400,
    min_elevation_deg=0.0
):
    """
    Full end-to-end satellite QKD simulation.

    Returns
    -------
    dict with full time series
    """

    # ======================================================
    # LOAD CONFIG
    # ======================================================
    scenario = load_yaml("config/scenario.yaml")
    detector_cfg = load_yaml("config/detector.yaml")
    sat_cfg = load_yaml("config/satellite.yaml")

    # ======================================================
    # LOAD TLE
    # ======================================================
    name, l1, l2 = load_tle(sat_cfg["tle_file"])

    satellite = create_satellite(name, l1, l2)

    # ======================================================
    # GROUND STATION (CONFIGURABILE)
    # ======================================================
    gs = create_ground_station(
        lat_deg=45.0,
        lon_deg=11.0,
        elevation_m=0.0
    )
    peak_time = find_pass_window(satellite, gs)
    # ======================================================
    # TIME GRID
    # ======================================================
    # finestra locale
    window_s = 1200  # 20 min totale

    # tempo centrale
    peak_dt = peak_time.utc_datetime()
    # inizio finestra centrata
    start_dt = peak_dt - timedelta(seconds=window_s / 2)

    # formato compatibile con generate_time_array
    start_utc = (
        start_dt.year,
        start_dt.month,
        start_dt.day,
        start_dt.hour,
        start_dt.minute,
        start_dt.second
    )

    # time grid finale
    times = generate_time_array(
        start_utc=start_utc,
        duration_s=window_s,
        step_s=step_s
    )

    # asse temporale locale
    t_seconds = np.arange(0, window_s, step_s)

    # ======================================================
    # ORBIT PROPAGATION (PHYSICAL)
    # ======================================================
    orbit = propagate_orbit_full(
        satellite,
        gs,
        times
    )

    R = orbit["distance"]
    elevation = orbit["elevation"]

    elevation_deg = np.rad2deg(elevation)
    print("Max elevation:", np.max(elevation_deg))
    print("Min elevation:", np.min(elevation_deg))

    # ======================================================
    # VISIBILITY MASK
    # ======================================================
    mask = visibility_mask(
        elevation_deg,
        min_elevation=min_elevation_deg
    )

    # DEBUG
    visible_fraction = np.mean(elevation_deg > 0)
    print("Fraction of visible time:", visible_fraction)

    # ------------------------------------------------------
    # NO VISIBILITY → EXIT
    # ------------------------------------------------------
    if not np.any(mask):
        print("ERROR: No visible satellite pass")
        print("Max elevation:", np.max(elevation_deg))
        return None

    # ------------------------------------------------------
    # APPLY MASK
    # ------------------------------------------------------
    R_vis = R[mask]
    elevation_vis = elevation[mask]
    t_vis = t_seconds[mask]

    # ------------------------------------------------------
    # FIND PEAK
    # ------------------------------------------------------
    peak_idx = np.argmax(elevation_vis)
    center_time = t_vis[peak_idx]

    # ------------------------------------------------------
    # ZOOM WINDOW (opzionale)
    # ------------------------------------------------------
    window = 300  # ±300 s
    zoom_mask = (t_seconds > center_time - window) & (t_seconds < center_time + window)


    # ======================================================        
    # CHANNEL
    # ======================================================
    channel = compute_link_budget(
        R_vis,
        elevation_vis,
        wavelength=scenario["wavelength"],
        tx_diameter=scenario["tx_diameter"],
        rx_diameter=scenario["rx_diameter"],
        config=scenario
    )

    eta = channel["eta_total"]

    # ======================================================
    # DETECTION (3 INTENSITIES)
    # ======================================================
    mu = scenario["mu"]
    nu = 0.1
    e_opt = scenario["e_opt"]

    dark_rate = detector_cfg["dark_rate"]
    gate_time = detector_cfg["gate_time"]

    # ⚠️ BACKGROUND DAL CHANNEL
    bg_rate = channel["background"]

    det_mu = compute_detection(
        eta,
        mu,
        dark_rate,
        gate_time,
        e_opt,
        bg_rate=bg_rate
    )

    det_nu = compute_detection(
        eta,
        nu,
        dark_rate,
        gate_time,
        e_opt,
        bg_rate=bg_rate
    )

    det_0 = compute_detection(
        eta,
        0.0,
        dark_rate,
        gate_time,
        e_opt,
        bg_rate=bg_rate
    )


    # ======================================================
    # QKD (DECOY)
    # ======================================================
    qkd = compute_key_rate(
        det_mu,
        det_nu,
        det_0,
        mu,
        nu
    )

    skr = qkd["skr"]
    print("\n--- VALIDATION ---")
    print("Max SKR:", np.max(skr))
    print("Mean SKR:", np.mean(skr))
    print("Max QBER:", np.max(det_mu["qber"]))
    print("Min QBER:", np.min(det_mu["qber"]))
    # ======================================================
    # OUTPUT (CONSISTENT & SAFE)
    # ======================================================
    return {
        # ================= FULL TIMELINE =================
        "full": {
            "time": t_seconds,
            "R": R,
            "elevation": elevation,
            "mask": mask,
        },

        # ================= VISIBLE ONLY =================
        "visible": {
            "time": t_vis,
            "R": R_vis,
            "elevation": elevation_vis,
            "eta": eta,
            "qber": det_mu["qber"],
            "p_click": det_mu["p_click"],
            "skr": skr,
        }
    }
if __name__ == "__main__":
    result = run_orbit_simulation()
    print("Simulation completed")
    print("Max SKR:", result["skr"].max())
