import numpy as np
from datetime import timedelta
from src.channel.solar import solar_position, sun_los_angle
from src.utils.io import load_yaml

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
from src.detection.temporal_effects import (
    apply_dead_time,
    apply_afterpulsing,
    apply_timing_jitter
)
from src.detection.noise_models import apply_intensity_noise
from src.qkd.qber import compute_qber

# QKD
from src.qkd.key_rate import compute_key_rate


# ==========================================================
# PASS FINDER
# ==========================================================
def find_pass_window(satellite, gs):
    times = generate_time_array(duration_s=86400, step_s=5)
    orbit = propagate_orbit_full(satellite, gs, times)

    elevation_deg = np.rad2deg(orbit["elevation"])

    visible = elevation_deg > 0

    if not np.any(visible):
        raise RuntimeError("No visible pass found")

    # prendi il PRIMO punto visibile, non il massimo
    idx = np.where(visible)[0][0]

    return times[idx]

# ==========================================================
# DETECTION PIPELINE (COERENTE)
# ==========================================================
def run_detection_pipeline(
    eta,
    bg_rate,
    mu,
    det_cfg,
    dt
):
    """
    Fully consistent detection pipeline.

    QBER is computed AFTER temporal effects.
    """


    # ======================================================
    # SYSTEM EFFICIENCY
    # ======================================================

    eta_det = det_cfg.get("efficiency", 1.0)
    eta_system = eta * eta_det

    # ======================================================
    # INTENSITY NOISE
    # ======================================================

    mu_array = mu * np.ones_like(eta_system)

    mu_eff = apply_intensity_noise(
        mu_array,
        det_cfg.get("intensity_noise", 0.0)
    )

    # ======================================================
    # BASE DETECTION (NO TEMPORAL EFFECTS)
    # ======================================================

    det = compute_detection(
        eta_total=eta_system,
        mu=mu_eff,
        gate_time=det_cfg.get("gate_time", 1e-9),
        dark_rate=det_cfg.get("dark_rate", 100),
        bg_rate=bg_rate,
        e_opt=det_cfg.get("e_opt", 0.02)
    )

    p_sig = det["p_sig"]
    p_noise = det["p_noise"]

    # ======================================================
    # BASE CLICK / ERROR PROBABILITIES
    # ======================================================

    p_click = det["p_click"]

    # error contributions
    p_error = (
        det_cfg.get("e_opt", 0.02) * p_sig +
        0.5 * p_noise
    )

    # ======================================================
    # TEMPORAL EFFECTS (APPLIED CONSISTENTLY)
    # ======================================================

    p_click_eff = apply_dead_time(
        p_click,
        dt,
        det_cfg.get("dead_time", 50e-9)
    )

    p_click_eff = apply_afterpulsing(
        p_click_eff,
        det_cfg.get("afterpulse_prob", 0.01)
    )

    p_click_eff = apply_timing_jitter(
        p_click_eff,
        det_cfg.get("jitter", 50e-12),
        dt
    )

    # ======================================================
    # SCALE ERROR CONSISTENTLY
    # ======================================================

    scale = np.zeros_like(p_click)

    valid = p_click > 1e-15
    scale[valid] = p_click_eff[valid] / p_click[valid]

    p_error_eff = p_error * scale

    # ======================================================
    # FINAL QBER (CONSISTENT)
    # ======================================================

    qber = np.zeros_like(p_click_eff)

    valid = p_click_eff > 1e-15
    qber[valid] = p_error_eff[valid] / p_click_eff[valid]

    # saturate (physical bounds)
    qber = np.clip(qber, 0.0, 0.5)

    return {
        "p_sig": p_sig,
        "p_noise": p_noise,
        "p_click": p_click_eff,
        "p_error": p_error_eff,
        "qber": qber
    }

# ==========================================================
# MAIN SIMULATION
# ==========================================================
def run_orbit_simulation(
    step_s=0.1,
    duration_s=86400,
    min_elevation_deg=0.0
):

    # ================= CONFIG =================
    scenario = load_yaml("config/scenario.yaml")
    det_cfg = load_yaml("config/detector.yaml")
    sat_cfg = load_yaml("config/satellite.yaml")

    mu = scenario["mu"]
    nu = 0.1

    # ================= ORBIT =================
    name, l1, l2 = load_tle(sat_cfg["tle_file"])
    satellite = create_satellite(name, l1, l2)

    gs = create_ground_station(45.0, 11.0, 0.0)
    lat = np.deg2rad(45.0)
    lon = np.deg2rad(11.0)
    peak_time = find_pass_window(satellite, gs)

    # ================= TIME GRID =================
    window_s = 1200
    peak_dt = peak_time.utc_datetime()
    start_dt = peak_dt - timedelta(seconds=window_s / 2)

    start_utc = (
        start_dt.year, start_dt.month, start_dt.day,
        start_dt.hour, start_dt.minute, start_dt.second
    )

    times = generate_time_array(
        start_utc=start_utc,
        duration_s=window_s,
        step_s=step_s
    )

    t_seconds = np.arange(0, window_s, step_s)

    # ================= PROPAGATION =================
    orbit = propagate_orbit_full(satellite, gs, times)

    R = orbit["distance"]
    elevation = orbit["elevation"]
    azimuth = orbit["azimuth"]

    elevation_deg = np.rad2deg(elevation)

    # ================= VISIBILITY =================
    mask = visibility_mask(elevation_deg, min_elevation=min_elevation_deg)

    print("Visible samples:", np.sum(mask))

    if not np.any(mask):
        print("ERROR: No visible pass")
        return None

    # ================= APPLY MASK =================
    R = R[mask]
    elevation = elevation[mask]
    azimuth = azimuth[mask]

    t_vis = t_seconds[mask]
    times_vis = times[mask]

    # ================= SOLAR =================
    sun_az = []
    sun_el = []
    sun_angle = []

    for t, el, az in zip(times_vis, elevation, azimuth):

        dt = t.utc_datetime()

        saz, sel = solar_position(dt, lat, lon)

        sun_az.append(saz)
        sun_el.append(sel)

        angle = sun_los_angle(saz, sel, az, el)
        sun_angle.append(angle)

    sun_az = np.array(sun_az)
    sun_el = np.array(sun_el)
    sun_angle = np.array(sun_angle)

    # ================= SOLAR CHECK =================
    print("\n--- SOLAR CHECK ---")

    if len(sun_el) == 0:
        print("⚠ No solar data")
    else:
        sun_deg = np.rad2deg(sun_el)
        print("Sun elevation (deg):", np.min(sun_deg), np.max(sun_deg))

    # ================= CHANNEL =================
    channel = compute_link_budget(
        R,
        elevation,
        wavelength=scenario["wavelength"],
        tx_diameter=scenario["tx_diameter"],
        rx_diameter=scenario["rx_diameter"],
        config=scenario,
        sun_angle=sun_angle,
        sun_elevation=sun_el
    )

    eta = channel["eta_total"]
    bg_rate = channel["background"]

    print("\n--- BACKGROUND CHECK ---")
    print("Background min/max:", np.min(bg_rate), np.max(bg_rate))

    # ================= DETECTION (MULTI-INTENSITY) =================
    dt = step_s  # coerente con simulazione

    det_mu = run_detection_pipeline(eta, bg_rate, mu, det_cfg, dt)
    det_nu = run_detection_pipeline(eta, bg_rate, nu, det_cfg, dt)
    det_0  = run_detection_pipeline(eta, bg_rate, 0.0, det_cfg, dt)

    # ================= QKD =================
    qkd = compute_key_rate(
        detection_mu=det_mu,
        detection_nu=det_nu,
        detection_0=det_0,
        mu=mu,
        nu=nu
    )

    skr = qkd["skr"]
    qber = det_mu["qber"]  # QBER osservabile
    # ======================================================
    # TEST 3 — DAY / NIGHT
    # ======================================================

    day_mask = sun_el > 0
    night_mask = sun_el <= 0

    print("\n--- DAY/NIGHT CHECK ---")

    if np.any(day_mask):
        print("QBER day:", np.mean(qber[day_mask]))
        print("SKR day:", np.mean(skr[day_mask]))
    else:
        print("No day samples")

    if np.any(night_mask):
        print("QBER night:", np.mean(qber[night_mask]))
        print("SKR night:", np.mean(skr[night_mask]))
    else:
        print("No night samples")


    # ======================================================
    # TEST 4 — ANGULAR DEPENDENCE
    # ======================================================

    print("\n--- ANGLE CORRELATION ---")

    idx = np.argsort(sun_angle)
    n = min(50, len(sun_angle)//2)

    print("Low angle bg:", np.mean(bg_rate[idx[:n]]))
    print("High angle bg:", np.mean(bg_rate[idx[-n:]]))


    # ======================================================
    # TEST 5 — TEMPORAL BEHAVIOR (PLOT)
    # ======================================================

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t_vis, bg_rate, label="background")
        plt.plot(t_vis, qber, label="qber")
        plt.xlabel("time [s]")
        plt.legend()
        plt.title("Temporal evolution: background & QBER")
        plt.show()

    except Exception as e:
        print("Plot skipped:", e)
    # ================= OUTPUT =================
    return {
        "time": t_vis,
        "R": R,
        "elevation": elevation,
        "eta": eta,
        "skr": skr,
        "qber": qber,
        "channel": channel,
        "detection": {
            "mu": det_mu,
            "nu": det_nu,
            "vacuum": det_0
        }
    }


if __name__ == "__main__":
    result = run_orbit_simulation()
    print("Simulation completed")
