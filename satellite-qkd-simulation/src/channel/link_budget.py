import numpy as np

from src.utils.constants import EPS
from src.utils.io import load_yaml
from src.channel.clouds import cloud_transmittance
# CHANNEL COMPONENTS
from src.channel.atmospheric import atmospheric_transmittance
from src.channel.turbulence import turbulence_fading
from src.channel.pointing import beam_waist, beam_radius, pointing_fading
from src.channel.background import background_photon_rate


# ==========================================================
# CONFIG
# ==========================================================

def load_channel_config(config_path: str = "config/scenario.yaml") -> dict:
    return load_yaml(config_path)


# ==========================================================
# APERTURE COUPLING (DIFFRACTION LIMITED)
# ==========================================================

def aperture_coupling(R, wavelength, tx_diameter, rx_diameter):
    """
    Gaussian beam → circular aperture coupling
    """

    R = np.asarray(R)

    w0 = beam_waist(tx_diameter)
    w = beam_radius(wavelength, w0, R)

    a = rx_diameter / 2.0

    eta = 1.0 - np.exp(-2.0 * (a**2) / (w**2 + EPS))

    return np.clip(eta, 0.0, 1.0)


# ==========================================================
# SYSTEM EFFICIENCY
# ==========================================================

def optical_efficiency(eta_tx, eta_rx, eta_misc):
    return eta_tx * eta_rx * eta_misc


# ==========================================================
# MAIN LINK BUDGET (PHYSICAL)
# ==========================================================

def compute_link_budget(
    R,
    elevation,
    wavelength,
    tx_diameter,
    rx_diameter,
    config=None
):
    """
    Fully physical link budget (no hacks).
    """
    wavelength = float(wavelength)
    tx_diameter = float(tx_diameter)
    rx_diameter = float(rx_diameter)
    if config is None:
        config = load_channel_config()

    # ======================================================
    # PARAMETERS
    # ======================================================
    eta_tx = config.get("eta_tx", 1.0)
    eta_rx = config.get("eta_rx", 1.0)
    eta_misc = config.get("eta_misc", 1.0)

    fov = float(config.get("fov", 1e-6))
    bandwidth = float(config.get("bandwidth", 1e-9))
    condition = config.get("condition", "day")

    # ======================================================
    # 1. GEOMETRIC (DIFFRACTION)
    # ======================================================
    eta_geo = aperture_coupling(
        R,
        wavelength,
        tx_diameter,
        rx_diameter
    )

    # ======================================================
    # 2. ATMOSPHERE (EXTINCTION)
    # ======================================================
    atm_config = config.get("atmosphere", {}).copy()
    atm_config["wavelength"] = float(wavelength)

    eta_atm = atmospheric_transmittance(
        elevation,
        R,
        config=atm_config
    )

    # ======================================================
    # 3. TURBULENCE (SCINTILLATION)
    # ======================================================
    turb_cfg = config.get("turbulence", {})

    eta_turb, sigma_R2 = turbulence_fading(
        elevation,
        R,
        wavelength,
        model=turb_cfg.get("model", "auto"),
        A=turb_cfg.get("A", 1.7e-14),
        v=turb_cfg.get("v", 21.0),
        n_steps=turb_cfg.get("n_steps", 300)
    )

    # ======================================================
    # 4. POINTING (INCLUDE BEAM WANDER)
    # ======================================================
    eta_point = pointing_fading(
        R,
        wavelength,
        elevation,
        tx_diameter,
        config=config.get("pointing", {})
    )

    # ======================================================
    # 5. SYSTEM
    # ======================================================
    eta_sys = optical_efficiency(
        eta_tx,
        eta_rx,
        eta_misc
    )

    # ======================================================
    # TOTAL LINK EFFICIENCY
    # ======================================================
    eta_total = (
        eta_geo *
        eta_atm *
        eta_turb *
        eta_point *
        eta_sys
    )
    # ======================================================
    # CLOUDS
    # ======================================================
    N = len(eta_total)

    eta_cloud, cloud_state = cloud_transmittance(
        N,
        config=config
    )
    print("CLOUD FRACTION:", np.mean(cloud_state))
    print("ETA_CLOUD MIN/MAX:", np.min(eta_cloud), np.max(eta_cloud))
    eta_total = eta_total * eta_cloud
    eta_total = np.clip(eta_total, 0.0, 1.0)

    # ======================================================
    # BACKGROUND
    # ======================================================
    background = background_photon_rate(
        wavelength=wavelength,
        rx_diameter=rx_diameter,
        fov=fov,
        bandwidth=bandwidth,
        elevation_rad=elevation,
        condition=condition,
        config=config
    )

    # ======================================================
    # DEBUG (PHYSICS CHECK)
    # ======================================================
    if config.get("debug", False):
        print("\n--- LINK BUDGET DEBUG ---")
        print("eta_geo   :", np.mean(eta_geo))
        print("eta_atm   :", np.mean(eta_atm))
        print("eta_turb  :", np.mean(eta_turb))
        print("eta_point :", np.mean(eta_point))
        print("eta_total :", np.mean(eta_total))
        print("sigma_R2  :", np.mean(sigma_R2))
        print("bg_rate   :", np.mean(background))

    return {
        "eta_total": eta_total,
        "eta_geo": eta_geo,
        "eta_atm": eta_atm,
        "eta_turb": eta_turb,
        "sigma_R2": sigma_R2,
        "eta_point": eta_point,
        "eta_sys": eta_sys,
        "background": background,
        "eta_cloud": eta_cloud,
        "cloud_state": cloud_state
    }


# ==========================================================
# dB SCALE
# ==========================================================

def to_dB(x):
    x = np.clip(x, 1e-15, None)
    return -10.0 * np.log10(x)


def link_budget_dB(results):
    return {
        k: to_dB(v)
        for k, v in results.items()
        if isinstance(v, np.ndarray)
    }
