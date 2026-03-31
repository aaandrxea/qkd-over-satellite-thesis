import numpy as np

from src.utils.constants import EPS
from src.utils.io import load_yaml

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
# APERTURE COUPLING
# ==========================================================

def aperture_coupling(
    R: np.ndarray,
    wavelength: float,
    tx_diameter: float,
    rx_diameter: float
) -> np.ndarray:

    R = np.asarray(R)

    w0 = beam_waist(tx_diameter)
    w = beam_radius(wavelength, w0, R)

    a = rx_diameter / 2.0

    eta = 1.0 - np.exp(-2.0 * (a**2) / (w**2 + EPS))

    return np.clip(eta, 0.0, 1.0)


# ==========================================================
# SYSTEM EFFICIENCY
# ==========================================================

def optical_efficiency(
    eta_tx: float,
    eta_rx: float,
    eta_misc: float
) -> float:
    return eta_tx * eta_rx * eta_misc


# ==========================================================
# MAIN LINK BUDGET (FIXED VERSION)
# ==========================================================

def compute_link_budget(
    R: np.ndarray,
    elevation: np.ndarray,   # ⚠️ DEVE essere in radianti
    wavelength: float,
    tx_diameter: float,
    rx_diameter: float,
    config: dict | None = None
) -> dict:

    if config is None:
        config = load_channel_config()

    # ----------------------------
    # System parameters
    # ----------------------------
    eta_tx = config.get("eta_tx", 1.0)
    eta_rx = config.get("eta_rx", 1.0)
    eta_misc = config.get("eta_misc", 1.0)

    fov = config.get("fov", 1e-6)
    bandwidth = config.get("bandwidth", 1e-9)
    condition = config.get("condition", "day")

    # ----------------------------
    # Aperture coupling
    # ----------------------------
    eta_aperture = aperture_coupling(
        R,
        wavelength,
        tx_diameter,
        rx_diameter
    )

    # ----------------------------
    # Atmospheric loss
    # ----------------------------
    eta_atm = atmospheric_transmittance(
        elevation,
        R,
        config=config.get("atmosphere", None)
    )

    # ----------------------------
    # Turbulence
    # ----------------------------
    eta_turb, sigma_R2 = turbulence_fading(
        elevation,
        R,
        wavelength,
        **config.get("turbulence", {})
    )

    # ----------------------------
    # Pointing
    # ----------------------------
    eta_point = pointing_fading(
        R,
        wavelength,
        elevation,
        tx_diameter,
        config=config.get("pointing", {})
    )

    # ----------------------------
    # System efficiency
    # ----------------------------
    eta_sys = optical_efficiency(
        eta_tx,
        eta_rx,
        eta_misc
    )

    # ----------------------------
    # TOTAL CHANNEL
    # ----------------------------
    eta_total = (
        eta_aperture *
        eta_atm *
        eta_point *
        eta_sys*
        eta_turb
    )


    eta_total = np.clip(eta_total, 0.0, 1.0)

    # ==========================================================
    # BACKGROUND (FINAL FIX)
    # ==========================================================

    background = background_photon_rate(
        wavelength=wavelength,
        rx_diameter=rx_diameter,
        fov=fov,
        bandwidth=bandwidth,
        elevation_rad=elevation,
        condition=condition
    )
    print("TURB:",
      np.mean(eta_turb),
      np.std(eta_turb))

    print("TOTAL:",
        np.mean(eta_total),
        np.std(eta_total))
    print("ETA TOTAL:",
      np.mean(eta_total),
      np.std(eta_total))
    return {
        "eta_aperture": eta_aperture,
        "eta_atm": eta_atm,
        "eta_turb": eta_turb,
        "sigma_R2": sigma_R2,
        "eta_point": eta_point,
        "eta_sys": eta_sys,
        "eta_total": eta_total,
        "background": background 
    }


# ==========================================================
# dB SCALE
# ==========================================================

def to_dB(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-15, None)
    return -10.0 * np.log10(x)


def link_budget_dB(results: dict) -> dict:
    return {k: to_dB(v) for k, v in results.items() if isinstance(v, np.ndarray)}
