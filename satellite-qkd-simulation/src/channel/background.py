import numpy as np
from src.channel.solar import solar_position, sun_los_angle
from src.utils.constants import H, C, EPS


# ==========================================================
# SKY RADIANCE MODEL (SEMPLIFIED BUT PHYSICAL)
# ==========================================================
import numpy as np

def sky_radiance(
    wavelength,
    elevation,
    sun_angle=None,
    sun_elevation=None,
    config=None
):
    """
    Fully vectorized sky radiance model (no external dependencies).
    """

    elevation = np.asarray(elevation)

    if sun_elevation is None:
        sun_elevation = np.zeros_like(elevation)
    else:
        sun_elevation = np.asarray(sun_elevation)

    if sun_angle is None:
        sun_angle = np.zeros_like(elevation)
    else:
        sun_angle = np.asarray(sun_angle)

    # ----------------------------
    # MASKS
    # ----------------------------
    day_mask = sun_elevation > 0
    night_mask = ~day_mask

    # ----------------------------
    # OUTPUT
    # ----------------------------
    L = np.zeros_like(elevation, dtype=float)

    # ----------------------------
    # NIGHT SKY
    # ----------------------------
    sin_el = np.maximum(np.sin(elevation[night_mask]), 0.1)
    L[night_mask] = 1e-6 / sin_el
    # ----------------------------
    # DAY SKY
    # ----------------------------
    if np.any(day_mask):

        # forward scattering model
        L_day = 1e-6 * np.exp(-sun_angle[day_mask])

        # airmass scaling
        sin_el = np.maximum(np.sin(elevation[night_mask]), 0.1)
        L[night_mask] = 1e-6 / sin_el
        L[day_mask] = L_day

    return L
# ==========================================================
# PHOTON CONVERSION
# ==========================================================

def photon_energy(wavelength):
    return H * C / wavelength


# ==========================================================
# MAIN BACKGROUND MODEL
# ==========================================================



h = 6.62607015e-34
c = 299792458.0


def compute_background(
    wavelength,
    elevation,
    rx_diameter,
    beam_radius,
    sun_angle=None,
    sun_elevation=None,
    config=None
):
    """
    ETH-grade background model with FOV and spectral bandwidth.
    """

    elevation = np.asarray(elevation)

    # =========================
    # CONFIG
    # =========================
    fov = config.get("fov_rad", 100e-6)          # radians
    delta_lambda = config.get("bandwidth_nm", 1.0)  # nm
    T_opt = config.get("optical_efficiency", 0.5)
    eta_det = config.get("detector_efficiency", 0.6)

    # =========================
    # SKY RADIANCE
    # =========================
    L = sky_radiance(
        wavelength,
        elevation,
        sun_angle=sun_angle,
        sun_elevation=sun_elevation,
        config=config
    )  # W / (m^2 sr nm)

    # =========================
    # RECEIVER AREA
    # =========================
    A_rx = np.pi * (rx_diameter / 2.0)**2

    # =========================
    # SOLID ANGLE (FOV)
    # =========================
    Omega = np.pi * fov**2

    # =========================
    # POWER COLLECTED
    # =========================
    P_bg = L * A_rx * Omega * delta_lambda * T_opt

    # =========================
    # PHOTON ENERGY
    # =========================
    lambda_m = wavelength
    E_ph = h * c / lambda_m

    # =========================
    # PHOTON RATE
    # =========================
    N_bg = P_bg / E_ph

    # =========================
    # DETECTOR EFFICIENCY
    # =========================
    N_bg *= eta_det

    return N_bg
