import numpy as np
from src.channel.solar import solar_position, sun_los_angle
from src.channel.atmospheric import atmospheric_transmittance
from src.channel.turbulence import compute_rytov_variance, sample_fading
from src.channel.pointing import pointing_fading
from src.channel.background import compute_background
from src.channel.clouds import cloud_attenuation_2d

EPS = 1e-15


# ==========================================================
# GAUSSIAN BEAM
# ==========================================================

def gaussian_beam_waist(tx_diameter: float) -> float:
    return tx_diameter / 2.0


def rayleigh_range(w0: float, wavelength: float) -> float:
    return np.pi * w0**2 / wavelength

def beam_radius(wavelength, tx_diameter, R):
    """
    Far-field Gaussian beam approximation (physically correct for satellite QKD)
    """

    R = np.asarray(R)

    theta = wavelength / (np.pi * tx_diameter / 2.0)

    w_R = theta * R

    return np.maximum(w_R, 1e-12)

def receiver_coupling_efficiency(w_R, rx_diameter):
    w_R = np.maximum(w_R, EPS)
    r_rx = rx_diameter / 2.0

    eta = 1.0 - np.exp(-2.0 * (r_rx**2) / (w_R**2))

    return np.clip(eta, 0.0, 1.0)


# ==========================================================
# GEOMETRIC LOSS (CORRETTA)
# ==========================================================

def geometric_loss(tx_diameter, rx_diameter, wavelength, R):

    R = np.asarray(R)

    w_R = beam_radius(wavelength, tx_diameter, R)

    eta_geo = receiver_coupling_efficiency(w_R, rx_diameter)

    beam = {
        "w_R": w_R,
        "w0": gaussian_beam_waist(tx_diameter),
        "z_R": rayleigh_range(
            gaussian_beam_waist(tx_diameter),
            wavelength
        )
    }

    return eta_geo, beam


# ==========================================================
# CHANNEL CORE
# ==========================================================
def compute_link_budget(
    R,
    elevation,
    wavelength,
    tx_diameter,
    rx_diameter,
    config,
    sun_angle=None,
    sun_elevation=None
):
    """
    Fully consistent optical channel model
    """

    R = np.asarray(R)
    elevation = np.asarray(elevation)

    # ======================================================
    # 1. GEOMETRY (beam + coupling)
    # ======================================================

    eta_geo, beam = geometric_loss(
        tx_diameter=tx_diameter,
        rx_diameter=rx_diameter,
        wavelength=wavelength,
        R=R
    )

    # ======================================================
    # 2. ATMOSPHERE
    # ======================================================

    eta_atm = atmospheric_transmittance(
        R,
        elevation,
        wavelength,
        config=config.get("atmosphere", {})
    )

    # ======================================================
    # 3. TURBULENCE
    # ======================================================

    sigma_R2 = compute_rytov_variance(
        wavelength,
        R,
        elevation,
        config=config.get("turbulence", {})
    )

    eta_turb = np.exp(-np.abs(sample_fading(sigma_R2)))
    # ======================================================
    # 4. POINTING (USA BEAM)
    # ======================================================

    eta_point = pointing_fading(
        R=R,
        wavelength=wavelength,
        elevation=elevation,
        w_R=beam["w_R"],
        config=config.get("pointing", {}),
        dt=config.get("simulation", {}).get("step_s", None)
    )

    # ======================================================
    # 5. CLOUDS
    # ======================================================

    cloud_cfg = config.get("clouds", {})

    cloud_state, eta_cloud = cloud_attenuation_2d(
        n_steps=len(R),
        elevation=elevation,
        dt=config.get("simulation", {}).get("step_s", 0.1),
        config=cloud_cfg
    )

    # ======================================================
    # 6. TOTAL CHANNEL
    # ======================================================

    eta_total = (
        eta_geo
        * eta_atm
        * eta_turb
        * eta_point
        * eta_cloud
    )

    eta_total = np.clip(eta_total, 0.0, 1.0)

    # ======================================================
    # 7. BACKGROUND (CORRETTO)
    # ======================================================
    background = compute_background(
        wavelength=wavelength,
        elevation=elevation,
        rx_diameter=rx_diameter,
        beam_radius=beam["w_R"],
        sun_angle=sun_angle,
        sun_elevation=sun_elevation,
        config=config.get("background", {})
    )


    # ======================================================
    # OUTPUT
    # ======================================================

    return {
        "eta_total": eta_total,
        "beam": beam,
        "background": background,

        "components": {
            "geometry": eta_geo,
            "atmosphere": eta_atm,
            "turbulence": eta_turb,
            "pointing": eta_point,
            "cloud": eta_cloud,
        },

        "state": {
            "sigma_R2": sigma_R2,
            "cloud_state": cloud_state,
        }
    }
