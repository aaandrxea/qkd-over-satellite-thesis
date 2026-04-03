import numpy as np

from src.utils.constants import EPS
from src.utils.io import load_yaml


# ==========================================================
# CONFIG
# ==========================================================

def load_pointing_config(config_path: str = "config/scenario.yaml") -> dict:
    cfg = load_yaml(config_path)
    return cfg.get("pointing", {})


# ==========================================================
# ANGULAR JITTER (PHYSICAL MODEL)
# ==========================================================

def angular_jitter_std(
    wavelength: float,
    R: np.ndarray,
    elevation: np.ndarray,
    sigma_platform: float,
    config=None
):
    """
    Total angular jitter (rad).

    Includes:
    - platform pointing jitter
    - turbulence-induced beam wander

    NOTE:
    Beam wander is derived from turbulence → correlated effect.
    """

    from src.channel.turbulence import beam_wander_std

    sigma_bw = beam_wander_std(
        wavelength,
        R,
        elevation,
        config=config
    )

    return np.sqrt(sigma_platform**2 + sigma_bw**2)


# ==========================================================
# STOCHASTIC OFFSET SAMPLING
# ==========================================================

def sample_pointing_offset(
    R: np.ndarray,
    sigma_theta: np.ndarray,
    static_offset: float = 0.0,
    rng=None
):
    """
    Sample radial displacement from angular jitter.

    θ_x, θ_y ~ N(0, σ²)
    r = R * sqrt(θ_x² + θ_y²)
    """

    if rng is None:
        rng = np.random

    theta_x = rng.normal(0.0, sigma_theta)
    theta_y = rng.normal(0.0, sigma_theta)

    theta = np.sqrt(theta_x**2 + theta_y**2)

    r_dynamic = R * theta

    r_total = np.sqrt(r_dynamic**2 + static_offset**2)

    return r_total


# ==========================================================
# GAUSSIAN MISALIGNMENT LOSS
# ==========================================================

def pointing_coupling(
    r: np.ndarray,
    w_R: np.ndarray
):
    """
    Gaussian beam misalignment loss:

    η = exp(-2 r² / w²)
    """

    w_R = np.maximum(w_R, EPS)

    return np.exp(-2.0 * (r / w_R)**2)


# ==========================================================
# TEMPORAL CORRELATION (OPTIONAL, ADVANCED)
# ==========================================================

def apply_temporal_correlation(
    sigma_theta: np.ndarray,
    correlation_time: float,
    dt: float,
    rng=None
):
    """
    Apply Ornstein-Uhlenbeck temporal correlation to angular jitter.

    This models mechanical inertia / control loop behavior.
    """

    if correlation_time is None or correlation_time <= 0:
        return sigma_theta

    if rng is None:
        rng = np.random

    alpha = np.exp(-dt / max(correlation_time, EPS))

    correlated = np.zeros_like(sigma_theta)

    for i in range(len(sigma_theta)):
        if i == 0:
            correlated[i] = sigma_theta[i]
        else:
            noise = rng.normal(0.0, sigma_theta[i])
            correlated[i] = alpha * correlated[i-1] + np.sqrt(1 - alpha**2) * noise

    return np.abs(correlated)


# ==========================================================
# MAIN POINTING MODEL
# ==========================================================

def pointing_fading(
    R,
    wavelength,
    elevation,
    w_R,                       # ← beam radius from link_budget
    config=None,
    dt=None,
    rng=None
):
    """
    Full physically consistent pointing model.

    Parameters
    ----------
    R : distance [m]
    wavelength : optical wavelength [m]
    elevation : angle [rad]
    w_R : beam radius at receiver [m]

    config:
        sigma_theta : platform jitter [rad]
        static_offset : static misalignment [m]
        correlation_time : optional temporal correlation [s]

    dt : timestep [s] (required for correlation)

    Returns
    -------
    eta_point : coupling efficiency
    """

    if config is None:
        config = load_pointing_config()

    sigma_platform = float(config.get("sigma_theta", 1e-6))
    static_offset = float(config.get("static_offset", 0.0))
    correlation_time = config.get("correlation_time", None)

    R = np.asarray(R)
    elevation = np.asarray(elevation)
    w_R = np.asarray(w_R)

    # ======================================================
    # TOTAL ANGULAR JITTER
    # ======================================================

    sigma_theta = angular_jitter_std(
        wavelength=wavelength,
        R=R,
        elevation=elevation,
        sigma_platform=sigma_platform,
        config=config.get("turbulence", {})
    )

    # ======================================================
    # TEMPORAL CORRELATION (OPTIONAL)
    # ======================================================

    if dt is not None:
        sigma_theta = apply_temporal_correlation(
            sigma_theta,
            correlation_time=correlation_time,
            dt=dt,
            rng=rng
        )

    # ======================================================
    # SAMPLE OFFSET
    # ======================================================

    r = sample_pointing_offset(
        R=R,
        sigma_theta=sigma_theta,
        static_offset=static_offset,
        rng=rng
    )

    # ======================================================
    # COUPLING LOSS
    # ======================================================

    eta_point = pointing_coupling(r, w_R)

    return np.clip(eta_point, 0.0, 1.0)
