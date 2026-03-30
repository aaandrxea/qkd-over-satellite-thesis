import numpy as np

from src.utils.constants import PI, EPS
from src.utils.io import load_yaml


# ==========================================================
# CONFIG
# ==========================================================

def load_pointing_config(config_path: str = "config/scenario.yaml") -> dict:
    cfg = load_yaml(config_path)
    return cfg.get("pointing", {})


# ==========================================================
# BEAM PROPAGATION (Gaussian beam)
# ==========================================================

def beam_waist(tx_diameter: float) -> float:
    """
    Waist at transmitter (approximation).

    w0 ≈ D / 2
    """
    return tx_diameter / 2.0


def beam_radius(wavelength: float, w0: float, z: np.ndarray) -> np.ndarray:
    """
    Gaussian beam radius:

    w(z) = w0 * sqrt(1 + (z / z_R)^2)

    where:
    z_R = π w0^2 / λ
    """

    z_R = PI * w0**2 / wavelength

    return w0 * np.sqrt(1 + (z / z_R)**2)


# ==========================================================
# POINTING ERROR (2D MODEL)
# ==========================================================

def pointing_offset(
    R: np.ndarray,
    sigma_theta: float,
    size=None
):
    """
    Generates radial pointing offset.

    Angular jitter → 2D Gaussian → radial Rayleigh

    r = R * θ

    Returns
    -------
    r : radial offset [m]
    """

    if sigma_theta < 1e-12:
        return np.zeros_like(R)

    # 2D Gaussian components
    theta_x = np.random.normal(0, sigma_theta, size=len(R))
    theta_y = np.random.normal(0, sigma_theta, size=len(R))

    theta = np.sqrt(theta_x**2 + theta_y**2)

    return R * theta


# ==========================================================
# COUPLING EFFICIENCY
# ==========================================================

def pointing_loss(
    r: np.ndarray,
    w: np.ndarray
):
    """
    Gaussian beam coupling:

    η = exp(-2 r^2 / w^2)
    """

    w = np.maximum(w, EPS)

    return np.exp(-2.0 * (r**2) / (w**2))


# ==========================================================
# MAIN INTERFACE
# ==========================================================

def pointing_fading(
    R: np.ndarray,
    wavelength: float,
    tx_diameter: float,
    config: dict | None = None
):
    """
    Full pointing loss model.

    Includes:
    - diffraction-limited beam propagation
    - 2D jitter
    - Gaussian coupling

    Returns
    -------
    eta_point : np.ndarray
    """

    if config is None:
        config = load_pointing_config()

    sigma_theta = float(config.get("sigma_theta", 1e-6))
    static_offset = float(config.get("static_offset", 0.0))
    R = np.asarray(R)

    # ----------------------------
    # Beam propagation
    # ----------------------------
    w0 = beam_waist(tx_diameter)
    w = beam_radius(wavelength, w0, R)

    # ----------------------------
    # Pointing offset
    # ----------------------------
    r_jitter = pointing_offset(R, sigma_theta)

    # static misalignment (converted to meters)
    r_static = R * static_offset

    r_total = np.sqrt(r_jitter**2 + r_static**2)

    # ----------------------------
    # Coupling
    # ----------------------------
    eta = pointing_loss(r_total, w)

    return np.clip(eta, 0.0, 1.0)
