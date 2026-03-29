import numpy as np

from src.utils.constants import PI, EPS
from src.utils.io import load_yaml


# ================================
# CONFIG
# ================================

def load_pointing_config(config_path: str = "config/scenario.yaml") -> dict:
    cfg = load_yaml(config_path)
    return cfg.get("pointing", {})


# ================================
# BEAM DIVERGENCE
# ================================

def beam_divergence(
    wavelength: float,
    tx_diameter: float
) -> float:
    """
    Divergenza limitata da diffrazione:

    theta ≈ lambda / (pi * w0), con w0 ≈ D/2
    """
    w0 = tx_diameter / 2.0
    return wavelength / (PI * w0)


# ================================
# JITTER ANGOLARE (GAUSSIANO)
# ================================

def pointing_jitter(
    n_samples: int,
    sigma_theta: float
) -> np.ndarray:
    """
    Genera errore angolare gaussiano (rad)

    theta_err ~ N(0, sigma_theta^2)
    """
    return np.random.normal(loc=0.0, scale=sigma_theta, size=n_samples)


# ================================
# OFFSET RADIALE
# ================================

def radial_offset(
    theta_error: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    r = R * theta_err
    """
    return R * theta_error


# ================================
# SPOT SIZE
# ================================

def beam_radius(
    R: np.ndarray,
    divergence: float
) -> np.ndarray:
    """
    w(R) = R * theta_div
    """
    return R * divergence


# ================================
# COUPLING GAUSSIANO
# ================================

def pointing_loss(theta_error, divergence):
    theta_error = np.asarray(theta_error)

    # Protezione numerica forte
    divergence = np.maximum(divergence, 1e-12)

    ratio = (theta_error / divergence)**2

    eta = np.exp(-2.0 * ratio)

    # Clipping fisico
    eta = np.clip(eta, 0.0, 1.0)

    return eta
# ================================
# MODELLO COMPLETO
# ================================

def pointing_fading(
    R: np.ndarray,
    wavelength: float,
    tx_diameter: float,
    config: dict | None = None
) -> np.ndarray:
    """
    Modello completo di pointing:

    1. calcola divergenza
    2. genera jitter
    3. calcola perdita

    Returns
    -------
    eta_point : np.ndarray
    """

    if config is None:
        config = load_pointing_config()

    sigma_theta = config.get("sigma_theta", 1e-6)  # rad

    # ----------------------------
    # Divergenza fascio
    # ----------------------------
    theta_div = beam_divergence(wavelength, tx_diameter)

    # ----------------------------
    # Jitter
    # ----------------------------
    if sigma_theta < 1e-10:
        theta_err = np.zeros_like(R)
    else:
        theta_err = pointing_jitter(len(R),sigma_theta)    
    # ----------------------------
    # Perdita
    # ----------------------------
    eta_point = pointing_loss(theta_err, theta_div)

    return eta_point
