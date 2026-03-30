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
    wavelength: float,
    elevation: np.ndarray,
    size=None
):
    """
    Generates radial pointing offset including beam wander.

    Angular jitter → 2D Gaussian → radial Rayleigh

    r = R * θ
    """

    from src.channel.turbulence import beam_wander_std

    R = np.asarray(R)
    elevation = np.asarray(elevation)

    # ----------------------------
    # Beam wander
    # ----------------------------
    sigma_bw = beam_wander_std(wavelength, R, elevation)

    # ----------------------------
    # Total jitter
    # ----------------------------
    sigma_total = np.sqrt(sigma_theta**2 + sigma_bw**2)

    # ----------------------------
    # Edge case
    # ----------------------------
    if np.all(sigma_total < 1e-12):
        return np.zeros_like(R)

    # ----------------------------
    # 2D Gaussian → Rayleigh
    # ----------------------------
    n_samples = 5000

    theta_x = np.random.normal(0, sigma_total, size=(n_samples, len(R)))
    theta_y = np.random.normal(0, sigma_total, size=(n_samples, len(R)))

    theta = np.sqrt(theta_x**2 + theta_y**2)

    r = R * theta
    return r

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
    R,
    wavelength,
    elevation,
    tx_diameter,
    config=None
):
    """
    Full pointing loss model.

    Includes:
    - diffraction-limited beam propagation
    - beam wander (turbulence)
    - 2D jitter
    - Gaussian coupling
    """

    import numpy as np

    if config is None:
        config = load_pointing_config()

    # ----------------------------
    # Parameters
    # ----------------------------
    sigma_theta = float(config.get("sigma_theta", 3e-7))
    static_offset = float(config.get("static_offset", 0.0))

    R = np.asarray(R)
    elevation = np.asarray(elevation)

    # ----------------------------
    # Beam propagation
    # ----------------------------
    w0 = tx_diameter / 2.0
    z_R = np.pi * w0**2 / wavelength
    w_z = w0 * np.sqrt(1 + (R / z_R)**2)

    # ----------------------------
    # Pointing jitter (WITH beam wander)
    # ----------------------------
    r_jitter = pointing_offset(
        R,
        sigma_theta,
        wavelength,
        elevation
    )

    # r_jitter ora è (n_samples, N)

    r_total = np.sqrt(r_jitter**2 + static_offset**2)

    eta_samples = np.exp(-2 * (r_total / w_z)**2)

    # MEDIA CORRETTA
    eta_point = np.mean(eta_samples, axis=0)
    return eta_point
