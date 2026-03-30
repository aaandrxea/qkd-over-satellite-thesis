import numpy as np

from src.utils.constants import RHO_0, H_SCALE, EPS, R_EARTH
from src.utils.io import load_yaml


# ==========================================================
# CONFIG
# ==========================================================

def load_atmosphere_config(config_path: str = "config/atmosphere.yaml") -> dict:
    return load_yaml(config_path)


# ==========================================================
# AIR DENSITY (Exponential atmosphere)
# ==========================================================

def air_density(h: np.ndarray) -> np.ndarray:
    """
    Exponential atmospheric density model.

    rho(h) = rho_0 * exp(-h / H)

    Valid up to ~20 km (sufficient for optical propagation).

    Parameters
    ----------
    h : altitude [m]

    Returns
    -------
    rho : [kg/m^3]
    """
    return RHO_0 * np.exp(-h / H_SCALE)


# ==========================================================
# GEOMETRY: SLANT PATH WITH EARTH CURVATURE
# ==========================================================

def slant_path_altitude(s: np.ndarray, elevation: float, h_ground: float = 0.0) -> np.ndarray:
    """
    Computes altitude along slant path using spherical Earth geometry.

    h(s) = sqrt(R_E^2 + s^2 + 2 R_E s sin(el)) - R_E

    Parameters
    ----------
    s : path coordinate [m]
    elevation : elevation angle [rad]
    h_ground : ground station altitude [m]

    Returns
    -------
    h : altitude along path [m]
    """
    return np.sqrt(R_EARTH**2 + s**2 + 2 * R_EARTH * s * np.sin(elevation)) - R_EARTH + h_ground


# ==========================================================
# EXTINCTION MODEL
# ==========================================================
def extinction_coefficient(
    h: np.ndarray,
    wavelength: float,
    config: dict
) -> np.ndarray:
    """
    High-fidelity atmospheric extinction model.

    Includes:
    - Rayleigh scattering (λ^-4)
    - Aerosol (Ångström law)
    - Humidity absorption (empirical)
    - Ozone absorption (simplified band model)

    Parameters
    ----------
    h : altitude [m]
    wavelength : [m]
    config : dict

    Returns
    -------
    alpha : [1/m]
    """

    # ------------------------------------------------------
    # Input
    # ------------------------------------------------------
    h = np.asarray(h)
    wavelength = float(wavelength)

    wl_um = wavelength * 1e6  # micrometri

    # ------------------------------------------------------
    # 1. RAYLEIGH SCATTERING
    # ------------------------------------------------------
    alpha_ray_0 = float(config.get("alpha_rayleigh_0", 1e-5))
    lambda_ref = float(config.get("lambda_ref", 800e-9))

    alpha_ray = (
        alpha_ray_0 *
        (lambda_ref / wavelength) ** 4 *
        (air_density(h) / RHO_0)
    )

    # ------------------------------------------------------
    # 2. AEROSOL (Ångström model)
    # ------------------------------------------------------
    alpha_aer_0 = float(config.get("alpha_aer_0", 5e-5))
    angstrom = float(config.get("angstrom_exponent", 1.3))
    H_aer = float(config.get("H_aer", 1200.0))

    alpha_aer = (
        alpha_aer_0 *
        (wl_um / 0.55) ** (-angstrom) *
        np.exp(-h / H_aer)
    )

    # ------------------------------------------------------
    # 3. HUMIDITY ABSORPTION (empirical)
    # ------------------------------------------------------
    humidity = float(config.get("humidity", 0.5))  # 0–1

    # absorption stronger in IR
    alpha_hum_0 = float(config.get("alpha_humidity_0", 1e-5))

    alpha_hum = (
        alpha_hum_0 *
        humidity *
        np.exp(-h / 2000.0) *
        (wl_um / 1.0)  # cresce verso IR
    )

    # ------------------------------------------------------
    # 4. OZONE ABSORPTION (simplified band)
    # ------------------------------------------------------
    alpha_oz_0 = float(config.get("alpha_ozone_0", 5e-6))

    # picco UV-vis (~0.6 μm), trascurabile in IR
    ozone_band = np.exp(-((wl_um - 0.6) / 0.2)**2)

    alpha_oz = alpha_oz_0 * ozone_band * np.exp(-h / 15000.0)

    # ------------------------------------------------------
    # TOTAL
    # ------------------------------------------------------
    alpha = alpha_ray + alpha_aer + alpha_hum + alpha_oz

    return np.clip(alpha, 0.0, None)
# ==========================================================
# TRANSMITTANCE (NUMERICAL INTEGRATION)
# ==========================================================

def atmospheric_transmittance(
    elevation: np.ndarray,
    R: np.ndarray,
    config: dict | None = None
) -> np.ndarray:
    """
    Computes atmospheric transmittance:

    η = exp(- ∫ α(h(s)) ds )

    Fully physical model with:
    - spherical geometry
    - altitude-dependent extinction

    Parameters
    ----------
    elevation : [rad]
    R : slant range [m]

    Returns
    -------
    eta_atm : np.ndarray
    """

    if config is None:
        config = load_atmosphere_config()

    wavelength = config.get("wavelength", 800e-9)
    n_steps = config.get("n_steps", 300)

    elevation = np.asarray(elevation)
    R = np.asarray(R)

    eta = np.zeros_like(R)

    for i in range(len(R)):

        s = np.linspace(0, R[i], n_steps)

        h = slant_path_altitude(s, elevation[i])

        alpha = extinction_coefficient(h, wavelength, config)

        integral = np.trapezoid(alpha, s)

        eta[i] = np.exp(-integral)

    return eta
