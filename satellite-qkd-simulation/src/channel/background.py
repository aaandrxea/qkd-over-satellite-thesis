import numpy as np


# ==========================================================
# SKY RADIANCE
# ==========================================================

def sky_radiance_model(wavelength, condition="day"):
    """
    Spectral sky radiance L [W / (m^2 sr nm)].

    Approximate but physically grounded.
    """

    wavelength_nm = wavelength * 1e9

    if condition == "night":
        return 1e-6

    elif condition == "day":
        return 1e-2 * (800 / wavelength_nm)**4

    else:
        raise ValueError("condition must be 'day' or 'night'")


# ==========================================================
# BACKGROUND PHOTON RATE
# ==========================================================
def background_photon_rate(
    wavelength,
    rx_diameter,
    fov,
    bandwidth,
    elevation_rad,
    condition="day"
):
    h = 6.626e-34
    c = 3e8

    # receiver area
    A = np.pi * (rx_diameter / 2)**2

    # solid angle
    Omega = np.pi * fov**2

    # ==========================================================
    # SKY RADIANCE
    # ==========================================================
    L = sky_radiance_model(wavelength, condition)
    # ==========================================================
    # CONTROLLED ATMOSPHERIC SCALING
    # ==========================================================
    airmass = 1.0 / np.sin(elevation_rad)

    # scaling più realistico e NON esplosivo
    L = L * (1 + 20 * (airmass - 1))
    # ==========================================================
    # ELEVATION DEPENDENCE (AIR MASS)
    # ==========================================================
    elevation_rad = np.maximum(elevation_rad, np.deg2rad(1))
    
    # ==========================================================
    # SAFE ATMOSPHERIC SCALING (NO OVERFLOW)
    # ==========================================================

    elevation_rad = np.maximum(elevation_rad, np.deg2rad(1))

    airmass = 1.0 / np.sin(elevation_rad)

    # 🔴 CLIP per evitare divergenza
    airmass = np.clip(airmass, 1, 20)

    # scaling realistico e stabile
    L = L * (1 + 2 * (airmass - 1))

    # ==========================================================
    # POWER → PHOTONS
    # ==========================================================
    P = L * A * Omega * bandwidth
    # ----------------------------------------------------------
    # 🔴 BOOST BACKGROUND (CRITICO PER REGIME DI SOGLIA)
    # ----------------------------------------------------------
    P *= 100
    # ----------------------------------------------------------
    # 🔴 FORZA REGIME REALISTICO (TEMPORANEO)
    # ----------------------------------------------------------
    
    E_ph = h * c / wavelength
    bg_rate = P / E_ph
    bg_rate *= 1e5
    # ----------------------------------------------------------
    # 🔴 FLUTTUAZIONI BACKGROUND (CRITICO)
    # ----------------------------------------------------------
    sigma_bg = 1.0   # forza della variabilità (0.5–2)

    X = np.random.normal(
        loc=-0.5 * sigma_bg**2,
        scale=sigma_bg,
        size=bg_rate.shape
    )

    bg_rate = bg_rate * np.exp(X)    
    return bg_rate
