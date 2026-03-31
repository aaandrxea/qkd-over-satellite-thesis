import numpy as np

# ==========================================================
# COSTANTI
# ==========================================================

h = 6.626e-34
c = 3e8

def background_photon_rate(
    wavelength,
    rx_diameter,
    fov,
    bandwidth,
    elevation_rad,
    condition="day",
    config=None
):
    elevation_rad = np.asarray(elevation_rad)

    if config is None:
        config = {}

    bg_cfg = config.get("background", {})

    g = float(bg_cfg.get("mie_g", 0.7))
    albedo = float(bg_cfg.get("albedo", 0.3))
    sigma = float(bg_cfg.get("fluct_sigma", 0.25))
    angstrom = float(bg_cfg.get("angstrom", 1.3))

    # COSTANTI
    h = 6.626e-34
    c = 3e8

    # GEOMETRIA
    A = np.pi * (rx_diameter / 2)**2
    Omega = np.pi * fov**2

    # SOLAR GEOMETRY
    theta_z = np.pi / 2 - elevation_rad
    cos_theta = np.clip(np.cos(theta_z), 0.0, 1.0)

    # RAYLEIGH
    wl_nm = wavelength * 1e9
    L_ray = 5e-3 * (800 / wl_nm)**4 * cos_theta

    # MIE
    wl_um = wavelength * 1e6
    phase = (1 - g**2) / (1 + g**2 - 2*g*cos_theta)**1.5
    L_mie = 1e-2 * (wl_um / 0.55)**(-angstrom) * phase

    # ALBEDO
    sin_el = np.sin(elevation_rad)
    sin_el = np.clip(sin_el, 0.05, 1.0)
    weight = (1 - sin_el)**2
    L_alb = albedo * 5e-3 * weight

    # TOTAL
    if condition == "night":
        L = np.full_like(elevation_rad, 1e-6)
    else:
        L = L_ray + L_mie + L_alb

    # AIR MASS
    airmass = 1.0 / sin_el
    L = L * airmass

    # POWER → PHOTONS
    P = L * A * Omega * bandwidth
    E_ph = h * c / wavelength
    bg_rate = P / E_ph

    # FLUTTUAZIONI
    X = np.random.normal(
        loc=-0.5 * sigma**2,
        scale=sigma,
        size=bg_rate.shape
    )
    bg_rate = bg_rate * np.exp(X)

    return np.clip(bg_rate, 0.0, None)
# ==========================================================
# SOLAR GEOMETRY
# ==========================================================

def solar_zenith_angle(elevation_rad):
    return np.pi / 2 - elevation_rad


# ==========================================================
# HENYEY-GREENSTEIN PHASE FUNCTION (MIE)
# ==========================================================

def hg_phase_function(theta, g=0.7):
    """
    Anisotropic scattering phase function.
    """

    cos_theta = np.cos(theta)

    return (1 - g**2) / (1 + g**2 - 2 * g * cos_theta)**1.5


# ==========================================================
# RAYLEIGH COMPONENT
# ==========================================================

def rayleigh_radiance(wavelength, elevation_rad):
    wl_nm = wavelength * 1e9

    theta_z = solar_zenith_angle(elevation_rad)
    cos_theta = np.clip(np.cos(theta_z), 0.0, 1.0)

    # λ^-4
    L0 = 5e-3 * (800 / wl_nm)**4

    return L0 * cos_theta


# ==========================================================
# MIE (AEROSOL) COMPONENT
# ==========================================================

def mie_radiance(wavelength, elevation_rad, g=0.7):
    wl_um = wavelength * 1e6

    theta_z = solar_zenith_angle(elevation_rad)

    # scattering angle approx (sole → linea di vista)
    theta_scat = theta_z

    phase = hg_phase_function(theta_scat, g=g)

    # Ångström scaling
    alpha = 1.3
    L0 = 1e-2 * (wl_um / 0.55)**(-alpha)

    return L0 * phase


# ==========================================================
# EARTH ALBEDO COMPONENT
# ==========================================================

def albedo_radiance(elevation_rad, albedo=0.3):
    """
    Ground-reflected light entering FOV.
    Dominant at low elevation.
    """

    sin_el = np.sin(elevation_rad)
    sin_el = np.clip(sin_el, 0.05, 1.0)

    # più forte a basse elevazioni
    weight = (1 - sin_el)**2

    L0 = 5e-3

    return albedo * L0 * weight


# ==========================================================
# TOTAL SKY RADIANCE
# ==========================================================

def sky_radiance_advanced(
    wavelength,
    elevation_rad,
    condition="day"
):
    elevation_rad = np.asarray(elevation_rad)

    if condition == "night":
        return np.full_like(elevation_rad, 1e-6)

    L_ray = rayleigh_radiance(wavelength, elevation_rad)
    L_mie = mie_radiance(wavelength, elevation_rad)
    L_alb = albedo_radiance(elevation_rad)

