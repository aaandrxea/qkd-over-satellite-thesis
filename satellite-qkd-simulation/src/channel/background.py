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
    condition="day"
):
    """
    Compute background photon rate at receiver.
    """

    h = 6.626e-34
    c = 3e8

    # receiver area
    A = np.pi * (rx_diameter / 2)**2

    # solid angle
    Omega = np.pi * fov**2

    # radiance
    L = sky_radiance_model(wavelength, condition)

    # optical power
    P = L * A * Omega * bandwidth

    # photon energy
    E_ph = h * c / wavelength

    return P / E_ph
