import numpy as np

EPS = 1e-15


# ==========================================================
# SCATTERING COEFFICIENTS
# ==========================================================

def rayleigh_coefficient(wavelength, alpha_0, lambda_ref):
    """
    Rayleigh scattering coefficient.

    α_R ∝ (λ_ref / λ)^4
    """
    return alpha_0 * (lambda_ref / wavelength)**4


def mie_coefficient(wavelength, alpha_0, lambda_ref, angstrom):
    """
    Aerosol (Mie) scattering using Ångström law.

    α_M ∝ (λ_ref / λ)^angstrom
    """
    return alpha_0 * (lambda_ref / wavelength)**angstrom


def absorption_coefficient(alpha_humidity, alpha_ozone):
    """
    Total absorption coefficient.
    """
    return alpha_humidity + alpha_ozone


# ==========================================================
# ALTITUDE PROFILES
# ==========================================================

def aerosol_profile(h, H_aer):
    """
    Exponential aerosol decay.

    α_M(h) ∝ exp(-h / H_aer)
    """
    return np.exp(-h / H_aer)


def humidity_profile(h, humidity):
    """
    Simplified humidity vertical profile.

    Strong near ground, decays exponentially.
    """
    return humidity * np.exp(-h / 2000.0)


# ==========================================================
# TOTAL EXTINCTION PROFILE
# ==========================================================

def extinction_profile(
    h,
    wavelength,
    config
):
    """
    Total extinction coefficient α(h).
    """

    lambda_ref = config.get("lambda_ref", wavelength)

    # base coefficients
    alpha_R0 = config.get("alpha_rayleigh_0", 5e-5)
    alpha_M0 = config.get("alpha_aer_0", 3e-4)
    alpha_H0 = config.get("alpha_humidity_0", 5e-5)
    alpha_O0 = config.get("alpha_ozone_0", 5e-6)

    angstrom = config.get("angstrom_exponent", 1.3)
    H_aer = config.get("H_aer", 1200)
    humidity = config.get("humidity", 0.5)

    # wavelength scaling
    alpha_R = rayleigh_coefficient(wavelength, alpha_R0, lambda_ref)
    alpha_M = mie_coefficient(wavelength, alpha_M0, lambda_ref, angstrom)

    # altitude scaling
    aer_profile = aerosol_profile(h, H_aer)
    hum_profile = humidity_profile(h, humidity)

    # components
    alpha_rayleigh = alpha_R
    alpha_mie = alpha_M * aer_profile
    alpha_abs = (alpha_H0 * hum_profile + alpha_O0)

    return alpha_rayleigh + alpha_mie + alpha_abs


# ==========================================================
# SLANT PATH INTEGRATION
# ==========================================================

def integrate_extinction(
    R,
    elevation,
    wavelength,
    config
):
    """
    Integrate extinction along slant path.

    ds = dz / sin(elevation)
    """

    n_steps = config.get("n_steps", 200)

    tau = np.zeros_like(R)

    for i in range(len(R)):

        z = np.linspace(0, R[i], n_steps)

        # altitude along path
        h = z * np.sin(elevation[i])

        alpha = extinction_profile(h, wavelength, config)

        integral = np.trapz(alpha, z)

        tau[i] = integral

    return tau


# ==========================================================
# MAIN ATMOSPHERIC TRANSMITTANCE
# ==========================================================

def atmospheric_transmittance(
    R,
    elevation,
    wavelength,
    config=None
):
    """
    Compute atmospheric transmission.

    η_atm = exp(-τ)
    """

    if config is None:
        config = {}

    R = np.asarray(R)
    elevation = np.asarray(elevation)

    tau = integrate_extinction(
        R,
        elevation,
        wavelength,
        config
    )

    eta = np.exp(-tau)

    return np.clip(eta, 0.0, 1.0)
