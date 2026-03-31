import numpy as np
from src.utils.constants import R_EARTH, EPS


# ==========================================================
# HUFNAGEL–VALLEY MODEL
# ==========================================================

def cn2_hufnagel_valley(h, A=1.7e-14, v=21.0):
    """
    Standard Hufnagel–Valley Cn^2 profile.

    Parameters
    ----------
    h : altitude [m]
    A : ground turbulence strength
    v : high-altitude wind speed [m/s]

    Returns
    -------
    Cn2 : [m^(-2/3)]
    """
    h = np.asarray(h)
    A = float(A)
    v = float(v)
    term1 = 0.00594 * (v / 27.0)**2 * (1e-5 * h)**10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A * np.exp(-h / 100.0)

    return term1 + term2 + term3


# ==========================================================
# GEOMETRY
# ==========================================================

def path_altitude(s, elevation):
    """
    Altitude along slant path (spherical Earth).
    """
    return np.sqrt(
        R_EARTH**2 + s**2 + 2 * R_EARTH * s * np.sin(elevation)
    ) - R_EARTH


# ==========================================================
# RYTOV VARIANCE (DOWNLINK)
# ==========================================================

def rytov_variance(
    wavelength,
    distance,
    elevation,
    A=1.7e-14,
    v=21.0,
    n_steps=300
):
    """
    Rytov variance for plane wave (downlink).

    σ_R^2 = 1.23 k^(7/6) ∫ Cn^2(h(s)) (s/L)^(5/6) ds
    """

    k = 2 * np.pi / wavelength

    s = np.linspace(0, distance, n_steps)
    h = path_altitude(s, elevation)

    cn2 = cn2_hufnagel_valley(h, A=A, v=v)

    weight = (s / (distance + EPS))**(5.0 / 6.0)

    integral = np.trapezoid(cn2 * weight, s)

    return 1.23 * k**(7.0 / 6.0) * integral


# ==========================================================
# LOGNORMAL (WEAK TURBULENCE)
# ==========================================================

def lognormal_fading(sigma_R2):
    """
    Lognormal fading for weak turbulence.

    σ_χ² = σ_R² / 4
    """
    sigma_chi2 = sigma_R2 / 4.0

    mu = -sigma_chi2 / 2.0
    sigma = np.sqrt(sigma_chi2)

    return np.random.lognormal(mean=mu, sigma=sigma)


# ==========================================================
# GAMMA-GAMMA (MODERATE/STRONG)
# ==========================================================

def gamma_gamma_parameters(sigma_R2):
    """
    Andrews & Phillips model.
    """

    sigma = np.sqrt(sigma_R2)

    alpha = np.exp(
        0.49 * sigma_R2 /
        (1 + 1.11 * sigma**(12.0/5.0))**(7.0/6.0)
    )

    beta = np.exp(
        0.51 * sigma_R2 /
        (1 + 0.69 * sigma**(12.0/5.0))**(5.0/6.0)
    )

    return alpha, beta


def gamma_gamma_fading(sigma_R2):
    alpha, beta = gamma_gamma_parameters(sigma_R2)

    X = np.random.gamma(alpha, 1/alpha)
    Y = np.random.gamma(beta, 1/beta)

    return X * Y


# ==========================================================
# MODEL SELECTION
# ==========================================================

def sample_fading(sigma_R2, model="auto"):
    """
    Select appropriate turbulence model.
    """

    if model == "auto":
        if sigma_R2 < 0.3:
            return lognormal_fading(sigma_R2)
        else:
            return gamma_gamma_fading(sigma_R2)

    elif model == "lognormal":
        return lognormal_fading(sigma_R2)

    elif model == "gamma-gamma":
        return gamma_gamma_fading(sigma_R2)

    else:
        raise ValueError(f"Unknown model: {model}")


# ==========================================================
# MAIN INTERFACE
# ==========================================================

def turbulence_fading(
    elevation,
    distance,
    wavelength,
    model="auto",
    A=1.7e-14,
    v=21.0,
    n_steps=300
):
    """
    Physically consistent turbulence fading.

    Returns
    -------
    eta_turb : fading coefficient
    sigma_R2 : Rytov variance
    """

    elevation = np.asarray(elevation)
    distance = np.asarray(distance)
    A = float(A)
    v = float(v)
    N = len(distance)

    eta = np.zeros(N)
    sigma_R2 = np.zeros(N)

    for i in range(N):

        sigma = rytov_variance(
            wavelength,
            distance[i],
            elevation[i],
            A=A,
            v=v,
            n_steps=n_steps
        )

        sigma_R2[i] = sigma

        fading = sample_fading(sigma, model=model)

        eta[i] = np.clip(fading, 0.0, None)

    return eta, sigma_R2


# ==========================================================
# BEAM WANDER (SEPARATO, NO CLIPPING ARTIFICIALE)
# ==========================================================

def beam_wander_std(
    wavelength,
    distance,
    elevation,
    Cn2_0=1e-14
):
    """
    Beam wander angular std [rad].

    Scaling: σ_bw ∝ sqrt(Cn² * L^3 / k^2)
    """

    wavelength = float(wavelength)

    distance = np.asarray(distance)
    elevation = np.asarray(elevation)

    k = 2 * np.pi / wavelength

    sin_el = np.sin(elevation)
    sin_el = np.clip(sin_el, 1e-3, None)

    sec_theta = 1.0 / sin_el

    # effective turbulence scaling
    Cn2_eff = Cn2_0 * sec_theta

    sigma_bw = np.sqrt(2.07 * Cn2_eff * distance**3 / k**2)

    return sigma_bw
