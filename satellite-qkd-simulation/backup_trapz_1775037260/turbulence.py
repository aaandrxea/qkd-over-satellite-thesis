import numpy as np

EPS = 1e-15


# ==========================================================
# CN2 PROFILE (HUFNAGEL-VALLEY)
# ==========================================================

def cn2_hufnagel_valley(h, A=1e-14, v=21.0):
    """
    Refractive index structure constant profile.

    Hufnagel-Valley model:

    Cn²(h) = 0.00594 (v/27)^2 (10^-5 h)^10 exp(-h/1000)
             + 2.7e-16 exp(-h/1500)
             + A exp(-h/100)

    Parameters
    ----------
    h : altitude [m]
    A : ground turbulence strength
    v : wind speed [m/s]
    """

    h = np.asarray(h)

    term1 = 0.00594 * (v / 27.0)**2 * (1e-5 * h)**10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A * np.exp(-h / 100.0)

    return term1 + term2 + term3


# ==========================================================
# RYTOV VARIANCE
# ==========================================================

def compute_rytov_variance(
    wavelength,
    R,
    elevation,
    config=None
):
    """
    Compute Rytov variance σ_R².

    For plane wave approximation:

        σ_R² = 1.23 k^(7/6) ∫ Cn²(h) z^(5/6) dz

    Approximated along slant path.

    Parameters
    ----------
    wavelength : [m]
    R : distance [m]
    elevation : [rad]
    """

    if config is None:
        config = {}

    A = config.get("A", 1e-14)
    v = config.get("v", 21.0)
    n_steps = config.get("n_steps", 200)

    k = 2.0 * np.pi / wavelength

    R = np.asarray(R)
    elevation = np.asarray(elevation)

    sigma_R2 = np.zeros_like(R)

    for i in range(len(R)):

        # path discretization
        z = np.linspace(0, R[i], n_steps)

        # altitude along slant path
        h = z * np.sin(elevation[i])

        cn2 = cn2_hufnagel_valley(h, A=A, v=v)

        integrand = cn2 * (z + EPS)**(5.0 / 6.0)

        integral = np.trapz(integrand, z)

        sigma_R2[i] = 1.23 * k**(7.0 / 6.0) * integral

    return np.maximum(sigma_R2, 0.0)


# ==========================================================
# DISTRIBUTIONS
# ==========================================================

def lognormal_parameters(sigma_R2):
    """
    Convert Rytov variance to lognormal variance.
    """
    sigma_ln2 = np.log(1.0 + sigma_R2)
    return sigma_ln2


def gamma_gamma_parameters(sigma_R2):
    """
    Gamma-Gamma parameters for strong turbulence.

    Valid for σ_R² > ~1
    """

    sigma_R2 = np.maximum(sigma_R2, EPS)

    alpha = (
        np.exp(
            0.49 * sigma_R2
            / (1 + 1.11 * sigma_R2**(12.0/5.0))**(7.0/6.0)
        ) - 1
    )**(-1)

    beta = (
        np.exp(
            0.51 * sigma_R2
            / (1 + 0.69 * sigma_R2**(12.0/5.0))**(5.0/6.0)
        ) - 1
    )**(-1)

    return alpha, beta


# ==========================================================
# SAMPLING
# ==========================================================

def sample_lognormal(sigma_R2, rng=None):
    """
    Lognormal fading (weak turbulence).
    """

    if rng is None:
        rng = np.random

    sigma_ln2 = lognormal_parameters(sigma_R2)
    sigma_ln = np.sqrt(sigma_ln2)

    mu_ln = -0.5 * sigma_ln2

    return rng.lognormal(mean=mu_ln, sigma=sigma_ln)


def sample_gamma_gamma(sigma_R2, rng=None):
    """
    Gamma-Gamma fading (strong turbulence).
    """

    if rng is None:
        rng = np.random

    alpha, beta = gamma_gamma_parameters(sigma_R2)

    X = rng.gamma(alpha, 1.0 / alpha)
    Y = rng.gamma(beta, 1.0 / beta)

    return X * Y


def sample_fading(
    sigma_R2,
    model="auto",
    rng=None
):
    """
    Sample turbulence fading.

    Models:
    - lognormal (weak)
    - gamma-gamma (strong)
    - auto (switch)

    Returns
    -------
    eta_turb (mean ≈ 1)
    """

    sigma_R2 = np.asarray(sigma_R2)

    if model == "lognormal":
        return sample_lognormal(sigma_R2, rng=rng)

    elif model == "gamma-gamma":
        return sample_gamma_gamma(sigma_R2, rng=rng)

    elif model == "auto":

        eta = np.zeros_like(sigma_R2)

        weak = sigma_R2 < 1.0

        eta[weak] = sample_lognormal(sigma_R2[weak], rng=rng)

        eta[~weak] = sample_gamma_gamma(sigma_R2[~weak], rng=rng)

        return eta

    else:
        raise ValueError(f"Unknown turbulence model: {model}")


# ==========================================================
# BEAM WANDER (USED BY POINTING)
# ==========================================================

def beam_wander_std(
    wavelength,
    R,
    elevation,
    config=None
):
    """
    Beam wander angular standard deviation.

    Approximate relation:
        σ_bw² ∝ σ_R² / k²

    Returns angular deviation [rad]
    """

    sigma_R2 = compute_rytov_variance(
        wavelength,
        R,
        elevation,
        config=config
    )

    k = 2.0 * np.pi / wavelength

    sigma_bw = np.sqrt(sigma_R2) / (k + EPS)

    return sigma_bw
