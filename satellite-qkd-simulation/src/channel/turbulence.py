import numpy as np


# ================================
# Cn^2 — Hufnagel-Valley
# ================================

def cn2_hufnagel_valley(h, A=1.7e-14, v=21.0):
    term1 = 0.00594 * (v / 27.0)*2 * (1e-5 * h)*10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A * np.exp(-h / 100.0)
    return term1 + term2 + term3


# ================================
# Geometria path
# ================================

def path_altitude(s, elevation, R_E=6371e3):
    """
    Altitude along curved Earth path

    Parameters
    ----------
    s : path coordinate [m]
    elevation : radians
    R_E : Earth radius [m]

    Returns
    -------
    h : altitude [m]
    """
    return np.sqrt(R_E*2 + s*2 + 2 * R_E * s * np.sin(elevation)) - R_E

# ================================
# Rytov variance (σ_R^2)
# ================================

def rytov_variance_integral(wavelength, distance, elevation,
                           A=1.7e-14, v=21.0, n_steps=500):

    k = 2 * np.pi / wavelength
    s = np.linspace(0, distance, n_steps)

    h = path_altitude(s, elevation)
    cn2 = cn2_hufnagel_valley(h, A=A, v=v)

    weight = (s / distance)**(5.0 / 6.0)
    integrand = cn2 * weight

    integral = np.trapezoid(integrand, s)

    return 1.23 * (k**(7.0 / 6.0)) * integral


# ================================
# LOG-NORMAL (weak turbulence)
# ================================

def lognormal_fading(sigma_R2):
    sigma_chi2 = sigma_R2 / 4.0
    mu = -sigma_chi2 / 2.0
    return np.exp(mu + sigma_chi2)


# ================================
# GAMMA-GAMMA (strong turbulence)
# ================================
def gamma_gamma_sample(alpha, beta, size=None):
    """
    Sample Gamma-Gamma fading
    """
    X = np.random.gamma(alpha, 1/alpha, size=size)
    Y = np.random.gamma(beta, 1/beta, size=size)
    return X * Y
def gamma_gamma_fading(sigma_R2):
    # Andrews & Phillips model

    sigma = np.sqrt(sigma_R2)

    alpha = np.exp(
        0.49 * sigma**2 /
        (1 + 1.11 * sigma*(12/5))*(7/6)
    )

    beta = np.exp(
        0.51 * sigma**2 /
        (1 + 0.69 * sigma*(12/5))*(5/6)
    )

    return (gamma_gamma_sample(alpha,beta))

# ================================
# MAIN INTERFACE
# ================================

def turbulence_fading(
    elevation,
    distance,
    wavelength,
    model="auto",
    A=1.7e-14,
    v=21.0,
    n_steps=500
):
    """
    Returns:
    --------
    eta_turb : attenuation factor (0–1)
    sigma_R2 : Rytov variance
    """

    elevation = np.asarray(elevation)
    distance = np.asarray(distance)

    sigma_R2 = np.zeros_like(distance)
    eta = np.zeros_like(distance)

    for i in range(len(distance)):

        sigma = rytov_variance_integral(
            wavelength,
            distance[i],
            elevation[i],
            A=A,
            v=v,
            n_steps=n_steps
        )

        sigma_R2[i] = sigma

        # ----------------------------
        # Regime selection
        # ----------------------------
        if model == "auto":
            if sigma < 0.3:
                val = lognormal_fading(sigma)
            else:
                val = gamma_gamma_fading(sigma)

        elif model == "lognormal":
            val = lognormal_fading(sigma)

        elif model == "gamma-gamma":
            val = gamma_gamma_fading(sigma)

        else:
            raise ValueError(f"Unknown model: {model}")

        # ----------------------------
        # CLAMP FISICO
        # ----------------------------
        eta[i] = np.clip(val, 0.0, 1.0)

    return eta, sigma_R2
