import numpy as np

from src.utils.constants import R_EARTH


# ==========================================================
# HUFNAGEL–VALLEY MODEL
# ==========================================================

def cn2_hufnagel_valley(h, A=1.7e-14, v=21.0):
    """
    Hufnagel–Valley Cn^2 profile.
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
    A = 1e-13,   #A=1.7e-14,
    v=21.0,
    n_steps=500
):
    """
    Rytov variance for plane wave (downlink).

    σ_R^2 = 1.23 k^(7/6) ∫ Cn^2(h(s)) (s/L)^(5/6) ds

    Returns
    -------
    sigma_R2
    """

    k = 2 * np.pi / wavelength

    s = np.linspace(0, distance, n_steps)

    h = path_altitude(s, elevation)

    cn2 = cn2_hufnagel_valley(h, A=A, v=v)

    weight = (s / distance)**(5.0 / 6.0)

    integral = np.trapezoid(cn2 * weight, s)

    return 1.23 * k**(7.0 / 6.0) * integral


# ==========================================================
# LOGNORMAL MODEL (WEAK TURBULENCE)
# ==========================================================

def lognormal_sample(sigma_R2, size=None):
    """
    Lognormal fading (weak turbulence).

    Intensity fluctuations:
    σ_chi^2 = σ_R^2 / 4
    """

    sigma_chi2 = sigma_R2 / 4.0

    mu = -sigma_chi2 / 2.0
    sigma = np.sqrt(sigma_chi2)

    return np.random.lognormal(mean=mu, sigma=sigma, size=size)


# ==========================================================
# GAMMA-GAMMA MODEL (STRONG TURBULENCE)
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


def gamma_gamma_sample(alpha, beta, size=None):
    X = np.random.gamma(alpha, 1/alpha, size=size)
    Y = np.random.gamma(beta, 1/beta, size=size)
    return X * Y


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
    n_steps=500,
    size=None
):
    """
    Returns
    -------
    eta_turb : stochastic fading
    sigma_R2 : Rytov variance
    """

    elevation = np.asarray(elevation)
    distance = np.asarray(distance)
    A = float(A)
    v = float(v)
    n_steps = int(n_steps)
    eta = np.zeros_like(distance)
    sigma_R2 = np.zeros_like(distance)

    for i in range(len(distance)):

        sigma = rytov_variance(
            wavelength,
            distance[i],
            elevation[i],
            A=A,
            v=v,
            n_steps=n_steps
        )
        
        sigma_R2[i] = sigma

        if model == "auto":
            if sigma < 0.3:
                val = lognormal_sample(sigma, size=size)
            else:
                alpha, beta = gamma_gamma_parameters(sigma)
                val = gamma_gamma_sample(alpha, beta, size=size)

        elif model == "lognormal":
            val = lognormal_sample(sigma, size=size)

        elif model == "gamma-gamma":
            alpha, beta = gamma_gamma_parameters(sigma)
            val = gamma_gamma_sample(alpha, beta, size=size)

        else:
            raise ValueError(f"Unknown model: {model}")

        if size is not None:
            val = val[0]

        eta[i] = np.clip(val, 0.0, None)

    return eta, sigma_R2
def beam_wander_std(wavelength, distance, elevation, Cn2_0=1e-18):
    """
    Beam wander standard deviation [rad].

    Scaled, physically realistic model.
    """


    wavelength = float(wavelength)
    distance = np.asarray(distance)
    elevation = np.asarray(elevation)

    k = 2 * np.pi / wavelength

    # evita divergenze
    sin_el = np.sin(elevation)
    sin_el = np.clip(sin_el, 1e-3, None)

    # slant path scaling
    sec_theta = 1.0 / sin_el

    # effective turbulence
    Cn2_eff = Cn2_0 * sec_theta

    # ----------------------------
    # CORREZIONE FISICA CRITICA
    # ----------------------------
    # normalizzazione empirica realistica
    L = distance

    sigma_bw = np.sqrt(2.91 * Cn2_eff * L) * (wavelength ** (-1/6))

    sigma_bw = np.clip(sigma_bw, 1e-8, 1e-4)
    return sigma_bw
