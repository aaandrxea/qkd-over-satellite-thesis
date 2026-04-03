import numpy as np

EPS = 1e-15


# ==========================================================
# LOG-NORMAL INTENSITY NOISE (PHYSICALLY CORRECT)
# ==========================================================

def lognormal_intensity_noise(
    mu,
    rel_std,
    rng=None
):
    """
    Multiplicative intensity noise (laser power fluctuations).

    Model:
        μ_eff = μ * X

    where:
        X ~ lognormal(mean=1, variance controlled by rel_std)

    This ensures:
    - positivity
    - realistic multiplicative fluctuations

    Parameters
    ----------
    mu : array
    rel_std : relative std deviation (e.g. 0.05)
    """

    if rng is None:
        rng = np.random

    mu = np.asarray(mu)

    if rel_std <= 0:
        return mu

    # convert relative std → lognormal parameters
    sigma2 = np.log(1.0 + rel_std**2)
    sigma = np.sqrt(sigma2)
    mean = -0.5 * sigma2

    noise = rng.lognormal(mean=mean, sigma=sigma, size=mu.shape)

    return mu * noise


# ==========================================================
# CORRELATED INTENSITY NOISE (OPTIONAL)
# ==========================================================

def correlated_intensity_noise(
    mu,
    rel_std,
    correlation_time,
    dt,
    rng=None
):
    """
    Time-correlated multiplicative noise.

    Simple Ornstein-Uhlenbeck-like approximation.

    Useful for:
    - slow laser drift
    - thermal fluctuations

    Parameters
    ----------
    correlation_time : [s]
    dt : timestep [s]
    """

    if rng is None:
        rng = np.random

    mu = np.asarray(mu)

    if rel_std <= 0:
        return mu

    alpha = np.exp(-dt / max(correlation_time, EPS))

    noise = np.zeros_like(mu)

    for i in range(len(mu)):
        if i == 0:
            noise[i] = rng.normal(0.0, rel_std)
        else:
            noise[i] = alpha * noise[i-1] + np.sqrt(1 - alpha**2) * rng.normal(0.0, rel_std)

    # convert to lognormal multiplicative factor
    factor = np.exp(noise - 0.5 * rel_std**2)

    return mu * factor


# ==========================================================
# UNIFIED INTERFACE
# ==========================================================

def apply_intensity_noise(
    mu,
    rel_std,
    model="lognormal",
    correlation_time=None,
    dt=None,
    rng=None
):
    """
    Unified intensity noise interface.

    Available models:
    - "none"
    - "lognormal" (default, recommended)
    - "correlated"

    Parameters
    ----------
    mu : array
    rel_std : relative std
    """

    if rel_std <= 0:
        return mu

    if model == "none":
        return mu

    elif model == "lognormal":
        return lognormal_intensity_noise(mu, rel_std, rng=rng)

    elif model == "correlated":
        if correlation_time is None or dt is None:
            raise ValueError("correlation_time and dt required for correlated model")

        return correlated_intensity_noise(
            mu,
            rel_std,
            correlation_time,
            dt,
            rng=rng
        )

    else:
        raise ValueError(f"Unknown noise model: {model}")
