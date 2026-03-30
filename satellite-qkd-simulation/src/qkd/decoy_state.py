import numpy as np


# ==========================================================
# Utility
# ==========================================================

def binary_entropy(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


# ==========================================================
# Yield estimation (3-intensity decoy)
# ==========================================================

def estimate_Y0(Q_0):
    """
    Vacuum yield (background)
    """
    return np.clip(Q_0, 0.0, 1.0)


def estimate_Y1_lower(mu, nu, Q_mu, Q_nu, Y0):
    """
    Lower bound on single-photon yield Y1
    (Lo–Ma–Chen, 3-intensity)
    """

    mu = float(mu)
    nu = float(nu)

    denom = mu * nu - nu**2
    denom = max(denom, 1e-15)

    term1 = Q_nu * np.exp(nu)
    term2 = (nu**2 / mu**2) * Q_mu * np.exp(mu)
    term3 = ((mu**2 - nu**2) / mu**2) * Y0

    Y1 = (mu / denom) * (term1 - term2 - term3)

    return np.clip(Y1, 0.0, 1.0)


def estimate_e1_upper(mu, nu, E_nu, Q_nu, Y0, Y1):
    """
    Upper bound on single-photon error rate e1
    """

    mu = float(mu)
    nu = float(nu)

    Y1 = np.maximum(Y1, 1e-15)

    numerator = E_nu * Q_nu * np.exp(nu) - 0.5 * Y0
    denominator = nu * Y1

    e1 = numerator / np.maximum(denominator, 1e-15)

    return np.clip(e1, 0.0, 0.5)


# ==========================================================
# Gain
# ==========================================================

def single_photon_gain(mu, Y1):
    """
    Q1 = Y1 * mu * exp(-mu)
    """
    return np.maximum(Y1 * mu * np.exp(-mu), 0.0)


# ==========================================================
# Key Rate (asymptotic decoy BB84)
# ==========================================================

def compute_key_rate_decoy(
    mu,
    nu,
    Q_mu,
    Q_nu,
    Q_0,
    E_mu,
    E_nu,
    f_ec=1.16,
    q=0.5
):
    """
    Full decoy-state key rate (3-intensity BB84)

    Parameters
    ----------
    mu : signal intensity
    nu : weak decoy intensity
    Q_mu : gain (signal)
    Q_nu : gain (decoy)
    Q_0 : vacuum gain
    E_mu : QBER (signal)
    E_nu : QBER (decoy)

    Returns
    -------
    R : secret key rate
    Y1 : single photon yield
    e1 : single photon error
    """

    # ----------------------------
    # Vacuum
    # ----------------------------
    Y0 = estimate_Y0(Q_0)

    # ----------------------------
    # Bounds
    # ----------------------------
    Y1 = estimate_Y1_lower(mu, nu, Q_mu, Q_nu, Y0)
    e1 = estimate_e1_upper(mu, nu, E_nu, Q_nu, Y0, Y1)

    # ----------------------------
    # Single photon gain
    # ----------------------------
    Q1 = single_photon_gain(mu, Y1)

    # ----------------------------
    # Key rate
    # ----------------------------
    R = q * (
        - Q_mu * f_ec * binary_entropy(E_mu)
        + Q1 * (1 - binary_entropy(e1))
    )

    return np.maximum(R, 0.0), Y1, e1
