import numpy as np


# ================================
# Utility
# ================================

def binary_entropy(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


# ================================
# Lo–Ma–Chen bounds
# ================================

def estimate_Y1_lower(mu, nu, Q_mu, Q_nu):
    """
    Lower bound on single-photon yield Y1
    """
    numerator = (
        Q_nu * np.exp(nu)
        - (nu*2 / mu*2) * Q_mu * np.exp(mu)
    )

    denominator = mu * nu - nu**2

    Y1 = numerator / denominator

    # stabilità numerica
    Y1 = np.maximum(Y1, 1e-15)

    return Y1


def estimate_e1_upper(mu, nu, E_nu, Q_nu, Y1):
    """
    Upper bound on single-photon error rate e1
    """
    e1 = (E_nu * Q_nu * np.exp(nu)) / (nu * Y1)

    # limiti fisici
    e1 = np.clip(e1, 0.0, 0.5)

    return e1


# ================================
# Key Rate (asintotico)
# ================================

def compute_key_rate_decoy(
    mu,
    nu,
    Q_mu,
    Q_nu,
    E_mu,
    E_nu,
    f_ec=1.16,
    q=0.5
):
    """
    Returns:
    --------
    skr_decoy : secret key rate
    Y1 : single photon yield
    e1 : single photon error rate
    """

    # ----------------------------
    # Bounds
    # ----------------------------

    Y1 = estimate_Y1_lower(mu, nu, Q_mu, Q_nu)
    e1 = estimate_e1_upper(mu, nu, E_nu, Q_nu, Y1)

    # ----------------------------
    # Single-photon gain
    # ----------------------------

    Q1 = Y1 * mu * np.exp(-mu)

    # ----------------------------
    # Key rate
    # ----------------------------

    R = q * (
        - Q_mu * f_ec * binary_entropy(E_mu)
        + Q1 * (1 - binary_entropy(e1))
    )

    # clamp fisico
    R = np.maximum(R, 0.0)

    return R, Y1, e1
