import numpy as np


# ================================
# POISSON DISTRIBUTION
# ================================

def poisson_prob(n, mu):
    return np.exp(-mu) * mu**n / np.math.factorial(n)


# ================================
# YIELD ESTIMATION
# ================================

def estimate_y1(Q_mu, Q_nu, mu, nu):
    """
    Lower bound on single-photon yield (simplified)
    """

    return (mu * Q_nu * np.exp(nu) - nu * Q_mu * np.exp(mu)) / (mu * nu * (mu - nu))


def estimate_e1(E_mu, Q_mu, Y1, mu):
    """
    Upper bound on single-photon error
    """

    return (E_mu * Q_mu) / (mu * np.exp(-mu) * Y1)


# ================================
# DECOY KEY RATE
# ================================

def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def decoy_key_rate(Q_mu, E_mu, Y1, e1, f_ec=1.16, q=0.5):
    """
    Decoy-state key rate
    """

    return q * (
        - Q_mu * f_ec * binary_entropy(E_mu)
        + Y1 * (1 - binary_entropy(e1))
    )


# ================================
# INTERFACCIA
# ================================

def compute_decoy_rate(detection_mu, detection_nu, mu, nu):
    """
    detection_mu: risultati per segnale
    detection_nu: risultati per decoy
    """

    Q_mu = detection_mu["p_click"]
    E_mu = detection_mu["qber"]

    Q_nu = detection_nu["p_click"]

    # ----------------------------
    # STIME
    # ----------------------------
    Y1 = estimate_y1(Q_mu, Q_nu, mu, nu)
    Y1 = np.maximum(Y1, 1e-12)

    e1 = estimate_e1(E_mu, Q_mu, Y1, mu)
    e1 = np.clip(e1, 0, 0.5)

    # ----------------------------
    # KEY RATE
    # ----------------------------
    R = decoy_key_rate(Q_mu, E_mu, Y1, e1)
    R = np.maximum(R,0)
    return {
        "skr_decoy": R,
        "Y1": Y1,
        "e1": e1
    }
