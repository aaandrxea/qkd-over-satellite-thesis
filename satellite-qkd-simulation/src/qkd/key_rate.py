import numpy as np

from src.qkd.decoy_state import compute_key_rate_decoy


# ==========================================================
# Utility
# ==========================================================

def binary_entropy(p):
    """
    Binary Shannon entropy.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# ==========================================================
# BASELINE (NO DECOY) — per confronto
# ==========================================================

def secret_key_rate_basic(p_click, qber, f_ec=1.16):
    """
    Standard BB84 (no decoy) — NON sicuro realisticamente,
    usato solo come baseline.
    """

    H = binary_entropy(qber)

    return np.maximum(
        p_click * (1.0 - f_ec * H - H),
        0.0
    )


# ==========================================================
# FULL QKD PIPELINE (DECOY)
# ==========================================================

def compute_key_rate(
    detection_mu,
    detection_nu,
    detection_0,
    mu,
    nu,
    f_ec=1.16,
    q=0.5
):
    """
    Full decoy-state BB84 key rate.

    Parameters
    ----------
    detection_mu : detection results for signal
    detection_nu : detection results for decoy
    detection_0  : detection results for vacuum

    mu : signal intensity
    nu : decoy intensity

    Returns
    -------
    dict with:
        skr
        Y1
        e1
        Q_mu
        Q_nu
        Q_0
    """

    # ----------------------------
    # Gains (Q)
    # ----------------------------
    Q_mu = detection_mu["p_click"]
    Q_nu = detection_nu["p_click"]
    Q_0  = detection_0["p_click"]

    # ----------------------------
    # QBER
    # ----------------------------
    E_mu = detection_mu["qber"]
    E_nu = detection_nu["qber"]

    # ----------------------------
    # Decoy key rate
    # ----------------------------
    R, Y1, e1 = compute_key_rate_decoy(
        mu,
        nu,
        Q_mu,
        Q_nu,
        Q_0,
        E_mu,
        E_nu,
        f_ec=f_ec,
        q=q
    )

    return {
        "skr": R,
        "Y1": Y1,
        "e1": e1,
        "Q_mu": Q_mu,
        "Q_nu": Q_nu,
        "Q_0": Q_0,
    }
