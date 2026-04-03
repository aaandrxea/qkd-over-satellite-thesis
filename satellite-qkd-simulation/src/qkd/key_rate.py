import numpy as np

from src.qkd.decoy_state import compute_key_rate_decoy

EPS = 1e-15


# ==========================================================
# NUMERICAL UTILITIES
# ==========================================================

def safe_clip(x, min_val=EPS, max_val=1.0):
    return np.clip(x, min_val, max_val)


def validate_probability(p):
    """
    Ensure valid probability range [0,1]
    """
    return np.clip(p, 0.0, 1.0)


# ==========================================================
# DETECTION PARSING
# ==========================================================

def extract_detection_quantities(detection):
    """
    Extract and validate detection quantities.

    Expected keys:
        - p_click
        - qber
    """

    Q = validate_probability(detection["p_click"])
    E = validate_probability(detection["qber"])

    return Q, E


# ==========================================================
# MAIN KEY RATE PIPELINE (DECOY BB84)
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
    Full decoy-state BB84 key rate (asymptotic).

    Pipeline:
        detection → gains → bounds → SKR

    Parameters
    ----------
    detection_mu : dict
    detection_nu : dict
    detection_0  : dict

    mu : signal intensity
    nu : decoy intensity

    Returns
    -------
    dict:
        skr
        Y1
        e1
        Q_mu
        Q_nu
        Q_0
        E_mu
        E_nu
    """

    # ======================================================
    # EXTRACT DETECTION DATA
    # ======================================================

    Q_mu, E_mu = extract_detection_quantities(detection_mu)
    Q_nu, E_nu = extract_detection_quantities(detection_nu)
    Q_0, _     = extract_detection_quantities(detection_0)

    # ======================================================
    # NUMERICAL SAFETY
    # ======================================================

    Q_mu = safe_clip(Q_mu)
    Q_nu = safe_clip(Q_nu)
    Q_0  = safe_clip(Q_0)

    E_mu = validate_probability(E_mu)
    E_nu = validate_probability(E_nu)

    # ======================================================
    # DECOY STATE ESTIMATION
    # ======================================================

    R, Y1, e1 = compute_key_rate_decoy(
        mu=mu,
        nu=nu,
        Q_mu=Q_mu,
        Q_nu=Q_nu,
        Q_0=Q_0,
        E_mu=E_mu,
        E_nu=E_nu,
        f_ec=f_ec,
        q=q
    )

    # ======================================================
    # POST-PROCESSING (STABILITY)
    # ======================================================

    R = np.maximum(R, 0.0)

    Y1 = np.clip(Y1, 0.0, 1.0)
    e1 = np.clip(e1, 0.0, 0.5)

    # ======================================================
    # OUTPUT
    # ======================================================

    return {
        "skr": R,

        # gains
        "Q_mu": Q_mu,
        "Q_nu": Q_nu,
        "Q_0": Q_0,

        # errors
        "E_mu": E_mu,
        "E_nu": E_nu,

        # decoy estimates
        "Y1": Y1,
        "e1": e1,
    }
