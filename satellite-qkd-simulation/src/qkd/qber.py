import numpy as np

EPS = 1e-15


# ==========================================================
# NUMERICAL UTILITIES
# ==========================================================

def safe_divide(num, den):
    return num / np.maximum(den, EPS)


def validate_probability(x):
    return np.clip(x, 0.0, 1.0)


# ==========================================================
# ERROR MODEL
# ==========================================================

def compute_error_probability(
    p_sig,
    p_noise,
    e_opt
):
    """
    Total error probability.

    Contributions:
    - signal: optical misalignment (e_opt)
    - noise: random (50%)

    p_err = e_opt * p_sig + 0.5 * p_noise
    """

    p_sig = validate_probability(p_sig)
    p_noise = validate_probability(p_noise)

    p_err = e_opt * p_sig + 0.5 * p_noise

    return np.clip(p_err, 0.0, 1.0)


# ==========================================================
# QBER FROM COMPONENTS
# ==========================================================

def compute_qber_from_components(
    p_sig,
    p_noise,
    p_click,
    e_opt
):
    """
    Compute QBER from signal + noise decomposition.
    """

    p_err = compute_error_probability(p_sig, p_noise, e_opt)

    qber = safe_divide(p_err, p_click)

    return np.clip(qber, 0.0, 0.5)


# ==========================================================
# QBER FROM DETECTION DICT (PIPELINE-COMPATIBLE)
# ==========================================================

def compute_qber_from_detection(
    detection,
    e_opt=None
):
    """
    Compute QBER using detection output.

    Compatible with new detector structure.

    Priority:
    1. If qber already present → return it
    2. Otherwise compute from components
    """

    # already computed (preferred)
    if "qber" in detection:
        return validate_probability(detection["qber"])

    # fallback (recompute)
    p_sig = detection["p_sig"]
    p_noise = detection["p_noise"]
    p_click = detection["p_click"]

    if e_opt is None:
        raise ValueError("e_opt required if qber not present in detection")

    return compute_qber_from_components(
        p_sig,
        p_noise,
        p_click,
        e_opt
    )


# ==========================================================
# DIRECT QBER (GENERIC)
# ==========================================================

def compute_qber(
    p_sig,
    p_noise,
    p_click,
    e_opt
):
    """
    Direct QBER computation (explicit interface).

    Useful for:
    - testing
    - validation
    - standalone usage
    """

    return compute_qber_from_components(
        p_sig,
        p_noise,
        p_click,
        e_opt
    )
