import numpy as np


# ================================
# SHANNON ENTROPY
# ================================

def binary_entropy(p):
    """
    H(p) = -p log2(p) - (1-p) log2(1-p)
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# ================================
# SECRET KEY RATE (BB84 base)
# ================================

def secret_key_rate(p_click, qber, f_ec=1.16):
    """
    R = p_click * (1 - f_ec * H(QBER) - H(QBER))
    """

    H = binary_entropy(qber)

    return p_click * (1.0 - f_ec * H - H)


# ================================
# INTERFACCIA
# ================================

def compute_key_rate(detection_results, f_ec=1.16):
    """
    detection_results = output di compute_detection
    """

    p_click = detection_results["p_click"]
    qber = detection_results["qber"]

    skr = secret_key_rate(p_click, qber, f_ec)

    return {
        "skr": skr
    }
