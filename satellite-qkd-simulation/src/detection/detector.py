import numpy as np

from src.detection.photon_model import signal_probability
from src.detection.noise import dark_count_probability
from src.detection.click_model import total_click_probability

def compute_detection(
    eta_total,
    mu,
    dark_rate,
    gate_time,
    e_opt
):
    """
    Complete detection pipeline.
    """

    eta_total = np.asarray(eta_total)

    # ----------------------------
    # SIGNAL
    # ----------------------------
    p_sig = signal_probability(mu, eta_total)

    # ----------------------------
    # DARK COUNTS
    # ----------------------------
    p_dark = dark_count_probability(
        dark_rate,
        gate_time,
        eta_total.shape
    )

    # ----------------------------
    # CLICK (corretto)
    # ----------------------------
    p_click = total_click_probability(p_sig, p_dark)

    # ----------------------------
    # ERROR (numeratore QBER)
    # ----------------------------
    p_err = e_opt * p_sig + 0.5 * p_dark

    # ----------------------------
    # QBER
    # ----------------------------
    p_click_safe = np.maximum(p_click, 1e-15)
    qber = p_err / p_click_safe

    return {
        "p_sig": p_sig,
        "p_dark": p_dark,
        "p_click": p_click,
        "p_err": p_err,
        "qber": qber,
    }
