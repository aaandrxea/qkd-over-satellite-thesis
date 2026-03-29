import numpy as np

from src.detection.photon_model import signal_probability
from src.detection.noise import dark_count_probability
from src.detection.click_model import (
    total_click_probability,
    error_probability,
    qber
)


def compute_detection(eta_total, mu, p_dark, e_opt):
    """
    Pipeline detection completa
    """

    eta_total = np.asarray(eta_total)

    # ----------------------------
    # SIGNAL
    # ----------------------------
    p_sig = signal_probability(mu, eta_total)

    # ----------------------------
    # DARK COUNTS
    # ----------------------------
    p_dark_arr = dark_count_probability(p_dark, eta_total.shape)

    # ----------------------------
    # CLICK
    # ----------------------------
    p_click = total_click_probability(p_sig, p_dark_arr)

    # ----------------------------
    # ERROR
    # ----------------------------
    p_err = error_probability(p_sig, p_dark_arr, e_opt)

    # ----------------------------
    # QBER
    # ----------------------------
    qber_val = qber(p_err, p_click)

    return {
        "p_sig": p_sig,
        "p_dark": p_dark_arr,
        "p_click": p_click,
        "p_err": p_err,
        "qber": qber_val
    }
