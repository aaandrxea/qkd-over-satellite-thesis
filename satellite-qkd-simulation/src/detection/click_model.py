import numpy as np

def total_click_probability(p_sig, p_dark):
    """
    p_click ≈ p_sig + p_dark
    """
    return p_sig + p_dark


def error_probability(p_sig, p_dark, e_opt):
    """
    p_err = 0.5 * p_dark + e_opt * p_sig
    """
    return 0.5 * p_dark + e_opt * p_sig


def qber(p_err, p_click):
    """
    QBER = p_err / p_click
    """
    p_click = np.maximum(p_click, 1e-15)
    return p_err / p_click
