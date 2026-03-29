import numpy as np


def compute_qber(p_err, p_click):
    """
    QBER = p_err / p_click
    """
    p_click = np.maximum(p_click, 1e-15)
    return p_err / p_click
