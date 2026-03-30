import numpy as np


# ==========================================================
# DARK COUNTS
# ==========================================================

def dark_count_probability(dark_rate, gate_time, shape):
    """
    Dark count probability per gate.
    """
    dark_rate = float(dark_rate)
    gate_time = float(gate_time)

    p_dark = 1.0 - np.exp(-dark_rate * gate_time)

    return np.full(shape, p_dark)


# ==========================================================
# BACKGROUND
# ==========================================================

def background_probability(bg_rate, gate_time, shape):
    """
    Background photon probability per gate.
    """
    bg_rate = float(bg_rate)
    gate_time = float(gate_time)

    p_bg = 1.0 - np.exp(-bg_rate * gate_time)

    return np.full(shape, p_bg)


# ==========================================================
# TOTAL NOISE
# ==========================================================

def total_noise_probability(p_dark, p_bg):
    """
    Combined noise probability.

    p_noise = p_dark + p_bg - p_dark * p_bg
    """
    return p_dark + p_bg - p_dark * p_bg
