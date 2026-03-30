import numpy as np


# ==========================================================
# SIGNAL DETECTION
# ==========================================================

def signal_probability(mu, eta):
    """
    Detection probability for signal photons (coherent state).

    p_sig = 1 - exp(-mu * eta)
    """
    mu = float(mu)
    eta = np.asarray(eta)

    return 1.0 - np.exp(-mu * eta)


# ==========================================================
# DARK COUNTS
# ==========================================================

def dark_count_probability(dark_rate, gate_time):
    """
    Dark count probability per gate.

    p_dark = 1 - exp(-R_dark * t_gate)
    """
    dark_rate = float(dark_rate)
    gate_time = float(gate_time)

    return 1.0 - np.exp(-dark_rate * gate_time)


# ==========================================================
# TOTAL CLICK PROBABILITY
# ==========================================================

def total_detection_probability(mu, eta, dark_rate, gate_time):
    """
    Total click probability:

    p_click = p_sig + p_dark - p_sig * p_dark
    """

    p_sig = signal_probability(mu, eta)
    p_dark = dark_count_probability(dark_rate, gate_time)

    return p_sig + p_dark - p_sig * p_dark


# ==========================================================
# ERROR PROBABILITY (QBER COMPONENT)
# ==========================================================

def error_probability(
    mu,
    eta,
    dark_rate,
    gate_time,
    e_opt=0.02
):
    """
    Total error probability.

    Contributions:
    - signal errors (optical misalignment)
    - dark counts (random → 50%)

    e = (e_opt * p_sig + 0.5 * p_dark) / p_click
    """

    p_sig = signal_probability(mu, eta)
    p_dark = dark_count_probability(dark_rate, gate_time)

    p_click = p_sig + p_dark - p_sig * p_dark

    # evita divisione per zero
    p_click = np.maximum(p_click, 1e-15)

    e = (e_opt * p_sig + 0.5 * p_dark) / p_click

    return e
