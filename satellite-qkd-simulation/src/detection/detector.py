import numpy as np


# ==========================================================
# MAIN DETECTION MODEL (PHYSICAL, POISSON-BASED)
# ==========================================================
def compute_detection(
    eta_total,
    mu,
    dark_rate,
    gate_time,
    e_opt,
    bg_rate=0.0,
    afterpulse_prob=0.02,
    dead_time=50e-9
    ):
    """
    Detection model with:
    - Poisson statistics
    - dead time (non-paralyzable)
    - afterpulsing
    """

    eta_total = np.asarray(eta_total)
    bg_rate = np.asarray(bg_rate)
    mu = float(mu)
    dark_rate = float(dark_rate)
    gate_time = float(gate_time)
    afterpulse_prob = float(afterpulse_prob)
    dead_time = float(dead_time)
    if bg_rate.shape == ():
        bg_rate = np.full_like(eta_total, bg_rate)

    # ======================================================
    # MEAN PHOTON NUMBERS
    # ======================================================
    lambda_sig = mu * eta_total
    lambda_dark = float(dark_rate) * float(gate_time)
    lambda_dark = np.full_like(lambda_sig, lambda_dark, dtype=float)
    lambda_bg = bg_rate * gate_time

    lambda_noise = lambda_dark + lambda_bg

    # ======================================================
    # POISSON PROBABILITIES
    # ======================================================
    lambda_total = lambda_sig + lambda_noise

    p_click_raw = 1 - np.exp(-lambda_total)

    # ======================================================
    # DEAD TIME EFFECT
    # ======================================================
    rate_raw = p_click_raw / gate_time

    rate_eff = rate_raw / (1 + rate_raw * dead_time)

    p_click = rate_eff * gate_time

    # ======================================================
    # AFTERPULSING
    # ======================================================
    p_after = p_click * afterpulse_prob

    p_noise = (1 - np.exp(-lambda_noise)) + p_after

    # ======================================================
    # SIGNAL COMPONENT
    # ======================================================
    p_sig = 1 - np.exp(-lambda_sig)

    # ======================================================
    # QBER
    # ======================================================
    error = 0.5 * p_noise + e_opt * p_sig

    qber = error / np.maximum(p_click, 1e-15)

    # ======================================================
    # SAFETY
    # ======================================================
    p_click = np.clip(p_click, 0.0, 1.0)
    qber = np.clip(qber, 0.0, 0.5)

    return {
        "p_sig": p_sig,
        "p_noise": p_noise,
        "p_click": p_click,
        "qber": qber,
    }

# ==========================================================
# MONTE CARLO VERSION (OPTIONAL, FOR VALIDATION)
# ==========================================================

def compute_detection_mc(
    eta_total,
    mu,
    dark_rate,
    gate_time,
    e_opt,
    bg_rate,
    n_pulses=10000
):
    """
    Monte Carlo detection model.

    Useful for validation of analytical model.
    """

    eta_total = np.asarray(eta_total)
    bg_rate = np.asarray(bg_rate)

    if bg_rate.shape == ():
        bg_rate = np.full_like(eta_total, bg_rate)

    # mean photon numbers
    lambda_sig = mu * eta_total
    lambda_dark = dark_rate * gate_time
    lambda_bg = bg_rate * gate_time

    lambda_dark = np.full_like(lambda_sig, lambda_dark)

    # sampling
    signal_counts = np.random.poisson(lambda_sig * n_pulses)
    dark_counts = np.random.poisson(lambda_dark * n_pulses)
    bg_counts = np.random.poisson(lambda_bg * n_pulses)

    noise_counts = dark_counts + bg_counts

    # total clicks
    total_counts = signal_counts + noise_counts
    p_click = total_counts / n_pulses

    # error model
    error_counts = (
        np.random.binomial(signal_counts, e_opt)
        + np.random.binomial(noise_counts, 0.5)
    )

    total_counts_safe = np.maximum(total_counts, 1)

    qber = error_counts / total_counts_safe

    return {
        "p_sig": signal_counts / n_pulses,
        "p_noise": noise_counts / n_pulses,
        "p_click": p_click,
        "qber": qber,
    }
