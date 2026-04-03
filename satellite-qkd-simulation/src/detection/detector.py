import numpy as np


# ==========================================================
# NUMERICAL SAFETY
# ==========================================================

EPS = 1e-15


# ==========================================================
# PHOTON STATISTICS (POISSON MODEL)
# ==========================================================

def poisson_click_probability(lmbda):
    """
    Probability of at least one detection event
    from Poisson process:

    P(click) = 1 - exp(-λ)
    """
    return 1.0 - np.exp(-lmbda)


# ==========================================================
# SIGNAL MODEL
# ==========================================================

def signal_mean_photons(mu, eta):
    """
    Mean detected photons from coherent state:

    λ_sig = μ * η
    """
    return mu * eta


# ==========================================================
# NOISE MODEL
# ==========================================================

def dark_mean_counts(dark_rate, gate_time):
    """
    Mean dark counts per gate:

    λ_dark = R_dark * t_gate
    """
    return dark_rate * gate_time


def background_mean_counts(bg_rate, gate_time):
    """
    Mean background photons per gate:

    λ_bg = R_bg * t_gate
    """
    return bg_rate * gate_time


# ==========================================================
# MAIN DETECTION MODEL
# ==========================================================

def compute_detection(
    eta_total,
    mu,
    gate_time,
    dark_rate,
    bg_rate=0.0,
    e_opt=0.02
):
    """
    Full physical detection model.

    Includes:
    - signal photons (Poisson)
    - dark counts
    - background photons
    - error model (QBER)

    Parameters
    ----------
    eta_total : channel efficiency
    mu : mean photon number
    gate_time : detection window
    dark_rate : dark counts [Hz]
    bg_rate : background photon rate [Hz]
    e_opt : optical error probability

    Returns
    -------
    dict with:
        p_sig
        p_dark
        p_bg
        p_noise
        p_click
        p_error
        qber
    """

    eta_total = np.asarray(eta_total)
    bg_rate = np.asarray(bg_rate)

    if bg_rate.shape == ():
        bg_rate = np.full_like(eta_total, bg_rate)

    # ======================================================
    # MEAN PHOTON NUMBERS
    # ======================================================

    lambda_sig = signal_mean_photons(mu, eta_total)
    lambda_dark = dark_mean_counts(dark_rate, gate_time)
    lambda_bg = background_mean_counts(bg_rate, gate_time)

    # total noise mean
    lambda_noise = lambda_dark + lambda_bg

    # ======================================================
    # CLICK PROBABILITIES
    # ======================================================

    p_sig = poisson_click_probability(lambda_sig)
    p_dark = poisson_click_probability(lambda_dark)
    p_bg = poisson_click_probability(lambda_bg)

    # combined noise (independent processes)
    p_noise = p_dark + p_bg - p_dark * p_bg

    # total detection probability
    p_click = 1.0 - (1.0 - p_sig) * (1.0 - p_noise)

    # ======================================================
    # ERROR MODEL (QBER)
    # ======================================================

    """
    Error contributions:
    - signal: e_opt (optical misalignment)
    - noise: 50% random
    """

    p_error = e_opt * p_sig + 0.5 * p_noise

    p_click_safe = np.maximum(p_click, EPS)

    qber = p_error / p_click_safe

    # ======================================================
    # OUTPUT
    # ======================================================

    return {
        "p_sig": p_sig,
        "p_dark": p_dark,
        "p_bg": p_bg,
        "p_noise": p_noise,
        "p_click": p_click,
        "p_error": p_error,
        "qber": qber
    }
