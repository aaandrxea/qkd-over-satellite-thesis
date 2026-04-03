import numpy as np

EPS = 1e-15


# ==========================================================
# DEAD TIME (NON-PARALYZABLE MODEL)
# ==========================================================

def apply_dead_time(
    p_click,
    dt,
    dead_time
):
    """
    Apply non-paralyzable dead time.

    Approximation:
    detection probability is reduced proportionally
    to recent detection activity.

    Model:
        p_eff(t) = p(t) * (1 - R_eff(t-1) * tau_d)

    where:
        R_eff ≈ p / dt

    Parameters
    ----------
    p_click : array
    dt : timestep [s]
    dead_time : detector dead time [s]
    """

    p_click = np.asarray(p_click)
    p_eff = np.zeros_like(p_click)

    for i in range(len(p_click)):

        if i == 0:
            p_eff[i] = p_click[i]
            continue

        # effective rate at previous step
        rate_prev = p_eff[i-1] / max(dt, EPS)

        suppression = 1.0 - rate_prev * dead_time
        suppression = np.clip(suppression, 0.0, 1.0)

        p_eff[i] = p_click[i] * suppression

    return p_eff


# ==========================================================
# AFTERPULSING (MEMORY EFFECT)
# ==========================================================

def apply_afterpulsing(
    p_click,
    afterpulse_prob
):
    """
    First-order afterpulsing model.

    Model:
        p_eff(t) = p(t) + p_ap * p(t-1)

    Stable formulation:
    - bounded to [0,1]
    - causal

    Parameters
    ----------
    p_click : array
    afterpulse_prob : probability of afterpulse per detection
    """

    p_click = np.asarray(p_click)
    p_eff = np.copy(p_click)

    for i in range(1, len(p_click)):
        p_eff[i] += afterpulse_prob * p_eff[i-1]

    return np.clip(p_eff, 0.0, 1.0)


# ==========================================================
# TIMING JITTER (TEMPORAL CONVOLUTION)
# ==========================================================

def apply_timing_jitter(
    p_click,
    jitter_std,
    dt
):
    """
    Apply timing jitter via Gaussian convolution.

    Physical meaning:
    detection time uncertainty spreads events.

    Parameters
    ----------
    jitter_std : timing jitter [s]
    dt : timestep [s]
    """

    p_click = np.asarray(p_click)

    # convert to samples
    sigma_samples = jitter_std / max(dt, EPS)

    # avoid degenerate kernel
    if sigma_samples < 1e-3:
        return p_click

    kernel_half = int(3 * sigma_samples)
    x = np.arange(-kernel_half, kernel_half + 1)

    kernel = np.exp(-0.5 * (x / sigma_samples)**2)
    kernel /= np.sum(kernel)

    return np.convolve(p_click, kernel, mode="same")


# ==========================================================
# FULL TEMPORAL PIPELINE
# ==========================================================

def apply_temporal_effects(
    p_click,
    dt,
    dead_time,
    afterpulse_prob,
    jitter_std
):
    """
    Full temporal detector model.

    Order is important:
    1. dead time (limits rate)
    2. afterpulsing (adds memory)
    3. jitter (smears in time)

    Returns
    -------
    p_click_eff
    """

    p = apply_dead_time(p_click, dt, dead_time)

    p = apply_afterpulsing(p, afterpulse_prob)

    p = apply_timing_jitter(p, jitter_std, dt)

    return np.clip(p, 0.0, 1.0)
