import numpy as np

from src.detection.photon_model import signal_probability
from src.detection.noise import dark_count_probability
from src.detection.click_model import total_click_probability


def compute_detection(
    eta_total,
    mu,
    dark_rate,
    gate_time,
    e_opt,
    bg_rate=0.0
):
    """
    Complete detection pipeline (physically consistent).

    Includes:
    - signal detection (Poisson model)
    - dark counts
    - background light
    - total click probability
    - QBER computation

    Parameters
    ----------
    eta_total : channel transmission
    mu : mean photon number
    dark_rate : detector dark count rate [Hz]
    gate_time : detection gate time [s]
    e_opt : optical error probability
    bg_rate : background photon rate [Hz] (default = 0)

    Returns
    -------
    dict:
        p_sig
        p_dark
        p_bg
        p_noise
        p_click
        p_err
        qber
    """

    import numpy as np

    from src.detection.photon_model import signal_probability
    from src.detection.noise import (
        dark_count_probability,
        background_probability,
        total_noise_probability
    )
    from src.detection.click_model import total_click_probability

    # ------------------------------------------------------
    # INPUT SANITIZATION
    # ------------------------------------------------------
    eta_total = np.asarray(eta_total)

    mu = float(mu)
    dark_rate = float(dark_rate)
    gate_time = float(gate_time)
    e_opt = float(e_opt)
    bg_rate = float(bg_rate)

    shape = eta_total.shape

    # ------------------------------------------------------
    # SIGNAL
    # ------------------------------------------------------
    p_sig = signal_probability(mu, eta_total)

    # ------------------------------------------------------
    # NOISE COMPONENTS
    # ------------------------------------------------------
    p_dark = dark_count_probability(dark_rate, gate_time, shape)
    p_bg   = background_probability(bg_rate, gate_time, shape)

    # total noise
    p_noise = total_noise_probability(p_dark, p_bg)

    # ------------------------------------------------------
    # CLICK PROBABILITY
    # ------------------------------------------------------
    p_click = total_click_probability(p_sig, p_noise)

    # ------------------------------------------------------
    # ERROR MODEL
    # ------------------------------------------------------
    # optical errors + random noise (50%)
    p_err = e_opt * p_sig + 0.5 * p_noise

    # ------------------------------------------------------
    # QBER
    # ------------------------------------------------------
    p_click_safe = np.maximum(p_click, 1e-15)
    qber = p_err / p_click_safe

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------
    return {
        "p_sig": p_sig,
        "p_dark": p_dark,
        "p_bg": p_bg,
        "p_noise": p_noise,
        "p_click": p_click,
        "p_err": p_err,
        "qber": qber,
    }

