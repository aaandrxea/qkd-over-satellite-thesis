import numpy as np
from src.detection.noise import total_noise_probability


def compute_detection(
    eta_total,
    mu,
    dark_rate,
    gate_time,
    e_opt,
    bg_rate=0.0
):
    """
    Detection model deterministico e consistente.

    Parametri:
    - eta_total: efficienza totale (array)
    - mu: numero medio di fotoni per impulso
    - dark_rate: rate dark counts (Hz)
    - gate_time: durata gate (s)
    - e_opt: errore ottico
    - bg_rate: rate background (Hz) → può essere array

    Output:
    dict con p_sig, p_noise, p_click, qber
    """

    # ======================================================
    # INPUT SANITIZATION
    # ======================================================
    eta_total = np.asarray(eta_total)
    bg_rate = np.asarray(bg_rate)

    shape = eta_total.shape

    mu = float(mu)
    dark_rate = float(dark_rate)
    gate_time = float(gate_time)
    e_opt = float(e_opt)

    # ======================================================
    # SIGNAL
    # ======================================================
    # probabilità di detection del segnale (Poisson → approx)
    p_sig = mu * eta_total

    # evita valori > 1
    p_sig = np.clip(p_sig, 0, 1)

    # ======================================================
    # NOISE
    # ======================================================
    # dark counts
    p_dark = dark_rate * gate_time

    # broadcast
    p_dark = np.full(shape, p_dark)

    # background
    p_bg = 1 - np.exp(-bg_rate * gate_time)
    # total noise (indipendenti)
    p_noise = p_dark + p_bg

    # evita >1
    p_noise = np.clip(p_noise, 0, 1)

    # ======================================================
    # CLICK PROBABILITY
    # ======================================================
    p_click = p_sig + p_noise

    p_click = np.clip(p_click, 0, 1)

    # ======================================================
    # QBER
    # ======================================================
    error = 0.5 * p_noise + e_opt * p_sig

    total = np.maximum(p_sig + p_noise, 1e-15)

    qber = error / total

    # ======================================================
    # DEBUG (leggero)
    # ======================================================
    print("DET:",
          "sig=", np.mean(p_sig),
          "noise=", np.mean(p_noise),
          "bg=", np.mean(p_bg))
    print("DET:",
      "sig=", np.mean(p_sig),
      "noise=", np.mean(p_noise),
      "bg=", np.mean(p_bg))
    # ======================================================
    # OUTPUT
    # ======================================================
    return {
        "p_sig": p_sig,
        "p_dark": p_dark,
        "p_bg": p_bg,
        "p_noise": p_noise,
        "p_click": p_click,
        "qber": qber,
    }


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
    Monte Carlo detection model (realistic).
    """

    eta_total = np.asarray(eta_total)
    bg_rate = np.asarray(bg_rate)

    shape = eta_total.shape

    # ======================================================
    # MEAN PHOTON NUMBERS
    # ======================================================
    lambda_sig = mu * eta_total
    lambda_dark = dark_rate * gate_time
    lambda_bg = bg_rate * gate_time

    # broadcast
    lambda_dark = np.full(shape, lambda_dark)

    # ======================================================
    # MONTE CARLO SAMPLING
    # ======================================================
    signal_counts = np.random.poisson(lambda_sig * n_pulses)
    dark_counts = np.random.poisson(lambda_dark * n_pulses)
    bg_counts = np.random.poisson(lambda_bg * n_pulses)

    noise_counts = dark_counts + bg_counts

    # ======================================================
    # CLICK PROBABILITY
    # ======================================================
    clicks = signal_counts + noise_counts
    p_click = clicks / n_pulses

    # ======================================================
    # ERROR MODEL
    # ======================================================
    error_counts = (
        np.random.binomial(signal_counts, e_opt)
        + np.random.binomial(noise_counts, 0.5)
    )

    total_counts = np.maximum(signal_counts + noise_counts, 1)

    qber = error_counts / total_counts

    # ======================================================
    # DEBUG (ridotto)
    # ======================================================
    print("MC DET:",
          "mean_sig=", np.mean(signal_counts / n_pulses),
          "mean_noise=", np.mean(noise_counts / n_pulses))

    return {
        "p_sig": signal_counts / n_pulses,
        "p_noise": noise_counts / n_pulses,
        "p_click": p_click,
        "qber": qber,
    }
