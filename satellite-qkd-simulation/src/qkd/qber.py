import numpy as np

def compute_qber(signal_counts, noise_counts, e_det=0.02):
    """
    QBER realistico:
    - noise → errori random (50%)
    - signal → errori sistematici (e_det)
    """
    print("DEBUG:",
          "signal=", np.mean(signal_counts),
        "noise=", np.mean(noise_counts))
    total = signal_counts + noise_counts

    # evita divisioni per zero
    total = np.maximum(total, 1e-15)

    error = 0.5 * noise_counts + e_det * signal_counts

    qber = error / total

    return qber
