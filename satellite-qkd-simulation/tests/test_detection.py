import numpy as np

from src.detection.detector import compute_detection


# ================================
# TEST BASE DETECTION
# ================================

def test_detection_basic():
    eta = np.linspace(1e-6, 1e-2, 1000)

    results = compute_detection(
        eta_total=eta,
        mu=0.5,
        p_dark=1e-6,
        e_opt=0.02
    )

    p_sig = results["p_sig"]
    p_click = results["p_click"]
    qber = results["qber"]

    # ----------------------------
    # CHECK RANGE
    # ----------------------------
    assert np.all(p_sig >= 0)
    assert np.all(p_click >= 0)

    # ----------------------------
    # QBER fisico
    # ----------------------------
    assert np.all(qber >= 0)
    assert np.all(qber <= 0.5)

    # ----------------------------
    # valori realistici
    # ----------------------------
    mean_qber = np.mean(qber)

    print("DEBUG QBER mean:", mean_qber)

    assert 0.0 < mean_qber < 0.2, "QBER fuori range realistico"

    print("✓ Detection basic OK")


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    test_detection_basic()
