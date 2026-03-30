import numpy as np

from src.channel.link_budget import compute_link_budget
from src.detection.detector import compute_detection


# ==========================================================
# SCENARIO
# ==========================================================

def generate_scenario(n_points=50):
    elevation = np.linspace(np.deg2rad(10), np.deg2rad(90), n_points)
    R = 500e3 / np.sin(elevation)
    return R, elevation


# ==========================================================
# TEST PRINCIPALE
# ==========================================================

def test_detection_pipeline():

    # ----------------------------
    # PARAMETRI
    # ----------------------------
    wavelength = 800e-9
    tx_diameter = 0.1
    rx_diameter = 0.5

    mu = 0.5
    dark_rate = 100        # Hz
    gate_time = 1e-9       # s
    e_opt = 0.02

    # ----------------------------
    # CHANNEL
    # ----------------------------
    R, elevation = generate_scenario()

    channel = compute_link_budget(
        R,
        elevation,
        wavelength,
        tx_diameter,
        rx_diameter
    )

    eta_total = channel["eta_total"]

    # ----------------------------
    # DETECTION
    # ----------------------------
    det = compute_detection(
        eta_total,
        mu,
        dark_rate,
        gate_time,
        e_opt
    )

    p_sig = det["p_sig"]
    p_dark = det["p_dark"]
    p_click = det["p_click"]
    qber = det["qber"]

    # ======================================================
    # 1. CHECK NUMERICI
    # ======================================================
    assert np.all(np.isfinite(p_click)), "NaN in p_click"
    assert np.all((p_click >= 0) & (p_click <= 1)), "p_click fuori range"
    assert np.all((qber >= 0) & (qber <= 0.5)), "QBER fuori range"

    # ======================================================
    # 2. CONSISTENZA FISICA
    # ======================================================
    assert np.all(p_click >= p_sig), "p_click < p_sig (impossibile)"

    # ======================================================
    # 3. SCALING CON PERDITA
    # ======================================================
    # zenith vs low elevation
    assert p_sig[-1] > p_sig[0], "p_sig non aumenta con elevazione"
    assert qber[0] > qber[-1], "QBER non diminuisce con elevazione"

    # ======================================================
    # 4. RANGE REALISTICI
    # ======================================================
    print("\n=== DETECTION TEST ===")

    print(f"p_sig mean: {np.mean(p_sig):.2e}")
    print(f"p_dark: {p_dark[0]:.2e}")
    print(f"p_click mean: {np.mean(p_click):.2e}")
    print(f"QBER mean: {np.mean(qber):.3f}")

    assert 1e-8 < np.mean(p_sig) < 1e-1, "p_sig non realistico"
    assert 1e-9 < p_dark[0] < 1e-4, "p_dark non realistico"
    assert 0.0 < np.mean(qber) < 0.2, "QBER non realistico"

    return det


# ==========================================================
# TEST LIMITE (HIGH LOSS)
# ==========================================================

def test_extreme_loss():

    eta = np.logspace(-10, -4, 50)
    det = compute_detection(
        eta,
        mu=0.01,
        dark_rate=100,
        gate_time=1e-9,
        e_opt=0.02
    )

    qber = det["qber"]

    print("\n=== EXTREME LOSS TEST ===")
    print(f"QBER min: {np.min(qber):.3f}")
    print(f"QBER max: {np.max(qber):.3f}")

    # QBER deve tendere a 0.5
    assert qber[0] > 0.3, "QBER non tende a 0.5 in high loss"


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    test_detection_pipeline()
    test_extreme_loss()

    print("\nALL DETECTION TESTS PASSED")
