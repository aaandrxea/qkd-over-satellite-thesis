import numpy as np

from src.channel.link_budget import compute_link_budget
from src.detection.detector import compute_detection


# ==========================================================
# SCENARIO GEOMETRICO
# ==========================================================

def generate_scenario(n_points=50):
    elevation = np.linspace(np.deg2rad(10), np.deg2rad(90), n_points)
    R = 500e3 / np.sin(elevation)
    return R, elevation


# ==========================================================
# PARAMETRI COMUNI
# ==========================================================

PARAMS = {
    "wavelength": 800e-9,
    "tx_diameter": 0.1,
    "rx_diameter": 0.5,
    "mu": 0.5,
    "dark_rate": 100,     # Hz
    "gate_time": 1e-9,    # s
    "e_opt": 0.02
}


# ==========================================================
# NIGHT SCENARIO (NO BACKGROUND)
# ==========================================================

def test_detection_night():

    print("\n=== DETECTION TEST: NIGHT ===")

    R, elevation = generate_scenario()

    channel = compute_link_budget(
        R,
        elevation,
        PARAMS["wavelength"],
        PARAMS["tx_diameter"],
        PARAMS["rx_diameter"]
    )

    eta = channel["eta_total"]

    det = compute_detection(
        eta,
        PARAMS["mu"],
        PARAMS["dark_rate"],
        PARAMS["gate_time"],
        PARAMS["e_opt"],
        bg_rate=0.0
    )

    qber = det["qber"]
    p_click = det["p_click"]

    print(f"QBER mean: {np.mean(qber):.3f}")
    print(f"p_click mean: {np.mean(p_click):.2e}")

    # ------------------------------------------------------
    # CHECK
    # ------------------------------------------------------
    assert np.all((qber >= 0) & (qber <= 0.5))
    assert np.mean(qber) < 0.1, "QBER notte troppo alto"

    return det


# ==========================================================
# DAY SCENARIO (WITH BACKGROUND)
# ==========================================================

def test_detection_day():

    print("\n=== DETECTION TEST: DAY ===")

    R, elevation = generate_scenario()

    channel = compute_link_budget(
        R,
        elevation,
        PARAMS["wavelength"],
        PARAMS["tx_diameter"],
        PARAMS["rx_diameter"]
    )

    eta = channel["eta_total"]

    # valore realistico ordine di grandezza
    bg_rate = 1e6  # Hz (daylight)

    det = compute_detection(
        eta,
        PARAMS["mu"],
        PARAMS["dark_rate"],
        PARAMS["gate_time"],
        PARAMS["e_opt"],
        bg_rate=bg_rate
    )

    qber = det["qber"]
    p_click = det["p_click"]

    print(f"QBER mean: {np.mean(qber):.3f}")
    print(f"p_click mean: {np.mean(p_click):.2e}")

    # ------------------------------------------------------
    # CHECK
    # ------------------------------------------------------
    assert np.mean(qber) > 0.05, "QBER giorno troppo basso (non realistico)"
    assert np.mean(qber) < 0.5

    return det


# ==========================================================
# EXTREME LOSS TEST
# ==========================================================

def test_extreme_loss():

    print("\n=== EXTREME LOSS TEST ===")

    eta = np.logspace(-10, -4, 50)

    det = compute_detection(
        eta,
        mu=0.01,
        dark_rate=100,
        gate_time=1e-9,
        e_opt=0.02,
        bg_rate=0.0
    )

    qber = det["qber"]

    print(f"QBER min: {np.min(qber):.3f}")
    print(f"QBER max: {np.max(qber):.3f}")

    # deve tendere a 0.5
    assert qber[0] > 0.3


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    night = test_detection_night()
    day = test_detection_day()
    test_extreme_loss()

    print("\nALL DETECTION TESTS PASSED")
