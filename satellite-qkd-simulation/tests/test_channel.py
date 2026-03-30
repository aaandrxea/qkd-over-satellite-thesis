import numpy as np

from src.channel.link_budget import compute_link_budget, link_budget_dB

# ==========================================================
# SCENARIO DI TEST (LEO DOWNLINK)
# ==========================================================

def generate_test_scenario(n_points=50):
    """
    Simula passaggio satellite:
    - elevazione da 10° a 90°
    - distanza coerente (~500–1500 km)
    """

    elevation = np.linspace(np.deg2rad(10), np.deg2rad(90), n_points)

    # distanza approssimata (slant range realistico)
    R = 500e3 / np.sin(elevation)

    return R, elevation


# ==========================================================
# TEST PRINCIPALE
# ==========================================================

def test_link_budget():

    wavelength = 800e-9      # 800 nm
    tx_diameter = 0.1        # 10 cm
    rx_diameter = 0.5        # 50 cm

    R, elevation = generate_test_scenario()

    results = compute_link_budget(
        R,
        elevation,
        wavelength,
        tx_diameter,
        rx_diameter
    )

    results_dB = link_budget_dB(results)

    eta_total = results["eta_total"]
    loss_dB = results_dB["eta_total"]

    # ------------------------------------------------------
    # 1. CHECK NUMERICO
    # ------------------------------------------------------
    assert np.all(np.isfinite(eta_total)), "NaN o Inf in eta_total"
    assert np.all((eta_total >= 0) & (eta_total <= 1)), "eta fuori range"

    # ------------------------------------------------------
    # 2. MONOTONICITÀ FISICA
    # ------------------------------------------------------
    # maggiore elevazione → meno perdita
    assert loss_dB[0] > loss_dB[-1], "La perdita NON diminuisce con elevazione"

    # ------------------------------------------------------
    # 3. RANGE REALISTICO
    # ------------------------------------------------------
    min_loss = np.min(loss_dB)
    max_loss = np.max(loss_dB)

    print("\n=== LINK BUDGET TEST ===")
    print(f"Min loss (zenith): {min_loss:.2f} dB")
    print(f"Max loss (low elevation): {max_loss:.2f} dB")

    assert 10 < min_loss < 60, "Loss zenith non realistica"
    assert max_loss > min_loss, "Comportamento non fisico"

    # ------------------------------------------------------
    # 4. CONTRIBUTI
    # ------------------------------------------------------
    print("\n--- CONTRIBUTI ---")
    for key in ["eta_aperture", "eta_atm", "eta_turb", "eta_point"]:
        val = results_dB[key]
        print(f"{key}: mean = {np.mean(val):.2f} dB")

    return results


# ==========================================================
# TEST LUNGHEZZA D’ONDA
# ==========================================================

def test_wavelength_dependence():

    R, elevation = generate_test_scenario()

    tx_diameter = 0.1
    rx_diameter = 0.5

    res_800 = compute_link_budget(
        R, elevation, 800e-9, tx_diameter, rx_diameter
    )

    res_1550 = compute_link_budget(
        R, elevation, 1550e-9, tx_diameter, rx_diameter
    )

    loss_800 = -10 * np.log10(res_800["eta_total"])
    loss_1550 = -10 * np.log10(res_1550["eta_total"])

    print("\n=== WAVELENGTH TEST ===")
    print(f"800 nm mean loss: {np.mean(loss_800):.2f} dB")
    print(f"1550 nm mean loss: {np.mean(loss_1550):.2f} dB")

    # atmosfera migliore a 1550 nm (meno Rayleigh)
    diff = np.mean(loss_1550) - np.mean(loss_800)

    print(f"Difference (1550 - 800): {diff:.2f} dB")

    # Effetti in competizione: Rayleigh vs diffrazione vs umidità
    assert -15 < diff < 15, "λ dependence fuori range fisico"

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    results = test_link_budget()

    test_wavelength_dependence()

    print("\nALL TESTS PASSED")
