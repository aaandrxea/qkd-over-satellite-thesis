import numpy as np

from src.channel.link_budget import compute_link_budget


# ================================
# TEST SETUP
# ================================

def generate_test_data(n=1000):
    # distanza: 200 km → 1000 km
    R = np.linspace(2e5, 1e6, n)

    # elevazione: 5° → 90°
    elevation = np.linspace(np.deg2rad(5), np.deg2rad(90), n)

    return R, elevation


# ================================
# TEST 1 — RANGE FISICO
# ================================

def test_physical_range():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    for key, val in results.items():
        if isinstance(val, np.ndarray):
            assert np.all(val >= 0), f"{key} < 0"
            if key != "eta_turb":
                assert np.all(val <= 1), f"{key} > 1"

    print("✓ Range fisico OK")


# ================================
# TEST 2 — MONOTONIA GEOMETRICA
# ================================

def test_geometric_monotonicity():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    eta_geo = results["eta_geo"]

    assert np.all(np.diff(eta_geo) <= 0), "eta_geo non decrescente"

    print("✓ Geometric loss OK")


# ================================
# TEST 3 — ATMOSFERA
# ================================

def test_atmospheric_monotonicity():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    eta_atm = results["eta_atm"]

    assert np.all(np.diff(eta_atm) >= 0), "eta_atm non crescente"

    print("✓ Atmosfera OK")


# ================================
# TEST 4 — TURBOLENZA
# ================================

def test_turbulence_statistics():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    eta_turb = results["eta_turb"]

    mean = np.mean(eta_turb)
    var = np.var(eta_turb)
    print("DEBUG TURBULENCE:")
    print("Mean:", mean)
    print("Std:", np.std(eta_turb))
    print("Min:", np.min(eta_turb))
    print("Max:", np.max(eta_turb))
    assert 0.5 < mean < 1.5, "Media turbolenza fuori range"
    assert var > 0, "Varianza nulla"

    print("✓ Turbolenza OK")


# ================================
# TEST 5 — POINTING LIMITE
# ================================

def test_pointing_limit():
    R, elevation = generate_test_data()

    config = {
        "eta_tx": 1.0,
        "eta_rx": 1.0,
        "eta_misc": 1.0,
        "pointing": {
            "sigma_theta": 1e-12
        }
    }

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5,
        config=config
    )

    eta_point = results["eta_point"]

    assert np.allclose(eta_point, 1.0, atol=1e-6), "Pointing non tende a 1"

    print("✓ Pointing limite OK")


# ================================
# TEST 6 — COERENZA TOTALE
# ================================

def test_total_consistency():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    eta_total = results["eta_total"]

    assert np.all(eta_total <= 1), "eta_total > 1"

    print("✓ Consistenza totale OK")


# ================================
# TEST 7 — STABILITÀ NUMERICA
# ================================

def test_numerical_stability():
    R, elevation = generate_test_data()

    results = compute_link_budget(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5
    )

    for key, val in results.items():
        if isinstance(val, np.ndarray):
            assert not np.isnan(val).any(), f"{key} contiene NaN"
            assert not np.isinf(val).any(), f"{key} contiene Inf"

    print("✓ Stabilità numerica OK")


# ================================
# MAIN (run manuale)
# ================================

if __name__ == "__main__":
    test_physical_range()
    test_geometric_monotonicity()
    test_atmospheric_monotonicity()
    test_turbulence_statistics()
    test_pointing_limit()
    test_total_consistency()
    test_numerical_stability()

    print("\nALL TEST PASSED ✅")
