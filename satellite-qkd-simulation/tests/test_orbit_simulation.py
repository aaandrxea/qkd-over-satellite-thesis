import numpy as np

from src.simulation.run_orbit_simulation import run_orbit_simulation


def test_orbit_simulation():

    results = run_orbit_simulation(
        tle_path="data/iss.tle",  # crea questo file!
        ground_lat=45.0,
        ground_lon=9.0,
        ground_alt=100,
        duration_s=7200,
        step_s=2,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5,
        mu=0.5,
        nu=0.1,
        p_dark=1e-6,
        e_opt=0.02
    )

    print("Mean QBER:", np.mean(results["qber"]))
    print("Mean SKR:", np.mean(results["skr"]))

    assert len(results["skr"]) > 0
    assert np.any(results["skr"] > 0), "Nessuna chiave generata"

    print("SKR > 0 fraction:", np.mean(results["skr"] > 0))
    print("✓ Orbit simulation OK")


if __name__ == "__main__":
    test_orbit_simulation()
