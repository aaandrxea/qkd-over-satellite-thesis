import numpy as np

from src.simulation.run_simulation import run_simulation


def test_simulation_basic():

    R = np.linspace(2e5, 1e6, 500)
    elevation = np.linspace(np.deg2rad(10), np.deg2rad(90), 500)

    results = run_simulation(
        R=R,
        elevation=elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5,
        mu=0.5,
        nu=0.1,
        p_dark=1e-6,
        e_opt=0.02
    )

    qber = results["qber"]
    skr = results["keyrate"]["skr"]
    skr_decoy = results["decoy"]["skr_decoy"]

    print("DEBUG QBER mean:", np.mean(qber))
    print("DEBUG SKR mean:", np.mean(skr))
    print("DEBUG Decoy SKR mean:", np.mean(skr_decoy))

    assert np.all(qber >= 0)
    assert np.all(skr >= 0)
    assert np.all(skr_decoy >= 0)

    print("✓ Simulation pipeline OK")


if __name__ == "__main__":
    test_simulation_basic()
