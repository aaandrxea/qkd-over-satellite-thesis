import numpy as np

from src.simulation.parameter_sweep import sweep_mu_nu


def test_sweep():

    R = np.linspace(2e5, 8e5, 200)
    elevation = np.linspace(np.deg2rad(20), np.deg2rad(90), 200)

    mu_values = [0.2, 0.5, 0.8]
    nu_values = [0.05, 0.1, 0.2]

    results = sweep_mu_nu(
        R,
        elevation,
        wavelength=800e-9,
        tx_diameter=0.1,
        rx_diameter=0.5,
        mu_values=mu_values,
        nu_values=nu_values,
        p_dark=1e-6,
        e_opt=0.02
    )

    for r in results:
        print(r)

    print("✓ Sweep OK")


if __name__ == "__main__":
    test_sweep()
