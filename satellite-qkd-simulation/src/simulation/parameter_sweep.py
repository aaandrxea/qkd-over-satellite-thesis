import numpy as np

from src.simulation.run_simulation import run_simulation


def sweep_mu_nu(
    R,
    elevation,
    wavelength,
    tx_diameter,
    rx_diameter,
    mu_values,
    nu_values,
    p_dark,
    e_opt
):
    """
    Sweep su μ e ν
    """

    results = []

    for mu in mu_values:
        for nu in nu_values:

            sim = run_simulation(
                R=R,
                elevation=elevation,
                wavelength=wavelength,
                tx_diameter=tx_diameter,
                rx_diameter=rx_diameter,
                mu=mu,
                nu=nu,
                p_dark=p_dark,
                e_opt=e_opt
            )

            skr = np.mean(sim["keyrate"]["skr"])
            skr_decoy = np.mean(sim["decoy"]["skr_decoy"])
            qber = np.mean(sim["qber"])

            results.append({
                "mu": mu,
                "nu": nu,
                "skr": skr,
                "skr_decoy": skr_decoy,
                "qber": qber
            })

    return results
