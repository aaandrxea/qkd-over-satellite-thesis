import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


def plot_orbit():

    results = run_orbit_simulation(
        tle_path="data/iss.tle",
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

    time = np.arange(len(results["skr"]))

    skr = results["skr"]
    qber = results["qber"]
    elevation = results["elevation"] * 180 / np.pi

    # SKR
    plt.figure()
    plt.plot(time, skr)
    plt.xlabel("Time index")
    plt.ylabel("SKR")
    plt.title("Secret Key Rate vs Time")
    plt.grid()

    # QBER
    plt.figure()
    plt.plot(time, qber)
    plt.xlabel("Time index")
    plt.ylabel("QBER")
    plt.title("QBER vs Time")
    plt.grid()

    # Elevation
    plt.figure()
    plt.plot(time, elevation)
    plt.xlabel("Time index")
    plt.ylabel("Elevation (deg)")
    plt.title("Elevation vs Time")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    plot_orbit()
