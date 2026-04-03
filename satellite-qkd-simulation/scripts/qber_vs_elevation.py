import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# CONFIG
# ======================================================
N_RUNS = 20
N_BINS = 50


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n=== QBER vs ELEVATION (MONTE CARLO) ===\n")

    all_elev = []
    all_qber = []
    all_sun = []

    valid_runs = 0

    # ==================================================
    # MONTE CARLO
    # ==================================================
    for i in range(N_RUNS):
        print(f"Run {i+1}/{N_RUNS}")

        result = run_orbit_simulation()

        if result is None:
            print("⚠ Skipped (no visible pass)")
            continue

        elev = np.rad2deg(result["elevation"])
        qber = result["qber"]
        sun_el = result.get("sun_elevation", None)

        if len(elev) == 0:
            print("⚠ Empty result")
            continue

        all_elev.append(elev)
        all_qber.append(qber)

        if sun_el is not None:
            all_sun.append(sun_el)

        valid_runs += 1

    if valid_runs == 0:
        print("❌ No valid runs")
        return

    print(f"\nValid runs: {valid_runs}/{N_RUNS}")

    # ==================================================
    # CONCAT
    # ==================================================
    elev_all = np.concatenate(all_elev)
    qber_all = np.concatenate(all_qber)

    sun_all = None
    if len(all_sun) > 0:
        sun_all = np.concatenate(all_sun)

    # ==================================================
    # FILTER (VISIBLE)
    # ==================================================
    mask = elev_all > 0

    elev_all = elev_all[mask]
    qber_all = qber_all[mask]

    if sun_all is not None:
        sun_all = sun_all[mask]

    if len(elev_all) == 0:
        print("❌ No visible samples")
        return

    # ==================================================
    # BINNING
    # ==================================================
    bins = np.linspace(0, 90, N_BINS)
    digitized = np.digitize(elev_all, bins)

    elev_bin = []
    qber_avg = []
    qber_std = []
    counts = []

    for i in range(1, len(bins)):
        mask_bin = digitized == i

        if not np.any(mask_bin):
            continue

        values = qber_all[mask_bin]

        elev_bin.append(np.mean(elev_all[mask_bin]))
        qber_avg.append(np.mean(values))
        qber_std.append(np.std(values))
        counts.append(np.sum(mask_bin))

    elev_bin = np.array(elev_bin)
    qber_avg = np.array(qber_avg)
    qber_std = np.array(qber_std)

    # ==================================================
    # DIAGNOSTICS
    # ==================================================
    print("\n--- DIAGNOSTICS ---")
    print("Total samples:", len(elev_all))
    print("QBER mean:", np.mean(qber_all))

    # ==================================================
    # PLOT MAIN
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.errorbar(
        elev_bin,
        qber_avg,
        yerr=qber_std,
        fmt='o-',
        capsize=3,
        label="QBER"
    )

    plt.xlabel("Elevation (deg)")
    plt.ylabel("QBER")
    plt.title("QBER vs Elevation")

    plt.ylim(0, 0.55)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/qber_vs_elevation.png", dpi=300)
    plt.close()

    # ==================================================
    # OPTIONAL: DAY vs NIGHT
    # ==================================================
    if sun_all is not None:
        print("\n--- DAY vs NIGHT ---")

        day = sun_all > 0
        night = sun_all <= 0

        if np.any(day) and np.any(night):
            print("QBER day:", np.mean(qber_all[day]))
            print("QBER night:", np.mean(qber_all[night]))

        # plot separato
        plt.figure(figsize=(8, 6))
        plt.scatter(elev_all[night], qber_all[night], s=5, label="Night", alpha=0.5)

        plt.xlabel("Elevation (deg)")
        plt.ylabel("QBER")
        plt.title("QBER vs Elevation (Day/Night)")

        plt.ylim(0, 0.55)
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/plots/qber_day_night.png", dpi=300)
        plt.close()


# ======================================================
# ENTRY
# ======================================================
if __name__ == "__main__":
    main()
