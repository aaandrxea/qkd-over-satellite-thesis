import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# CONFIG
# ======================================================
N_RUNS = 30
N_BINS = 50


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n=== SKR vs ELEVATION ===\n")

    all_elev = []
    all_skr = []
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
        skr = result["skr"]
        sun_el = result.get("sun_elevation", None)

        if len(elev) == 0:
            print("⚠ Empty result")
            continue

        all_elev.append(elev)
        all_skr.append(skr)

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
    skr_all = np.concatenate(all_skr)

    sun_all = None
    if len(all_sun) > 0:
        sun_all = np.concatenate(all_sun)

    # ==================================================
    # FILTER
    # ==================================================
    mask = elev_all > 0

    elev_all = elev_all[mask]
    skr_all = skr_all[mask]

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
    skr_mean = []
    skr_std = []
    skr_p10 = []
    skr_p90 = []
    outage = []

    for i in range(1, len(bins)):
        mask_bin = digitized == i

        if not np.any(mask_bin):
            continue

        values = skr_all[mask_bin]

        elev_bin.append(np.mean(elev_all[mask_bin]))
        skr_mean.append(np.mean(values))
        skr_std.append(np.std(values))
        skr_p10.append(np.percentile(values, 10))
        skr_p90.append(np.percentile(values, 90))
        outage.append(np.mean(values <= 0))

    elev_bin = np.array(elev_bin)
    skr_mean = np.array(skr_mean)
    skr_std = np.array(skr_std)
    skr_p10 = np.array(skr_p10)
    skr_p90 = np.array(skr_p90)
    outage = np.array(outage)

    # ==================================================
    # DIAGNOSTICS
    # ==================================================
    print("\n--- DIAGNOSTICS ---")
    print("Total samples:", len(elev_all))
    print("Mean SKR:", np.mean(skr_all))
    print("Outage mean:", np.mean(outage))

    # ==================================================
    # MAIN PLOT (PAPER STYLE)
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.plot(elev_bin, skr_mean, label="Mean SKR")
    plt.fill_between(elev_bin, skr_p10, skr_p90, alpha=0.3, label="10–90 percentile")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("SKR")
    plt.title("SKR vs Elevation")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/plots/skr_vs_elevation.png", dpi=300)
    plt.close()

    # ==================================================
    # OUTAGE PLOT
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.plot(elev_bin, outage, 'o-', label="Outage Probability")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("Outage Probability")
    plt.title("Outage vs Elevation (from SKR)")

    plt.ylim(0, 1)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/plots/skr_outage.png", dpi=300)
    plt.close()

    # ==================================================
    # DAY vs NIGHT
    # ==================================================
    if sun_all is not None:
        print("\n--- DAY vs NIGHT ---")

        day = sun_all > 0
        night = sun_all <= 0

        if np.any(day) and np.any(night):
            print("SKR day:", np.mean(skr_all[day]))
            print("SKR night:", np.mean(skr_all[night]))

        plt.figure(figsize=(8, 6))

        plt.scatter(elev_all[day], skr_all[day], s=5, alpha=0.5, label="Day")
        plt.scatter(elev_all[night], skr_all[night], s=5, alpha=0.5, label="Night")

        plt.xlabel("Elevation (deg)")
        plt.ylabel("SKR")
        plt.title("SKR vs Elevation (Day/Night)")

        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig("results/plots/skr_day_night.png", dpi=300)
        plt.close()


# ======================================================
# ENTRY
# ======================================================
if __name__ == "__main__":
    main()
