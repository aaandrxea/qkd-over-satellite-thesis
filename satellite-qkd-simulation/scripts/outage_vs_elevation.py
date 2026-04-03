import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# CONFIG
# ======================================================
N_RUNS = 30
N_BINS = 40


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n=== OUTAGE vs ELEVATION (MONTE CARLO) ===\n")

    all_elev = []
    all_skr = []

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

        if len(elev) == 0:
            print("⚠ Empty result")
            continue

        all_elev.append(elev)
        all_skr.append(skr)

        valid_runs += 1

    if valid_runs == 0:
        print("❌ No valid runs")
        return

    print(f"\nValid runs: {valid_runs}/{N_RUNS}")

    # ==================================================
    # CONCATENATION
    # ==================================================
    elev_all = np.concatenate(all_elev)
    skr_all = np.concatenate(all_skr)

    # ==================================================
    # FILTER (VISIBLE ONLY)
    # ==================================================
    mask = elev_all > 0

    elev_all = elev_all[mask]
    skr_all = skr_all[mask]

    if len(elev_all) == 0:
        print("❌ No visible samples after filtering")
        return

    # ==================================================
    # BINNING
    # ==================================================
    bins = np.linspace(0, 90, N_BINS)
    digitized = np.digitize(elev_all, bins)

    elev_bin = []
    outage_prob = []
    counts = []

    for i in range(1, len(bins)):
        mask_bin = digitized == i

        if not np.any(mask_bin):
            continue

        skr_bin = skr_all[mask_bin]

        # outage definition
        outage = skr_bin <= 0

        p_out = np.mean(outage)

        elev_bin.append(np.mean(elev_all[mask_bin]))
        outage_prob.append(p_out)
        counts.append(np.sum(mask_bin))

    elev_bin = np.array(elev_bin)
    outage_prob = np.array(outage_prob)
    counts = np.array(counts)

    # ==================================================
    # DIAGNOSTICS
    # ==================================================
    print("\n--- DIAGNOSTICS ---")
    print("Total samples:", len(elev_all))
    print("Outage mean:", np.mean(outage_prob))

    # ==================================================
    # PLOT
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.plot(elev_bin, outage_prob, 'o-', label="Outage probability")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("Outage Probability")
    plt.title("Outage vs Elevation")

    plt.ylim(0, 1)
    plt.grid()

    plt.legend()
    plt.tight_layout()

    plt.savefig("results/plots/outage_vs_elevation.png", dpi=300)
    plt.show()


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
