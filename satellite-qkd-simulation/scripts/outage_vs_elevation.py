import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# PARAMETRI
# ======================================================
N = 30   # numero realizzazioni Monte Carlo

# ======================================================
# MONTE CARLO RUNS
# ======================================================
all_skr = []
all_elev = None

for i in range(N):
    print(f"Run {i+1}/{N}")

    result = run_orbit_simulation()

    vis = result["visible"]

    elev = np.rad2deg(vis["elevation"])
    skr = vis["skr"]

    if all_elev is None:
        all_elev = elev

    all_skr.append(skr)

all_skr = np.array(all_skr)  # shape: (N, time)

# ======================================================
# FILTRO VISIBILITÀ
# ======================================================
mask = all_elev > 0

elev = all_elev[mask]
skr_all = all_skr[:, mask]

# ======================================================
# BINNING
# ======================================================
bins = np.linspace(0, 90, 40)
digitized = np.digitize(elev, bins)

elev_bin = []
outage_prob = []

for i in range(1, len(bins)):
    mask_bin = digitized == i

    if np.any(mask_bin):
        values = skr_all[:, mask_bin]   # shape: (N, n_points)

        # outage per ogni realizzazione 
        outage_runs = np.mean(values <= 0, axis=1)

        # poi media tra run
        p_out = np.mean(outage_runs)
        outage_prob.append(p_out)
        elev_bin.append(np.mean(elev[mask_bin]))

# ======================================================
# PLOT
# ======================================================
plt.figure()

plt.plot(elev_bin, outage_prob, 'o-')

plt.xlabel("Elevation (deg)")
plt.ylabel("Outage Probability")
plt.title("Outage vs Elevation")

plt.ylim(0, 1)
plt.grid()

plt.show()
