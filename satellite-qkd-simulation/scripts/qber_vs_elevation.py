import numpy as np
import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# RUN
# ======================================================
result = run_orbit_simulation()
vis = result["visible"]

t = vis["time"]
elev = np.rad2deg(vis["elevation"])
qber = vis["qber"]
skr = vis["skr"]
cloud = vis["cloud_state"]


# ======================================================
# FIGURE MULTIPANEL
# ======================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)


# ================== CLOUD =============================
axs[0].plot(t, cloud, lw=1)
axs[0].set_title("Cloud State")
axs[0].set_ylabel("State")
axs[0].grid()


# ================== QBER TIME =========================
axs[1].plot(t, qber, lw=1)
axs[1].set_title("QBER vs Time")
axs[1].set_ylabel("QBER")
axs[1].set_ylim(0, 0.55)
axs[1].grid()


# ================== SKR TIME ==========================
axs[2].plot(t, skr, lw=1)
axs[2].set_title("SKR vs Time")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("SKR")
axs[2].grid()


plt.tight_layout()
plt.savefig("results/plots/time_series.png", dpi=300)
plt.close()


# ======================================================
# QBER vs ELEVATION
# ======================================================
mask = elev > 0

elev_vis = elev[mask]
qber_vis = qber[mask]

# sorting
idx = np.argsort(elev_vis)
elev_vis = elev_vis[idx]
qber_vis = qber_vis[idx]

# binning
bins = np.linspace(0, 90, 50)
digitized = np.digitize(elev_vis, bins)

qber_avg = []
qber_std = []
elev_bin = []

for i in range(1, len(bins)):
    mask_bin = digitized == i
    if np.any(mask_bin):
        qber_avg.append(np.mean(qber_vis[mask_bin]))
        qber_std.append(np.std(qber_vis[mask_bin]))
        elev_bin.append(np.mean(elev_vis[mask_bin]))


# ======================================================
# PLOT ELEVATION
# ======================================================
plt.figure(figsize=(8, 6))

plt.errorbar(
    elev_bin,
    qber_avg,
    yerr=qber_std,
    fmt='o-',
    capsize=3
)

plt.xlabel("Elevation (deg)")
plt.ylabel("QBER")
plt.title("QBER vs Elevation")
plt.ylim(0, 0.55)
plt.grid()

plt.tight_layout()
plt.savefig("results/plots/qber_vs_elevation.png", dpi=300)
plt.close()
