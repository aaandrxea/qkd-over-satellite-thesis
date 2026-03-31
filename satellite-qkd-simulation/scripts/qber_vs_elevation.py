import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# RUN SIMULATION
# ======================================================
result = run_orbit_simulation()

# ======================================================
# ACCESSO CORRETTO AI DATI
# ======================================================
vis = result["visible"]

t = vis["time"]
elev = np.rad2deg(vis["elevation"])
qber = vis["qber"]


# ======================================================
# ================== PLOT 1 =============================
# QBER vs TIME  (U fisica)
# ======================================================
plt.figure()

plt.plot(t, qber)

plt.xlabel("Time (s)")
plt.ylabel("QBER")
plt.title("QBER vs Time")

plt.grid()


# ======================================================
# ================== PLOT 2 =============================
# QBER vs ELEVATION
# ======================================================
mask = elev > 0

elev_vis = elev[mask]
qber_vis = qber[mask]

# ordina
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

plt.figure()

plt.errorbar(
    elev_bin,
    qber_avg,
    yerr=qber_std,
    fmt='o-'
)

plt.xlabel("Elevation (deg)")
plt.ylabel("QBER")
plt.title("QBER vs Elevation")

plt.ylim(0, 0.55)
plt.grid()


# ======================================================
# SHOW
# ======================================================
plt.show()
