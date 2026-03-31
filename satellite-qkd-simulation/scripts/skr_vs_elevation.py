import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


result = run_orbit_simulation()

# ======================================================
# DATI CORRETTI
# ======================================================
vis = result["visible"]

elev = np.rad2deg(vis["elevation"])
skr = vis["skr"]

# ======================================================
# FILTRO VISIBILITÀ (extra safety)
# ======================================================
mask = elev > 0
elev = elev[mask]
skr = skr[mask]

# ======================================================
# 🔴 ORDINA (IMPORTANTE)
# ======================================================
idx = np.argsort(elev)
elev = elev[idx]
skr = skr[idx]

# ======================================================
# BINNING
# ======================================================
bins = np.linspace(0, 90, 50)
digitized = np.digitize(elev, bins)

skr_avg = []
skr_std = []
elev_bin = []

for i in range(1, len(bins)):
    mask_bin = digitized == i

    if np.any(mask_bin):
        skr_avg.append(np.mean(skr[mask_bin]))
        skr_std.append(np.std(skr[mask_bin]))
        elev_bin.append(np.mean(elev[mask_bin]))

# ======================================================
# PLOT
# ======================================================
plt.figure()

plt.errorbar(
    elev_bin,
    skr_avg,
    yerr=skr_std,
    fmt='o-'
)

plt.xlabel("Elevation (deg)")
plt.ylabel("SKR")
plt.title("SKR vs Elevation")

plt.grid()
plt.show()
