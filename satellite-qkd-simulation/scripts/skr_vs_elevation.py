import numpy as np
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


N = 1   # numero realizzazioni

all_skr = []
all_elev = None

for i in range(N):
    result = run_orbit_simulation()

    vis = result["visible"]

    elev = np.rad2deg(vis["elevation"])
    skr = vis["skr"]

    if all_elev is None:
        all_elev = elev

    all_skr.append(skr)

all_skr = np.array(all_skr)  # shape: (N, time)
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

elev = all_elev
idx = elev > 0

elev = elev[idx]
skr_all = all_skr[:, idx]

digitized = np.digitize(elev, bins)

skr_avg = []
skr_std = []
elev_bin = []

for i in range(1, len(bins)):
    mask = digitized == i

    if np.any(mask):
        values = skr_all[:, mask]   # 🔴 tutte le realizzazioni

        skr_avg.append(np.mean(values))
        skr_std.append(np.std(values))
        elev_bin.append(np.mean(elev[mask]))
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
