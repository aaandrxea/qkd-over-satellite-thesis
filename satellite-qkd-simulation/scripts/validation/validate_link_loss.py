import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# RUN
result = run_orbit_simulation()
vis = result["visible"]

eta = vis["eta"]

# PLOT
plt.figure(figsize=(8, 5))
plt.hist(eta, bins=100, density=True)

plt.yscale("log")
plt.title("Fading Distribution")
plt.xlabel("Transmission η")
plt.ylabel("PDF")
plt.grid()

plt.tight_layout()
plt.savefig("results/plots/validation_fading.png", dpi=300)
plt.close()


# DEBUG
print("eta min:", np.min(eta))
print("eta max:", np.max(eta))
