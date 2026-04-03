import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation.run_orbit_simulation import run_orbit_simulation


# RUN
result = run_orbit_simulation()
vis = result["visible"]

t = vis["time"]
sigma = vis["sigma_R2"]

# PLOT
plt.figure(figsize=(8, 5))
plt.plot(t, sigma)
plt.title("Rytov Variance vs Time")
plt.xlabel("Time (s)")
plt.ylabel("σ_R²")
plt.grid()

plt.tight_layout()
plt.savefig("results/plots/validation_rytov.png", dpi=300)
plt.close()


# DEBUG
print("σ_R² min:", np.min(sigma))
print("σ_R² max:", np.max(sigma))
print("σ_R² mean:", np.mean(sigma))
