import numpy as np
import matplotlib.pyplot as plt

from src.channel.link_budget import geometric_loss


# ======================================================
# PARAMETRI FISICI (coerenti con il tuo scenario)
# ======================================================

wavelength = 8.0e-7      # 800 nm
tx_diameter = 0.1        # 10 cm
rx_diameter = 0.5        # 50 cm

# range tipico LEO (500–2000 km)
R = np.linspace(5e5, 2e6, 500)


# ======================================================
# CALCOLO
# ======================================================

eta_geo = geometric_loss(
    tx_diameter=tx_diameter,
    rx_diameter=rx_diameter,
    wavelength=wavelength,
    R=R
)


# ======================================================
# CHECK NUMERICI
# ======================================================

print("\n=== GEOMETRIC LOSS CHECK ===")
print(f"Min eta: {eta_geo.min():.3e}")
print(f"Max eta: {eta_geo.max():.3e}")

# monotonicità (deve decrescere)
is_monotonic = np.all(np.diff(eta_geo) <= 1e-12)
print(f"Monotonic decreasing: {is_monotonic}")


# ======================================================
# PLOT LINEARE
# ======================================================

plt.figure()
plt.plot(R / 1e3, eta_geo)
plt.xlabel("Distance [km]")
plt.ylabel("Geometric efficiency")
plt.title("Geometric Loss vs Distance")
plt.grid()


# ======================================================
# PLOT LOG (fondamentale)
# ======================================================

plt.figure()
plt.semilogy(R / 1e3, eta_geo)
plt.xlabel("Distance [km]")
plt.ylabel("Geometric efficiency (log scale)")
plt.title("Geometric Loss vs Distance (log)")
plt.grid()


# ======================================================
# SCALING CHECK ~ 1/R^2
# ======================================================

# normalizzo rispetto al primo punto
eta_norm = eta_geo / eta_geo[0]
R_norm = R / R[0]

plt.figure()
plt.loglog(R_norm, eta_norm, label="Simulated")

# riferimento teorico ~ 1/R^2
plt.loglog(R_norm, R_norm**-2, '--', label="~1/R^2")

plt.xlabel("Normalized distance")
plt.ylabel("Normalized eta")
plt.title("Scaling Check")
plt.legend()
plt.grid()


plt.show()
