import numpy as np

# ================================
# COSTANTI FISICHE
# ================================

C = 299792458.0                 # velocità della luce [m/s]
H = 6.62607015e-34              # costante di Planck [J*s]
K_B = 1.380649e-23              # costante di Boltzmann [J/K]

# ================================
# COSTANTI OTTICHE
# ================================

PI = np.pi

# ================================
# ATMOSFERA
# ================================

RHO_0 = 1.225                   # kg/m^3 (livello del mare)
H_SCALE = 8000.0                # m

# ================================
# QKD
# ================================

QBER_MAX = 0.5
SECURITY_FACTOR = 0.5           # base BB84 (basis sifting)

# ================================
# NUMERICHE
# ================================

EPS = 1e-12
