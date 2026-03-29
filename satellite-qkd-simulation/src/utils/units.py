import numpy as np

# ================================
# ANGOLI
# ================================

def deg2rad(x):
    return np.deg2rad(x)

def rad2deg(x):
    return np.rad2deg(x)

# ================================
# DISTANZE
# ================================

def km2m(x):
    return x * 1e3

def m2km(x):
    return x / 1e3

# ================================
# POTENZA / dB
# ================================

def linear_to_db(x):
    x = np.clip(x, 1e-15, None)
    return 10 * np.log10(x)

def db_to_linear(x):
    return 10 ** (x / 10)

# ================================
# TEMPO
# ================================

def ns2s(x):
    return x * 1e-9

def s2ns(x):
    return x * 1e9
