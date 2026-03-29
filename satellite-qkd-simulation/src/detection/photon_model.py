import numpy as np

def signal_probability(mu, eta):
    """
    p_sig = 1 - exp(-mu * eta)
    """
    return 1.0 - np.exp(-mu * eta)
