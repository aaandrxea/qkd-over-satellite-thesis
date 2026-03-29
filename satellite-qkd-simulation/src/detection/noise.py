import numpy as np

def dark_count_probability(p_dark, shape):
    """
    Dark counts costanti
    """
    return np.full(shape, p_dark)
