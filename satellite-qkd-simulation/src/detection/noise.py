import numpy as np

def dark_count_probability(dark_rate, gate_time, shape):
    p_dark = 1.0 - np.exp(-dark_rate * gate_time)
    return np.full(shape, p_dark)
