import numpy as np


# ==========================================================
# MARKOV CLOUD MODEL
# ==========================================================

def simulate_cloud_state(N, p_clear_to_cloud=0.01, p_cloud_to_clear=0.1):
    """
    Generate cloud state with temporal correlation.

    0 = clear
    1 = cloudy
    """

    state = np.zeros(N, dtype=int)

    i = 0
    current = 0  # start in clear state

    while i < N:
        if current == 0:
            duration = np.random.geometric(p_clear_to_cloud)
        else:
            duration = np.random.geometric(p_cloud_to_clear)

        end = min(i + duration, N)
        state[i:end] = current

        i = end
        current = 1 - current

    return state

# ==========================================================
# CLOUD ATTENUATION
# ==========================================================

def cloud_attenuation(state, min_loss_db=20, max_loss_db=80):
    """
    Convert cloud state → transmission factor.
    """

    N = len(state)
    eta = np.ones(N)

    # random attenuation for cloudy points
    cloudy = state == 1

    loss_db = np.random.uniform(min_loss_db, max_loss_db, size=N)

    eta[cloudy] = 10 ** (-loss_db[cloudy] / 10)

    return eta


# ==========================================================
# MAIN INTERFACE
# ==========================================================

def cloud_transmittance(N, config=None):
    """
    Returns eta_cloud(t)
    """

    if config is None:
        config = {}

    cloud_cfg = config.get("clouds", {})

    p_clear_to_cloud = float(cloud_cfg.get("p_clear_to_cloud", 0.01))
    p_cloud_to_clear = float(cloud_cfg.get("p_cloud_to_clear", 0.1))

    min_loss_db = float(cloud_cfg.get("min_loss_db", 20))
    max_loss_db = float(cloud_cfg.get("max_loss_db", 80))

    state = simulate_cloud_state(
        N,
        p_clear_to_cloud,
        p_cloud_to_clear
    )

    eta = cloud_attenuation(
        state,
        min_loss_db,
        max_loss_db
    )

    return eta, state
