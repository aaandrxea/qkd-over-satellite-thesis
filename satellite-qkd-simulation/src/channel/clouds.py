import numpy as np

EPS = 1e-15


# ==========================================================
# MARKOV STATE (TEMPORAL CORRELATION)
# ==========================================================

def generate_cloud_state(n_steps, p_clear, tau_corr, dt, rng):
    state = np.zeros(n_steps)

    p_switch = dt / max(tau_corr, EPS)

    for i in range(1, n_steps):
        if rng.random() < p_switch:
            state[i] = 1 - state[i-1]
        else:
            state[i] = state[i-1]

    # enforce clear probability
    mask = rng.random(n_steps) < p_clear
    state[mask] = 0

    return state


# ==========================================================
# NORMAL CLOUD LOSS
# ==========================================================

def sample_normal_cloud(min_db, max_db, size, rng):
    return rng.uniform(min_db, max_db, size=size)


# ==========================================================
# EXTREME EVENTS (HEAVY TAIL)
# ==========================================================

def sample_extreme_cloud(size, config, rng):
    """
    Rare heavy attenuation events.
    """

    p_extreme = config.get("p_extreme", 0.01)

    extreme_mask = rng.random(size) < p_extreme

    loss = np.zeros(size)

    if np.any(extreme_mask):
        # lognormal heavy tail
        mean = config.get("extreme_mean_db", 15.0)
        sigma = config.get("extreme_sigma", 0.7)

        loss_ext = rng.lognormal(mean=np.log(mean), sigma=sigma, size=np.sum(extreme_mask))

        loss[extreme_mask] = loss_ext

    return loss, extreme_mask


# ==========================================================
# MAIN MODEL
# ==========================================================

def cloud_attenuation_2d(
    n_steps,
    elevation,
    dt,
    config=None,
    rng=None
):
    if rng is None:
        rng = np.random

    if config is None:
        config = {}

    # ----------------------------
    # PARAMETERS
    # ----------------------------

    p_clear = config.get("p_clear", 0.7)
    tau_corr = config.get("correlation_time", 20.0)

    min_db = config.get("min_loss_db", 1.0)
    max_db = config.get("max_loss_db", 8.0)

    efficiency = config.get("efficiency", 1.0)

    elevation = np.asarray(elevation)
    sin_el = np.maximum(np.sin(elevation), 0.2)

    # ----------------------------
    # STATE
    # ----------------------------

    cloud_state = generate_cloud_state(
        n_steps, p_clear, tau_corr, dt, rng
    )

    # ----------------------------
    # BASE LOSS
    # ----------------------------

    loss_db = sample_normal_cloud(min_db, max_db, n_steps, rng)

    # ----------------------------
    # EXTREME EVENTS (RARE)
    # ----------------------------

    extreme_loss, extreme_mask = sample_extreme_cloud(n_steps, config, rng)

    # combine
    loss_db = np.where(extreme_mask, extreme_loss, loss_db)

    # ----------------------------
    # ELEVATION SCALING (MILD)
    # ----------------------------

    slant = 1.0 / sin_el
    loss_db *= (1 + 0.2 * (slant - 1))

    # hard cap (VERY IMPORTANT)
    max_cap = config.get("hard_max_db", 25.0)
    loss_db = np.clip(loss_db, 0.0, max_cap)

    # ----------------------------
    # APPLY ONLY WHEN CLOUD PRESENT
    # ----------------------------

    eta_cloud = np.ones(n_steps)

    mask = cloud_state > 0.5

    eta_cloud[mask] = 10 ** (-loss_db[mask] / 10.0)

    # efficiency scaling
    eta_cloud *= efficiency

    return cloud_state, np.clip(eta_cloud, 0.0, 1.0)
