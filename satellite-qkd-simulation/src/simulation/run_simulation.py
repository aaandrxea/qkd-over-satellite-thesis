import numpy as np

# CHANNEL
from src.channel.link_budget import compute_link_budget

# DETECTION
from src.detection.detector import compute_detection

# QKD
from src.qkd.qber import compute_qber
from src.qkd.key_rate import compute_key_rate

# DECOY
from src.qkd.decoy_state import compute_decoy_rate


def run_simulation(
    R,
    elevation,
    wavelength,
    tx_diameter,
    rx_diameter,
    mu,
    nu,
    p_dark,
    e_opt,
    config=None
):
    """
    Pipeline completa QKD:
    channel → detection → qkd → decoy → keyrate
    """

    # ============================
    # CHANNEL
    # ============================
    channel = compute_link_budget(
        R,
        elevation,
        wavelength,
        tx_diameter,
        rx_diameter,
        config=config
    )

    eta_total = channel["eta_total"]

    # ============================
    # DETECTION (SIGNAL μ)
    # ============================
    det_mu = compute_detection(
        eta_total=eta_total,
        mu=mu,
        p_dark=p_dark,
        e_opt=e_opt
    )

    # ============================
    # DETECTION (DECOY ν)
    # ============================
    det_nu = compute_detection(
        eta_total=eta_total,
        mu=nu,
        p_dark=p_dark,
        e_opt=e_opt
    )

    # ============================
    # QBER (da detection μ)
    # ============================
    qber = compute_qber(
        det_mu["p_err"],
        det_mu["p_click"]
    )

    # ============================
    # KEY RATE (BASE)
    # ============================
    keyrate = compute_key_rate({
        "p_click": det_mu["p_click"],
        "qber": qber
    })

    # ============================
    # DECOY KEY RATE
    # ============================
    decoy = compute_decoy_rate(
        detection_mu={
            "p_click": det_mu["p_click"],
            "qber": qber
        },
        detection_nu={
            "p_click": det_nu["p_click"],
            "qber": qber
        },
        mu=mu,
        nu=nu
    )

    return {
        "channel": channel,
        "detection_mu": det_mu,
        "detection_nu": det_nu,
        "qber": qber,
        "keyrate": keyrate,
        "decoy": decoy
    }
