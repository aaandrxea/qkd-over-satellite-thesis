import numpy as np

from src.utils.constants import PI, EPS
from src.utils.io import load_yaml

from src.channel.atmospheric import atmospheric_transmittance
from src.channel.turbulence import turbulence_fading
from src.channel.pointing import pointing_fading, beam_divergence


# ================================
# CONFIG
# ================================

def load_channel_config(config_path: str = "config/scenario.yaml") -> dict:
    return load_yaml(config_path)


# ================================
# BEAM RADIUS
# ================================

def beam_radius(
    R: np.ndarray,
    divergence: float
) -> np.ndarray:
    """
    w(R) = R * theta_div
    """
    return R * divergence


# ================================
# PERDITE GEOMETRICHE (GAUSSIANO)
# ================================

def geometric_loss(
    R: np.ndarray,
    wavelength: float,
    tx_diameter: float,
    rx_diameter: float
) -> np.ndarray:
    """
    Coupling gaussiano realistico:

    eta_geo = 1 - exp( -2 * (a^2 / w^2) )

    dove:
    a = raggio ricevitore
    w = raggio fascio
    """

    theta_div = beam_divergence(wavelength, tx_diameter)
    w_R = beam_radius(R, theta_div)

    a = rx_diameter / 2.0

    eta = 1.0 - np.exp(-2.0 * (a*2) / (w_R*2 + EPS))

    return np.clip(eta, 0.0, 1.0)


# ================================
# EFFICIENZA OTTICA
# ================================

def optical_efficiency(
    eta_tx: float,
    eta_rx: float,
    eta_misc: float
) -> float:
    return eta_tx * eta_rx * eta_misc


# ================================
# LINK BUDGET COMPLETO
# ================================

def compute_link_budget(
    R: np.ndarray,
    elevation: np.ndarray,
    wavelength: float,
    tx_diameter: float,
    rx_diameter: float,
    config: dict | None = None
) -> dict:
    """
    Calcolo completo del link budget.

    Returns
    -------
    dict:
        eta_geo
        eta_atm
        eta_turb
        eta_point
        eta_sys
        eta_total
    """

    if config is None:
        config = load_channel_config()

    # ----------------------------
    # Parametri sistema
    # ----------------------------
    eta_tx = config.get("eta_tx", 1.0)
    eta_rx = config.get("eta_rx", 1.0)
    eta_misc = config.get("eta_misc", 1.0)

    # ----------------------------
    # Geometria
    # ----------------------------
    eta_geo = geometric_loss(
        R,
        wavelength,
        tx_diameter,
        rx_diameter
    )

    # ----------------------------
    # Atmosfera (deterministica)
    # ----------------------------
    eta_atm = atmospheric_transmittance(
        elevation,
        R
    )

    # ----------------------------
    # Turbolenza (stocastica)
    # ----------------------------
    eta_turb = turbulence_fading(
        elevation,
        R,
        wavelength
    )

    # ----------------------------
    # Pointing (stocastico)
    # ----------------------------
    eta_point = pointing_fading(
        R,
        wavelength,
        tx_diameter,
        config = config.get("pointing", )
)

    # ----------------------------
    # Efficienza sistema
    # ----------------------------
    eta_sys = optical_efficiency(
        eta_tx,
        eta_rx,
        eta_misc
    )

    # ----------------------------
    # Trasmissione totale
    # ----------------------------
    eta_total = (
        eta_geo *
        eta_atm *
        eta_turb *
        eta_point *
        eta_sys
    )

    eta_total = np.clip(eta_total, 0.0, 1.0)

    return {
        "eta_geo": eta_geo,
        "eta_atm": eta_atm,
        "eta_turb": eta_turb,
        "eta_point": eta_point,
        "eta_sys": eta_sys,
        "eta_total": eta_total
    }


# ================================
# VERSIONE dB
# ================================

def to_dB(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-15, None)
    return -10.0 * np.log10(x)


def link_budget_dB(results: dict) -> dict:
    return {k: to_dB(v) for k, v in results.items()}
