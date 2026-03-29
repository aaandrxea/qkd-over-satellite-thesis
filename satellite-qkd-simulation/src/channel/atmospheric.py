import numpy as np

# ================================
# COSTANTI FISICHE
# ================================

RHO_0 = 1.225          # kg/m^3, densità aria al livello del mare
H_SCALE = 8000.0       # m, altezza di scala atmosferica

# Parametri attenuazione (approssimati, modificabili)
TAU_0 = 0.1            # optical depth verticale

# Parametri Hufnagel-Valley
A_CN2 = 1.7e-14        # coefficiente near-ground
V_WIND = 21.0          # m/s velocità vento tipica


# ================================
# DENSITÀ DELL’ARIA
# ================================

def air_density(h: np.ndarray | float) -> np.ndarray | float:
    """
    Densità dell'aria in funzione dell'altitudine.

    Parameters
    ----------
    h : float or np.ndarray
        Altitudine [m]

    Returns
    -------
    rho : float or np.ndarray
        Densità [kg/m^3]
    """
    h = np.asarray(h)
    return RHO_0 * np.exp(-h / H_SCALE)


# ================================
# TRASMITTANZA ATMOSFERICA
# ================================

def atmospheric_transmittance(elevation: np.ndarray) -> np.ndarray:
    """
    Trasmittanza atmosferica in funzione dell'elevazione.

    Modello:
    eta_atm = exp(-tau_0 / sin(el))

    Parameters
    ----------
    elevation : np.ndarray
        Angolo di elevazione [rad]

    Returns
    -------
    eta_atm : np.ndarray
        Trasmittanza atmosferica [0,1]
    """
    elevation = np.asarray(elevation)

    # Evita divisioni per zero o valori negativi
    elevation = np.clip(elevation, 1e-6, np.pi / 2)

    air_mass = 1.0 / np.sin(elevation)

    eta_atm = np.exp(-TAU_0 * air_mass)

    return eta_atm


# ================================
# PROFILO Cn^2 (TURBOLENZA)
# ================================

def cn2_profile(h: np.ndarray | float) -> np.ndarray | float:
    """
    Profilo Cn^2 secondo Hufnagel-Valley.

    Parameters
    ----------
    h : float or np.ndarray
        Altitudine [m]

    Returns
    -------
    cn2 : float or np.ndarray
        Struttura della turbolenza [m^(-2/3)]
    """
    h = np.asarray(h)

    term1 = 0.00594 * (V_WIND / 27.0) * 2 * (1e-5 * h) * 10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A_CN2 * np.exp(-h / 100.0)

    return term1 + term2 + term3


# ================================
# RYTOV VARIANCE (SCINTILLAZIONE)
# ================================

def rytov_variance(
    wavelength: float,
    cn2: np.ndarray,
    path_length: np.ndarray
) -> np.ndarray:
    """
    Varianza di Rytov (scintillazione)

    Parameters
    ----------
    wavelength : float
        Lunghezza d'onda [m]
    cn2 : np.ndarray
        Profilo Cn^2
    path_length : np.ndarray
        Lunghezza percorso [m]

    Returns
    -------
    sigma_R2 : np.ndarray
        Varianza di Rytov
    """
    k = 2 * np.pi / wavelength

    return 1.23 * cn2 * (k * (7.0 / 6.0)) * (path_length * (11.0 / 6.0))


# ================================
# FADING TURBOLENTO (LOG-NORMAL)
# ================================

def turbulence_fading(
    sigma_R2: np.ndarray,
    size: int | None = None
) -> np.ndarray:
    """
    Genera fading log-normale dovuto a turbolenza.

    Parameters
    ----------
    sigma_R2 : np.ndarray
        Varianza di Rytov
    size : int, optional
        Numero campioni (se None, usa shape di sigma_R2)

    Returns
    -------
    fading : np.ndarray
        Fattore moltiplicativo (>0)
    """
    sigma_lnI = np.sqrt(sigma_R2)

    mu_lnI = -0.5 * sigma_lnI**2

    if size is None:
        size = sigma_R2.shape

    return np.exp(
        mu_lnI + sigma_lnI * np.random.normal(size=size)
    )


# ================================
# TRASMITTANZA COMPLETA ATMOSFERA
# ================================

def atmospheric_channel(
    elevation: np.ndarray,
    wavelength: float,
    path_length: np.ndarray,
    include_turbulence: bool = False
) -> np.ndarray:
    """
    Trasmittanza atmosferica completa.

    Parameters
    ----------
    elevation : np.ndarray
        Elevazione [rad]
    wavelength : float
        Lunghezza d'onda [m]
    path_length : np.ndarray
        Distanza [m]
    include_turbulence : bool
        Se includere fading turbolento

    Returns
    -------
    eta_atm_total : np.ndarray
    """
    eta_atm = atmospheric_transmittance(elevation)

    if not include_turbulence:
        return eta_atm

    # Stima altitudine media (semplificazione)
    h_eff = 5000.0  # m

    cn2 = cn2_profile(h_eff)

    sigma_R2 = rytov_variance(wavelength, cn2, path_length)

    fading = turbulence_fading(sigma_R2)

    return eta_atm * fading
