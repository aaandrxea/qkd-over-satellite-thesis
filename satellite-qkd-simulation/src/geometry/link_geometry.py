import numpy as np


# ================================
# VISIBILITY MASK
# ================================

def visibility_mask(elevation_deg, min_elevation=0.0):
    """
    True se satellite sopra orizzonte
    """
    return elevation_deg > min_elevation


# ================================
# FILTER VISIBLE PASS
# ================================

def filter_visibility(elevation_deg, distance_m, min_elevation=0.0):
    """
    Filtra solo i punti visibili
    """

    mask = visibility_mask(elevation_deg, min_elevation)

    elevation_visible = elevation_deg[mask]
    distance_visible = distance_m[mask]

    return elevation_visible, distance_visible, mask


# ================================
# CONVERSION FOR CHANNEL
# ================================

def prepare_channel_inputs(elevation_deg, distance_m):
    """
    Converte per il channel model
    """

    elevation_rad = np.deg2rad(elevation_deg)
    R = distance_m

    return R, elevation_rad


# ================================
# FULL PIPELINE
# ================================

def compute_link_geometry(elevation_deg, distance_m, min_elevation=0.0):
    """
    Pipeline completa geometry → channel inputs
    """

    elev_vis, dist_vis, mask = filter_visibility(
        elevation_deg,
        distance_m,
        min_elevation
    )

    R, elevation_rad = prepare_channel_inputs(
        elev_vis,
        dist_vis
    )

    return {
        "R": R,
        "elevation": elevation_rad,
        "mask": mask,
        "elevation_deg": elev_vis,
        "distance_m": dist_vis
    }
