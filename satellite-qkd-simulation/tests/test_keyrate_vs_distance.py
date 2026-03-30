import numpy as np
import matplotlib.pyplot as plt

from src.channel.link_budget import compute_link_budget
from src.detection.detector import compute_detection
from src.qkd.key_rate import compute_key_rate


# ==========================================================
# SCENARIO
# ==========================================================

def generate_scenario(n_points=100):

    elevation = np.linspace(np.deg2rad(10), np.deg2rad(90), n_points)

    # evita divergenze
    sin_el = np.sin(elevation)
    sin_el = np.clip(sin_el, 1e-3, None)

    R = 500e3 / sin_el
# ordina per distanza crescente
    idx = np.argsort(R)
    R = R[idx]
    elevation = elevation[idx]
    return R, elevation

# ==========================================================
# PARAMETRI
# ==========================================================

PARAMS = {
    "wavelength": 800e-9,
    "tx_diameter": 0.1,
    "rx_diameter": 0.5,
    "mu": 0.5,
    "nu": 0.1,
    "dark_rate": 100,
    "gate_time": 1e-9,
    "e_opt": 0.02,
}


# ==========================================================
# PIPELINE
# ==========================================================

def run_qkd(R, elevation, bg_rate):

    channel = compute_link_budget(
        R,
        elevation,
        PARAMS["wavelength"],
        PARAMS["tx_diameter"],
        PARAMS["rx_diameter"]
    )

    eta = channel["eta_total"]

    # ----------------------------
    # SIGNAL
    # ----------------------------
    det_mu = compute_detection(
        eta,
        PARAMS["mu"],
        PARAMS["dark_rate"],
        PARAMS["gate_time"],
        PARAMS["e_opt"],
        bg_rate=bg_rate
    )

    # ----------------------------
    # DECOY
    # ----------------------------
    det_nu = compute_detection(
        eta,
        PARAMS["nu"],
        PARAMS["dark_rate"],
        PARAMS["gate_time"],
        PARAMS["e_opt"],
        bg_rate=bg_rate
    )

    # ----------------------------
    # VACUUM
    # ----------------------------
    det_0 = compute_detection(
        eta,
        0.0,
        PARAMS["dark_rate"],
        PARAMS["gate_time"],
        PARAMS["e_opt"],
        bg_rate=bg_rate
    )

    # ----------------------------
    # KEY RATE
    # ----------------------------
    kr = compute_key_rate(
        det_mu,
        det_nu,
        det_0,
        PARAMS["mu"],
        PARAMS["nu"]
    )

    return kr["skr"]


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    R, elevation = generate_scenario()

    # ----------------------------
    # NIGHT
    # ----------------------------
    R_night = run_qkd(R, elevation, bg_rate=0.0)

    # ----------------------------
    # DAY
    # ----------------------------
    R_day = run_qkd(R, elevation, bg_rate=1e6)
    # ----------------------------
    # PLOT
    # ----------------------------
    distance_km = R / 1e3
    def max_distance(R, d):
        idx = np.where(R > 0)[0]
        return d[idx[-1]] if len(idx) > 0 else 0

    print("\n=== DISTANCE LIMITS ===")
    print("Night:", max_distance(R_night, distance_km))
    print("Day:", max_distance(R_day, distance_km))
    print("\nDEBUG DISTANCE:")
    print("min:", np.min(R)/1e3, "km")
    print("max:", np.max(R)/1e3, "km")
    plt.figure(figsize=(8, 5))

    plt.semilogy(distance_km, R_night, label="Night")
    plt.semilogy(distance_km, np.maximum(R_day, 1e-12), label="Day")
    plt.xlabel("Distance [km]")
    plt.ylabel("Secret Key Rate")
    plt.title("QKD Key Rate vs Distance")

    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # DEBUG INFO
    # ----------------------------
    print("\n=== SUMMARY ===")
    print(f"Night max distance (R>0): {distance_km[np.where(R_night>0)[0][-1]]:.1f} km")
    print(f"Day max distance (R>0): {distance_km[np.where(R_day>0)[0][-1]]:.1f} km")
