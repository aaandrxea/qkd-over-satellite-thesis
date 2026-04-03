import numpy as np

from src.simulation.run_orbit_simulation import run_orbit_simulation


# ======================================================
# UTILS
# ======================================================

def safe_stats(name, x):
    if x is None or len(x) == 0:
        print(f"{name}: EMPTY")
        return
    print(f"{name}: mean={np.mean(x):.3e}, min={np.min(x):.3e}, max={np.max(x):.3e}")


# ======================================================
# MAIN TEST
# ======================================================

def main():
    print("\n=== RUNNING FULL PIPELINE TEST ===\n")

    result = run_orbit_simulation()

    if result is None:
        print("❌ Simulation failed (no visible pass)")
        return

    eta = result["eta"]
    qber = result["qber"]
    skr = result["skr"]
    elevation = result["elevation"]

    # optional
    sun_el = result.get("sun_elevation", None)
    sun_angle = result.get("sun_angle", None)

    channel = result["channel"]
    comp = channel["components"]

    bg = channel.get("background", None)

    det = result["detection"]["mu"]
    p_sig = det["p_sig"]
    p_noise = det["p_noise"]

    # ======================================================
    # 1. DATA VALIDATION
    # ======================================================

    print("\n--- DATA CHECK ---")

    if len(eta) == 0:
        print("❌ Empty simulation output")
        return

    print(f"Samples: {len(eta)}")

    # ======================================================
    # 2. CHANNEL COMPONENTS
    # ======================================================

    print("\n--- CHANNEL COMPONENTS ---")

    for k, v in comp.items():
        safe_stats(k, v)

    # ======================================================
    # 3. BASIC METRICS
    # ======================================================

    print("\n--- BASIC METRICS ---")

    safe_stats("η", eta)
    safe_stats("QBER", qber)
    safe_stats("SKR", skr)

    # ======================================================
    # 4. PHYSICAL SANITY
    # ======================================================

    print("\n--- PHYSICAL CHECK ---")

    if np.any(eta > 1):
        print("❌ η > 1 → unphysical")

    if np.all(eta < 1e-12):
        print("❌ η too low → dead link")

    if np.all(qber >= 0.49):
        print("❌ QBER ≈ 0.5 → pure noise")

    if np.any(qber < 0):
        print("❌ QBER < 0")

    if np.all(skr <= 0):
        print("⚠ SKR always zero")

    # ======================================================
    # 5. SIGNAL VS NOISE
    # ======================================================

    print("\n--- SIGNAL VS NOISE ---")

    ratio = np.mean(p_sig / (p_noise + 1e-15))
    print(f"Signal/Noise ratio: {ratio:.3e}")

    if ratio < 1:
        print("⚠ Noise-dominated")
    else:
        print("✔ Signal-dominated")

    # ======================================================
    # 6. SOLAR / BACKGROUND
    # ======================================================

    print("\n--- SOLAR / BACKGROUND ---")

    if sun_el is None or len(sun_el) == 0:
        print("⚠ No solar data available")
    else:
        print("Sun elevation range (deg):",
              np.min(np.rad2deg(sun_el)),
              np.max(np.rad2deg(sun_el)))

        day = sun_el > 0
        night = sun_el <= 0

        if np.any(day) and np.any(night):
            print("QBER day:", np.mean(qber[day]))
            print("QBER night:", np.mean(qber[night]))

            print("SKR day:", np.mean(skr[day]))
            print("SKR night:", np.mean(skr[night]))
        else:
            print("⚠ Only day or only night samples")

    if bg is not None and len(bg) > 0:
        safe_stats("Background", bg)

        if sun_angle is not None and len(sun_angle) > 0:
            idx = np.argsort(sun_angle)
            n = min(50, len(bg)//2)

            print("Low angle bg:", np.mean(bg[idx[:n]]))
            print("High angle bg:", np.mean(bg[idx[-n:]]))

    # ======================================================
    # 7. ELEVATION CHECK
    # ======================================================

    print("\n--- ELEVATION CHECK ---")

    elev_deg = np.rad2deg(elevation)

    idx_max = np.argmax(elev_deg)

    print(f"Max elevation: {np.max(elev_deg):.2f} deg")
    print(f"QBER at peak: {qber[idx_max]:.3f}")
    print(f"SKR at peak: {skr[idx_max]:.3e}")

    # ======================================================
    # 8. NUMERICAL STABILITY
    # ======================================================

    print("\n--- NUMERICAL CHECK ---")

    if np.any(np.isnan(qber)) or np.any(np.isnan(skr)):
        print("❌ NaN detected")
    else:
        print("✔ No NaN")

    if np.any(np.isinf(qber)) or np.any(np.isinf(skr)):
        print("❌ Inf detected")
    else:
        print("✔ No Inf")

    print("\n=== TEST COMPLETED ===\n")


if __name__ == "__main__":
    main()
