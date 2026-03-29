import numpy as np

from src.qkd.decoy_state import compute_decoy_rate


def test_decoy_basic():

    # finti dati realistici
    Q_mu = np.linspace(1e-6, 1e-2, 1000)
    Q_nu = 0.5 * Q_mu

    detection_mu = {
        "p_click": Q_mu,
        "qber": np.full_like(Q_mu, 0.02)
    }

    detection_nu = {
        "p_click": Q_nu,
        "qber": np.full_like(Q_mu, 0.02)
    }

    results = compute_decoy_rate(
        detection_mu,
        detection_nu,
        mu=0.5,
        nu=0.1
    )

    skr = results["skr_decoy"]

    assert np.all(skr >= 0)

    print("DEBUG Decoy SKR mean:", np.mean(skr))
    print("✓ Decoy OK")


if __name__ == "__main__":
    test_decoy_basic()
