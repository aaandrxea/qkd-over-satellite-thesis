import numpy as np

from src.qkd.decoy_state import compute_key_rate_decoy


def test_decoy():

    mu = 0.5
    nu = 0.1

    Q_mu = 1e-3
    Q_nu = 2e-4

    E_mu = 0.02
    E_nu = 0.03

    R, Y1, e1 = compute_key_rate_decoy(
        mu, nu,
        Q_mu, Q_nu,
        E_mu, E_nu
    )

    print("R:", R)
    print("Y1:", Y1)
    print("e1:", e1)

    assert R >= 0
    assert 0 <= e1 <= 0.5
    assert Y1 > 0

    print("✓ Decoy OK")


if __name__ == "__main__":
    test_decoy()
