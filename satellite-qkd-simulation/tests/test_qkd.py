import numpy as np

from src.qkd.key_rate import compute_key_rate


def test_key_rate_basic():

    p_click = np.linspace(1e-6, 1e-2, 1000)
    qber = np.full_like(p_click, 0.02)

    detection = {
        "p_click": p_click,
        "qber": qber
    }

    results = compute_key_rate(detection)

    skr = results["skr"]

    # ----------------------------
    # CHECK
    # ----------------------------
    assert np.all(skr >= 0)

    mean_skr = np.mean(skr)

    print("DEBUG SKR mean:", mean_skr)

    assert mean_skr > 0

    print("✓ Key rate OK")


if __name__ == "__main__":
    test_key_rate_basic()
