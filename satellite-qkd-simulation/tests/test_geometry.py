import numpy as np

from src.geometry.link_geometry import compute_link_geometry


def test_geometry_basic():

    elevation = np.linspace(-10, 80, 1000)
    distance = np.linspace(2e5, 1e6, 1000)

    geom = compute_link_geometry(elevation, distance)

    R = geom["R"]
    elev = geom["elevation"]

    print("Visible points:", len(R))

    assert len(R) > 0
    assert np.all(R > 0)
    assert np.all(elev >= 0)

    print("✓ Geometry OK")


if __name__ == "__main__":
    test_geometry_basic()
