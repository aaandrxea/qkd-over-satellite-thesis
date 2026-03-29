from src.orbit.tle_loader import load_tle_from_string
from src.orbit.orbit_propagation import (
    create_satellite,
    generate_time_array,
    create_ground_station,
    propagate_satellite
)

import numpy as np


def test_orbit_basic():

    tle = """ISS (ZARYA)
1 25544U 98067A   21275.51005787  .00001264  00000-0  29604-4 0  9993
2 25544  51.6442  21.5183 0003654  91.5623  36.7850 15.48915334299929
"""

    name, l1, l2 = load_tle_from_string(tle)

    sat = create_satellite(name, l1, l2)

    times = generate_time_array(duration_s=600)

    gs = create_ground_station(45.0, 9.0, 100)  # Milano approx

    elevation, distance = propagate_satellite(sat, gs, times)

    print("Elevation mean:", np.mean(elevation))
    print("Distance mean:", np.mean(distance))

    assert len(elevation) > 0
    assert np.all(distance > 0)

    print("✓ Orbit propagation OK")


if __name__ == "__main__":
    test_orbit_basic()
