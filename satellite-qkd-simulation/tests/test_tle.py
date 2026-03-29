from src.orbit.tle_loader import load_tle_from_string


def test_tle_loader():

    tle = """ISS (ZARYA)
1 25544U 98067A   21275.51005787  .00001264  00000-0  29604-4 0  9993
2 25544  51.6442  21.5183 0003654  91.5623  36.7850 15.48915334299929
"""

    name, l1, l2 = load_tle_from_string(tle)

    print("Name:", name)
    print("Line1:", l1)
    print("Line2:", l2)

    assert len(l1) > 0
    assert len(l2) > 0

    print("✓ TLE loader OK")


if __name__ == "__main__":
    test_tle_loader()
