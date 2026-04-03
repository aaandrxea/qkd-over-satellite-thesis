import numpy as np

# ==========================================================
# TIME → JULIAN DATE
# ==========================================================

def datetime_to_julian(dt):
    """
    Convert datetime (UTC) to Julian date.
    """

    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + dt.minute/60 + dt.second/3600)/24

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    JD = int(365.25*(year + 4716)) + int(30.6001*(month + 1)) + day + B - 1524.5

    return JD


# ==========================================================
# SOLAR POSITION (NOAA MODEL)
# ==========================================================

def solar_position(dt, lat, lon):
    """
    Compute solar azimuth and elevation (radians).

    lat, lon in radians
    """

    JD = datetime_to_julian(dt)
    T = (JD - 2451545.0) / 36525.0

    # Mean longitude
    L0 = (280.46646 + 36000.76983*T) % 360

    # Mean anomaly
    M = np.deg2rad((357.52911 + 35999.05029*T) % 360)

    # Ecliptic longitude
    C = (1.914602 - 0.004817*T)*np.sin(M) + 0.019993*np.sin(2*M)
    lam = np.deg2rad(L0 + C)

    # Obliquity
    eps = np.deg2rad(23.439 - 0.0000004*T)

    # Declination
    delta = np.arcsin(np.sin(eps) * np.sin(lam))

    # Equation of time (approx)
    y = np.tan(eps/2)**2
    E = 4 * np.rad2deg(
        y*np.sin(2*np.deg2rad(L0))
        - 2*0.0167*np.sin(M)
    )

    # Solar time
    time = dt.hour + dt.minute/60 + dt.second/3600
    true_time = (time*60 + E + 4*np.rad2deg(lon)) % 1440

    hour_angle = np.deg2rad(true_time/4 - 180)

    # Elevation
    elev = np.arcsin(
        np.sin(lat)*np.sin(delta) +
        np.cos(lat)*np.cos(delta)*np.cos(hour_angle)
    )

    # Azimuth
    az = np.arctan2(
        -np.sin(hour_angle),
        np.tan(delta)*np.cos(lat) - np.sin(lat)*np.cos(hour_angle)
    )

    return az, elev
# ==========================================================
# SUN-SATELLITE ANGLE
# ==========================================================

def sun_los_angle(sun_az, sun_el, sat_az, sat_el):
    """
    Angle between sun direction and line-of-sight (radians)
    """

    # convert to unit vectors
    def sph_to_cart(az, el):
        return np.array([
            np.cos(el)*np.cos(az),
            np.cos(el)*np.sin(az),
            np.sin(el)
        ])

    s = sph_to_cart(sun_az, sun_el)
    l = sph_to_cart(sat_az, sat_el)

    cos_angle = np.dot(s, l)

    return np.arccos(np.clip(cos_angle, -1.0, 1.0))
