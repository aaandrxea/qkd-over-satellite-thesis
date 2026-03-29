import os


def load_tle(file_path):
    """
    Carica TLE da file

    Returns
    -------
    name, line1, line2
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TLE file not found: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    # rimuovi newline
    lines = [l.strip() for l in lines if l.strip()]

    if len(lines) < 3:
        raise ValueError("TLE file must contain at least 3 lines")

    name = lines[0]
    line1 = lines[1]
    line2 = lines[2]

    return name, line1, line2


def load_tle_from_string(tle_string):
    """
    Utile per test rapidi
    """

    lines = [l.strip() for l in tle_string.split("\n") if l.strip()]

    if len(lines) < 3:
        raise ValueError("Invalid TLE string")

    return lines[0], lines[1], lines[2]
