import matplotlib.pyplot as plt


def plot_results(result):

    t = result["time_visible"]
    elev = result["elevation"]
    skr = result["skr"]
    qber = result["qber"]

    plt.figure()
    plt.plot(t, elev)
    plt.title("Elevation vs Time")

    plt.figure()
    plt.plot(t, skr)
    plt.title("SKR vs Time")

    plt.figure()
    plt.plot(t, qber)
    plt.title("QBER vs Time")

    plt.show()
