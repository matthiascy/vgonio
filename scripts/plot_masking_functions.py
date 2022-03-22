import sys
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 1:
        print("no file provided!", file=sys.stderr)
        exit(-1)

    filename = sys.argv[1]
    data = np.array(open(filename, 'r').read().rstrip().split(' ')).astype(np.float32)
    data = data.reshape((360, 91))

    sb.set_palette("pastel")

    xs = np.array([i - 90 for i in range(0, 181)])
    # for phi in range(0, 180):
    phi = 0
    ys = np.concatenate((data[phi, ::-1], data[phi+180, 1:]))
    plt.clf()
    sb.lineplot(x=xs, y=ys)
    plt.title(f"Measured Masking Function, azimuth = {phi}")
    plt.ylabel("Percentage")
    plt.show()

