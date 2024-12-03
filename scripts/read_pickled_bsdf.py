import pickle
import numpy as np
from sys import argv

if __name__ == "__main__":
    filename = argv[1]
    with open(filename, "rb") as f:
        bsdf = pickle.load(f)
    print(type(bsdf))
    print(len(bsdf.keys()))

    unique_keys = set()

    n_spectra = 0
    for i, (key, value) in enumerate(bsdf.items()):
        if type(key) == tuple:
            unique_keys.add(key[0])
        else:
            unique_keys.add(key)

        if key == "phi_i_res":
            print(f"phi_i_res: {value}")
        elif key == "theta_i_res":
            print(f"theta_i_res: {value}")
        elif key == "slice_res":
            print(f"slice_res: {value}")

        if key[0] == "spectra":
            n_spectra += 1
            print(f"Spectra {i}: {key}")

    print(f"Number of spectra: {n_spectra}")
    print(f"Unique keys: {unique_keys}")
