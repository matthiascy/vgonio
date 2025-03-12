import gc
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rich

matplotlib.rcParams['figure.max_open_warning'] = 0


# python scripts/per_wavelength.py ./per_wavelength --fitted ./fitting-baseline.csv --output ./per_wavelength/plot/

def plot_per_wavelength_err_single(title, wavelengths, alphas, errors, fitted, out_dir=None, separate=False):
    x = wavelengths
    y = alphas

    fitted_alpha = np.full_like(x, fitted)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    ax[0].plot(x, y, ".-", label="Per wavelength roughness α")
    ax[0].plot(x, fitted_alpha, "--", label="Fitted (as whole) α = " + str(fitted))

    ax[0].set_xlabel("Wavelengths (nm)")
    ax[0].set_ylabel(f"Roughness α")
    ax[0].legend()
    # ax[0].grid(True)

    ax[1].plot(x, errors, ".-", label="Per wavelength error (MSE)")
    ax[1].set_xlabel("Wavelengths (nm)")
    ax[1].set_ylabel("Error (MSE)")
    ax[1].legend()

    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{title.replace(' ', '_')}.png"))
    else:
        fig.savefig(f"{title.replace(' ', '_')}.png")

    # plt.show()


def plot_per_wavelength_err_anisotropic(title, wavelengths, alphas_x, alphas_y, errors, mses, out_dir=None):
    x = wavelengths
    y0 = alphas_x
    y1 = alphas_y
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title)

    ax.plot(x, y0, ".-", label=r"$\alpha_x$")
    ax.plot(x, y1, ".-", label=r"$\alpha_y$")
    ax.set_xlabel("Wavelengths (nm)")
    ax.set_ylabel(f"Roughness α")
    ax.legend()

    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{title.replace(' ', '_')}.png"))
    else:
        fig.savefig(f"{title.replace(' ', '_')}.png")


def list_all_files(path: str, ext: str):
    """Recursively list all files in a directory with a specific extension together with the subfolder name"""
    import os
    files = {}
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if f.endswith(ext):
                if root not in files:
                    files[root] = []
                files[root].append(os.path.abspath(os.path.join(root, f)))
    return files


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Per wavelength analysis")
    parser.add_argument("input", help="Input file or directory containing the CSV files")
    parser.add_argument("--fitted", help="Fitted value stored inside a CSV file")
    parser.add_argument("--output", help="Output directory to store the plots")

    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--separate", action="store_true", help="Create separate plots for each file")
    exclusive_group.add_argument("--compact", action="store_true", help="Create compact plots")

    parser.add_argument("--aniso", action="store_true", help="Values are anisotropic.")

    args = parser.parse_args()

    # temporary plot for anisotropic
    if args.aniso:
        with open(args.input, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            kind = ""
            surface = ""
            weighting = ""
            distro = ""
            wavelengths = []
            alphas_x = []
            alphas_y = []
            errors = []
            mses = []
            for i, row in enumerate(reader):
                if i == 0:
                    surface = row[0]
                    kind = row[1]
                    weighting = row[2]
                    distro = row[3]
                if row[4] == "none":
                    continue
                wavelengths.append(float(row[4].split(' ')[0]))
                alphas_x.append(float(row[5]))
                alphas_y.append(float(row[6]))
                errors.append(float(row[7]))
                if len(row) > 8:
                    mses.append(float(row[8]))
            plot_per_wavelength_err_anisotropic(f"{surface} {kind} {weighting} {distro}", wavelengths, alphas_x,
                                                alphas_y, errors, mses)
        exit(0)

    with open(args.fitted) as f:
        reader = csv.DictReader(f)
        # read all alpha values from the fitted file ordered by kind, surface, distribution, weighting
        fitted = {}
        for row in reader:
            kind = row["kind"]
            surface = row["surface"]
            weighting = row["weighting"]
            distribution = row["distribution"]
            alpha = row["alpha_x"]
            fitted[kind] = fitted.get(kind, {})
            fitted[kind][surface] = fitted[kind].get(surface, {})
            fitted[kind][surface][weighting] = fitted[kind][surface].get(weighting, {})
            fitted[kind][surface][weighting][distribution] = alpha

        files_to_process = {}
        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            print(f"File or directory {input_path} does not exist")
            exit(1)

        if os.path.isdir(input_path):
            files_to_process = list_all_files(input_path, ".csv")
        else:
            files_to_process[os.path.dirname(input_path)] = [input_path]

        rich.print(files_to_process)

        if not files_to_process:
            print(f"No files found in {input_path}")
            exit(1)

        # create the output directory if it does not exist
        if args.output:
            if not os.path.exists(args.output):
                os.makedirs(args.output)

        for (d, files) in files_to_process.items():
            figures = {}
            for file in files:
                print(f"Processing {file}")
                with open(file) as f2:

                    reader = csv.reader(f2)
                    header = next(reader)

                    kind = ""
                    surface = ""
                    weighting = ""
                    distro = ""
                    wavelengths = []
                    alphas = []
                    errors = []

                    for i, row in enumerate(reader):
                        if i == 0:
                            surface = row[0]
                            kind = row[1]
                            weighting = row[2]
                            distro = row[3]
                        wavelengths.append(float(row[4].split(' ')[0]))
                        alphas.append(float(row[5]))
                        errors.append(float(row[7]))

                    # find the fitted value for this kind, surface, weighting, distro
                    surface_key = list(filter(lambda x: x if x == surface else None, fitted[kind].keys()))[0]

                    if surface_key is None:
                        print(f"Could not find a surface key for {surface}")
                        continue
                    else:
                        fitted_alpha = float(fitted[kind][surface_key][weighting][distro])
                        if args.separate:
                            title = f"{surface} {kind} {weighting} {distro}"
                            plot_per_wavelength_err_single(title, wavelengths, alphas, errors, fitted_alpha,
                                                           args.output,
                                                           args.separate)
                        elif args.compact:
                            title = f"{kind}"
                            # weighting-none  |    0,0      |     0,1     |
                            # weighting-lncos |    0,1      |     1,1     |
                            #                   distro-bk  distro-tr
                            rich.print(f"Adding figure for {title}")
                            fig, axes = figures.get(title, plt.subplots(2, 2, figsize=(12, 12)))
                            figures[title] = (fig, axes)
                            fig.suptitle(title)
                            c = 0 if distro == "bk" else 1
                            r = 0 if weighting == "none" else 1
                            ax = axes[r, c]
                            ax.set_xlabel("Wavelengths (nm)")
                            ax.set_ylabel(f"Roughness α")
                            ax.plot(wavelengths, alphas, ".-", label=f"{surface_key} {weighting} {distro} α")
                            ax.legend()
                            del fig, axes
                        else:
                            title = f"{surface} {kind}"
                            # weighting-none  |    0,0      |     0,1     |
                            # weighting-lncos |    0,1      |     1,1     |
                            #                   distro-bk  distro-tr
                            rich.print(f"Adding figure for {title}")
                            fig, axes = figures.get(title, plt.subplots(2, 2, figsize=(12, 12)))
                            figures[title] = (fig, axes)
                            fig.suptitle(title)
                            c = 0 if distro == "bk" else 1
                            r = 0 if weighting == "none" else 1
                            ax = axes[r, c]
                            ax.set_xlabel("Wavelengths (nm)")
                            ax.set_ylabel(f"Roughness α")
                            ax.plot(wavelengths, alphas, ".-", label=f"{weighting} {distro} α")
                            ax.legend()
                            del fig, axes

            rich.print(figures)

            if not args.separate and not args.compact:
                for fig, _ in figures.values():
                    if fig is None:
                        continue
                    filename = os.path.join(args.output, f"{fig._suptitle._text.replace(' ', '_')}.png")
                    print(f"Saving {filename}")
                    fig.savefig(filename)
                    plt.close(fig)

                del figures
                gc.collect()
                figures = {}

            if args.compact:
                for title, (fig, axes) in figures.items():
                    filename = os.path.join(args.output, f"{title}.png")
                    print(f"Saving {filename}")
                    fig.savefig(filename)
                    plt.close(fig)

                del figures
                gc.collect()
                figures = {}
