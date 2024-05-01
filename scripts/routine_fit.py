import os.path
import subprocess
import re
from datetime import datetime

import sys


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode('utf-8')


def parse_nlls_output(output):
    best_alpha = None
    best_error = None

    result = re.search(
        r'Best model: Some\(MicrofacetBrdf { distro: (\w+) { alpha_x: ([\d.]+), alpha_y: ([\d.]+) } }\), Err: ([\d.]+)',
        output,
        re.MULTILINE)

    if result:
        alpha_x = float(result.group(2))
        alpha_y = float(result.group(3))
        error = float(result.group(4))
        best_alpha = (alpha_x, alpha_y)
        best_error = error

    return best_alpha, best_error


def parse_bruteforce_output(output):
    min_error = None
    min_alpha = None

    # Extract the lines containing MSE values and minimum error
    min_error_line = re.search(r'Minimum error: (.*?) at alpha = (.*?)$', output, re.MULTILINE)

    if min_error_line:
        # Extract the minimum error and corresponding alpha
        min_error = float(min_error_line.group(1))
        min_alpha = float(min_error_line.group(2))

    return min_error, min_alpha


def process_brdf_fitting(inputs, distributions, f):
    for input in inputs:
        if os.path.exists(input):
            if os.path.isdir(input):
                for file in os.listdir(input):
                    if file.endswith(".vgmo"):
                        for distro in distributions:
                            brdf_fitting(os.path.join(input, file), distro, f)
            elif input.endswith(".vgmo"):
                for distro in distributions:
                    brdf_fitting(input, distro, f)


def brdf_fitting(input, distro, f):
    bruteforce_command = f"./target/release/vgonio fit -i {input} -s 0.01 -e 0.6 -p 0.01 --family microfacet --isotropy isotropic --distro {distro} --method bruteforce"
    bruteforce_output = run_command(bruteforce_command)
    min_error, min_alpha = parse_bruteforce_output(bruteforce_output)
    s = max(min_alpha - 0.025, 0.001)
    e = min(min_alpha + 0.025, 0.6)
    p = 0.001
    if min_alpha is not None:
        nlls_command = f"./target/release/vgonio fit -i {input} -s {s} -e {e} -p {p} --family microfacet --isotropy isotropic --distro {distro} --method nlls"
        nlls_output = run_command(nlls_command)
        best_alpha, best_error = parse_nlls_output(nlls_output)
        # Print results
        print(f"Results for file: {input}, distribution: {distro}, s: {s}, e: {e}, p: {p}", file=f)
        print(f"-- Bruteforce alpha: {min_alpha}", file=f)
        print(f"-- Non-linear least squares: {best_alpha}, err: {best_error}", file=f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file/directory> ...")
        sys.exit(1)

    # List of input files from command line arguments
    inputs = sys.argv[1:]

    # List of distributions
    distributions = ['bk', 'tr']

    # open a file to write the results with the date and time as the filename
    with open(f"rountine_fit_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", 'w') as f:
        process_brdf_fitting(inputs, distributions, f)
