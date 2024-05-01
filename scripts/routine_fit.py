import subprocess
import re
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


def process_files(input_files, distributions):
    results = {}
    for file in input_files:
        for distro in distributions:
            bruteforce_command = f"./target/release/vgonio fit -i {file} -s 0.01 -e 0.5 -p 0.01 --family microfacet --isotropy isotropic --distro {distro} --method bruteforce"
            bruteforce_output = run_command(bruteforce_command)
            min_error, min_alpha = parse_bruteforce_output(bruteforce_output)
            s = max(min_alpha - 0.05, 0.001)
            e = min(min_alpha + 0.05, 0.5)
            p = 0.001
            print(f"File: {file}, Distribution: {distro}, s: {s}, e: {e}, p: {p}")
            if min_alpha is not None:
                nlls_command = f"./target/release/vgonio fit -i {file} -s {s} -e {e} -p {p} --family microfacet --isotropy isotropic --distro {distro} --method nlls"
                nlls_output = run_command(nlls_command)
                best_alpha, best_error = parse_nlls_output(nlls_output)
                results[(file, distro)] = {
                    'bf_min_alpha': min_alpha,
                    'nl_min_alpha': best_alpha,
                    'nl_min_error': best_error,
                }
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file1> ...")
        sys.exit(1)

    # List of input files from command line arguments
    input_files = sys.argv[1:]

    # List of distributions
    distributions = ['bk', 'tr']

    # Process files
    results = process_files(input_files, distributions)

    # Print results
    for key, data in results.items():
        file, distro = key
        print(f"Results for file: {file}, distribution: {distro}")
        print("-- Bruteforce alpha:", data['bf_min_alpha'])
        print("-- Non-linear least squares:", data['nl_min_alpha'], data['nl_min_error'])
