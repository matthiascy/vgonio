# Read a space separated file and write a comma separated file
# Usage: python tidy_csv.py input_file output_file

import sys


def tidy_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line.replace(' ', ','))
    print(f'Wrote {output_file}')


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    tidy_csv(input_file, output_file)
