if __name__ == "__main__":
    # Read a csv file with the fields: surface,kind,weighting,distro,wavelength,alphax,alphay,error,mse
    # and generate a txt file with the alpha x and y ranges for each record in the csv file.
    # The output file will have the following format:
    # start:end:step
    # where start is the minimum value of the range, end is the maximum value of the range and step is the step size.
    # The ranges are generated based on the alphax and alphay values in the csv file.
    # The output contains two files, one for the alpha x and one for the alpha y values.
    # The user should also provide the decimal precision to be used for the ranges.

    import argparse
    import pandas as pd
    import numpy as np
    import os

    parser = argparse.ArgumentParser(description='Generate alpha ranges from a csv file.')
    parser.add_argument('--input', type=str, help='Input csv file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--input-precision', type=int, help='Decimal precision')
    parser.add_argument('--output-precision', type=int, help='Decimal precision')
    args = parser.parse_args()

    # Read the csv file
    df = pd.read_csv(args.input)

    # Create the output directory
    os.makedirs(args.output, exist_ok=True)

    print('Generating alpha ranges...')

    precision = 1.0 / 10 ** args.input_precision
    print("Precision: ", precision)

    # Generate the alpha x and alpha y ranges
    with open(os.path.join(args.output, f'alphax.txt'), 'w') as fx:
        with open(os.path.join(args.output, f'alphay.txt'), 'w') as fy:
            for index, row in df.iterrows():
                alphax = row['alphax']
                alphay = row['alphay']
                alpha = np.array([alphax, alphay])
                start = alpha - 2.0 * precision
                end = alpha + 2.0 * precision
                step = 1.0 / 10 ** args.output_precision
                fx.write(f'{start[0]}:{end[0]}:{step}\n')
                fy.write(f'{start[1]}:{end[1]}:{step}\n')

    print('Done')
