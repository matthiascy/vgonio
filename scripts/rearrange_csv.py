if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Rearrange CSV files')
    parser.add_argument('input', type=str, help='Input CSV file')
    parser.add_argument('output', type=str, help='Output CSV file')
    args = parser.parse_args()

    # Read the input CSV file
    df = pd.read_csv(args.input)
    # Check if the required columns exit
    for col in ['surface', 'distro', 'weighting', 'wavelength', 'alphax', 'alphay', 'mse']:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column.")

    # Rearrange the columns
    df_rearranged = df[['surface', 'distro', 'weighting', 'wavelength', 'alphax', 'alphay', 'mse']]

    # Save the rearranged DataFrame to a new CSV file
    df_rearranged.to_csv(args.output, index=False)
