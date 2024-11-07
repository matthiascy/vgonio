#!/bin/sh

# This script measures the Wave Optics BRDF model.
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

input_dir=$1
output_dir=$2

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop over each file in the input directory with '_grounded' suffix
for input_file in "$input_dir"/*_grounded.exr; do
  # Check if the file exists
  if [ ! -e "$input_file" ]; then
    echo "No files matching *_grounded.exr found in $input_dir"
    exit 1
  fi

  # Extract the base name of the file
  base_name=$(basename "$input_file" _grounded.exr)

  output_file="$output_dir"/"$base_name"_brdf_rohs.exr

  # Measure the Wave Optics BRDF model
  ./gen_brdf -m Wave -d ROHS -i "$input_file" -o "$output_file" -w 0.5 -p 10.0 -r 512 -q -z -s 5 -t 30

  echo "Measured Wave Optics BRDF model for $input_file"

  if [ $? -eq 0 ]; then
    echo "Processed $input_file -> $output_file"
  else
    echo "Failed to process $input_file"
  fi
done