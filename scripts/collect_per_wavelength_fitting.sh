#!/bin/bash

# Base directory containing input files
BASE_DIR="$HOME/Documents/virtual-gonio/measured/brdfs/clausen/"
if [ -z "$1" ]; then
  KIND=""
else
  KIND="$1"
fi



# -o -name "*.binary"
# -o -name "*.bsdf"
# Iterate over supported file types in all subdirectories
find "$BASE_DIR" -type f \( -name "*.json" -o -name "*.vgmo" \) | while read -r FILE; do
    # Extract filename without extension
    BASENAME=$(basename "$FILE")
    DIRNAME=$(basename $(dirname "$FILE"))

    echo "Processing file: $FILE with $BASENAME and $DIRNAME"

    # Use the subdirectory name as the kind if the first argument is not provided
    if [ -z "$KIND" ]; then
        KIND="$DIRNAME"
    fi

    # If the kind is vgonio, add extra options
    if [ "$KIND" = "vgonio" ]; then
      LVL="--level l0"
    else
      LVL=""
    fi

    # Run the command with different distro and weighting values
    for DISTRO in tr bk; do
        for WEIGHTING in lncos none; do
            cargo run -F embree,fitting --bin vgonio-comp -- fit "$FILE" \
                --err mse --weighting "$WEIGHTING" --method brute --distro "$DISTRO" --symmetry iso \
                --family microfacet --kind "$KIND" --per-wavelength $LVL
        done
    done
done
