#!/bin/bash

# Define input folders and distribution options
BASE_DIR="$HOME/Documents/virtual-gonio/measured/brdfs"
SUB_DIRS=("clausen") # "rgl" "vgonio" "clausen" "merl" "yan2018")
DISTROS=("bk" "tr")
WEIGHTING=("none" "lncos")

# Check if fd exists
if ! command -v fd &> /dev/null; then
    exit 1
fi

TOTAL_FILES=0
PROCESSED_FILES=0
BRUTE_FORCE=1

if [ "$1" = "nllsq" ]; then
    BRUTE_FORCE=0
fi

echo "Running fitting with brute force: $BRUTE_FORCE"
# Output CSV file
if [ $BRUTE_FORCE -eq 1 ]; then
    OUTPUT_CSV="fitting_results_brute.csv"
else
    OUTPUT_CSV="fitting_results_nllsq.csv"
fi

HEADER="# run date: $(date)"
if [ $BRUTE_FORCE -eq 1 ]; then
    HEADER="$HEADER\nkind,surface,distribution,weighting,alpha_x,alpha_y,mse"
else
    HEADER="$HEADER\nkind,surface,distribution,weighting,alpha_x,alpha_y,nllsq_err,mse"
fi

# Only write header if file does not exist otherwise append to it
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Creating new file: $OUTPUT_CSV"
    echo -e "$HEADER" > "$OUTPUT_CSV"
else
    echo "Appending to existing file: $OUTPUT_CSV"
    # Add a newline to separate the new results and date of the run
    echo "" >> "$OUTPUT_CSV"
    echo -e "$HEADER" >> "$OUTPUT_CSV"
fi

# Count total files for progress
for SUB_DIR in "${SUB_DIRS[@]}"; do
  DIR="$BASE_DIR/$SUB_DIR"
  case "$SUB_DIR" in
    "vgonio") EXT="vgmo" ;;
    "rgl") EXT="bsdf" ;;
    "clausen") EXT="json" ;;
    "yan2018") EXT="exr" ;;
    "merl") EXT="binary" ;;
    *) EXT="*" ;;
  esac
  if [ -d "$DIR" ]; then
      COUNT=$(cd "$DIR" && fd . "$DIR" -t f -e "$EXT" | wc -l)
      TOTAL_FILES=$((TOTAL_FILES + $COUNT))
      echo "Found $COUNT files in $DIR"
  else
      echo "Directory $DIR does not exist"
  fi
done

echo "Total files to process: $TOTAL_FILES"

# Iterate over input folders
for SUB_DIR in "${SUB_DIRS[@]}"; do
    DIR="${BASE_DIR}/${SUB_DIR}"
    KIND="$SUB_DIR"

    # Detremine file extension based on kind
    case "$KIND" in
        "vgonio")  EXT="vgmo"   ;;
        "rgl")     EXT="bsdf"   ;;
        "clausen") EXT="json"   ;;
        "yan2018") EXT="exr"    ;;
        "merl")    EXT="binary" ;;
        *)         exit "Unknown kind: $KIND"; exit 1 ;;
    esac

    if [ ! -d "$DIR" ]; then
        echo "Directory $DIR does not exist"
        continue
    fi

    # Add target/release temporarily to PATH
    export PATH="$PATH:$HOME/Documents/repos/vgonio/target/release"

    ISOTROPY="iso"

    # Find all matching files recursively
    # Not using pipe here to as it spawns a subshell and we need to update the PROCESSED_FILES variable
    while read -r FILE; do
        [ -z "$FILE" ] && continue

        echo "Processing file: $FILE"

        # If the file name contains "aniso", set the isotropic flag
        if echo "$FILE" | grep -q "aniso"; then
            ISOTROPY="aniso"
        else
            ISOTROPY="iso"
        fi


        BASE_NAME=$(basename "$FILE")
        # Extract relevant file name for vgonio
        if [ "$KIND" == "vgonio" ]; then
            SURF_NAME=$(echo "$BASE_NAME" | grep -o "al[0-9]\+")
            if [ -z "$SURF_NAME" ]; then
                # case: bsdf_isotropic_2024-12-04T21-36-04.vgmo, take the isotropic part
                SURF_NAME=$(echo "$BASE_NAME" | grep -o "bsdf_\([a-z]\+\)" | sed "s/bsdf_//")
            fi
            SUB_PATH=$(dirname "$FILE" | sed "s|$BASE_DIR/$KIND/||")
            if [ "$SUB_PATH" != "$FILE" ]; then
                FILE_NAME="${SURF_NAME}_$(basename "$SUB_PATH")"
            else
                FILE_NAME="${SURF_NAME}"
            fi
        elif [ "$KIND" == "yan2018" ]; then
            # Extract the surface name from the file name
            SURF_NAME=$(echo "$BASE_NAME" | grep -o "al[0-9]\+")
            # Extract the resolution from the file name
            RES_NAME=$(echo "$BASE_NAME" | grep -o "rohs_[0-9]\+" | grep -o "[0-9]\+")
            FILE_NAME="${SURF_NAME}_${RES_NAME}"
        elif [ "$KIND" == "clausen" ]; then
            SURF_NAME=$(echo "$BASE_NAME" | grep -o "al[0-9]\+")
            FILE_NAME="${SURF_NAME}"
        else
            FILE_NAME="${BASE_NAME%.*}"
        fi

        # Use alternative config file for RGL data as the wavelength range is different
        if [ "$KIND" == "rgl" ]; then
            CONFIG="./scripts/vgonio-ior-alt.toml"
        else
            CONFIG="./vgonio.toml"
        fi

        # Iterate over weighting options
        for WEIGHT in "${WEIGHTING[@]}"; do
            # Iterate over distributions
            for DISTRO in "${DISTROS[@]}"; do
                # Run the fitting command
                if [ $BRUTE_FORCE -eq 1 ]; then
                    OUTPUT=$(vgonio-comp -c "$CONFIG" fit -i "$FILE" -f microfacet -k "$KIND" --isotropy "$ISOTROPY" -d "$DISTRO" -m brute --weighting "$WEIGHT" --err mse 2>&1)
                else
                    # Skip files with resolution 512, 256
                    if [ "$RES_NAME" == "512" ]; then
                        echo "  - Skipping Yan2018 with resolution 512: $FILE"
                        PROCESSED_FILES=$((PROCESSED_FILES + 1))
                        continue
                    else
                        OUTPUT=$(vgonio-comp -c "$CONFIG" fit -i "$FILE" -f microfacet -k "$KIND" --isotropy "$ISOTROPY" -d "$DISTRO" -m nllsq --weighting "$WEIGHT" 2>&1)
                    fi
                fi

                # Extract the best model and error from the output
                BEST_MODEL=$(echo "$OUTPUT" | grep "Best model:")
                ALPHAX=$(echo "$BEST_MODEL" | grep -o "α_x: [0-9.]\+" | grep -o "[0-9.]\+")
                ALPHAY=$(echo "$BEST_MODEL" | grep -o "α_y: [0-9.]\+" | grep -o "[0-9.]\+")
                ERROR=$(echo "$OUTPUT" | grep "Best model:" | sed -n 's/.*err: \([^,]*\),.*/\1/p')

                # Extract the mse error if nllsq is used
                if [ $BRUTE_FORCE -eq 0 ]; then
                    MSE=$(echo "$OUTPUT" | grep "Best model:" | sed -n 's/.*mse: \([^,]*\).*/\1/p')
                fi

                # Write results to CSV
                if [ $BRUTE_FORCE -eq 1 ]; then
                    echo "$KIND,$FILE_NAME,$DISTRO,$WEIGHT,$ALPHAX,$ALPHAY,$ERROR" >> "$OUTPUT_CSV"
                else
                    echo "$KIND,$FILE_NAME,$DISTRO,$WEIGHT,$ALPHAX,$ALPHAY,$ERROR,$MSE" >> "$OUTPUT_CSV"
                fi
            done
        done

        # Update progress
        PROCESSED_FILES=$((PROCESSED_FILES + 1))
        echo "Progress: $PROCESSED_FILES / $TOTAL_FILES files processed."
    done < <(fd . "$DIR" -t file -e "$EXT")

    cd - > /dev/null || exit
done

# Print completion message
echo "Fitting completed. Results saved to $OUTPUT_CSV."
