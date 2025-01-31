#!/usr/bin/env bash

# Base directory for BRDF files
BASE_DIR="$HOME/Documents/virtual-gonio/measured/brdfs"

# Set the environment variable for the vgonio-comp executable
export RUSTFLAGS="-Awarnings"

VGONIO_COMP_CMD="cargo run -q -r --bin vgonio-comp --features embree,fitting -- -c $HOME/Documents/repos/vgonio/vgonio.toml plot"

# Base directory for output plots
OUTPUT_BASE_DIR="plots"

# Create the base output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Function to get the file path based on kind and surface
get_file_path() {
    local kind="$1"
    local surface="$2"
    
    case "$kind" in
        # "vgonio")
        #     # Handle special case for isotropic_vgonio
        #     if [ "$surface" = "isotropic_vgonio" ]; then
        #         echo "$BASE_DIR/vgonio/bsdf_isotropic_2024-12-04T21-36-04.vgmo"
        #         return
        #     fi
            
        #     # Extract base name (al0, al1, etc) and folder type
        #     local base=$(echo "$surface" | grep -o "al[0-9]\+")
        #     local folder_type=$(echo "$surface" | sed "s/${base}_//")
            
        #     if [ "$folder_type" = "original" ]; then
        #         # Files in original folder
        #         echo "$BASE_DIR/vgonio/original/bsdf_${base}bar100_*.vgmo"
        #     elif [[ "$folder_type" =~ wiggly_l[0-9]+_[0-9]+\.[0-9]+ ]]; then
        #         # Files in wiggly folders
        #         echo "$BASE_DIR/vgonio/${folder_type}/bsdf_${base}bar100_*.vgmo"
        #     else
        #         # Files in root folder (shouldn't happen with current data)
        #         echo "$BASE_DIR/vgonio/bsdf_${base}_*.vgmo"
        #     fi
        #     ;;
        "clausen")
            echo "$BASE_DIR/clausen/${surface}bar.json"
            ;;
        # "yan2018")
        #     # Extract base name and resolution (e.g., from al65_128 get al65 and 128)
        #     local base=$(echo "$surface" | sed 's/_[0-9]\+$//')
        #     local res=$(echo "$surface" | grep -o '[0-9]\+$')
            
        #     # Add bar100 suffix for al1-al65
        #     if [ "$base" != "al0" ]; then
        #         base="${base}bar100"
        #     fi
            
        #     echo "$BASE_DIR/yan2018/${base}_brdf_rohs_${res}_t5_p30.exr"
        #     ;;
        # "merl")
        #     echo "$BASE_DIR/merl/${surface}.binary"
        #     ;;
        *)
            echo ""
            ;;
    esac
}

# Read the CSV file line by line
while IFS=, read -r kind surface distribution weighting alpha_x alpha_y mse || [ -n "$kind" ]; do
    # Skip header lines and empty lines
    [[ "$kind" =~ ^#.*$ || "$kind" == "kind" || -z "$kind" ]] && continue
    
    # Get the input file path
    input_file=$(get_file_path "$kind" "$surface")
    
    # For vgonio files with wildcards, get the first matching file
    if [[ "$input_file" == *"*"* ]]; then
        input_file=$(ls $input_file 2>/dev/null | head -n 1)
    fi
    
    # Skip if file doesn't exist
    if [ ! -f "$input_file" ]; then
        echo "Warning: File not found: $input_file of kind $kind and surface $surface"
        continue
    fi
    
    # Create output directory for this surface
    output_dir="$OUTPUT_BASE_DIR/${kind}_${surface}/${distribution}_${weighting}"
    mkdir -p "$output_dir"
    
    # Change to the output directory
    cd "$output_dir" || continue
    
    echo "Generating plots for $kind/$surface ($distribution, $weighting)"
    echo "  Input file: $input_file"
    echo "  Output dir: $output_dir"
    echo "  Parameters: alpha_x=$alpha_x alpha_y=$alpha_y"

    # Print cwd
    echo "Current working directory: $(pwd)"
    
    # Run the plot command
    $VGONIO_COMP_CMD "$input_file" \
        --kind brdf-fitting \
        --err mse \
        --weighting "$weighting" \
        --alpha "$alpha_x" "$alpha_y" \
        --model "$distribution" \
        --parallel
    
    # Return to original directory
    cd - > /dev/null || exit
    
done < fitting-baseline.csv

echo "Plot generation completed. Results are in the '$OUTPUT_BASE_DIR' directory."
