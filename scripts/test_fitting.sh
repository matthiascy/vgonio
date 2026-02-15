#!/bin/bash
# Helper script for running fitting tests and benchmarks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${GREEN}==== $1 ====${NC}"
}

function print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Parse command line arguments
COMMAND=${1:-help}

case "$COMMAND" in
    test)
        print_header "Running fitting tests"
        RUN_FITTING_TESTS=1 cargo test --package vgonio-bxdf --features fitting
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --package vgonio-app --features fitting --test cmd_fit_tests
        ;;

    test-all)
        print_header "Running all tests (including ignored)"
        RUN_FITTING_TESTS=1 cargo test --package vgonio-bxdf --features fitting -- --include-ignored
        RUN_FITTING_TESTS=1 PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --package vgonio-app --features fitting --test cmd_fit_tests -- --include-ignored
        ;;

    bench)
        print_header "Running benchmarks"
        if ! rustup toolchain list | grep -q "nightly-2024-11-11"; then
            print_error "Required nightly toolchain not found. Install with: rustup toolchain install nightly-2024-11-11"
            exit 1
        fi
        RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting
        ;;

    bench-cuda)
        print_header "Running CUDA benchmarks"
        if ! command -v nvcc &> /dev/null; then
            print_error "CUDA not found. Please install CUDA toolkit."
            exit 1
        fi
        RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting,cuda
        ;;

    profile)
        print_header "Running performance profiling"

        if [ -z "$2" ]; then
            print_error "Usage: $0 profile <input_file> [level]"
            exit 1
        fi

        INPUT_FILE="$2"
        LEVEL="${3:-l0}"
        if [ ! -f "$INPUT_FILE" ]; then
            print_error "Input file not found: $INPUT_FILE"
            exit 1
        fi

        # Build release binary
        print_header "Building release binary with debug info"
        cargo build --features fitting,cuda --profile rel-with-dbg-info

        # Run profiling script
        if command -v python3 &> /dev/null; then
            python3 scripts/profile_fitting.py \
                --vgonio-bin target/rel-with-dbg-info/vgn_app \
                --input "$INPUT_FILE" \
                --level "$LEVEL" \
                --output profiling_results.csv
        else
            print_error "Python 3 not found"
            exit 1
        fi
        ;;

    perf)
        print_header "Running perf profiling"

        if ! command -v perf &> /dev/null; then
            print_error "perf not found. Install with: sudo apt install linux-tools-generic"
            exit 1
        fi

        if [ -z "$2" ]; then
            print_error "Usage: $0 perf <input_file> [output.data]"
            exit 1
        fi

        INPUT_FILE="$2"
        OUTPUT=${3:-perf.data}

        # Build with debug info
        cargo build --release --features fitting --profile rel-with-dbg-info

        # Run perf record
        sudo perf record -g -o "$OUTPUT" \
            target/rel-with-dbg-info/vgn_app fit \
            --kind vgonio \
            --family microfacet \
            --distro trowbridge \
            --symmetry isotropic \
            --method brute \
            --a 0.1:0.5:0.001 \
            --err mse \
            "$INPUT_FILE"

        print_header "Profiling complete. View with: perf report -i $OUTPUT"
        ;;

    generate-test-data)
        print_header "Generating synthetic test data"

        ALPHA=${2:-0.25}
        OUTPUT=${3:-test_data/synthetic_${ALPHA}.json}

        mkdir -p test_data

        python3 scripts/generate_test_data.py \
            --output "$OUTPUT" \
            --alpha "$ALPHA" \
            --incident-samples 10 \
            --outgoing-theta 16 \
            --outgoing-phi 32

        print_header "Test data generated: $OUTPUT"
        ;;

    test-cuda)
        print_header "Running CUDA tests"

        if ! command -v nvcc &> /dev/null; then
            print_error "CUDA not found"
            exit 1
        fi

        RUN_CUDA_TESTS=1 RUN_FITTING_TESTS=1 cargo test --features fitting,cuda -- --include-ignored
        ;;

    clean)
        print_header "Cleaning test artifacts"
        rm -f profiling_results.csv
        rm -f benchmark_results.txt
        rm -f perf.data*
        rm -rf test_data/
        print_header "Clean complete"
        ;;

    build-all)
        print_header "Building all variants"
        cargo build --release --features fitting
        cargo build --release --features fitting,cuda
        print_header "Build complete"
        ;;

    check)
        print_header "Running checks"
        cargo check --features fitting
        cargo clippy --features fitting -- -D warnings
        cargo fmt -- --check
        ;;

    help|*)
        cat << EOF
Fitting Test and Benchmark Helper Script

Usage: $0 <command> [arguments]

Commands:
  test                  Run standard fitting tests
  test-all              Run all tests including ignored ones
  bench                 Run benchmarks (requires nightly toolchain)
  bench-cuda            Run CUDA benchmarks
  profile <input> [level]  Run comprehensive profiling on input file (vgonio level defaults to l0)
  perf <input> [out]    Profile with Linux perf tool
  generate-test-data [alpha] [output]  Generate synthetic test data
  test-cuda             Run CUDA-specific tests
  clean                 Clean test artifacts
  build-all             Build all variants (CPU and GPU)
  check                 Run cargo check, clippy, and fmt
  help                  Show this help message

Examples:
  $0 test                                    # Run quick tests
  $0 bench                                   # Run benchmarks
  $0 profile meas/sample.vgmo                # Profile fitting at level l0
  $0 profile meas/sample.vgmo l1             # Profile fitting at level l1
  $0 generate-test-data 0.25 data.json      # Generate test data
  $0 perf meas/sample.vgmo perf_out.data    # Perf profiling

Environment Variables:
  RUN_FITTING_TESTS=1   Enable fitting tests
  RUN_CUDA_TESTS=1      Enable CUDA tests

Requirements:
  - Rust toolchain (stable + nightly-2024-11-11 for benchmarks)
  - Python 3 with psutil, numpy (for profiling scripts)
  - CUDA toolkit (for GPU features)
  - perf (for detailed CPU profiling)

For more information, see docs/FITTING_TESTS.md
EOF
        ;;
esac
