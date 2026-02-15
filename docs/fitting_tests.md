# BRDF Fitting Tests and Benchmarks

This is the canonical reference for fitting tests, benchmarks, and profiling.

## Quick Start

```bash
# 1) Run core tests
./scripts/test_fitting.sh test

# 2) Run all tests (including ignored)
./scripts/test_fitting.sh test-all

# 3) Run benchmarks
./scripts/test_fitting.sh bench

# 4) Profile on a measurement file
./scripts/test_fitting.sh profile meas/your_file.vgmo
```

## Project Layout

```text
crates/vgonio-bxdf/tests/
  fitting_tests.rs                  # Synthetic fitting correctness tests

crates/vgonio-app/tests/
  cmd_fit_tests.rs                  # CLI argument/parser tests

crates/vgonio-bxdf/benches/
  fitting_bench.rs                  # Fitting microbenchmarks

scripts/
  test_fitting.sh                   # Test/bench/profile helper
  profile_fitting.py                # Automated profiling driver
  generate_test_data.py             # Synthetic data generator
```

## Running Tests

### Unit + fitting correctness tests (`vgonio-bxdf`)

```bash
# run all fitting tests
cargo test -p vgonio-bxdf --features fitting

# run one test
cargo test -p vgonio-bxdf --features fitting test_brute_force_fitting_recovers_known_isotropic_parameters

# include ignored tests
RUN_FITTING_TESTS=1 cargo test -p vgonio-bxdf --features fitting -- --include-ignored
```

### CLI integration tests (`vgonio-app`)

```bash
# run parser/integration tests
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p vgonio-app --features fitting --test cmd_fit_tests

# include ignored tests (if any)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p vgonio-app --features fitting --test cmd_fit_tests -- --include-ignored
```

### CUDA test path

```bash
# CPU baseline
cargo test -p vgonio-bxdf --features fitting

# CUDA-specific tests
RUN_CUDA_TESTS=1 RUN_FITTING_TESTS=1 cargo test -p vgonio-bxdf --features fitting,cuda -- --include-ignored
```

## Running Benchmarks

Benchmarks are under `crates/vgonio-bxdf/benches/fitting_bench.rs`.

```bash
# run all fitting benches
RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting

# isotropic brute fit benchmark
RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting -- bench_brute_force_isotropic

# anisotropic windowed benchmark
RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting -- bench_brute_force_anisotropic_windowed

# save benchmark output
RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting 2>&1 | tee benchmark_results.txt
```

CUDA benches:

```bash
RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --bench fitting_bench --features fitting,cuda
```

## Performance Profiling

### Through helper script

```bash
./scripts/test_fitting.sh profile meas/some_measurement.vgmo
```

This builds `target/rel-with-dbg-info/vgn_app` and runs automated profiling.

### Direct Python profiler

```bash
python3 scripts/profile_fitting.py \
  --vgonio-bin target/rel-with-dbg-info/vgn_app \
  --input meas/some_measurement.vgmo \
  --output profiling_results.csv \
  --tests all
```

`--tests` options: `precision`, `symmetry`, `wavelength`, `gpu`, `all`.

### Manual CPU profiling with `perf`

```bash
cargo build --release --features fitting --profile rel-with-dbg-info

perf record -g target/rel-with-dbg-info/vgn_app fit \
  --kind vgonio \
  --family microfacet \
  --distro trowbridge \
  --symmetry isotropic \
  --method brute \
  --a 0.1:0.5:0.001 \
  --err mse \
  input_file.vgmo

perf report
```

### Manual GPU profiling

```bash
cargo build --release --features fitting,cuda

nvprof --print-gpu-trace \
  target/release/vgn_app fit \
  --kind vgonio \
  --family microfacet \
  --distro trowbridge \
  --symmetry isotropic \
  --method brute \
  --a 0.1:0.5:0.001 \
  --cuda \
  --per-wl \
  input_file.vgmo
```

## Synthetic Data Generation

```bash
# isotropic
python3 scripts/generate_test_data.py \
  --output test_data/isotropic_025.json \
  --alpha 0.25 \
  --incident-samples 10 \
  --outgoing-theta 16 \
  --outgoing-phi 32

# anisotropic
python3 scripts/generate_test_data.py \
  --output test_data/anisotropic.json \
  --alpha-x 0.1 \
  --alpha-y 0.5 \
  --incident-samples 10 \
  --outgoing-theta 16 \
  --outgoing-phi 32
```

## Expected Behavior

- Synthetic isotropic recovery should typically land near the injected roughness.
- Per-wavelength fitting should be stable when synthetic roughness is wavelength-invariant.
- CUDA runs should match CPU results within tolerance and usually complete faster.

## CI Snippet

```yaml
- name: Fitting tests
  run: |
    cargo test -p vgonio-bxdf --features fitting
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p vgonio-app --features fitting --test cmd_fit_tests

- name: Fitting benchmarks
  run: |
    RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench -p vgonio-bxdf --features fitting --bench fitting_bench --no-fail-fast
```

## Troubleshooting

### Benchmark command fails on nightly or toolchain

- Install pinned toolchain: `rustup toolchain install nightly-2024-11-11`
- Use the same invocation as script: `RUSTC_WRAPPER= cargo +nightly-2024-11-11 bench ...`

### App tests fail due to PyO3 Python-version check

- Use: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 ...`

### CUDA tests/benches are skipped or fail

- Confirm CUDA tooling: `nvcc --version` and `nvidia-smi`
- Build with CUDA feature: `--features fitting,cuda`

## References

- Rust Testing: <https://doc.rust-lang.org/book/ch11-00-testing.html>
- Unstable bench API: <https://doc.rust-lang.org/unstable-book/library-features/test.html>
- CUDA profiling: <https://docs.nvidia.com/cuda/profiler-users-guide/>
