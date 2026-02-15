#!/usr/bin/env python3
"""
Performance profiling script for BRDF fitting

This script runs the fitting process with different configurations and
collects performance metrics including timing, memory usage, and convergence.

Usage:
    python profile_fitting.py --input <data_file> --output <results.csv>

Requirements:
    - vgonio binary built with release profile
    - psutil for memory monitoring
"""

import argparse
import csv
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import re

try:
    import psutil
except ImportError:
    print("psutil not found. Install with: pip install psutil")
    sys.exit(1)

FLOAT_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class FittingProfiler:
    """Profile BRDF fitting performance"""

    def __init__(self, vgonio_bin: Path, input_file: Path, level: str = "l0"):
        self.vgonio_bin = vgonio_bin
        self.input_file = input_file
        self.level = level
        self.results = []
        self._cuda_flag_supported: bool | None = None

    @staticmethod
    def _extract_first_float(text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first_non_empty_line(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _clean_log_text(text: str) -> str:
        cleaned = ANSI_ESCAPE.sub("", text)
        return cleaned.replace("\r\n", "\n").replace("\r", "\n")

    def _supports_cuda_flag(self) -> bool:
        if self._cuda_flag_supported is not None:
            return self._cuda_flag_supported

        try:
            proc = subprocess.run(
                [str(self.vgonio_bin), "fit", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            help_text = f"{proc.stdout}\n{proc.stderr}"
            self._cuda_flag_supported = "--cuda" in help_text
        except Exception:
            self._cuda_flag_supported = False

        return self._cuda_flag_supported

    @classmethod
    def _build_error_context(
        cls,
        stderr: str,
        stdout: str,
        max_lines_per_stream: int = 40,
        max_chars: int = 8000,
    ) -> str | None:
        streams = []
        stderr_clean = cls._clean_log_text(stderr).strip()
        stdout_clean = cls._clean_log_text(stdout).strip()
        if stderr_clean:
            streams.append(("stderr", stderr_clean))
        if stdout_clean:
            streams.append(("stdout", stdout_clean))

        if not streams:
            return None

        chunks = []
        for name, text in streams:
            lines = text.splitlines()
            if len(lines) > max_lines_per_stream:
                omitted = len(lines) - max_lines_per_stream
                lines = lines[:max_lines_per_stream]
                lines.append(f"... ({omitted} more lines)")
            chunks.append(f"[{name}]\n" + "\n".join(lines))

        context = "\n\n".join(chunks)
        if len(context) > max_chars:
            context = context[: max_chars - 16] + "\n... [truncated]"

        # Keep CSV single-line friendly.
        return context.replace("\n", "\\n")

    @classmethod
    def _parse_fitting_output(cls, stdout: str, stderr: str) -> Dict[str, Any]:
        text = "\n".join(part for part in (stdout, stderr) if part)
        text = ANSI_ESCAPE.sub("", text)

        alpha_symbol = "\u03b1"
        alpha_x = cls._extract_first_float(
            text,
            rf"(?:{alpha_symbol}x|alpha[_\s]?x)\s*:\s*{FLOAT_PATTERN}",
        )
        alpha_y = cls._extract_first_float(
            text,
            rf"(?:{alpha_symbol}y|alpha[_\s]?y)\s*:\s*{FLOAT_PATTERN}",
        )
        alpha_iso = cls._extract_first_float(
            text,
            rf"(?:alpha|a)\s*:\s*{FLOAT_PATTERN}",
        )

        if alpha_iso is not None:
            if alpha_x is None:
                alpha_x = alpha_iso
            if alpha_y is None:
                alpha_y = alpha_iso

        # Keep the legacy column "best_alpha" for isotropic-friendly output.
        best_alpha = alpha_x

        obj_fn = cls._extract_first_float(text, rf"obj_fn\s*:\s*{FLOAT_PATTERN}")
        legacy_error = cls._extract_first_float(
            text,
            rf"(?:\berr(?:or)?\b)\s*:\s*{FLOAT_PATTERN}",
        )
        mse = cls._extract_first_float(text, rf"\bmse\s*:\s*{FLOAT_PATTERN}")

        # Prefer objective value from current output; fall back to older formats.
        best_error = obj_fn
        if best_error is None:
            best_error = legacy_error if legacy_error is not None else mse

        return {
            "best_alpha": best_alpha,
            "best_alpha_x": alpha_x,
            "best_alpha_y": alpha_y,
            "best_error": best_error,
            "best_mse": mse,
        }

    @staticmethod
    def _result_summary(result: Dict[str, Any]) -> str:
        if result["success"]:
            return f"{result['time_seconds']:.2f}s"

        error_message = result.get("error_message")
        if error_message:
            return f"{result['time_seconds']:.2f}s [FAILED: {error_message}]"

        return (
            f"{result['time_seconds']:.2f}s "
            f"[FAILED: return_code={result.get('return_code', 'N/A')}]"
        )

    def run_fitting(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run fitting with given configuration and collect metrics"""

        # Build command
        cmd = [
            str(self.vgonio_bin),
            "fit",
            "--kind",
            config["kind"],
        ]

        if config["kind"] == "vgonio":
            cmd.extend(["--level", str(config.get("level", self.level))])

        cmd.extend(
            [
                "--family",
                config["family"],
                "--distro",
                config["distro"],
                "--symmetry",
                config["symmetry"],
                "--method",
                config["method"],
                "--err",
                config["error_metric"],
                "--weighting",
                config["weighting"],
            ]
        )

        # Add roughness parameters
        if config["symmetry"] == "isotropic":
            cmd.extend(["--a", config["alpha_range"]])
        else:
            cmd.extend(["--ax", config["alpha_x_range"]])
            cmd.extend(["--ay", config["alpha_y_range"]])

        # Add brute force precision if applicable
        if config["method"] == "brute":
            cmd.extend(["--bprec", str(config["precision"])])

        # Add optional flags
        if config.get("per_wavelength"):
            cmd.append("--per-wl")

        if config.get("cuda"):
            if not self._supports_cuda_flag():
                return {
                    "config": config,
                    "command": " ".join(cmd + ["--cuda", "--", str(self.input_file)]),
                    "success": False,
                    "return_code": None,
                    "time_seconds": 0.0,
                    "memory_delta_gb": 0.0,
                    "memory_peak_gb": 0.0,
                    "best_alpha": None,
                    "best_alpha_x": None,
                    "best_alpha_y": None,
                    "best_error": None,
                    "best_mse": None,
                    "error_message": "GPU profiling requested, but this binary does not support `--cuda`.",
                    "error_context": (
                        "[profiler]\\nThe `fit` command has no `--cuda` option in this build. "
                        "Rebuild with `--features fitting,cuda`."
                    ),
                    "stdout": "",
                    "stderr": "",
                }
            cmd.append("--cuda")

        if config.get("theta_limit"):
            cmd.extend(["--theta-limit", str(config["theta_limit"])])

        # Keep options before positional inputs; pass explicit terminator to avoid
        # any ambiguity when positional values can start with '-'.
        cmd.extend(["--", str(self.input_file)])

        print(f"Running: {' '.join(cmd)}")

        # Start process and monitor
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)  # GB

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Monitor peak memory
            peak_memory = start_memory
            while process.poll() is None:
                current_memory = psutil.virtual_memory().used / (1024**3)
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)

            stdout, stderr = process.communicate()
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)

        except Exception as e:
            return {
                "config": config,
                "success": False,
                "error": str(e),
            }

        # Parse output for fitting results
        fitting_time = end_time - start_time
        memory_delta = end_memory - start_memory
        memory_peak = peak_memory - start_memory

        parsed = self._parse_fitting_output(stdout, stderr)
        error_message = None
        error_context = None
        if process.returncode != 0:
            stderr_clean = self._clean_log_text(stderr)
            stdout_clean = self._clean_log_text(stdout)
            error_message = (
                self._first_non_empty_line(stderr_clean)
                or self._first_non_empty_line(stdout_clean)
                or f"Command exited with code {process.returncode}"
            )
            error_context = self._build_error_context(stderr_clean, stdout_clean)

        result = {
            "config": config,
            "command": " ".join(cmd),
            "success": process.returncode == 0,
            "return_code": process.returncode,
            "time_seconds": fitting_time,
            "memory_delta_gb": memory_delta,
            "memory_peak_gb": memory_peak,
            "best_alpha": parsed["best_alpha"],
            "best_alpha_x": parsed["best_alpha_x"],
            "best_alpha_y": parsed["best_alpha_y"],
            "best_error": parsed["best_error"],
            "best_mse": parsed["best_mse"],
            "error_message": error_message,
            "error_context": error_context,
            "stdout": stdout,
            "stderr": stderr,
        }

        return result

    def profile_precision_scaling(self):
        """Profile how fitting time scales with precision"""
        print("\n=== Profiling Precision Scaling ===")

        base_config = {
            "kind": "vgonio",
            "family": "microfacet",
            "distro": "trowbridge",
            "symmetry": "isotropic",
            "method": "brute",
            "error_metric": "mse",
            "weighting": "none",
            "alpha_range": "0.1:0.5:0.01",
        }

        for precision in [2, 3, 4, 5, 6]:
            config = base_config.copy()
            config["precision"] = precision
            config["test_name"] = f"precision_{precision}"

            result = self.run_fitting(config)
            self.results.append(result)

            print(f"Precision {precision}: {self._result_summary(result)}")

    def profile_isotropic_vs_anisotropic(self):
        """Compare isotropic vs anisotropic fitting performance"""
        print("\n=== Profiling Isotropic vs Anisotropic ===")

        # Isotropic
        iso_config = {
            "kind": "vgonio",
            "family": "microfacet",
            "distro": "trowbridge",
            "symmetry": "isotropic",
            "method": "brute",
            "error_metric": "mse",
            "weighting": "none",
            "alpha_range": "0.1:0.5:0.01",
            "precision": 3,
            "test_name": "isotropic",
        }

        iso_result = self.run_fitting(iso_config)
        self.results.append(iso_result)
        print(f"Isotropic: {self._result_summary(iso_result)}")

        # Anisotropic
        aniso_config = {
            "kind": "vgonio",
            "family": "microfacet",
            "distro": "trowbridge",
            "symmetry": "anisotropic",
            "method": "brute",
            "error_metric": "mse",
            "weighting": "none",
            "alpha_x_range": "0.1:0.5:0.02",
            "alpha_y_range": "0.1:0.5:0.02",
            "precision": 3,
            "test_name": "anisotropic",
        }

        aniso_result = self.run_fitting(aniso_config)
        self.results.append(aniso_result)
        print(f"Anisotropic: {self._result_summary(aniso_result)}")

    def profile_per_wavelength(self):
        """Profile per-wavelength fitting overhead"""
        print("\n=== Profiling Per-Wavelength Fitting ===")

        base_config = {
            "kind": "vgonio",
            "family": "microfacet",
            "distro": "trowbridge",
            "symmetry": "isotropic",
            "method": "brute",
            "error_metric": "mse",
            "weighting": "none",
            "alpha_range": "0.1:0.5:0.02",
            "precision": 3,
        }

        # Without per-wavelength
        config = base_config.copy()
        config["test_name"] = "all_wavelengths"
        config["per_wavelength"] = False

        result = self.run_fitting(config)
        self.results.append(result)
        print(f"All wavelengths: {self._result_summary(result)}")

        # With per-wavelength
        config = base_config.copy()
        config["test_name"] = "per_wavelength"
        config["per_wavelength"] = True

        result = self.run_fitting(config)
        self.results.append(result)
        print(f"Per wavelength: {self._result_summary(result)}")

    def profile_cpu_vs_gpu(self):
        """Compare CPU vs GPU fitting performance"""
        print("\n=== Profiling CPU vs GPU ===")

        base_config = {
            "kind": "vgonio",
            "family": "microfacet",
            "distro": "trowbridge",
            "symmetry": "isotropic",
            "method": "brute",
            "error_metric": "mse",
            "weighting": "none",
            "alpha_range": "0.1:0.5:0.01",
            "precision": 4,
        }

        # CPU
        config = base_config.copy()
        config["test_name"] = "cpu"
        config["cuda"] = False

        cpu_result = self.run_fitting(config)
        self.results.append(cpu_result)
        print(f"CPU: {self._result_summary(cpu_result)}")

        # GPU
        config = base_config.copy()
        config["test_name"] = "gpu"
        config["cuda"] = True

        gpu_result = self.run_fitting(config)
        self.results.append(gpu_result)
        print(f"GPU: {self._result_summary(gpu_result)}")

        if cpu_result["success"] and gpu_result["success"]:
            speedup = cpu_result["time_seconds"] / gpu_result["time_seconds"]
            print(f"GPU Speedup: {speedup:.2f}x")

    def save_results(self, output_file: Path):
        """Save profiling results to CSV"""
        if not self.results:
            print("No results to save")
            return

        fieldnames = [
            "test_name",
            "success",
            "return_code",
            "command",
            "level",
            "time_seconds",
            "memory_delta_gb",
            "memory_peak_gb",
            "best_alpha",
            "best_alpha_x",
            "best_alpha_y",
            "best_error",
            "best_mse",
            "error_message",
            "error_context",
            "method",
            "symmetry",
            "distro",
            "precision",
        ]

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "test_name": result["config"].get("test_name", "unknown"),
                    "success": result["success"],
                    "return_code": result.get("return_code", "N/A"),
                    "command": result.get("command"),
                    "level": result["config"].get("level", self.level),
                    "time_seconds": result["time_seconds"],
                    "memory_delta_gb": result["memory_delta_gb"],
                    "memory_peak_gb": result["memory_peak_gb"],
                    "best_alpha": result.get("best_alpha"),
                    "best_alpha_x": result.get("best_alpha_x"),
                    "best_alpha_y": result.get("best_alpha_y"),
                    "best_error": result.get("best_error"),
                    "best_mse": result.get("best_mse"),
                    "error_message": result.get("error_message"),
                    "error_context": result.get("error_context"),
                    "method": result["config"]["method"],
                    "symmetry": result["config"]["symmetry"],
                    "distro": result["config"]["distro"],
                    "precision": result["config"].get("precision", "N/A"),
                }
                writer.writerow(row)

        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile BRDF fitting performance")
    parser.add_argument(
        "--vgonio-bin",
        type=Path,
        default="target/release/vgn_app",
        help="Path to vgonio app binary",
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input measurement file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="profiling_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["precision", "symmetry", "wavelength", "gpu", "all"],
        default=["all"],
        help="Which profiling tests to run",
    )
    parser.add_argument(
        "--level",
        default="l0",
        help="Level of measured BRDF data when fitting --kind vgonio (e.g. l0, l1, l2).",
    )

    args = parser.parse_args()

    if not args.vgonio_bin.exists():
        print(f"Error: vgonio binary not found at {args.vgonio_bin}")
        print(
            "Build with: cargo build --release --features fitting,cuda (if GPU tests are needed)"
        )
        sys.exit(1)

    if not args.input.exists():
        print(f"Error: input file not found at {args.input}")
        sys.exit(1)

    profiler = FittingProfiler(args.vgonio_bin, args.input, args.level)

    if "all" in args.tests or "precision" in args.tests:
        profiler.profile_precision_scaling()

    if "all" in args.tests or "symmetry" in args.tests:
        profiler.profile_isotropic_vs_anisotropic()

    if "all" in args.tests or "wavelength" in args.tests:
        profiler.profile_per_wavelength()

    if "all" in args.tests or "gpu" in args.tests:
        profiler.profile_cpu_vs_gpu()

    profiler.save_results(args.output)


if __name__ == "__main__":
    main()
