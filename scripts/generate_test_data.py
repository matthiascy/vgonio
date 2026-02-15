#!/usr/bin/env python3
"""
Generate synthetic BRDF test data for fitting validation

This script creates synthetic BRDF measurements using known analytical models,
which can be used to verify that the fitting process recovers the correct parameters.

Usage:
    python generate_test_data.py --output test_data.json --alpha 0.25 --samples 100
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List
import sys


def microfacet_ggx(
    wi: np.ndarray,
    wo: np.ndarray,
    alpha_x: float,
    alpha_y: float,
    n_i: float = 1.0,
    n_t: float = 1.5,
) -> float:
    """
    Evaluate GGX/Trowbridge-Reitz microfacet BRDF

    Args:
        wi: Incident direction (normalized)
        wo: Outgoing direction (normalized)
        alpha_x: Roughness in x direction
        alpha_y: Roughness in y direction
        n_i: IOR of incident medium
        n_t: IOR of transmitted medium

    Returns:
        BRDF value
    """
    # Half vector
    h = wi + wo
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-8:
        return 0.0
    h = h / h_norm

    # Cosines
    cos_theta_i = max(wi[2], 0.0)
    cos_theta_o = max(wo[2], 0.0)
    cos_theta_h = max(h[2], 0.0)

    if cos_theta_i < 1e-8 or cos_theta_o < 1e-8:
        return 0.0

    # GGX normal distribution
    tan_theta_h_sq = (1.0 - cos_theta_h**2) / (cos_theta_h**2 + 1e-8)

    # Anisotropic term
    h_x = h[0]
    h_y = h[1]
    h_z = h[2]

    if abs(h_z) < 1e-8:
        return 0.0

    exponent = (h_x / alpha_x) ** 2 + (h_y / alpha_y) ** 2
    D = 1.0 / (np.pi * alpha_x * alpha_y * cos_theta_h**4 * (1.0 + exponent) ** 2)

    # Smith masking-shadowing (simplified)
    def lambda_ggx(v, alpha_x, alpha_y):
        cos_theta = v[2]
        if cos_theta < 1e-8:
            return 0.0
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        if sin_theta < 1e-8:
            return 1.0
        tan_theta = sin_theta / cos_theta
        cos_phi = v[0] / (sin_theta + 1e-8)
        sin_phi = v[1] / (sin_theta + 1e-8)
        alpha = np.sqrt(cos_phi**2 * alpha_x**2 + sin_phi**2 * alpha_y**2)
        a = 1.0 / (alpha * tan_theta + 1e-8)
        return (-1.0 + np.sqrt(1.0 + 1.0 / (a**2 + 1e-8))) / 2.0

    G = 1.0 / (
        1.0 + lambda_ggx(wi, alpha_x, alpha_y) + lambda_ggx(wo, alpha_x, alpha_y)
    )

    # Fresnel (Schlick approximation)
    F0 = ((n_i - n_t) / (n_i + n_t)) ** 2
    cos_theta_d = np.dot(wo, h)
    F = F0 + (1.0 - F0) * (1.0 - cos_theta_d) ** 5

    # Microfacet BRDF
    brdf = (D * G * F) / (4.0 * cos_theta_i * cos_theta_o + 1e-8)

    return max(0.0, brdf)


def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to Cartesian"""
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def generate_brdf_samples(
    alpha_x: float,
    alpha_y: float,
    num_incident: int,
    num_outgoing_theta: int,
    num_outgoing_phi: int,
    wavelengths: List[float],
    add_noise: bool = False,
    noise_level: float = 0.01,
) -> dict:
    """
    Generate synthetic BRDF samples

    Args:
        alpha_x: Roughness in x direction
        alpha_y: Roughness in y direction
        num_incident: Number of incident angle samples
        num_outgoing_theta: Number of outgoing theta samples
        num_outgoing_phi: Number of outgoing phi samples
        wavelengths: List of wavelengths in nm
        add_noise: Whether to add Gaussian noise
        noise_level: Standard deviation of noise

    Returns:
        Dictionary containing BRDF data
    """
    print(f"Generating BRDF data:")
    print(f"  alpha_x = {alpha_x}, alpha_y = {alpha_y}")
    print(f"  Incident angles: {num_incident}")
    print(f"  Outgoing angles: {num_outgoing_theta}θ × {num_outgoing_phi}φ")
    print(f"  Wavelengths: {len(wavelengths)}")

    # Generate incident angles
    theta_i_samples = np.linspace(0, np.pi / 2 * 0.9, num_incident)

    # Generate outgoing angles (hemisphere)
    theta_o_samples = np.linspace(0, np.pi / 2, num_outgoing_theta)
    phi_o_samples = np.linspace(0, 2 * np.pi, num_outgoing_phi, endpoint=False)

    data = {
        "metadata": {
            "alpha_x": alpha_x,
            "alpha_y": alpha_y,
            "model": "trowbridge-reitz",
            "incident_medium_ior": 1.0,
            "transmitted_medium_ior": 1.5,
            "noise_level": noise_level if add_noise else 0.0,
        },
        "wavelengths": wavelengths,
        "incident_angles": theta_i_samples.tolist(),
        "outgoing_theta": theta_o_samples.tolist(),
        "outgoing_phi": phi_o_samples.tolist(),
        "values": [],
    }

    # Generate BRDF values
    total_samples = (
        len(wavelengths)
        * len(theta_i_samples)
        * len(theta_o_samples)
        * len(phi_o_samples)
    )
    print(f"Generating {total_samples} samples...")

    for wl_idx, wavelength in enumerate(wavelengths):
        wl_data = []
        for theta_i in theta_i_samples:
            wi = spherical_to_cartesian(theta_i, 0.0)
            incident_data = []

            for theta_o in theta_o_samples:
                theta_data = []
                for phi_o in phi_o_samples:
                    wo = spherical_to_cartesian(theta_o, phi_o)

                    # Evaluate BRDF
                    value = microfacet_ggx(wi, wo, alpha_x, alpha_y)

                    # Add noise if requested
                    if add_noise:
                        noise = np.random.normal(0, noise_level * value)
                        value = max(0.0, value + noise)

                    theta_data.append(value)

                incident_data.append(theta_data)

            wl_data.append(incident_data)

        data["values"].append(wl_data)

    print("Generation complete!")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic BRDF test data")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Roughness value (isotropic, default: 0.25)",
    )
    parser.add_argument(
        "--alpha-x", type=float, help="Roughness in x direction (anisotropic)"
    )
    parser.add_argument(
        "--alpha-y", type=float, help="Roughness in y direction (anisotropic)"
    )
    parser.add_argument(
        "--incident-samples",
        type=int,
        default=10,
        help="Number of incident angle samples",
    )
    parser.add_argument(
        "--outgoing-theta",
        type=int,
        default=16,
        help="Number of outgoing theta samples",
    )
    parser.add_argument(
        "--outgoing-phi", type=int, default=32, help="Number of outgoing phi samples"
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[450.0, 550.0, 650.0],
        help="Wavelengths in nm",
    )
    parser.add_argument(
        "--noise", action="store_true", help="Add Gaussian noise to data"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.01,
        help="Noise level (fraction of signal)",
    )

    args = parser.parse_args()

    # Determine alpha values
    if args.alpha_x is not None and args.alpha_y is not None:
        alpha_x = args.alpha_x
        alpha_y = args.alpha_y
    else:
        alpha_x = alpha_y = args.alpha

    # Generate data
    data = generate_brdf_samples(
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        num_incident=args.incident_samples,
        num_outgoing_theta=args.outgoing_theta,
        num_outgoing_phi=args.outgoing_phi,
        wavelengths=args.wavelengths,
        add_noise=args.noise,
        noise_level=args.noise_level,
    )

    # Save to file
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done! Test data saved to {args.output}")
    print(f"\nExpected fitting results:")
    print(f"  alpha_x should be close to: {alpha_x}")
    print(f"  alpha_y should be close to: {alpha_y}")


if __name__ == "__main__":
    main()
