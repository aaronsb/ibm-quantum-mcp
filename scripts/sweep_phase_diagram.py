#!/usr/bin/env python3
"""
Generate phase diagram heatmaps for the Mertens spin glass.

Sweeps (lambda, gamma) parameter space using exact diagonalization,
producing 2x2 heatmap panels: ground state energy, energy gap,
max Mertens deviation, and frustration index.

Usage:
    uv run python scripts/sweep_phase_diagram.py --N 12
    uv run python scripts/sweep_phase_diagram.py --N 8 --grid 30 --lambda-max 25
    uv run python scripts/sweep_phase_diagram.py --N 5 8 10 12 15
"""

import argparse
import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mertens_utils import (
    build_mertens_hamiltonian,
    extract_ground_state_info,
    compute_frustration_index,
    mertens,
    mertens_table,
)


def sweep_phase(
    N: int,
    lam_min: float = 0.0,
    lam_max: float | None = None,
    gam_min: float = 0.0,
    gam_max: float = 3.0,
    grid: int = 25,
    J: float = 1.0,
    epsilon: float = 0.01,
) -> dict:
    """Run phase diagram sweep for a single N.

    If lam_max is None, auto-scales based on the predicted lambda_c.
    """
    M_N = abs(mertens(N))
    predicted_lc = 2.0 * J * N ** (1 + 2 * epsilon) / max(M_N, 1)

    if lam_max is None:
        lam_max = predicted_lc * 3.0

    lambdas = np.linspace(lam_min, lam_max, grid)
    gammas = np.linspace(gam_min, gam_max, grid)

    energies = np.zeros((grid, grid))
    gaps = np.zeros((grid, grid))
    deviations = np.zeros((grid, grid))
    frustrations = np.zeros((grid, grid))

    total = grid * grid
    for i, lam in enumerate(lambdas):
        for j, gam in enumerate(gammas):
            H, meta = build_mertens_hamiltonian(N, J, lam, gam, epsilon)
            mat = H.to_matrix(sparse=True)
            k = min(2, mat.shape[0] - 2)
            vals, vecs = eigsh(mat, k=k, which="SA")
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]

            energies[i, j] = vals[0]
            gaps[i, j] = vals[1] - vals[0] if len(vals) > 1 else 0

            info = extract_ground_state_info(vecs[:, 0], N)
            if info["top_configurations"]:
                deviations[i, j] = info["top_configurations"][0]["max_mertens_deviation"]

            frustrations[i, j] = compute_frustration_index(
                vecs[:, 0], meta["petri_net"]["prime_edges"], N
            )

        pct = (i + 1) / grid * 100
        print(f"  {pct:.0f}%", end="\r", flush=True)

    print(f"  done ({grid}x{grid} = {total} points)")

    return {
        "N": N,
        "lambdas": lambdas,
        "gammas": gammas,
        "energies": energies,
        "gaps": gaps,
        "deviations": deviations,
        "frustrations": frustrations,
        "lambda_range": (lam_min, lam_max),
        "gamma_range": (gam_min, gam_max),
    }


def plot_phase_diagram(result: dict, output_path: str):
    """Generate 2x2 heatmap figure."""
    N = result["N"]
    lam_min, lam_max = result["lambda_range"]
    gam_min, gam_max = result["gamma_range"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mertens Spin Glass Phase Diagram (N={N})", fontsize=16)

    panels = [
        (result["energies"], "Ground State Energy", "RdYlBu_r"),
        (result["gaps"], "Energy Gap (E1 - E0)", "viridis"),
        (result["deviations"], "Max |M(x)| Deviation", "hot"),
        (result["frustrations"], "Frustration Index", "coolwarm"),
    ]

    for ax, (data, label, cmap) in zip(axes.flat, panels):
        im = ax.imshow(
            data, origin="lower", aspect="auto", cmap=cmap,
            extent=[gam_min, gam_max, lam_min, lam_max],
        )
        ax.set_xlabel("Gamma (transverse field)", fontsize=11)
        ax.set_ylabel("Lambda (Mertens penalty)", fontsize=11)
        ax.set_title(label, fontsize=12)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase diagram sweep for the Mertens spin glass"
    )
    parser.add_argument("--N", type=int, nargs="+", required=True, help="N values")
    parser.add_argument("--grid", type=int, default=25, help="Grid points per axis")
    parser.add_argument("--lambda-max", type=float, default=None, help="Max lambda (auto if omitted)")
    parser.add_argument("--gamma-max", type=float, default=3.0, help="Max gamma")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "mertens-spin-glass",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    for N in sorted(args.N):
        table = mertens_table(N)
        nq = table["num_qubits"]
        print(f"N={N} ({nq} qubits): sweeping {args.grid}x{args.grid} grid...")

        result = sweep_phase(
            N, lam_max=args.lambda_max, gam_max=args.gamma_max, grid=args.grid,
        )

        output_path = os.path.join(args.output_dir, f"phase_diagram_N{N}.png")
        plot_phase_diagram(result, output_path)


if __name__ == "__main__":
    main()
