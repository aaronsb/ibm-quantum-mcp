#!/usr/bin/env python3
"""
Stage 2: Hardware-efficient VQE lambda sweep on statevector simulator.

Uses RealAmplitudes ansatz (Ry + CX layers) instead of QAOA. This ansatz
converges far better on the Mertens Hamiltonian because it doesn't try to
exponentiate the full cost operator — it just provides a flexible parameterized
circuit that the optimizer can shape to the ground state.

Key finding: RealAmplitudes r=6 with COBYLA + warm-start achieves 99.9-100%
of exact ground state energy at N=12 (8 qubits) and correctly detects the
frustration transition. QAOA with the same optimizer could only reach 50-65%.

Usage:
    uv run python scripts/scan_lambda_c_vqe.py --N 12
    uv run python scripts/scan_lambda_c_vqe.py --N 12 --reps 6 --maxiter 1000
    uv run python scripts/scan_lambda_c_vqe.py --N 15 --reps 8 --points 15
    uv run python scripts/scan_lambda_c_vqe.py --N 20 --ansatz EfficientSU2
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SparseEfficiencyWarning.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from mertens_utils import (
    build_mertens_hamiltonian,
    compute_frustration_index,
    mertens,
    mertens_table,
)


def run_vqe_at_lambda(
    N: int,
    lam: float,
    ansatz_name: str = "RealAmplitudes",
    reps: int = 6,
    max_iter: int = 1000,
    J: float = 1.0,
    gamma: float = 0.0,
    epsilon: float = 0.01,
    seed: int = 42,
    initial_params: np.ndarray = None,
) -> dict:
    """Run VQE at a single (N, lambda) point."""
    H, meta = build_mertens_hamiltonian(N, J, lam, gamma, epsilon)
    nq = meta["num_qubits"]

    if ansatz_name == "EfficientSU2":
        ansatz = EfficientSU2(nq, reps=reps)
    else:
        ansatz = RealAmplitudes(nq, reps=reps)

    estimator = StatevectorEstimator()
    energies = []

    def cost_func(params):
        qc = ansatz.assign_parameters(params)
        job = estimator.run([(qc, H)])
        energy = float(job.result()[0].data.evs)
        energies.append(energy)
        return energy

    if initial_params is not None and len(initial_params) == ansatz.num_parameters:
        init = initial_params
    else:
        np.random.seed(seed)
        init = np.random.random(ansatz.num_parameters) * 2 * np.pi

    t0 = time.perf_counter()
    result = minimize(cost_func, init, method='COBYLA', options={'maxiter': max_iter})
    elapsed = time.perf_counter() - t0

    # Frustration from VQE state
    qc_final = ansatz.assign_parameters(result.x)
    sv = Statevector(qc_final)
    vqe_frust = compute_frustration_index(
        sv.data, meta["petri_net"]["prime_edges"], N
    )

    # Exact reference (if feasible)
    exact_energy = None
    exact_frust = None
    if nq <= 20:
        mat = H.to_matrix(sparse=True)
        vals, vecs = eigsh(mat, k=2, which='SA')
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        exact_energy = float(vals[0])
        exact_frust = compute_frustration_index(
            vecs[:, 0], meta["petri_net"]["prime_edges"], N
        )

    return {
        "lambda": lam,
        "vqe_energy": float(result.fun),
        "exact_energy": exact_energy,
        "energy_ratio": float(result.fun) / exact_energy if exact_energy else None,
        "vqe_frustration": vqe_frust,
        "exact_frustration": exact_frust,
        "num_iterations": len(energies),
        "elapsed_s": elapsed,
        "optimal_params": result.x.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hardware-efficient VQE lambda sweep for Mertens spin glass (Stage 2)"
    )
    parser.add_argument("--N", type=int, required=True, help="System size")
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes",
                        choices=["RealAmplitudes", "EfficientSU2"])
    parser.add_argument("--reps", type=int, default=6, help="Ansatz repetitions")
    parser.add_argument("--maxiter", type=int, default=1000, help="COBYLA iterations per point")
    parser.add_argument("--points", type=int, default=13, help="Number of lambda points")
    parser.add_argument("--lam-min", type=float, default=None)
    parser.add_argument("--lam-max", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Transverse field (0 = classical limit)")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        default=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    N = args.N
    M_N = abs(mertens(N))
    table = mertens_table(N)
    nq = table["num_qubits"]
    epsilon = 0.01
    J = 1.0

    predicted_lc = 2.0 * J * N ** (1 + 2 * epsilon) / max(M_N, 1)

    lam_min = args.lam_min if args.lam_min is not None else 0.0
    lam_max = args.lam_max if args.lam_max is not None else predicted_lc * 1.5
    lambdas = np.linspace(lam_min, lam_max, args.points)

    print(f"N={N}, {nq} qubits, |M(N)|={M_N}, predicted λ_c={predicted_lc:.2f}")
    print(f"Sweep: λ={lam_min:.2f}..{lam_max:.2f}, {args.points} points")
    print(f"VQE: {args.ansatz} r={args.reps}, COBYLA maxiter={args.maxiter}, "
          f"warm_start={args.warm_start}, Γ={args.gamma}")
    print()
    print(f"{'λ':>6s}  {'exact_E':>9s}  {'vqe_E':>9s}  {'ratio':>6s}  "
          f"{'ex_fr':>5s}  {'vqe_fr':>6s}  {'iters':>5s}  {'time':>5s}")

    results = []
    prev_params = None
    for i, lam in enumerate(lambdas):
        r = run_vqe_at_lambda(
            N, lam,
            ansatz_name=args.ansatz,
            reps=args.reps,
            max_iter=args.maxiter,
            gamma=args.gamma,
            seed=42,
            initial_params=np.array(prev_params) if (args.warm_start and prev_params) else None,
        )
        results.append(r)
        if args.warm_start:
            prev_params = r["optimal_params"]

        ratio_str = f"{r['energy_ratio']*100:5.1f}%" if r['energy_ratio'] else "  N/A"
        ex_e = f"{r['exact_energy']:9.4f}" if r['exact_energy'] else "      N/A"
        ex_f = f"{r['exact_frustration']:5.3f}" if r['exact_frustration'] is not None else "  N/A"
        print(f"{lam:6.1f}  {ex_e}  {r['vqe_energy']:9.4f}  {ratio_str}  "
              f"{ex_f}  {r['vqe_frustration']:6.3f}  {r['num_iterations']:5d}  {r['elapsed_s']:5.1f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    lams = [r["lambda"] for r in results]
    vqe_frusts = [r["vqe_frustration"] for r in results]
    vqe_energies = [r["vqe_energy"] for r in results]
    exact_energies = [r["exact_energy"] for r in results if r["exact_energy"] is not None]
    exact_frusts = [r["exact_frustration"] for r in results if r["exact_frustration"] is not None]
    exact_lams = [r["lambda"] for r in results if r["exact_energy"] is not None]

    # Frustration comparison
    ax = axes[0]
    ax.plot(lams, vqe_frusts, 'o-', color='red', linewidth=2, markersize=6, label='VQE')
    if exact_frusts:
        ax.plot(exact_lams, exact_frusts, 's--', color='blue', linewidth=1.5, markersize=5,
                alpha=0.7, label='Exact')
    ax.axvline(predicted_lc, color='gray', linestyle=':', alpha=0.7, label=f'predicted λ_c={predicted_lc:.1f}')
    ax.set_xlabel("Lambda", fontsize=12)
    ax.set_ylabel("Frustration Index", fontsize=12)
    ax.set_title(f"Frustration vs Lambda (N={N})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy comparison
    ax = axes[1]
    ax.plot(lams, vqe_energies, 'o-', color='red', linewidth=2, markersize=6, label='VQE')
    if exact_energies:
        ax.plot(exact_lams, exact_energies, 's--', color='blue', linewidth=1.5, markersize=5,
                alpha=0.7, label='Exact')
    ax.axvline(predicted_lc, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel("Lambda", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.set_title(f"Ground State Energy (N={N})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy ratio
    ax = axes[2]
    ratios = [r["energy_ratio"] * 100 for r in results if r["energy_ratio"] is not None]
    ratio_lams = [r["lambda"] for r in results if r["energy_ratio"] is not None]
    if ratios:
        ax.plot(ratio_lams, ratios, 'o-', color='green', linewidth=2, markersize=6)
        ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(min(ratios) - 1, 101)
    ax.axvline(predicted_lc, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel("Lambda", fontsize=12)
    ax.set_ylabel("VQE / Exact Energy (%)", fontsize=12)
    ax.set_title(f"Energy Accuracy (N={N})", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "docs", "mertens-spin-glass")
    output_path = args.output or os.path.join(output_dir, f"vqe_sweep_N{N}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")

    # JSON
    json_path = args.json or output_path.replace(".png", ".json")
    summary = {
        "N": N,
        "num_qubits": nq,
        "M_N": M_N,
        "predicted_lambda_c": predicted_lc,
        "ansatz": args.ansatz,
        "reps": args.reps,
        "maxiter": args.maxiter,
        "warm_start": args.warm_start,
        "gamma": args.gamma,
        "results": [{k: v for k, v in r.items() if k != "optimal_params"} for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
