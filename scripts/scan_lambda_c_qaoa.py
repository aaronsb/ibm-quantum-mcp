#!/usr/bin/env python3
"""
Stage 2: QAOA lambda_c sweep on statevector simulator.

Tests whether QAOA can detect the classical phase transition at lambda_c.
Sweeps lambda at fixed N, runs QAOA at each point, and checks if the
frustration index jumps where classical exact diag says it should.

This is the go/no-go gate for quantum hardware (Stage 3).

Usage:
    uv run python scripts/scan_lambda_c_qaoa.py --N 12
    uv run python scripts/scan_lambda_c_qaoa.py --N 12 --layers 4 --optimizer SPSA
    uv run python scripts/scan_lambda_c_qaoa.py --N 30 --layers 4 --points 13 --maxiter 200
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

from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize as scipy_minimize

from mertens_utils import (
    build_cost_operator,
    build_mertens_hamiltonian,
    compute_frustration_index,
    mertens,
    mertens_table,
)


def run_qaoa_at_lambda(
    N: int,
    lam: float,
    num_layers: int = 4,
    max_iter: int = 150,
    optimizer: str = "SPSA",
    J: float = 1.0,
    gamma: float = 0.0,
    epsilon: float = 0.01,
    seed: int = 42,
    initial_params: np.ndarray = None,
) -> dict:
    """Run QAOA at a single (N, lambda) point and return energy + frustration.

    If initial_params is provided, use as warm start instead of random init.
    """
    from qiskit.quantum_info import Statevector

    # Full Hamiltonian (with transverse field if gamma > 0)
    H_full, meta = build_mertens_hamiltonian(N, J, lam, gamma, epsilon)

    # Z-only cost operator drives QAOA
    cost_op = build_cost_operator(N, J, lam, epsilon)

    ansatz_raw = QAOAAnsatz(cost_operator=cost_op, reps=num_layers)
    # Decompose evolved-operator gates to native gates for fast statevector simulation
    ansatz = ansatz_raw.decompose().decompose().decompose()
    estimator = StatevectorEstimator()

    energies = []

    def cost_func(params):
        qc = ansatz.assign_parameters(params)
        job = estimator.run([(qc, H_full)])
        energy = float(job.result()[0].data.evs)
        energies.append(energy)
        return energy

    if initial_params is None:
        np.random.seed(seed)
        initial_params = np.random.random(ansatz_raw.num_parameters) * 2 * np.pi

    t0 = time.perf_counter()

    if optimizer == "SPSA":
        from qiskit_algorithms.optimizers import SPSA
        spsa = SPSA(maxiter=max_iter)
        result = spsa.minimize(cost_func, initial_params)
        best_energy = float(result.fun)
        best_params = result.x
    else:
        result = scipy_minimize(
            cost_func, initial_params,
            method='COBYLA',
            options={'maxiter': max_iter},
        )
        best_energy = float(result.fun)
        best_params = result.x

    elapsed = time.perf_counter() - t0

    # Get frustration from the optimized state
    qc_final = ansatz.assign_parameters(best_params)
    sv = Statevector(qc_final)
    frust = compute_frustration_index(
        sv.data, meta["petri_net"]["prime_edges"], N
    )

    return {
        "lambda": lam,
        "optimal_params": best_params.tolist(),
        "qaoa_energy": best_energy,
        "frustration": frust,
        "num_iterations": len(energies),
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="QAOA lambda sweep for Mertens spin glass (Stage 2)"
    )
    parser.add_argument("--N", type=int, required=True, help="System size")
    parser.add_argument("--layers", type=int, default=4, help="QAOA layers (p)")
    parser.add_argument("--maxiter", type=int, default=150, help="Optimizer iterations per point")
    parser.add_argument("--optimizer", type=str, default="SPSA", choices=["SPSA", "COBYLA"])
    parser.add_argument("--points", type=int, default=13, help="Number of lambda points")
    parser.add_argument("--lam-min", type=float, default=None, help="Lambda min (auto if omitted)")
    parser.add_argument("--lam-max", type=float, default=None, help="Lambda max (auto if omitted)")
    parser.add_argument("--gamma", type=float, default=0.0, help="Transverse field (0 = classical limit)")
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="Use previous lambda's params as initial guess (default: on)")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="Use random initialization at each lambda point")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    N = args.N
    M_N = abs(mertens(N))
    table = mertens_table(N)
    nq = table["num_qubits"]
    epsilon = 0.01
    J = 1.0

    predicted_lc = 2.0 * J * N ** (1 + 2 * epsilon) / max(M_N, 1)

    # Auto-bracket: sweep from 0.5x to 1.5x predicted lambda_c
    lam_min = args.lam_min if args.lam_min is not None else max(0, predicted_lc * 0.5)
    lam_max = args.lam_max if args.lam_max is not None else predicted_lc * 1.5
    lambdas = np.linspace(lam_min, lam_max, args.points)

    print(f"N={N}, {nq} qubits, |M(N)|={M_N}, predicted λ_c={predicted_lc:.2f}")
    print(f"Sweep: λ={lam_min:.2f}..{lam_max:.2f}, {args.points} points")
    print(f"QAOA: p={args.layers}, {args.optimizer}, maxiter={args.maxiter}, Γ={args.gamma}, warm_start={args.warm_start}")
    print()

    results = []
    prev_params = None
    for i, lam in enumerate(lambdas):
        print(f"  [{i+1}/{args.points}] λ={lam:.2f}: ", end="", flush=True)
        r = run_qaoa_at_lambda(
            N, lam,
            num_layers=args.layers,
            max_iter=args.maxiter,
            optimizer=args.optimizer,
            gamma=args.gamma,
            seed=42 + i,
            initial_params=np.array(prev_params) if (args.warm_start and prev_params is not None) else None,
        )
        results.append(r)
        prev_params = r["optimal_params"]
        print(f"E={r['qaoa_energy']:.4f}, frust={r['frustration']:.3f}, {r['elapsed_s']:.1f}s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    lams = [r["lambda"] for r in results]
    frusts = [r["frustration"] for r in results]
    energies = [r["qaoa_energy"] for r in results]

    ax = axes[0]
    ax.plot(lams, frusts, 'o-', color='red', linewidth=2, markersize=6)
    ax.axvline(predicted_lc, color='blue', linestyle='--', alpha=0.7, label=f'predicted λ_c={predicted_lc:.2f}')
    ax.set_xlabel("Lambda (Mertens penalty)", fontsize=12)
    ax.set_ylabel("Frustration Index", fontsize=12)
    ax.set_title(f"QAOA Frustration vs Lambda (N={N}, p={args.layers}, {args.optimizer})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(lams, energies, 's-', color='steelblue', linewidth=2, markersize=6)
    ax.axvline(predicted_lc, color='blue', linestyle='--', alpha=0.7, label=f'predicted λ_c={predicted_lc:.2f}')
    ax.set_xlabel("Lambda (Mertens penalty)", fontsize=12)
    ax.set_ylabel("QAOA Energy", fontsize=12)
    ax.set_title(f"QAOA Energy vs Lambda (N={N})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "mertens-spin-glass")
    output_path = args.output or os.path.join(output_dir, f"qaoa_sweep_N{N}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")

    json_path = args.json or output_path.replace(".png", ".json")
    summary = {
        "N": N,
        "num_qubits": nq,
        "M_N": M_N,
        "predicted_lambda_c": predicted_lc,
        "qaoa_layers": args.layers,
        "optimizer": args.optimizer,
        "maxiter": args.maxiter,
        "gamma": args.gamma,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
