#!/usr/bin/env python3
"""
Scan the frustration index vs lambda at Gamma=0 (classical limit)
for a range of N values. Identifies the critical lambda_c where the
Mobius-obedient phase gives way to the penalty-obedient phase.

Produces a two-panel plot:
  Left:  Frustration vs lambda for each N
  Right: Measured lambda_c vs N, compared to analytical prediction

Usage:
    uv run python scripts/scan_lambda_c.py
    uv run python scripts/scan_lambda_c.py --n-values 5 8 10 12 15 20
    uv run python scripts/scan_lambda_c.py --n-values 5 8 10 12 --points 200 --output results.png
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mertens_utils import (
    build_mertens_hamiltonian,
    compute_frustration_index,
    mertens,
    mertens_table,
)


def scan_lambda_c(
    N: int,
    J: float = 1.0,
    epsilon: float = 0.01,
    num_points: int = 150,
    lam_max_factor: float = 4.0,
) -> dict:
    """Scan frustration vs lambda at Gamma=0 for a single N.

    Returns dict with lambdas, frustrations, predicted and measured lambda_c.
    """
    M_N = abs(mertens(N))
    predicted = 2.0 * J * N ** (1 + 2 * epsilon) / max(M_N, 1)
    lam_max = min(predicted * lam_max_factor, 200)
    lambdas = np.linspace(0, lam_max, num_points)

    table = mertens_table(N)
    nq = table["num_qubits"]

    frusts = []
    for lam in lambdas:
        H, meta = build_mertens_hamiltonian(
            N, J_coupling=J, lambda_penalty=lam,
            gamma_transverse=0.0, epsilon=epsilon,
        )
        mat = H.to_matrix(sparse=True)
        k = min(2, mat.shape[0] - 2)
        vals, vecs = eigsh(mat, k=k, which="SA")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        f = compute_frustration_index(
            vecs[:, 0], meta["petri_net"]["prime_edges"], N
        )
        frusts.append(f)

    frusts = np.array(frusts)
    transition_idx = np.where(frusts > 0)[0]
    lam_c = float(lambdas[transition_idx[0]]) if len(transition_idx) > 0 else None

    return {
        "N": N,
        "num_qubits": nq,
        "M_N": M_N,
        "predicted_lambda_c": predicted,
        "measured_lambda_c": lam_c,
        "lambdas": lambdas,
        "frustrations": frusts,
        "max_frustration": float(frusts.max()),
    }


def plot_results(results: list[dict], output_path: str):
    """Generate the two-panel scaling plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: frustration vs lambda
    ax = axes[0]
    for r in results:
        if r["measured_lambda_c"] is not None:
            ax.plot(
                r["lambdas"], r["frustrations"],
                label=f'N={r["N"]} (|M|={r["M_N"]})', linewidth=1.5,
            )
    ax.set_xlabel("Lambda (Mertens penalty)", fontsize=12)
    ax.set_ylabel("Frustration Index", fontsize=12)
    ax.set_title("Frustration vs Lambda at Gamma=0 (classical limit)", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: lambda_c vs N
    ax = axes[1]
    with_trans = [r for r in results if r["measured_lambda_c"] is not None]
    without_trans = [r for r in results if r["measured_lambda_c"] is None]

    if with_trans:
        ns = [r["N"] for r in with_trans]
        lcs = [r["measured_lambda_c"] for r in with_trans]
        pcs = [r["predicted_lambda_c"] for r in with_trans]
        ms = [r["M_N"] for r in with_trans]

        ax.scatter(ns, lcs, s=100, c="red", zorder=5, label="Measured lambda_c")
        ax.scatter(
            ns, pcs, s=80, c="blue", marker="x", zorder=5,
            linewidths=2, label="Predicted (2J·N^{1+2ε}/|M(N)|)",
        )
        for i, N in enumerate(ns):
            ax.annotate(
                f"|M|={ms[i]}", (N, lcs[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=8,
            )

    for r in without_trans:
        ax.scatter(
            r["N"], r["predicted_lambda_c"],
            s=60, c="gray", marker="v", zorder=4, alpha=0.6,
        )
        ax.annotate(
            f'N={r["N"]}\n|M|={r["M_N"]}\nno trans.',
            (r["N"], r["predicted_lambda_c"]),
            textcoords="offset points", xytext=(8, -15), fontsize=7, color="gray",
        )

    all_ns = [r["N"] for r in results]
    ns_range = np.linspace(min(all_ns) - 1, max(all_ns) + 1, 50)
    ax.plot(ns_range, 2 * np.sqrt(ns_range), "--", color="gray", alpha=0.4, label="2√N")
    ax.plot(ns_range, ns_range, ":", color="gray", alpha=0.4, label="N")

    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("lambda_c", fontsize=12)
    ax.set_title(f"Phase Boundary vs System Size (N={min(all_ns)}..{max(all_ns)})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan lambda_c phase boundary for the Mertens spin glass"
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+",
        default=[5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20],
        help="N values to scan",
    )
    parser.add_argument("--points", type=int, default=150, help="Lambda scan points per N")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    parser.add_argument("--json", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "mertens-spin-glass", "lambda_c_scaling.png",
        )

    results = []
    for N in sorted(args.n_values):
        table = mertens_table(N)
        nq = table["num_qubits"]
        M_N = abs(mertens(N))
        print(f"N={N:2d} ({nq:2d} qubits, |M|={M_N}): scanning...", end=" ", flush=True)

        r = scan_lambda_c(N, num_points=args.points)
        results.append(r)

        if r["measured_lambda_c"] is not None:
            print(f"lambda_c={r['measured_lambda_c']:.1f} (predicted {r['predicted_lambda_c']:.1f})")
        else:
            print(f"no transition (predicted {r['predicted_lambda_c']:.1f})")

    plot_results(results, args.output)

    if args.json:
        serializable = []
        for r in results:
            s = {k: v for k, v in r.items() if k not in ("lambdas", "frustrations")}
            serializable.append(s)
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"JSON saved: {args.json}")


if __name__ == "__main__":
    main()
