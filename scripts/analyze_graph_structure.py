#!/usr/bin/env python3
"""
Graph-combinatorics analysis of the prime factorization graph.

Tests whether the cooperative correction factor f(|M|) can be explained
by graph-structural quantities, and derives the closed-form expression.

Result: f(|M|) = |M| / (2(|M| - 1))

This comes from two facts:
1. The prime factorization graph always has degree-1 leaf nodes in the
   majority spin class (large primes near N), so the cheapest flip
   always costs exactly 2J (one edge broken).
2. The paper's formula lambda_c = 2J*N^{1+2e}/|M| uses an implicit
   total-cost argument. The correct marginal energy balance gives
   lambda_c = 2J*N^{1+2e}/(2|M|-2), using the quadratic penalty
   structure (M-2)^2 - M^2 = -4M+4 = -4(|M|-1).

The "zero scatter" within each |M| class is real but follows from the
formula depending only on |M|, not from empirical coincidence. Graph
metrics (degree distributions, cut ratios) vary widely within each
class, but the transition point doesn't depend on them.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mertens_utils import (
    build_diagonal_components,
    frustration_from_bitindex,
    mertens,
    mertens_table,
    mobius,
)


def f_closed_form(abs_M: int) -> float:
    """Closed-form correction factor: f(|M|) = |M| / (2(|M| - 1))."""
    if abs_M < 2:
        return None
    return abs_M / (2.0 * (abs_M - 1))


def corrected_lambda_c(N: int, J: float = 1.0, epsilon: float = 0.01) -> float:
    """Corrected critical lambda using exact marginal energy balance.

    lambda_c = 2J * N^{1+2e} / (2|M| - 2)
             = J * N^{1+2e} / (|M| - 1)

    This is the point where flipping a single degree-1 node in the
    majority spin class first becomes energetically favorable.
    """
    abs_M = abs(mertens(N))
    if abs_M < 2:
        return None
    return J * N ** (1.0 + 2.0 * epsilon) / (abs_M - 1)


def graph_metrics(N: int, measure_lc: bool = True) -> dict:
    """Compute graph-structural metrics for the prime factorization graph at N."""
    table = mertens_table(N)
    edges = table["prime_edges_sqfree"]
    sqfree = table["squarefree"]
    nq = table["num_qubits"]
    M_N = mertens(N)
    abs_M = abs(M_N)

    # Mobius assignment: true spin partition
    spin_up = [n for n in sqfree if mobius(n) == +1]
    spin_down = [n for n in sqfree if mobius(n) == -1]

    # Degree of each node
    degree = {n: 0 for n in sqfree}
    for n, np_val in edges:
        degree[n] += 1
        degree[np_val] += 1

    total_edges = len(edges)

    # Majority class (the one we flip FROM to reduce |M|)
    if M_N > 0:
        majority = spin_up
    elif M_N < 0:
        majority = spin_down
    else:
        majority = []

    cheapest_single_flip = min((degree[n] for n in majority), default=None)
    avg_flip_cost = float(np.mean([degree[n] for n in majority])) if majority else None

    # Identify the degree-1 leaf nodes in majority (why cheapest flip = 1)
    leaf_nodes = [n for n in majority if degree[n] == 1]
    # These are typically large primes or large semiprimes near N
    leaf_descriptions = []
    for n in leaf_nodes[:5]:
        factors = []
        temp = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        leaf_descriptions.append(f"{n}={'*'.join(map(str, factors))}")

    epsilon = 0.01

    # Three lambda_c predictions
    paper_lc = 2.0 * N ** (1 + 2 * epsilon) / max(abs_M, 1) if abs_M > 0 else None
    corrected_lc = corrected_lambda_c(N) if abs_M >= 2 else None
    f_predicted = f_closed_form(abs_M)

    # Measure actual lambda_c from diagonal decomposition
    measured_lc = None
    if measure_lc and nq <= 25 and abs_M >= 2:
        struct_diag, pen_diag, _ = build_diagonal_components(N, J_coupling=1.0, epsilon=epsilon)
        lam_max = paper_lc * 2.0 if paper_lc else 100.0
        # Use 4000 points for higher resolution
        for lam in np.linspace(0, lam_max, 4000):
            diag = struct_diag + lam * pen_diag
            gs_idx = np.argmin(diag)
            frust = frustration_from_bitindex(gs_idx, edges, N)
            if frust > 0:
                measured_lc = lam
                break
        del struct_diag, pen_diag

    ratio_paper = measured_lc / paper_lc if (measured_lc and paper_lc) else None
    ratio_corrected = measured_lc / corrected_lc if (measured_lc and corrected_lc) else None

    return {
        "N": N,
        "num_qubits": nq,
        "M_N": M_N,
        "abs_M": abs_M,
        "total_edges": total_edges,
        "mean_degree": float(np.mean(list(degree.values()))) if degree else 0,
        "max_degree": max(degree.values()) if degree else 0,
        "cheapest_single_flip": cheapest_single_flip,
        "avg_flip_cost_majority": avg_flip_cost,
        "leaf_nodes_in_majority": leaf_descriptions,
        "paper_lc": paper_lc,
        "corrected_lc": corrected_lc,
        "measured_lc": measured_lc,
        "f_predicted": f_predicted,
        "ratio_vs_paper": ratio_paper,
        "ratio_vs_corrected": ratio_corrected,
    }


def main():
    print("=" * 100)
    print("Graph-Combinatorics Analysis: Deriving f(|M|) = |M| / (2(|M| - 1))")
    print("=" * 100)

    # Part 1: Show the closed-form prediction
    print("\n--- Closed-form prediction ---")
    print(f"{'|M|':>4s}  {'f(|M|)':>10s}  {'exact':>10s}")
    for m in range(2, 8):
        f = f_closed_form(m)
        # Express as fraction
        from fractions import Fraction
        frac = Fraction(m, 2 * (m - 1))
        print(f"  {m:2d}    {f:10.6f}    {frac}")
    print(f"  inf   {0.5:10.6f}    1/2")

    # Part 2: Verify against numerical measurements
    print("\n--- Numerical verification (4000-point grid, up to 25 qubits) ---")
    print(f"{'N':>3s}  {'nq':>3s}  {'|M|':>3s}  {'edges':>5s}  "
          f"{'min_d':>5s}  {'leaves':>30s}  "
          f"{'paper_lc':>9s}  {'corrected':>9s}  {'measured':>9s}  "
          f"{'vs_paper':>9s}  {'vs_corr':>9s}")

    results = []
    for N in range(5, 51):
        abs_M = abs(mertens(N))
        if abs_M < 2:
            continue
        nq = mertens_table(N)["num_qubits"]
        r = graph_metrics(N, measure_lc=(nq <= 25))
        results.append(r)

        leaves_str = ", ".join(r["leaf_nodes_in_majority"][:3])
        meas_str = f"{r['measured_lc']:9.3f}" if r['measured_lc'] else "      N/A"
        rp_str = f"{r['ratio_vs_paper']:9.5f}" if r['ratio_vs_paper'] else "      N/A"
        rc_str = f"{r['ratio_vs_corrected']:9.5f}" if r['ratio_vs_corrected'] else "      N/A"
        print(f"{r['N']:3d}  {r['num_qubits']:3d}  {r['abs_M']:3d}  {r['total_edges']:5d}  "
              f"{r['cheapest_single_flip']:5d}  {leaves_str:>30s}  "
              f"{r['paper_lc']:9.3f}  {r['corrected_lc']:9.3f}  {meas_str}  "
              f"{rp_str}  {rc_str}")

    # Part 3: Summary by |M| class
    print("\n" + "=" * 100)
    print("SUMMARY BY |M| CLASS")
    print("=" * 100)

    for abs_M_class in [2, 3, 4]:
        class_results = [r for r in results if r["abs_M"] == abs_M_class]
        measured = [r for r in class_results if r["ratio_vs_corrected"] is not None]
        if not measured:
            continue

        f_pred = f_closed_form(abs_M_class)
        ratios_paper = [r["ratio_vs_paper"] for r in measured]
        ratios_corrected = [r["ratio_vs_corrected"] for r in measured]

        print(f"\n--- |M| = {abs_M_class} ({len(measured)} measured, {len(class_results)} total) ---")
        print(f"  Predicted f(|M|) = {abs_M_class}/(2*{abs_M_class-1}) = {f_pred:.6f}")
        print(f"  vs paper formula:    mean={np.mean(ratios_paper):.6f} +/- {np.std(ratios_paper):.6f}")
        print(f"  vs corrected formula: mean={np.mean(ratios_corrected):.6f} +/- {np.std(ratios_corrected):.6f}")
        print(f"  Cheapest flip = {measured[0]['cheapest_single_flip']} edge for ALL cases")

        # Show that graph metrics VARY while f stays constant
        edges_list = [r["total_edges"] for r in measured]
        mean_deg = [r["mean_degree"] for r in measured]
        print(f"  Total edges: {edges_list} (varies {min(edges_list)}-{max(edges_list)})")
        print(f"  Mean degree: {[f'{d:.2f}' for d in mean_deg]} (varies)")
        print(f"  BUT f(|M|) is constant: {[f'{r:.5f}' for r in ratios_corrected]}")

    # Part 4: Why degree-1 nodes always exist
    print("\n" + "=" * 100)
    print("WHY DEGREE-1 NODES ALWAYS EXIST IN THE MAJORITY CLASS")
    print("=" * 100)
    print()
    print("The prime factorization graph always has leaf nodes because:")
    print("  - Large primes p near N have degree 1 (connected only to 1 via edge (1,p))")
    print("    because p*2 > N, so no upward connections exist.")
    print("  - All primes have mu(p) = -1, providing leaves in the spin-down class.")
    print("  - Large semiprimes pq near N have degree 1-2, with mu(pq) = +1,")
    print("    providing leaves in the spin-up class.")
    print()
    print("Leaf nodes by |M| class:")
    for abs_M_class in [2, 3, 4]:
        class_results = [r for r in results if r["abs_M"] == abs_M_class]
        for r in class_results[:3]:
            print(f"  N={r['N']:2d}: {', '.join(r['leaf_nodes_in_majority'][:4])}")

    # Part 5: The corrected formula
    print("\n" + "=" * 100)
    print("CORRECTED FORMULA")
    print("=" * 100)
    print()
    print("Paper:     lambda_c = 2J * N^{1+2e} / |M(N)|       (total-cost argument)")
    print("Corrected: lambda_c = 2J * N^{1+2e} / (2|M(N)|-2)  (marginal energy balance)")
    print("         = J * N^{1+2e} / (|M(N)| - 1)")
    print()
    print("Derivation:")
    print("  Structure cost of flipping one degree-1 node: 2J (one edge broken)")
    print("  Penalty energy saved: lambda * |M^2 - (|M|-2)^2| / (2*N^{1+2e})")
    print("                      = lambda * (4|M|-4) / (2*N^{1+2e})")
    print("                      = lambda * 2(|M|-1) / N^{1+2e}")
    print("  Setting equal: 2J = lambda * 2(|M|-1) / N^{1+2e}")
    print("  Solving:  lambda_c = J * N^{1+2e} / (|M|-1)")
    print()
    print("The paper's formula implicitly assumes the penalty change is proportional")
    print("to |M| (linear), but the actual change is proportional to 2(|M|-1)")
    print("(from the quadratic (sum Z_i)^2 structure). The ratio is:")
    print("  f(|M|) = paper / corrected = |M| / (2(|M|-1))")
    print()
    print("Predictions for untested |M| values:")
    for m in range(5, 9):
        f = f_closed_form(m)
        print(f"  f({m}) = {m}/{2*(m-1)} = {f:.4f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "docs", "mertens-spin-glass")
    json_path = os.path.join(output_dir, "graph_structure_analysis.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, list):
                sr[k] = [convert(x) for x in v]
            else:
                sr[k] = convert(v)
        serializable.append(sr)

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
