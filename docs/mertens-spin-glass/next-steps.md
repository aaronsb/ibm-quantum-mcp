# Next Steps: Mertens Spin Glass Experiment

## What We Have

- Corrected analytical formula: λ_c = J·N^{1+2ε}/(|M(N)|-1), confirmed across 29 data points (N=5..37)
- Closed-form correction factor: f(|M|) = |M|/(2(|M|-1)), derived from quadratic penalty structure
  - f(2) = 1, f(3) = 3/4, f(4) = 2/3, f(∞) → 1/2
- Graph-combinatorics analysis proving the transition is always single-spin-flip via degree-1 leaf nodes
  - Bertrand's postulate guarantees leaf nodes exist for all N
  - No cooperative multi-spin effects (zero shared edges between cheapest flips)
  - Graph metrics vary wildly within |M| classes but f is invariant — correction is algebraic, not topological
- |M|=0,1: no transition (14 cases) — correctly predicted by formula (denominator vanishes)
- Diagonal decomposition enables scans in seconds (no eigensolvers needed at Γ=0)
- Phase diagrams showing order-by-disorder quantum protection of the Möbius phase
- VQE (RealAmplitudes, warm-start) detects the level crossing at N=5,12,17,20
- Reproducible scripts with `--parallel` support for multi-core scaling

## What This Means for RH

Almost nothing. At N ≤ 50, M(N) is exactly and trivially computable, |M(N)|/sqrt(N) is far below any bound of interest, and the epsilon parameter is inert (N^{0.02} ≈ 1.08). The connection to RH is motivational: it explains why we chose this Hamiltonian, not what the Hamiltonian proves. Encoding a known function into a finite system and recovering its known values provides no information about asymptotic growth rates.

What we have is a **complete analytical theory** of a quantum spin system on a number-theoretic graph. The formula λ_c = J·N^{1+2ε}/(|M|-1) contains no fitting parameters and is derived from first principles (quadratic penalty structure + Bertrand's postulate guaranteeing leaf nodes). This is cleaner than the original claim of empirically discovered correction factors.

## Completed: Stage 1 — Classical Exact Diag to N=50

**Result**: Exceeded the original N=25 target by 2×. Diagonal decomposition with magnetization trick and chunked memory bypassed eigsh entirely, enabling N=50 (31 qubits, 2.1 billion states, 34 GB) in ~26 minutes per N value.

## Completed: Stage 1.5 — Graph-Combinatorics Analysis

**Result**: Derived the closed-form correction factor f(|M|) = |M|/(2(|M|-1)).

The original "cooperative correction" was a linearization error: the naive formula approximated Δ(M²) as proportional to |M|, but the actual quadratic penalty gives Δ(M²) = 4(|M|-1). The corrected formula matches all 29 measured data points to within grid resolution.

Key structural finding: Bertrand's postulate guarantees degree-1 leaf nodes in the prime factorization graph for all N, ensuring the cheapest spin flip always costs exactly 2J (one edge). This kills any possibility of graph-topology-dependent transition points.

**Predictions**: f(5) = 5/8, f(6) = 3/5, f(7) = 7/12. Falsifiable when |M|=5 data becomes available.

## Completed: Stage 2 — Variational Quantum Verification

**Goal**: Check whether a variational algorithm can detect the level crossing on a noiseless simulator.

**QAOA result: FAIL.** QAOA with QAOAAnsatz cannot find the ground state at N≥8 (6+ qubits). The all-to-all ZZ penalty creates a rugged cost landscape that traps QAOA in local minima.

**VQE result: PASS.** RealAmplitudes ansatz (Ry + CX layers, r=6-8) with COBYLA + warm-start detects the level crossing correctly at N=5,12,17,20 (up to 13 qubits, 99.5-100% accuracy).

**Scripts**: `scripts/scan_lambda_c_vqe.py` (VQE), `scripts/scan_lambda_c_qaoa.py` (QAOA), `scripts/analyze_graph_structure.py` (graph analysis)

## Plan: Two Stages Before Spending Money

### Stage 3: Calibration Run on Torino (N=12, 8 qubits)

**Goal**: Verify real hardware can reproduce a known result.

**What to do**:
- Pick N=12 where we know exact λ_c = 12.61
- Run VQE (RealAmplitudes r=6) on Torino at 3-4 lambda values: one below λ_c, one near, one above
- Compare hardware frustration index against exact diag
- Use error mitigation (resilience level 1-2) and dynamical decoupling

**Cost estimate**:
- 3 lambda values × VQE r=6 × ~200 iterations × 1024 shots = ~600k circuit executions
- On IBM free tier: may require multiple sessions
- On pay-as-you-go: ~$100-200

**What to look for**:
- Does the hardware frustration index show the level crossing?
- How much does noise smear the boundary?
- Is the signal-to-noise ratio viable for larger N?

**Only after all three stages succeed does it make sense to spend real money on N=50+ runs.**

## Technical Prerequisites

Before any hardware runs:
- [x] Split `mcp_vqe_server_local.py` (1042 lines → extract Mertens handlers)
- [x] Derive closed-form f(|M|) — corrected formula eliminates empirical constants
- [ ] Add transpilation-aware Hamiltonian construction (map to heavy-hex topology)
- [ ] Wire up the quantum hardware server (`mcp_vqe_server_quantum.py`) for Mertens tools
- [ ] Add cost estimation tool (estimate QPU seconds before running)

## Open Science Questions

1. **Order-by-disorder quantification**: The transverse field shifts the phase boundary upward, but the quantitative dependence Γ_c(λ) has not been derived analytically. Is there a counterpart to the λ_c formula for the quantum boundary?

2. **Thermodynamic limit**: Does the level crossing sharpen into a true quantum phase transition in some appropriate N→∞ limit? The graph topology changes qualitatively with N, making standard finite-size scaling inapplicable.

3. **f(5) = 5/8 prediction**: The first |M|=5 case appears at N values beyond our current computational range. Reaching it would test whether the corrected formula continues to hold.

4. **Eigenspectrum structure**: The band structure in the low-lying spectrum may encode information about zeta zeros. Speculative but testable.
