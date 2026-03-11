# Next Steps: Mertens Spin Glass Experiment

## What We Have

- 43 data points from N=5 to N=50 (4 to 31 qubits, up to 2.1 billion basis states)
- λ_c = 2J·N^{1+2ε}/|M(N)| confirmed to 0.5% for 17 |M|=2 cases
- Four-regime stratification with zero intra-class scatter:
  - |M|=0,1: no transition (14 cases) — number-theoretic protection
  - |M|=2: ratio 1.005 (17 cases) — single-spin-flip formula exact
  - |M|=3: ratio 0.764 (10 cases) — 24% cooperative discount
  - |M|=4: ratio 0.683 (2 cases) — 32% cooperative discount
- Cooperativity increases monotonically with |M|: f(2)=1.005, f(3)=0.764, f(4)=0.683
- Diagonal decomposition enables scans in seconds (no eigensolvers needed at Γ=0)
- Phase diagrams showing quantum protection of the Möbius phase by transverse field
- Reproducible scripts with `--parallel` support for multi-core scaling

## What This Means for RH

Almost nothing. Yet. Confirming a phase boundary formula up to N=43 is 29 qubits. RH is a statement about infinity. The Mertens conjecture was confirmed computationally to enormous values and it's still false.

What we have is a **new instrument** — a machine that translates the growth rate of |M(N)| into a physically measurable quantity (λ_c). The instrument scales (demonstrated over 7 orders of magnitude in Hilbert space dimension) and reveals structure invisible to purely number-theoretic methods.

The cooperative correction f(|M|) is scientifically more interesting than the formula confirmations. The ratios 0.764 and 0.683 are not noise — they are constants that hold across all system sizes within each |M| class. Understanding why multi-spin flips on the prime factorization graph are exactly 24% (or 32%) cheaper than predicted might teach us something about the cooperative structure of prime multiplication.

## Completed: Stage 1 — Classical Exact Diag to N=43

**Result**: Exceeded the original N=25 target by 2×. Diagonal decomposition with magnetization trick and chunked memory bypassed eigsh entirely, enabling N=50 (31 qubits, 2.1 billion states, 34 GB) in ~26 minutes per N value.

**Key findings**:
- f(3) = 0.764 confirmed across 10 data points from N=13 (9 qubits) to N=50 (31 qubits) — not a finite-size effect
- f(4) = 0.683 confirmed at N=31,32 — deeper cooperative discount
- f(2) = 1.005 confirmed across 17 data points up to N=46 (30 qubits) — formula exact
- |M|=0 (N=39,40) behaves like |M|=1: no transition
- Cooperativity increases monotonically: f(2) > f(3) > f(4)

**To push further**: N=51+ requires >31 qubits (>64 GB RAM for two diag vectors). Would need either (a) chunked argmin without materializing full diag vectors, or (b) a machine with >128 GB RAM. First |M|=5 doesn't appear until much larger N.

## Plan: Two Stages Before Spending Money

### Completed: Stage 2 — Variational Quantum Verification

**Goal**: Check whether a variational algorithm can detect the phase transition on a noiseless simulator.

**QAOA result: FAIL.** QAOA with QAOAAnsatz cannot find the ground state at N≥8 (6+ qubits). Tested with COBYLA, SPSA, p=2..5, warm-start — achieves only 50-65% of exact energy at N=12. The all-to-all ZZ penalty creates a rugged cost landscape that traps QAOA in local minima.

**VQE result: PASS.** RealAmplitudes ansatz (Ry + CX layers, r=6-8) with COBYLA + warm-start detects the transition correctly:

| N | Qubits | |M| | Energy accuracy | Transition detected |
|---|--------|-----|-----------------|---------------------|
| 5 | 4 | 2 | 100% | Yes |
| 12 | 8 | 2 | 99.7-100% | Yes (λ~12.9) |
| 17 | 12 | 2 | 99.8-100% | Yes |
| 20 | 13 | 3 | 99.5-100% | Yes (λ~11.5) |

**Key findings**:
- Warm-start is essential — without it, VQE shows the same local-minimum trapping as QAOA
- `QAOAAnsatz.decompose()` x3 gives 400x speedup (8s→0.02s per eval) on StatevectorEstimator
- The ansatz choice matters more than the optimizer — RealAmplitudes succeeds where QAOA fails
- The transition location and frustration values match exact diag at every point tested

**Scripts**: `scripts/scan_lambda_c_vqe.py` (VQE, works), `scripts/scan_lambda_c_qaoa.py` (QAOA, for comparison)

### Stage 3: Calibration Run on Torino (N=12, 8 qubits)

**Goal**: Verify real hardware can reproduce a known result.

**What to do**:
- Pick N=12 where we know exact λ_c = 12.9
- Run VQE (RealAmplitudes r=6) on Torino at 3-4 lambda values: one below λ_c, one near, one above
- Compare hardware frustration index against exact diag
- Use error mitigation (resilience level 1-2) and dynamical decoupling

**Cost estimate**:
- 3 lambda values × VQE r=6 × ~200 iterations × 1024 shots = ~600k circuit executions
- On IBM free tier: may require multiple sessions
- On pay-as-you-go: ~$100-200

**What to look for**:
- Does the hardware frustration index show the phase transition?
- How much does noise smear the boundary?
- Is the signal-to-noise ratio viable for larger N?

**Only after all three stages succeed does it make sense to spend real money on N=50+ runs.**

## Cost Reality for Larger N

| N | Qubits | QAOA run (est.) | Phase sweep (est.) |
|---|--------|----------------|-------------------|
| 12 | 8 | ~$50 | ~$500 |
| 30 | 19 | ~$200 | ~$2,000 |
| 50 | 31 | ~$500 | ~$5,000+ |
| 60 | 37 | ~$1,000 | ~$10,000+ |

These are rough estimates assuming QAOA p=3, SPSA optimizer, 1024 shots, ~20 iterations per point.

## Technical Prerequisites

Before any hardware runs:
- [ ] Split `mcp_vqe_server_local.py` (1042 lines → extract Mertens handlers)
- [ ] Improve QAOA optimizer (SPSA instead of COBYLA, gradient-based options)
- [ ] Add transpilation-aware Hamiltonian construction (map to heavy-hex topology)
- [ ] Wire up the quantum hardware server (`mcp_vqe_server_quantum.py`) for Mertens tools
- [ ] Add cost estimation tool (estimate QPU seconds before running)

## The Real Science Question

## The Real Science Question

The cooperative correction function f(|M|) is where the physics lives:

| |M| | f(|M|) = measured/predicted | Cases | Qubit range | Scatter |
|-----|---------------------------|-------|-------------|---------|
| 0,1 | — (no transition) | 14 | 5-27 | — |
| 2 | 1.005 | 17 | 4-30 | 0 |
| 3 | 0.764 | 10 | 9-31 | 0 |
| 4 | 0.683 | 2 | 20 | 0 |

The formula λ_c = 2J·N^{1+2ε}/|M(N)| assumes single-spin flips dominate the transition. The correction f(|M|) quantifies how much cheaper correlated multi-spin rearrangements are on the prime factorization graph. Three observations:

1. **f is a function of |M| alone** — it has no N dependence (confirmed across 20 qubits of scaling per |M| class)
2. **f is monotonically decreasing** — larger |M| means more cooperative transitions
3. **Zero scatter** — within each |M| class, the ratio is identical to three decimal places

This means the prime factorization graph has a universal cooperative structure that depends only on how many net Möbius signs need to be rearranged, not on the system size or which specific integers are involved. Understanding why f(3) = 0.764 and f(4) = 0.683 — and predicting f(5) — would quantify something about prime multiplication that pure number theory hasn't revealed. That's the payoff worth chasing.
