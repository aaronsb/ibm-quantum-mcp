# Next Steps: Mertens Spin Glass Experiment

## What We Have

- Confirmed λ_c = 2J·N^{1+2ε}/|M(N)| to within 2% for 8 independent data points (|M|=2, N=5..18)
- Three-regime stratification: |M|=2 exact match, |M|=1 no transition, |M|=3 systematic 25% undershoot
- Phase diagrams showing quantum protection of the Möbius phase by transverse field
- Reproducible scripts and MCP tools for interactive exploration
- All results at N ≤ 20 (≤ 13 square-free qubits)

## What This Means for RH

Almost nothing. Yet. Confirming a phase boundary formula up to N=20 is 13 qubits. RH is a statement about infinity. The Mertens conjecture was confirmed computationally to enormous values and it's still false.

What we have is a **new instrument** — a machine that translates the growth rate of |M(N)| into a physically measurable quantity (λ_c). If the instrument scales, it becomes a new way to *study* the Mertens function. Studying is not proving.

The |M|=3 discrepancy is scientifically more interesting than the |M|=2 confirmations. It reveals cooperative physics beyond the single-spin-flip theory. Understanding why multi-spin flips are 25% cheaper than predicted might teach us something about the structure of prime factorization that pure number theory hasn't revealed.

## Plan: Three Stages Before Spending Money

### Stage 1: Push Classical Exact Diag to N=25

**Goal**: More data points on the |M|=3+ discrepancy. First |M|=4 cases.

**What to do**:
- Run `scan_lambda_c.py` for N=21..25
- N=25 has 16 square-free qubits → 2^16 = 65536 dimensional matrix
- Each eigsh call ~1-5 seconds, 150 lambda points → ~10 min per N
- Total: ~1 hour of CPU time

**What to look for**:
- Does the |M|=3 discrepancy grow, stabilize, or shrink?
- Do we see |M|=4 or |M|=5 cases? What's their discrepancy?
- Does the energy gap keep shrinking with N? Is there a trend toward gap closure?

**Key observable — discrepancy direction**: For all three |M|=3 cases (N=13,19,20), measured λ_c is *below* predicted — the system transitions more easily than the single-spin-flip theory expects. The prime graph finds correlated multi-spin rearrangements that are collectively cheaper than flipping one spin at a time. If |M|=4 cases show the same or stronger undershoot, it means prime factorization graphs become *more* cooperative at higher |M|, not less. That trend — cooperativity increasing with |M| — would be the most interesting finding from Stage 1, because it reveals structure in how primes compose that isn't visible to purely number-theoretic methods.

**Key N values**:
| N  | Square-free qubits | |M(N)| | Notes |
|----|-------------------|--------|-------|
| 21 | 14 | 2 | Should match formula |
| 22 | 14 | 1 | No transition expected |
| 23 | 15 | 2 | Should match formula |
| 24 | 15 | 2 | Should match formula |
| 25 | 16 | 2 | Should match formula, near memory limit |

### Stage 2: QAOA on Simulator at N=30-40

**Goal**: Check whether QAOA can even detect the phase transition on a noiseless simulator.

**What to do**:
- Run `run_mertens_qaoa` at N=30 (19 qubits) with p=3..5 layers
- StatevectorEstimator, no noise — this is the upper bound on QAOA performance
- Compare QAOA energy against exact energy (if we can compute it) or at least check frustration
- Sweep lambda across the predicted λ_c and see if QAOA's ground state shows the frustration transition

**What to look for**:
- Can QAOA distinguish the two phases?
- How many layers (p) are needed to see the transition?
- What optimizer works best? (COBYLA is struggling; try SPSA, L-BFGS-B)

**If QAOA can't see the transition on a perfect simulator, scaling to hardware is pointless.**

### Stage 3: Calibration Run on Torino (N=12, 8 qubits)

**Goal**: Verify real hardware can reproduce a known result.

**What to do**:
- Pick N=12 where we know exact λ_c = 12.9
- Run QAOA on Torino at 3-4 lambda values: one below λ_c, one near, one above
- Compare hardware frustration index against exact diag
- Use error mitigation (resilience level 1-2) and dynamical decoupling

**Cost estimate**:
- 3 lambda values × QAOA p=3 × ~50 iterations × 1024 shots = ~150k circuit executions
- On IBM free tier: fits within 10 min/month if circuits are short
- On pay-as-you-go: ~$50-100

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

The |M|=3 discrepancy at 25% below prediction is where the physics lives. The formula λ_c = 2J·N^{1+2ε}/|M(N)| assumes single-spin flips dominate the transition. When |M| is larger, the system finds correlated multi-spin rearrangements that are collectively cheaper. This is a statement about the connectivity structure of the prime factorization graph — how efficiently can you reduce net magnetization by flipping connected clusters?

If we can characterize the correction factor as a function of |M| and N, we learn something about the cooperative structure of prime multiplication that may not be visible to purely number-theoretic methods. That's the payoff worth chasing.
