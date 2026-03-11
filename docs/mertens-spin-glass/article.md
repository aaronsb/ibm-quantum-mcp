# The Transverse-Field Mobius Ising Model: A Quantum Spin Glass from Prime Factorization

## Abstract

We construct a novel quantum spin system — the Transverse-Field Mobius Ising Model — whose interaction graph encodes the multiplicative structure of the integers through prime factorization. Each qubit represents a square-free integer, antiferromagnetic ZZ couplings connect integers related by multiplication by a prime, and a penalty term suppresses growth of the Mertens function M(x) = sum of mu(k) for k=1..x. Exact diagonalization of systems up to N=50 (31 qubits, 2.1 billion basis states) reveals a first-order quantum phase transition between a "Mobius-obedient" phase (ground state matches the true Mobius function) and a "penalty-obedient" phase (ground state rearranges to minimize |M(x)|). The critical penalty strength lambda_c matches the analytical prediction lambda_c = 2J * N^{1+2epsilon} / |M(N)| to within 0.5% for all |M(N)|=2 cases (17 independent data points, up to 30 qubits). For |M(N)|>=3, the system transitions *below* the predicted lambda_c — the prime factorization graph finds cooperative multi-spin rearrangements that are collectively cheaper than the single-spin-flip theory predicts. The cooperative discount increases monotonically with |M|: 24% at |M|=3, 32% at |M|=4, with zero scatter within each class. The transverse field stabilizes the Mobius phase, shifting the phase boundary upward — quantum fluctuations protect the number-theoretic structure.

## 1. Introduction

The Mobius function mu(n) lies at the heart of analytic number theory. Defined as:

- mu(1) = 1
- mu(n) = (-1)^k if n is a product of k distinct primes
- mu(n) = 0 if n has a squared prime factor

its cumulative sum M(x) = sum_{k=1}^{x} mu(k) — the Mertens function — controls the distribution of primes through the identity 1/zeta(s) = sum_{n=1}^{infty} mu(n)/n^s. The growth rate of M(x) is intimately connected to the Riemann Hypothesis: the bound |M(x)| = O(x^{1/2+epsilon}) for all epsilon > 0 is equivalent to RH.

We ask: what happens when the multiplicative structure of the integers is encoded as the interaction graph of a quantum spin system? The prime factorization map n -> n*p defines a natural directed graph (a Petri net) over the integers. Antiferromagnetic couplings along these edges create frustration whenever the Mobius sign pattern cannot be perfectly realized — and a penalty on the Mertens sum introduces direct competition between local structure and global cancellation.

The resulting Hamiltonian is, to our knowledge, novel. It is not a standard spin glass (the interaction graph has number-theoretic structure, not randomness), nor a standard Ising model on a lattice (the graph is irregular, with connectivity dictated by primes). Its phase diagram reveals a sharp quantum phase transition whose critical point is analytically predictable and numerically confirmed.

## 2. Construction

### 2.1 Encoding: Square-Free Integers as Qubits

A key observation simplifies the encoding: integers with mu(n) = 0 contribute nothing to M(x) and are decoupled from the relevant physics. We adopt the **square-free encoding (Option C)**: only square-free integers receive qubits. This eliminates ~35-40% of qubits for typical N, with no loss of information.

The encoding convention is:

- Qubit in |0> (spin +1) represents mu = +1
- Qubit in |1> (spin -1) represents mu = -1

| N | Square-free qubits | Savings | Prime edges | M(N) | \|M(N)\| / sqrt(N) |
|---|-------------------|---------|-------------|------|-------------------|
| 5 | 4 | 20% | 3 | -2 | 0.894 |
| 8 | 6 | 25% | 6 | -2 | 0.707 |
| 10 | 7 | 30% | 8 | -1 | 0.316 |
| 12 | 8 | 33% | 9 | -2 | 0.577 |
| 15 | 11 | 27% | 14 | -1 | 0.258 |
| 20 | 13 | 35% | 16 | -3 | 0.671 |
| 30 | 19 | 37% | 27 | -3 | 0.548 |
| 60 | 37 | 38% | 57 | -1 | 0.129 |

For all N in this table, |M(N)|/sqrt(N) < 1, consistent with the conjectured O(sqrt(x)) growth rate implied by RH.

### 2.2 The Hamiltonian

The full Hamiltonian consists of three competing terms:

```
H = H_structure + H_penalty + H_transverse
```

**Structure term (antiferromagnetic prime couplings):**

```
H_structure = J * sum_{(n, n*p)} Z_n Z_{n*p}
```

where the sum runs over all pairs (n, n*p) where both n and n*p are square-free and n*p <= N, and p is prime. The coupling J > 0 is antiferromagnetic: aligned spins (same Mobius sign) are penalized, opposite spins (different Mobius sign) are rewarded.

This is physically motivated: multiplication by a prime always changes the parity of the number of prime factors, so mu(n) and mu(n*p) always have opposite signs when both are nonzero. The true Mobius assignment is therefore an **unfrustrated ground state** of H_structure alone — every prime edge is satisfied.

**Mertens penalty term:**

```
H_penalty = (lambda / N^{1+2*epsilon}) * (sum_i Z_i)^2
```

Expanding the square and dropping the constant diagonal (Z_i^2 = I):

```
H_penalty = (lambda / N^{1+2*epsilon}) * sum_{i<j} Z_i Z_j
```

This all-to-all ZZ coupling penalizes configurations where the net magnetization (proportional to M(N)) is large. The scaling denominator N^{1+2*epsilon} normalizes the penalty relative to the system size, with epsilon controlling the sensitivity. At epsilon = 0.01, this closely tracks an x^{1/2+epsilon} bound on the Mertens function.

**Transverse field (quantum mixing):**

```
H_transverse = Gamma * sum_i X_i
```

applied to all qubits. This non-commuting term is critical: without it, H is diagonal in the computational basis and the problem is classical. The transverse field introduces quantum fluctuations that allow tunneling between spin configurations and gives rise to a genuine quantum phase transition.

### 2.3 The Petri Net Interpretation

The interaction graph of H_structure is a directed graph where:
- **Places** are the square-free integers (qubits)
- **Transitions** are prime multiplications (edges)

This is precisely a Petri net whose firing rules correspond to prime factorization. The structure is bipartite: every prime edge connects a mu = +1 integer to a mu = -1 integer (since multiplication by a single prime flips the parity). This bipartiteness guarantees that the true Mobius assignment is a perfect (zero-frustration) ground state of the structure term.

![Mobius Structure / Petri Net for N=15](mobius_structure_N15.png)

Blue nodes represent mu = +1, red nodes represent mu = -1. Colored edges correspond to different primes (x2, x3, x5, x7). The bipartite structure is visible: every edge connects blue to red.

## 3. Phase Structure

### 3.1 Competing Energy Scales

The Hamiltonian has three energy scales that compete:

1. **J** (structure): rewards spin configurations consistent with the multiplicative structure
2. **lambda** (penalty): rewards spin configurations with small net magnetization (small |M(N)|)
3. **Gamma** (transverse): rewards superposition states, drives quantum fluctuations

At J >> lambda, Gamma: the ground state is the true Mobius assignment (or its global spin-flip partner).

At lambda >> J, Gamma: the ground state minimizes |sum Z_i|^2, which may require violating some prime edges.

At Gamma >> J, lambda: the ground state approaches the uniform superposition |+>^N.

### 3.2 Analytical Prediction for lambda_c

The phase transition between the Mobius-obedient and penalty-obedient phases occurs when the energy cost of flipping a spin to reduce |M| exceeds the energy gained from maintaining the prime edge structure.

In the Mobius-obedient phase at Gamma = 0, the net magnetization is M(N) = sum mu(k) for square-free k up to N. The penalty energy is:

```
E_penalty = lambda * M(N)^2 / N^{1+2*epsilon}
```

Flipping a single spin changes the magnetization by 2 and costs at most 2J per prime edge connected to that spin. The critical lambda at which it becomes energetically favorable to break a prime edge to reduce |M| is:

```
lambda_c = 2J * N^{1+2*epsilon} / |M(N)|
```

This prediction has a crucial dependence on |M(N)|: when |M(N)| is small (the Mertens function is well-cancelled), the penalty must be enormous to overcome the structure. When |M(N)| is large, the transition occurs at more modest lambda.

### 3.3 Numerical Verification

We exploit a key property of the Gamma = 0 (classical) limit: the Hamiltonian is diagonal in the computational basis. We decompose H(lambda) = H_structure + lambda * H_penalty, precompute both diagonals once per N using the magnetization identity (sum_{i<j} Z_i Z_j = (m^2 - n)/2 where m is the total magnetization), then sweep lambda with vector addition and argmin. This avoids matrix construction and eigensolvers entirely, enabling scans up to N = 43 (29 qubits, 537 million basis states) in minutes on a single CPU core.

For each N, we scanned lambda from 0 to 4 * lambda_c at Gamma = 0 and identified the first lambda at which the frustration index (fraction of prime edges unsatisfied in the dominant ground state configuration) becomes nonzero.

**Selected results (full dataset in lambda_c_results.json, 43 data points, N=5..50):**

| N | Qubits | \|M(N)\| | Predicted lambda_c | Measured lambda_c | Ratio | Regime |
|---|--------|---------|-------------------|------------------|-------|--------|
| 5 | 4 | 2 | 5.16 | 5.19 | 1.005 | Single-flip |
| 12 | 8 | 2 | 12.61 | 12.67 | 1.005 | Single-flip |
| 13 | 9 | 3 | 9.12 | 6.97 | 0.764 | Cooperative |
| 21 | 14 | 2 | 22.28 | 22.39 | 1.005 | Single-flip |
| 29 | 18 | 2 | 31.04 | 31.19 | 1.005 | Single-flip |
| 30 | 19 | 3 | 21.40 | 16.35 | 0.764 | Cooperative |
| 31 | 20 | 4 | 16.57 | 11.32 | 0.683 | Cooperative |
| 32 | 20 | 4 | 17.13 | 11.70 | 0.683 | Cooperative |
| 37 | 24 | 2 | 39.82 | 40.02 | 1.005 | Single-flip |
| 42 | 28 | 2 | 45.29 | 45.51 | 1.005 | Single-flip |
| 43 | 29 | 3 | 30.87 | 23.58 | 0.764 | Cooperative |
| 46 | 30 | 2 | 49.72 | 49.97 | 1.005 | Single-flip |
| 47 | 31 | 3 | 33.78 | 25.81 | 0.764 | Cooperative |
| 50 | 31 | 3 | 36.01 | 27.51 | 0.764 | Cooperative |

The results stratify into four regimes based on |M(N)|, with zero scatter within each class:

**|M(N)| = 2 (17 data points, N = 5..46):** The measured lambda_c agrees with the single-spin-flip prediction at a ratio of 1.005 across all 17 cases, from 4 to 30 qubits. The formula lambda_c = 2J * N^{1+2*epsilon} / |M(N)| is essentially exact. The consistency of this ratio across a 7x range in system size establishes that the transition mechanism is correctly captured by the analytical argument.

**|M(N)| = 0 or 1 (10 data points, N = 6..41):** No transition is observed even at lambda = 4 * lambda_c. This is physically correct: when |M(N)| <= 1, the Mertens function is already nearly minimized by the true Mobius assignment. The penalty has almost nothing to gain by rearranging spins, so the structure dominates at all lambda in the scanned range. These are the "fortified" N values where the number theory itself provides protection.

**|M(N)| = 3 (10 data points, N = 13..50):** The transition occurs at a ratio of 0.764 (24% below predicted) across all 10 cases, from 9 to 31 qubits. This systematic undershoot indicates that the single-spin-flip argument overestimates the energy barrier: correlated multi-spin rearrangements can collectively reduce the penalty at lower cost. The remarkable consistency of the 0.764 ratio across system sizes suggests this is a structural property of the prime factorization graph, not a finite-size effect.

**|M(N)| = 4 (2 data points, N = 31, 32):** The transition occurs at a ratio of 0.683 (32% below predicted). This deeper cooperative discount confirms the monotonic trend: larger |M| enables more efficient collective rearrangements. The correction factor decreases as 1.005 → 0.764 → 0.683 with increasing |M|, indicating that the prime factorization graph becomes *more* cooperative, not less, as the Mertens function grows.

The monotonic cooperativity trend is the most significant finding. It means that the departure from single-spin-flip theory is not noise — it is a systematic, |M|-dependent correction that reveals how efficiently connected clusters of square-free integers can be collectively rearranged. This cooperative structure is encoded in the topology of the prime factorization graph and may not be visible to purely number-theoretic methods.

![Lambda_c scaling analysis](lambda_c_scaling.png)

Left: Frustration index vs lambda at Gamma = 0 for each N. The transitions are discontinuous step functions — hallmarks of a first-order phase transition. Right: Measured lambda_c (red dots) vs predicted lambda_c (blue crosses). The points coincide exactly for N = 5, 8, 12.

### 3.4 The Phase Diagram

Sweeping both lambda and Gamma on a 25x25 grid reveals the full two-parameter phase diagram. The frustration index serves as the order parameter: 0 in the Mobius-obedient phase, nonzero in the penalty-obedient phase.

**N = 5 (4 qubits):**

![Phase diagram N=5](phase_diagram_N5.png)

The frustration panel (bottom right) shows a clean phase boundary at lambda ~5 for Gamma = 0, curving upward to lambda ~7 at Gamma = 3. The energy gap (top right) is large throughout, consistent with a small system. The ground state energy (top left) shows smooth variation with no singularity.

**N = 8 (6 qubits):**

![Phase diagram N=8](phase_diagram_N8.png)

The boundary shifts to lambda ~8 and exhibits a more pronounced upward curve with Gamma. The staircase structure in the frustration boundary reflects the discrete nature of the spin rearrangements — each step corresponds to one additional prime edge becoming frustrated.

**N = 10 (7 qubits):**

![Phase diagram N=10](phase_diagram_N10.png)

No phase transition is visible in the scanned range (lambda up to 63). The Mobius phase dominates everywhere. This is the |M(N)| = 1 anomaly: the penalty cannot overcome the structure because the Mertens function is already nearly optimal.

**N = 12 (8 qubits):**

![Phase diagram N=12](phase_diagram_N12.png)

The richest phase diagram. The frustration boundary starts at lambda ~12.7 (Gamma = 0) and curves upward to lambda ~25 (Gamma = 3). The energy gap panel shows near-zero gap along the phase boundary at low Gamma — the system is nearly degenerate between competing ground states. As Gamma increases, the gap opens, indicating that quantum fluctuations resolve the near-degeneracy and stabilize the Mobius phase.

**N = 15 (11 qubits):**

![Phase diagram N=15](phase_diagram_N15.png)

Like N = 10, no transition is observed (lambda up to 95). Again, |M(N)| = 1. The energy gap is small at low Gamma for all lambda, reflecting the increased density of states in a larger system.

### 3.5 The Role of the Transverse Field

A consistent feature across all system sizes: **the transverse field raises the phase boundary**. Increasing Gamma stabilizes the Mobius-obedient phase by:

1. Mixing computational basis states, making it harder for the penalty to lock into a single rearranged configuration
2. Opening the energy gap, which protects the ground state from perturbative corrections due to the penalty

This is a quantum protection effect. The transverse field does not merely add noise — it structurally protects the number-theoretic ground state. The staircase shape of the boundary (visible in N = 12) shows that this protection increases stepwise as Gamma crosses thresholds where specific spin-flip excitations become gapped.

## 4. Eigenspectrum Structure

![Eigenspectrum for N=12](eigenspectrum_N12.png)

The low-lying eigenspectrum at N = 12 (default parameters: J = 1, lambda = 0.5, Gamma = 0.5) shows structured energy bands rather than a random distribution. The ground state energy is E_0 = -9.52 with a gap of Delta = 0.18 to the first excited state. The band structure suggests approximate symmetries inherited from the prime factorization graph.

## 5. Methodology

### 5.1 Implementation

The Hamiltonian is constructed using Qiskit 2.3.0's `SparsePauliOp` for the full quantum case (Gamma > 0). For classical scans (Gamma = 0), we bypass Qiskit entirely: the diagonal is computed directly from bit operations using the magnetization identity, enabling systems up to 29 qubits (537M states, 8.6 GB) on a single CPU core. Phase diagram sweeps (Gamma > 0) use `SparsePauliOp.to_matrix(sparse=True)` with scipy's ARPACK eigensolver.

The implementation is exposed as MCP (Model Context Protocol) tools, allowing interactive exploration through natural language queries. Six tools are provided:

- `get_mertens_info`: Number theory exploration
- `build_mertens_hamiltonian`: Hamiltonian construction and caching
- `run_mertens_exact`: Sparse exact diagonalization
- `run_mertens_qaoa`: QAOA variational solver using QAOAAnsatz
- `sweep_mertens_phase`: Phase diagram generation
- `get_mertens_plot`: Visualization

### 5.2 Validation

The number theory primitives (mobius, mertens) are validated at import time against OEIS sequences A008683 and A002321 for n = 1..30. The Hamiltonian is verified to be Hermitian (H = H^dagger). In the classical limit (Gamma = 0, lambda = 0), the ground state has zero frustration index, confirming that the antiferromagnetic coupling sign is correct and the bipartite structure is unfrustrated.

### 5.3 Limitations

- **System size**: Classical (Gamma = 0) scans reach N = 50 (31 qubits, 2.1 billion states). Full quantum phase sweeps (Gamma > 0) are practical up to N ~ 20 with sparse diagonalization. GPU-accelerated dense diagonalization could extend quantum sweeps to N ~ 23 (16 qubits).
- **QAOA performance**: QAOA with few layers (p = 2-3) and COBYLA optimization finds energies significantly above the exact ground state for this Hamiltonian. The energy landscape appears to have many local minima, consistent with the spin-glass character. More layers and better optimizers (e.g., gradient-based) may improve convergence.
- **Finite-size effects**: The strong dependence of lambda_c on |M(N)| — which fluctuates erratically with N — means that finite-size scaling analysis is complicated. The N = 10 and N = 15 anomalies (|M(N)| = 1) are genuine number-theoretic effects, not numerical artifacts.

## 6. Discussion

### 6.1 What This Is Not

This work does not claim to "prove the Riemann Hypothesis" or to reduce it to a quantum computation. The Mertens function's growth rate is a statement about asymptotic behavior (N -> infinity), while our simulations reach N = 50. The connection to RH is motivational, not operational.

### 6.2 What This Is

This is a new quantum spin system with several genuinely interesting properties:

1. **Analytically tractable phase transition**: The critical lambda_c is predicted by a simple formula and confirmed numerically. This is rare for spin systems with irregular interaction graphs.

2. **Number-theoretic structure in the spectrum**: The eigenvalue bands and the anomalous behavior at |M(N)| = 1 are direct consequences of the arithmetic structure of the Mobius function. These are not generic features of random spin glasses.

3. **Quantum protection of classical structure**: The transverse field stabilizes the Mobius-obedient phase, shifting lambda_c upward. This is a concrete example of quantum fluctuations protecting a classical ground state configuration against a competing interaction — a phenomenon relevant to quantum error correction and topological protection.

4. **Novel interaction graph**: The Petri net topology from prime factorization is neither a lattice nor a random graph. It has properties of both (local structure from small primes, long-range connections from large primes) and may be of independent interest for studying quantum dynamics on arithmetic graphs.

5. **The phase boundary is the number theory.** The critical penalty strength lambda_c = 2J * N^{1+2*epsilon} / |M(N)| directly encodes the growth rate of the Mertens function. The phase diagram is not merely *inspired by* number theory — it *is* number theory, rewritten as a competition between energy scales. If |M(N)| grows slower than N^{1/2+epsilon} (the RH-equivalent bound), then lambda_c grows faster than N^{1/2}, and the Mobius-obedient phase occupies an expanding region of parameter space. The phase boundary's shape as a function of N is a physical encoding of the same arithmetic cancellation that governs the distribution of primes.

### 6.3 Sonification and Visualization

An unexplored direction: mapping the eigenspectrum to audio. The structured energy bands (Section 4) have characteristic spacings that vary with N and with position in the phase diagram. Assigning pitch to eigenvalue and timbre to degeneracy would produce a "sound" for each point in parameter space — the phase transition would be audible as a change in harmonic structure. This is science communication rather than science, but it leverages the same data and could make the phase transition viscerally accessible.

### 6.4 Open Questions

- **Scaling to large N**: Can tensor network methods (MPS/DMRG) handle the irregular interaction graph for N > 100? The graph has bounded-but-growing treewidth, which may limit applicability.
- **Quantum hardware**: The N = 12 system (8 qubits) is within reach of current quantum devices. Can QAOA on real hardware reproduce the phase transition? The circuit depth with p >= 3 layers on a heavy-hex topology (IBM Torino) requires careful transpilation.
- **The cooperative correction factor**: The ratios 1.005, 0.764, 0.683 for |M| = 2, 3, 4 are remarkably clean. Is there a closed-form expression f(|M|) that predicts these? Does f(5) follow the trend? Understanding this function would quantify the cooperative structure of the prime factorization graph.
- **The |M(N)| = 1 anomaly**: Why do N = 10, 15, 22, 26, 27, 28, 35, 36, 38, 41 resist the transition? Is this purely the trivial effect of |M| being too small to profit from rearrangement, or does it reflect deeper structure?
- **Connection to zeta zeros**: The eigenspectrum band structure may encode information about the zeros of zeta(s) through the Mobius inversion formula. This is speculative but testable.

## 7. Reproducing These Results

All code is available in the repository. Core modules:

- `mertens_utils.py` — Number theory primitives and Hamiltonian construction
- `mcp_vqe_server_local.py` — MCP server with 6 interactive tools
- `scripts/scan_lambda_c.py` — Lambda_c phase boundary scanner (produces Fig. 4)
- `scripts/sweep_phase_diagram.py` — Phase diagram generator (produces Figs. 1-3)

```bash
# Install dependencies
uv sync

# Reproduce the full lambda_c scan (N=5..43, ~15 min on modern CPU)
uv run python scripts/scan_lambda_c.py --n-values $(seq 5 43) --points 200

# Faster: run N values in parallel (4 workers)
uv run python scripts/scan_lambda_c.py --n-values $(seq 5 43) --points 200 --parallel 4

# Reproduce a phase diagram (Gamma > 0 sweep)
uv run python scripts/sweep_phase_diagram.py --N 12 --grid 25

# Quick verification: confirm f(|M|=4) = 0.683 at N=31
uv run python scripts/scan_lambda_c.py --n-values 31 --points 200

# Interactive exploration via MCP
claude mcp add qiskit-vqe-local -- uv --directory . run python mcp_vqe_server_local.py
```

## References

1. F. Mertens, "Uber eine zahlentheoretische Funktion," Sitzungsberichte der Kaiserlichen Akademie der Wissenschaften, 1897.
2. A. Odlyzko and H. te Riele, "Disproof of the Mertens Conjecture," Journal fur die reine und angewandte Mathematik, 357:138-160, 1985.
3. E. Farhi et al., "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028, 2014.
4. OEIS A008683 (Mobius function), A002321 (Mertens function).
