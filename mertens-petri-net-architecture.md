# Möbius Spin Glass: Quantum Phase Diagram of Prime Factorization Topology

## The Big Idea

Instead of computing individual zeros of the zeta function, we encode the
multiplicative structure of integers as a quantum spin system and study its
phase diagram. The interaction graph comes from prime factorization — each
prime multiplication edge creates an antiferromagnetic coupling. We then
probe: does this system exhibit a phase transition between a "Möbius-obedient"
regime (where spins follow the true μ values) and a "Mertens-obedient" regime
(where the running sum is forced to stay small)?

The character of that phase transition — sharp or smooth, first-order or
continuous — tells us something about whether the cancellation between
μ(n) = +1 and μ(n) = -1 is structural or accidental.

### Critical Correction: The Mertens Conjecture is False

The strict bound |M(x)| < √x was disproven by Odlyzko and te Riele in 1985.
The first counterexample exists somewhere around 10^(10^40) — far beyond
computational reach, but mathematically certain.

The Riemann Hypothesis is equivalent to the weaker *asymptotic* bound:

    M(x) = O(x^(1/2 + ε))  for any ε > 0

This means the cat DOES eventually knock the glass off the shelf — but only
barely, and it always climbs back. The question is whether deviation *scales*
faster than x^(1/2+ε). Our experiment probes the scaling behavior, not a
strict cutoff.


## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Petri Net Layer                     │
│                                                     │
│  Places: integers 1..N                              │
│  Tokens: μ(n) ∈ {+1, 0, -1} encoded as spin        │
│  Transitions: multiplicative relationships          │
│    - p × q → pq  (prime factorization edges)        │
│    - Firing rules preserve Möbius structure          │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │ maps to
┌──────────────────────▼──────────────────────────────┐
│            Quantum Hamiltonian Layer                  │
│                                                     │
│  Each integer n gets a qubit (or qubit pair)         │
│  μ(n) encoded as:                                   │
│    |0⟩ → μ=0  (repeated prime factor)               │
│    |↑⟩ → μ=+1 (even # distinct prime factors)       │
│    |↓⟩ → μ=-1 (odd # distinct prime factors)        │
│                                                     │
│  Interaction terms from multiplicative structure:    │
│    H = H_local + H_structure + H_penalty + H_mixing │
│                                                     │
│  H_mixing = Γ * Σ X_i  (transverse field)           │
│    → induces quantum fluctuations                   │
│    → makes the problem genuinely quantum             │
│    → without this, H is diagonal / classical         │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │ runs on
┌──────────────────────▼──────────────────────────────┐
│            QAOA / VQE Eigensolver                    │
│                                                     │
│  QAOA preferred: naturally separates classical       │
│    cost (ZZ terms) from quantum mixing (X terms)    │
│  Cost function: deviation scaling M(x)² / x^(1+2ε) │
│  Classical baseline: NumPyMinimumEigensolver        │
│    for N ≤ 30 to validate QPU results               │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │ results
┌──────────────────────▼──────────────────────────────┐
│        Phase Diagram / Analysis / Sonification       │
│                                                     │
│  Sweep parameters: N, λ (penalty), Γ (mixing)       │
│  Ground state energy → minimum cancellation          │
│  Energy gap → structural bound strength             │
│  Phase boundary → where Möbius order breaks down    │
│  Frustration metrics → how primes create disorder   │
│  Sonify the eigenspectrum → the "ringing" test      │
│                                                     │
└─────────────────────────────────────────────────────┘
```


## Phase 1: Encoding the Möbius Function as a Hamiltonian

### The Petri Net Structure

For integers 1 through N, the Petri net has:

- **Places**: One per integer n ∈ {1, ..., N}
- **Tokens**: Spin value representing μ(n)
- **Transitions**: Edges encoding prime factorization
  - For each prime p and integer n where p|n: edge from n/p to n
  - These encode "n is built from n/p by multiplying by p"

### Mapping to Qubits

**Option A: One qubit per integer (simplest)**
- N qubits for integers 1..N
- μ(n) = +1 → |0⟩, μ(n) = -1 → |1⟩, μ(n) = 0 → decouple
- Mertens function M(x) = Σ (count of |0⟩ minus count of |1⟩) up to x
- Pro: Simple. Con: Wastes qubits on μ=0 values.

**Option B: Two qubits per integer (richer encoding)**
- First qubit: "is square-free" flag (|0⟩ = square-free, |1⟩ = not)
- Second qubit: parity of prime factor count
- Pro: Encodes full μ structure. Con: 2N qubits needed.

**Option C: Encode only square-free integers (efficient)**
- Skip integers with repeated prime factors (they contribute 0)
- ~60% of integers are square-free, so significant savings
- Each qubit is +1 or -1, directly mapping to μ
- Pro: Maximizes qubit efficiency. Con: Loses some structural info.

**Recommendation for proof of concept: Option A with N ≤ 60**
- Uses at most 60 qubits (within IBM Torino's 133)
- Simple mapping
- μ=0 values get fixed as |+⟩ (decoupled from optimization)


### The Hamiltonian

```
H = H_mobius + H_structure + H_mertens_penalty + H_transverse

H_mobius (local fields):
  - h_n * Z_n for each qubit
  - h_n encodes the "preferred" spin direction based on μ(n)
  - For μ(n) = 0: strong field pinning qubit to a known state

H_structure (Petri net transitions — antiferromagnetic prime couplings):
  - For each prime p and each pair (n, n*p) both ≤ N:
    J_{n,np} * Z_n ⊗ Z_{np}
  - Coupling strength J encodes multiplicative relationship
  - Key insight: primes CREATE anti-correlation
    (if n has even # factors → μ=+1, then np has odd # → μ=-1)
  - J is NEGATIVE (antiferromagnetic) for prime multiplication edges

H_mertens_penalty (deviation scaling penalty):
  - Penalizes configurations where |M(x)| grows too fast
  - Implemented as: λ * (Σ_n Z_n)² / N^(1+2ε)
  - Uses x^(1/2+ε) scaling, NOT strict √x (Mertens conjecture is false!)
  - The ground state MINIMIZES deviation scaling
  - Excited states MAXIMIZE deviation — worst-case configurations

H_transverse (quantum mixing field):
  - Γ * Σ_n X_n
  - THIS IS CRITICAL: without it, all terms commute and the
    Hamiltonian is diagonal — a classical Ising model, not quantum
  - The transverse field induces quantum fluctuations between
    spin-up and spin-down, creating superpositions of integer states
  - This makes the problem a genuine quantum many-body system
  - Γ is a tunable parameter: the phase diagram lives in (λ, Γ) space
```

### Why the Transverse Field Matters

Without H_transverse, the Hamiltonian is entirely Pauli-Z operators.
All terms commute: [H_i, H_j] = 0. The eigenstates are just classical
bitstrings. You don't need a quantum computer — a classical solver
finds the ground state trivially.

Adding Γ * Σ X_i breaks this commutativity. Now the system explores
quantum superpositions of classical configurations. The ground state
is no longer a single bitstring but an entangled state. THIS is where
quantum hardware earns its keep — exploring the entangled landscape
that classical computers can't efficiently simulate.

The transverse field also enables the key physics: it creates a
competition between quantum fluctuations (Γ) and classical order
(the Z-Z couplings). This competition is what produces phase
transitions, and studying those transitions is the real experiment.


### What We're Actually Studying: A Phase Diagram

This is NOT "testing if RH is true." This is studying the phase
diagram of a novel spin glass whose interaction topology comes from
prime factorization. Three competing forces:

1. **H_structure** wants spins to obey the multiplicative rules
   (antiferromagnetic alignment along prime edges)

2. **H_penalty** wants the total magnetization to stay small
   (Mertens function close to zero)

3. **H_transverse** wants quantum disorder
   (superpositions, fluctuations, entanglement)

The scientifically interesting questions:

- **Phase boundary**: As you increase λ (penalty strength), the system
  transitions from "Möbius-obedient" (spins follow true μ values) to
  "Mertens-obedient" (spins rearrange to minimize running sum). WHERE
  does this transition happen? Is it sharp (first-order) or smooth?

- **Frustration**: The prime multiplication edges will create geometric
  frustration — configurations where not all couplings can be satisfied
  simultaneously. How does the quantum system resolve this frustration?
  This is directly related to why primes create "apparent randomness."

- **Scaling of the energy gap**: As N increases, does the gap between
  ground and first excited state grow, shrink, or stay constant?
  This tells us about the structural robustness of the cancellation.

- **Critical exponents**: If there IS a phase transition, what are its
  critical exponents? Do they match any known universality class?
  A NEW universality class from prime topology would be remarkable.


## Phase 2: Implementation Plan

### New MCP Tools Needed

```python
# New tools to add to the MCP server:

"build_mertens_hamiltonian"
  - Input: N (max integer), J_coupling, lambda_penalty,
           gamma_transverse, epsilon
  - Output: SparsePauliOp encoding the full Hamiltonian
  - This is where the Petri net → Hamiltonian mapping lives

"run_mertens_qaoa"
  - Input: N, num_layers (p), max_iterations, etc.
  - Output: ground state energy, optimal parameters,
            M(x) values at optimal parameters,
            comparison with x^(1/2+ε) scaling

"run_mertens_classical_baseline"
  - Input: N (max ~25 for exact diag)
  - Output: exact eigenvalues via NumPyMinimumEigensolver
  - Purpose: validate QPU results against exact answers

"sweep_phase_diagram"
  - Input: N, lambda_range, gamma_range, grid_resolution
  - Output: ground state energy surface, gap surface,
            phase boundary location

"analyze_mertens_spectrum"
  - Input: N, num_eigenvalues
  - Output: lowest K eigenvalues, energy gaps,
            max |M(x)| for each eigenstate,
            scaling analysis, frustration index

"sonify_mertens_spectrum"
  - Input: eigenvalue data
  - Output: frequency mapping for sonification
            (the "does it ring?" test)
```

### Concrete Hamiltonian Construction (Python pseudocode)

```python
def build_mertens_hamiltonian(N, J_coupling=1.0, lambda_penalty=0.5,
                               gamma_transverse=0.5, epsilon=0.01):
    """
    Build the Transverse-Field Möbius Ising Hamiltonian.

    The Petri net topology becomes interaction terms:
    - Prime multiplication edges → antiferromagnetic ZZ couplings
    - Local fields → pin μ(n)=0 qubits
    - Penalty term → penalize large |M(x)| scaling
    - Transverse field → quantum fluctuations (makes it non-classical)

    Note: Qiskit uses little-endian qubit ordering.
    Qubit 0 is the rightmost character in a Pauli string.
    """
    from qiskit.quantum_info import SparsePauliOp
    import numpy as np

    num_qubits = N
    pauli_terms = []

    # Helper: build Pauli string with Z at given indices
    # Reverses for Qiskit little-endian convention
    def make_z_pauli(z_indices):
        p = ['I'] * num_qubits
        for idx in z_indices:
            p[idx] = 'Z'
        return ''.join(p)[::-1]

    # Helper: build Pauli string with X at given index
    def make_x_pauli(x_index):
        p = ['I'] * num_qubits
        p[x_index] = 'X'
        return ''.join(p)[::-1]

    # 1. Local fields: pin known μ=0 values
    for n in range(1, N+1):
        mu_n = mobius(n)
        if mu_n == 0:
            # Strong local field to pin this qubit
            pauli_terms.append((make_z_pauli([n-1]), 5.0))

    # 2. Multiplicative structure (Petri net transitions)
    primes = [p for p in range(2, N+1) if is_prime(p)]
    for p in primes:
        for n in range(1, N+1):
            np_val = n * p
            if np_val <= N:
                # Antiferromagnetic coupling
                pauli_terms.append((make_z_pauli([n-1, np_val-1]),
                                    -J_coupling))

    # 3. Mertens penalty: λ * (Σ Z_i)² / N^(1+2ε)
    # Expanding (Σ Z_i)² = Σ Z_i² + Σ_{i≠j} Z_i Z_j
    # Z_i² = I (constant, ignore). Keep pairwise ZZ terms.
    scaling_denominator = N ** (1.0 + 2.0 * epsilon)
    for i in range(N):
        for j in range(i+1, N):
            mu_i = mobius(i+1)
            mu_j = mobius(j+1)
            if mu_i != 0 and mu_j != 0:  # only active qubits
                pauli_terms.append((make_z_pauli([i, j]),
                                    lambda_penalty / scaling_denominator))

    # 4. CRITICAL: Transverse field (quantum mixing)
    # Without this, the Hamiltonian is entirely diagonal (classical)
    for n in range(N):
        mu_n = mobius(n+1)
        if mu_n != 0:  # only apply to active qubits
            pauli_terms.append((make_x_pauli(n), gamma_transverse))

    return SparsePauliOp.from_list(pauli_terms).simplify()
```

### Ansatz Design

**Recommended: QAOA (QAOAAnsatz)**

QAOA is the natural choice for this problem because:
- It naturally separates the classical cost function (ZZ terms)
  from quantum mixing (X terms) into alternating layers
- The Hamiltonian already has this structure built in
- Layer count (p) controls the trade-off between accuracy and depth
- Qiskit provides QAOAAnsatz out of the box

**Alternative: Hardware-Efficient Ansatz (HEA)**
- For larger N on real hardware, HEA with connectivity matching
  the Torino/Heron topology may transpile better
- Less physically motivated but more hardware-friendly

**Validation: Classical exact diagonalization**
- For N ≤ 25, use NumPyMinimumEigensolver as ground truth
- Compare QAOA/VQE results against exact eigenvalues
- This catches bugs in the Hamiltonian construction

**Optimizer choice:**
- Simulator: COBYLA (reliable, noise-free)
- Real hardware: SPSA (robust to quantum noise)


## Phase 3: Scaling and Analysis

### Experiment Matrix

| N   | Qubits | Active (μ≠0) | Hardware     | Method          |
|-----|--------|--------------|--------------|-----------------|
| 10  | 10     | 7            | Simulator    | Exact + QAOA    |
| 15  | 15     | 10           | Simulator    | Exact + QAOA    |
| 20  | 20     | 13           | Simulator    | Exact + QAOA    |
| 25  | 25     | 16           | Simulator    | Exact + QAOA    |
| 30  | 30     | 20           | Simulator    | QAOA only       |
| 50  | 50     | 31           | Torino       | QAOA            |
| 60  | 60     | 37           | Torino       | QAOA            |
| 100 | 100    | 61           | Torino       | QAOA            |

### Phase Diagram Sweep Parameters

For each N, sweep:
- **λ** (penalty weight): 0.0 → 2.0 in ~20 steps
- **Γ** (transverse field): 0.0 → 2.0 in ~20 steps
- **ε** (scaling exponent): fixed at 0.01 initially, then vary

This produces a 2D phase diagram for each system size.
Overlay the phase boundaries for different N to see scaling.

### Key Metrics to Track

1. Ground state energy E₀(N, λ, Γ)
2. First excited state E₁(N, λ, Γ)
3. Energy gap Δ(N) = E₁ - E₀
4. Max |M(x)| in ground state vs x^(1/2+ε) scaling
5. Scaling of Δ(N) with N — the money plot
6. Phase boundary location λ_c(Γ) for each N
7. Frustration index: fraction of unsatisfied prime-edge couplings
8. Entanglement entropy of the ground state across a bipartition

### The "Does It Ring?" Test

If we sonify the eigenspectrum:
- **Clear fundamental frequency** → structural bound exists
- **Harmonic overtones** → the bound has internal structure
- **Noise/silence** → no structural bound, cancellation is accidental

The frequency of the fundamental tone ∝ energy gap Δ(N).
If Δ(N) → 0 as N → ∞, the tone drops in pitch until silence.
If Δ(N) → const > 0, the tone stabilizes. The shelf holds.

Note: On real QPU hardware, noise will blur excited state energies.
Run exact diagonalization for small N first to establish the
spectral structure, then verify QPU reproduces it before scaling up.


## Phase 4: What Would Be Novel

Even as a curiosity-driven experiment, the following contributions
don't exist in the literature:

1. **Transverse-field Möbius Ising model** — a quantum spin glass
   whose interaction graph encodes prime factorization topology.
   This specific Hamiltonian construction is new.

2. **Petri net formulation of multiplicative number theory** — mapping
   the Möbius function's structure to a reachability problem in a
   concurrent system. This framing doesn't exist.

3. **Phase diagram of prime topology** — studying the competition
   between multiplicative order, Mertens cancellation, and quantum
   fluctuations. Is there a sharp phase transition? What universality
   class? Nobody has asked this question.

4. **Energy gap scaling as RH probe** — empirical data on whether
   the structural bound strengthens or weakens with N. Even negative
   results (gap closes quickly) would be informative.

5. **Geometric frustration from primes** — primes create a specific
   frustration pattern in the antiferromagnetic couplings. How does
   the system resolve this? This connects number theory to condensed
   matter physics in a concrete way.

6. **Sonification of number-theoretic structure** — the "ringing"
   test as an intuitive probe. Good science communication even if
   the physics results are modest.

Title if you ever want to write it up:
"Spin-Glass Phase Diagram of the Möbius Function:
 Prime Factorization Topology as a Quantum Interaction Graph"


## Getting Started with Claude Code

### Step 1: Add Möbius/Mertens utilities
- `mobius(n)`, `mertens(n)`, `is_prime(n)`, `factorize(n)`
- Test against known values (OEIS A002321 for Mertens)
- Verify: M(1)=1, M(2)=0, M(3)=-1, M(4)=-1, M(5)=-2, ...

### Step 2: Build the Hamiltonian constructor
- Implement `build_mertens_hamiltonian(N)` with all four terms
- Include the transverse field!
- Use Qiskit little-endian convention (qubit 0 = rightmost)
- Verify: print Hamiltonian for N=5, sanity check term count

### Step 3: Classical baseline (N ≤ 25)
- Run NumPyMinimumEigensolver on the Hamiltonian
- Extract exact eigenvalues, energy gap, ground state configuration
- This is your ground truth for validating everything else

### Step 4: Add QAOA/VQE MCP tools
- Wire up Hamiltonian → QAOAAnsatz → optimizer → results
- Start with N=10 on local simulator
- Compare against classical baseline from Step 3

### Step 5: Phase diagram sweep
- Sweep (λ, Γ) for N=10, 15, 20
- Look for phase transitions
- Plot ground state energy landscapes

### Step 6: Scale to hardware
- Run N=50, 60 on IBM Torino
- Compare QAOA results with simulator trends
- Track energy gap scaling

### Step 7: Sonification
- Map eigenspectrum to audio
- Listen for the ringing
- Record the sound of prime numbers

### Step 8 (optional): Write it up
- "Spin-Glass Phase Diagram of the Möbius Function"
- arXiv: quant-ph / math-ph / cond-mat.dis-nn crosslist
- Or just post it on your blog. Whatever. It's your experiment.
