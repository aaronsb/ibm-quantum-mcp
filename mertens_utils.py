"""
Number theory utilities and Hamiltonian construction for the
Transverse-Field Möbius Ising Model (Mertens Spin Glass).

Encodes the multiplicative structure of integers as a quantum spin system
whose interaction graph comes from prime factorization (Petri net topology).

Encoding: Option C (square-free only). Each qubit represents a square-free
integer n with μ(n) ∈ {+1, -1}. Integers with μ(n)=0 are excluded entirely —
they contribute nothing to M(x) and waste qubits.

Convention: qubit in |0⟩ → spin +1 → μ=+1, qubit in |1⟩ → spin -1 → μ=-1.
"""

from qiskit.quantum_info import SparsePauliOp
import numpy as np


# ---------------------------------------------------------------------------
# Number theory primitives
# ---------------------------------------------------------------------------

def factorize(n: int) -> list[int]:
    """Return prime factors of n with multiplicity."""
    if n < 2:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def is_squarefree(n: int) -> bool:
    """True if n has no repeated prime factors."""
    if n < 1:
        return False
    factors = factorize(n)
    return len(factors) == len(set(factors))


def mobius(n: int) -> int:
    """Möbius function μ(n): +1, -1, or 0."""
    if n < 1:
        raise ValueError(f"mobius undefined for n={n}")
    if n == 1:
        return 1
    factors = factorize(n)
    if len(factors) != len(set(factors)):
        return 0  # repeated prime factor
    return (-1) ** len(factors)


def mertens(n: int) -> int:
    """Mertens function M(n) = Σ_{k=1}^{n} μ(k)."""
    return sum(mobius(k) for k in range(1, n + 1))


def primes_up_to(n: int) -> list[int]:
    """All primes up to n."""
    return [p for p in range(2, n + 1) if is_prime(p)]


def mertens_table(N: int) -> dict:
    """Precompute all Möbius/Mertens data for 1..N.

    Returns dict with:
        mobius: {n: μ(n)} for all n in 1..N
        mertens: {n: M(n)} for all n in 1..N
        squarefree: sorted list of square-free integers in 1..N
        sqfree_to_qubit: {n: qubit_index} mapping for square-free n
        qubit_to_sqfree: {qubit_index: n} reverse mapping
        num_qubits: number of qubits (= number of square-free integers)
        prime_edges_sqfree: [(n, n*p)] where both n and n*p are square-free and ≤ N
        primes: list of primes up to N
    """
    mob = {n: mobius(n) for n in range(1, N + 1)}
    mert = {}
    running = 0
    for n in range(1, N + 1):
        running += mob[n]
        mert[n] = running

    sqfree = sorted(n for n in range(1, N + 1) if mob[n] != 0)
    sqfree_to_qubit = {n: i for i, n in enumerate(sqfree)}
    qubit_to_sqfree = {i: n for i, n in enumerate(sqfree)}
    primes = primes_up_to(N)

    # Prime edges: only between square-free integers
    # n*p is square-free iff p does not divide n (since n is already square-free)
    edges = []
    for p in primes:
        for n in sqfree:
            np_val = n * p
            if np_val <= N and np_val in sqfree_to_qubit:
                edges.append((n, np_val))

    return {
        "mobius": mob,
        "mertens": mert,
        "squarefree": sqfree,
        "sqfree_to_qubit": sqfree_to_qubit,
        "qubit_to_sqfree": qubit_to_sqfree,
        "num_qubits": len(sqfree),
        "prime_edges_sqfree": edges,
        "primes": primes,
    }


# ---------------------------------------------------------------------------
# Validation against known values (OEIS A008683, A002321)
# ---------------------------------------------------------------------------

_KNOWN_MOBIUS = {
    1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0, 9: 0, 10: 1,
    11: -1, 12: 0, 13: -1, 14: 1, 15: 1, 16: 0, 17: -1, 18: 0, 19: -1, 20: 0,
    21: 1, 22: 1, 23: -1, 24: 0, 25: 0, 26: 1, 27: 0, 28: 0, 29: -1, 30: -1,
}

_KNOWN_MERTENS = {
    1: 1, 2: 0, 3: -1, 4: -1, 5: -2, 6: -1, 7: -2, 8: -2, 9: -2, 10: -1,
    11: -2, 12: -2, 13: -3, 14: -2, 15: -1, 16: -1, 17: -2, 18: -2, 19: -3,
    20: -3, 21: -2, 22: -1, 23: -2, 24: -2, 25: -2, 26: -1, 27: -1, 28: -1,
    29: -2, 30: -3,
}


def validate_number_theory(N: int = 30) -> bool:
    """Validate mobius/mertens against known OEIS values. Raises on failure."""
    for n in range(1, min(N, 30) + 1):
        mu = mobius(n)
        if mu != _KNOWN_MOBIUS[n]:
            raise AssertionError(
                f"mobius({n}) = {mu}, expected {_KNOWN_MOBIUS[n]}"
            )
        m = mertens(n)
        if m != _KNOWN_MERTENS[n]:
            raise AssertionError(
                f"mertens({n}) = {m}, expected {_KNOWN_MERTENS[n]}"
            )
    return True


# Run validation at import time
validate_number_theory()


# ---------------------------------------------------------------------------
# Hamiltonian construction (square-free encoding, corrected signs)
# ---------------------------------------------------------------------------

def _pauli_z_string(num_qubits: int, z_indices: list[int]) -> str:
    """Build Pauli string with Z at given qubit indices (Qiskit little-endian)."""
    p = ['I'] * num_qubits
    for idx in z_indices:
        p[idx] = 'Z'
    return ''.join(p)[::-1]  # reverse for little-endian


def _pauli_x_string(num_qubits: int, x_index: int) -> str:
    """Build Pauli string with X at given qubit index (Qiskit little-endian)."""
    p = ['I'] * num_qubits
    p[x_index] = 'X'
    return ''.join(p)[::-1]


def build_mertens_hamiltonian(
    N: int,
    J_coupling: float = 1.0,
    lambda_penalty: float = 0.5,
    gamma_transverse: float = 0.5,
    epsilon: float = 0.01,
) -> tuple[SparsePauliOp, dict]:
    """Build the Transverse-Field Möbius Ising Hamiltonian.

    Square-free encoding (Option C): only square-free integers get qubits.
    No pinning fields needed — μ=0 integers are simply absent.

    Coupling signs:
      H_structure uses +J (antiferromagnetic): Z_n Z_{np} with J > 0.
      With |0⟩→+1, |1⟩→-1: same spins give +J (penalized), opposite give -J (rewarded).
      Since prime multiplication always flips parity (μ sign), the true Möbius
      assignment is an unfrustrated ground state of H_structure alone.

    Returns:
        (hamiltonian, metadata) where metadata includes term counts,
        qubit mapping, and the Petri net structure.
    """
    table = mertens_table(N)
    sq = table["sqfree_to_qubit"]
    edges = table["prime_edges_sqfree"]
    num_qubits = table["num_qubits"]

    terms_structure = []
    terms_penalty = []
    terms_transverse = []

    # 1. Multiplicative structure (Petri net transitions)
    #    Antiferromagnetic: +J * Z_n Z_{np} (positive J penalizes aligned spins)
    for n, np_val in edges:
        qi = sq[n]
        qj = sq[np_val]
        terms_structure.append((
            _pauli_z_string(num_qubits, [qi, qj]),
            abs(J_coupling),
        ))

    # 2. Mertens penalty: λ * (Σ Z_i)² / N^(1+2ε)
    #    All qubits are active (square-free encoding), so all pairs contribute.
    #    Z_i² = I (constant, dropped). Keep pairwise ZZ terms.
    scaling_denom = N ** (1.0 + 2.0 * epsilon)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            terms_penalty.append((
                _pauli_z_string(num_qubits, [i, j]),
                lambda_penalty / scaling_denom,
            ))

    # 3. Transverse field: Γ * Σ X_i on all qubits
    #    CRITICAL: without this the Hamiltonian is diagonal (classical)
    for i in range(num_qubits):
        terms_transverse.append((_pauli_x_string(num_qubits, i), gamma_transverse))

    all_terms = terms_structure + terms_penalty + terms_transverse
    if not all_terms:
        # Edge case: N=1 has 1 qubit, no edges, no penalty pairs
        all_terms = [(_pauli_z_string(num_qubits, []), 0.0)]

    hamiltonian = SparsePauliOp.from_list(all_terms).simplify()

    metadata = {
        "num_qubits": num_qubits,
        "active_qubits": num_qubits,  # all qubits are active in Option C
        "encoding": "square-free (Option C)",
        "sqfree_to_qubit": sq,
        "qubit_to_sqfree": table["qubit_to_sqfree"],
        "term_counts": {
            "structure": len(terms_structure),
            "penalty": len(terms_penalty),
            "transverse": len(terms_transverse),
            "total_raw": len(all_terms),
            "total_simplified": len(hamiltonian),
        },
        "petri_net": {
            "num_places": N,
            "num_squarefree": num_qubits,
            "num_transitions": len(edges),
            "prime_edges": edges,
            "primes": table["primes"],
        },
        "parameters": {
            "N": N,
            "J_coupling": J_coupling,
            "lambda_penalty": lambda_penalty,
            "gamma_transverse": gamma_transverse,
            "epsilon": epsilon,
        },
        "table": table,
    }

    return hamiltonian, metadata


def build_cost_operator(
    N: int,
    J_coupling: float = 1.0,
    lambda_penalty: float = 0.5,
    epsilon: float = 0.01,
) -> SparsePauliOp:
    """Build Z-only cost operator for QAOA (no transverse field).

    This is the classical part of the Hamiltonian — the cost function
    that QAOA's alternating layers optimize against.
    """
    hamiltonian, _ = build_mertens_hamiltonian(
        N, J_coupling, lambda_penalty,
        gamma_transverse=0.0,  # no X terms
        epsilon=epsilon,
    )
    return hamiltonian


def extract_ground_state_info(eigvec: np.ndarray, N: int) -> dict:
    """Decode an eigenvector to extract implied Möbius assignments.

    Uses square-free encoding: each qubit maps to a specific square-free integer.
    The eigenvector lives in a 2^(num_qubits) dimensional Hilbert space.
    """
    table = mertens_table(N)
    q2n = table["qubit_to_sqfree"]
    num_qubits = table["num_qubits"]
    true_mob = table["mobius"]
    probs = np.abs(eigvec) ** 2

    # Top configurations by probability
    top_indices = np.argsort(probs)[::-1][:5]
    top_configs = []
    for idx in top_indices:
        if probs[idx] < 1e-10:
            break
        bits = format(idx, f'0{num_qubits}b')

        # Decode: qubit i → square-free integer n, bit 0 → μ=+1, bit 1 → μ=-1
        implied_mobius = {}
        for qi in range(num_qubits):
            bit = int(bits[num_qubits - 1 - qi])  # little-endian decode
            n = q2n[qi]
            implied_mobius[n] = 1 if bit == 0 else -1

        # Fill in μ=0 for non-square-free integers
        full_mobius = {}
        for n in range(1, N + 1):
            if true_mob[n] == 0:
                full_mobius[n] = 0
            else:
                full_mobius[n] = implied_mobius[n]

        # Compute implied Mertens running sum
        implied_mertens = {}
        running = 0
        for n in range(1, N + 1):
            running += full_mobius[n]
            implied_mertens[n] = running

        max_dev = max(abs(v) for v in implied_mertens.values())

        # Count matches with true Möbius values
        matches = sum(1 for n in implied_mobius if implied_mobius[n] == true_mob[n])

        top_configs.append({
            "bitstring": bits,
            "probability": float(probs[idx]),
            "implied_mobius": {str(k): v for k, v in implied_mobius.items()},
            "implied_mertens_final": implied_mertens[N],
            "max_mertens_deviation": float(max_dev),
            "mobius_matches": matches,
            "mobius_total": len(implied_mobius),
        })

    return {
        "top_configurations": top_configs,
        "dominant_probability": float(probs[top_indices[0]]),
        "is_classical": float(probs[top_indices[0]]) > 0.95,
    }


def _popcount_array(indices: np.ndarray, nq: int) -> np.ndarray:
    """Compute popcount (number of set bits) for each element in indices.

    Uses byte-level lookup for efficiency on large arrays.
    """
    # Byte-level popcount lookup table
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    count = np.zeros(len(indices), dtype=np.int32)
    remaining = indices.copy()
    for _ in range((nq + 7) // 8):
        count += lut[(remaining & 0xFF).astype(np.int64)]
        remaining >>= 8
    return count


def build_diagonal_components(
    N: int,
    J_coupling: float = 1.0,
    epsilon: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Precompute diagonal vectors for H_structure and H_penalty at Gamma=0.

    At Gamma=0 the Hamiltonian is diagonal in the computational basis:
        H(lambda) = H_structure + lambda * H_penalty

    Returns (structure_diag, penalty_diag, table) where each diagonal is a
    numpy array of length 2^num_qubits. For a given lambda, the full diagonal
    is structure_diag + lambda * penalty_diag, and the ground state is argmin.

    The penalty term uses the magnetization identity:
        Σ_{i<j} Z_i Z_j = ((Σ Z_i)² - n) / 2 = (m² - n) / 2
    where m = n - 2·popcount(b). This is O(2^n · n) instead of O(2^n · n²).
    """
    table = mertens_table(N)
    sq = table["sqfree_to_qubit"]
    edges = table["prime_edges_sqfree"]
    nq = table["num_qubits"]
    dim = 1 << nq

    indices = np.arange(dim, dtype=np.int64)
    structure_diag = np.zeros(dim, dtype=np.float64)
    scaling = N ** (1.0 + 2.0 * epsilon)

    # Structure: +|J| * Z_i Z_j for each prime edge (antiferromagnetic)
    for n, np_val in edges:
        qi = sq[n]
        qj = sq[np_val]
        bit_i = (indices >> qi) & 1
        bit_j = (indices >> qj) & 1
        zz = 1 - 2 * (bit_i ^ bit_j)  # +1 if same spin, -1 if opposite
        structure_diag += abs(J_coupling) * zz

    # Penalty: Σ_{i<j} Z_i Z_j / scaling = (m² - n) / (2 · scaling)
    # where m = n - 2·popcount(b) is the total magnetization
    pc = _popcount_array(indices, nq)
    magnetization = nq - 2 * pc  # m(b) = n_qubits - 2·popcount(b)
    penalty_diag = (magnetization.astype(np.float64) ** 2 - nq) / (2.0 * scaling)

    return structure_diag, penalty_diag, table


def frustration_from_bitindex(bit_index: int, prime_edges: list, N: int) -> float:
    """Compute frustration index directly from a basis state index.

    Faster than compute_frustration_index when you already know the dominant
    basis state (e.g. from argmin of a diagonal Hamiltonian at Gamma=0).
    """
    table = mertens_table(N)
    sq = table["sqfree_to_qubit"]

    if not prime_edges:
        return 0.0

    unsatisfied = 0
    for n, np_val in prime_edges:
        qi = sq[n]
        qj = sq[np_val]
        spin_n = (bit_index >> qi) & 1
        spin_np = (bit_index >> qj) & 1
        if spin_n == spin_np:
            unsatisfied += 1

    return unsatisfied / len(prime_edges)


def compute_frustration_index(eigvec: np.ndarray, prime_edges: list, N: int) -> float:
    """Fraction of prime edges unsatisfied in the ground state.

    For the dominant configuration, check if each prime edge (n, np)
    has antiferromagnetic alignment (opposite spins).
    """
    table = mertens_table(N)
    sq = table["sqfree_to_qubit"]
    num_qubits = table["num_qubits"]

    probs = np.abs(eigvec) ** 2
    dominant_idx = np.argmax(probs)
    bits = format(dominant_idx, f'0{num_qubits}b')

    if not prime_edges:
        return 0.0

    unsatisfied = 0
    for n, np_val in prime_edges:
        qi = sq[n]
        qj = sq[np_val]
        # Little-endian: qubit i is bit position i from the right
        spin_n = int(bits[num_qubits - 1 - qi])
        spin_np = int(bits[num_qubits - 1 - qj])
        # Antiferromagnetic wants opposite spins (different bits)
        if spin_n == spin_np:
            unsatisfied += 1

    return unsatisfied / len(prime_edges)
