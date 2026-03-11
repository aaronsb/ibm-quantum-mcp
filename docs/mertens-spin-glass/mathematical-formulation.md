# Mathematical Formulation of the Transverse-Field Möbius Ising Model

This document provides the formal mathematical construction of the Hamiltonian,
the derivation of the critical penalty strength $\lambda_c$, and the empirical correction
function $f$. For narrative context and physical interpretation, see [article.md](article.md).

## Encoding

The square-free integers up to $N$ form our computational basis. Define the index set:

$$S_N = \\{n \leq N : \mu(n) \neq 0\\}$$

Each $n \in S_N$ gets a qubit. The Hilbert space is $\mathcal{H} = (\mathbb{C}^2)^{\otimes \|S_N\|}$. The Pauli operators $Z_n$ and $X_n$ act on the qubit corresponding to integer $n$.

The Möbius function assigns each qubit a preferred spin:

$$\mu(n) = +1 \enspace \textrm{(even number of distinct prime factors)}, \quad -1 \enspace \textrm{(odd number)}$$

The Mertens function is the running sum:

$$M(N) = \sum_{n=1}^{N} \mu(n) = \sum_{n \in S_N} \mu(n)$$

The Riemann Hypothesis is equivalent to: for any $\epsilon > 0$, there exists $C_\epsilon$ such that $\|M(x)\| \leq C_\epsilon \, x^{1/2+\epsilon}$ for all $x$.

## The Hamiltonian

$$H = H_\textrm{structure} + H_\textrm{penalty} + H_\textrm{transverse}$$

### Term 1 — Antiferromagnetic prime couplings

$$H_\textrm{structure} = J \sum_{(n,\, np) \in E} Z_n \, Z_{np}$$

where $E$ is the set of prime edges: pairs $(n, np)$ with $p$ prime and both $n, np \in S_N$. The coupling $J > 0$ is antiferromagnetic.

Since $Z_n Z_{np}$ has eigenvalue $+1$ when both spins agree and $-1$ when they disagree, this term penalizes agreement. Multiplication by a prime always flips the parity of the prime factor count, so $\mu(n)$ and $\mu(np)$ always have opposite signs. Therefore the true Möbius assignment gives $Z_n Z_{np} = -1$ on every edge — every coupling is satisfied, and:

$$E_\textrm{structure}^\textrm{(ground)} = -J \cdot \|E\|$$

The interaction graph is bipartite. Partition $A = \\{n \in S_N : \mu(n) = +1\\}$ and $B = \\{n \in S_N : \mu(n) = -1\\}$. Every prime edge connects $A$ to $B$.

### Term 2 — Mertens penalty (all-to-all antiferromagnet)

$$H_\textrm{penalty} = \frac{\lambda}{N^{1+2\epsilon}} \biggl(\sum_{n \in S_N} Z_n\biggr)^2$$

Expanding the square and using $Z_n^2 = I$:

$$\biggl(\sum_n Z_n\biggr)^2 = \|S_N\| \cdot I + 2\sum_{i < j} Z_i Z_j$$

Dropping the constant:

$$H_\textrm{penalty} = \frac{2\lambda}{N^{1+2\epsilon}} \sum_{i < j} Z_i Z_j$$

In the Möbius ground state, $\sum_n Z_n = \|A\| - \|B\| = M(N)$, so:

$$E_\textrm{penalty}^\textrm{(Möbius)} = \frac{\lambda \, M(N)^2}{N^{1+2\epsilon}}$$

This is zero when $M(N) = 0$, small when $\|M(N)\| = 1$, and increasingly costly as $\|M(N)\|$ grows.

### Term 3 — Transverse field (quantum mixing)

$$H_\textrm{transverse} = \Gamma \sum_{n \in S_N} X_n$$

$X_n$ flips qubit $n$ between $\|0\rangle$ and $\|1\rangle$. Because $[Z_i Z_j, X_k] \neq 0$ when $k \in \\{i,j\\}$, the transverse field does not commute with the other terms. The eigenstates are no longer classical bitstrings but entangled superpositions. This makes the problem a genuine quantum many-body system.

## The Phase Transition

At $\lambda = 0$, the ground state is the true Möbius assignment (every prime edge satisfied, $M(N)$ unconstrained).

As $\lambda$ increases, the penalty pressures the system to reduce $\|M(N)\|$. To do so, it must flip spins, which breaks prime edges at cost $2J$ per edge.

The cheapest spins to flip are the "leaves" — large primes $p \approx N$ that connect to only one other node (the integer 1 via $1 \times p = p$, or similar). Flipping one leaf changes $M$ by 2 and costs $2J$.

The transition occurs when the penalty reduction from flipping equals the structure cost:

$$\frac{\lambda_c}{N^{1+2\epsilon}} \cdot 2\|M(N)\| \cdot 2 = 2J$$

Solving:

$$\lambda_c = \frac{2J \cdot N^{1+2\epsilon}}{\|M(N)\|}$$

## The Cooperative Correction

The true transition includes cooperative multi-spin rearrangements not captured by the single-flip argument. Empirically:

$$\lambda_c^\textrm{(measured)} = f(\|M\|) \cdot \frac{2J \cdot N^{1+2\epsilon}}{\|M(N)\|}$$

where $f$ is a correction factor that depends only on $\|M(N)\|$:

| \|M(N)\| | f(\|M\|) | Data points | Qubit range | N range |
|-----------|----------|-------------|-------------|---------|
| 0 | no transition | 2 | 26 | 39–40 |
| 1 | no transition | 12 | 5–27 | 6–41 |
| 2 | 1.005 | 17 | 4–30 | 5–46 |
| 3 | 0.764 | 10 | 9–31 | 13–50 |
| 4 | 0.683 | 2 | 20 | 31–32 |

Every value of $f$ is exact to three decimal places across all system sizes tested. No $N$-dependence. No drift. No scatter. These appear to be universal constants of the prime factorization graph.

## Connection to the Riemann Hypothesis

Rearranging the formula:

$$\|M(N)\| = f(\|M\|) \cdot \frac{2J \cdot N^{1+2\epsilon}}{\lambda_c}$$

If $\lambda_c$ can be measured (via the phase transition) at large $N$, the growth rate of $\|M(N)\|$ can be extracted directly. The Riemann Hypothesis constrains this growth to $O(N^{1/2+\epsilon})$, which would require:

$$\lambda_c = \Omega(N^{1/2+\epsilon})$$

That is: if RH is true, the phase boundary must grow at least as fast as $\sqrt{N}$. If the boundary grows slower, the Mertens function is escaping its bound. The phase diagram *is* the number theory.
