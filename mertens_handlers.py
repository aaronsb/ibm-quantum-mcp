"""
Mertens Spin Glass MCP tool handlers.

Extracted from mcp_vqe_server_local.py to keep the server file focused on VQE.
Provides a mixin class with tool definitions, dispatch, and all handler methods.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mcp.types as types

from mertens_utils import (
    mertens_table, build_mertens_hamiltonian, build_cost_operator,
    extract_ground_state_info, compute_frustration_index,
)


def mertens_tool_definitions() -> List[types.Tool]:
    """Return the list of Mertens spin glass MCP tool definitions."""
    return [
        types.Tool(
            name="get_mertens_info",
            description="Inspect Mobius/Mertens number theory structure: values, prime edges, and statistics for integers 1..N",
            inputSchema={
                "type": "object",
                "properties": {
                    "N": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 100,
                        "description": "Maximum integer to analyze"
                    }
                },
                "required": ["N"]
            }
        ),
        types.Tool(
            name="build_mertens_hamiltonian",
            description="Build the Transverse-Field Mobius Ising Hamiltonian encoding prime factorization topology as a quantum spin system. Caches the result for subsequent solver calls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "N": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 60,
                        "description": "Maximum integer (number of qubits)"
                    },
                    "J_coupling": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Antiferromagnetic coupling strength for prime edges"
                    },
                    "lambda_penalty": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Mertens deviation penalty weight"
                    },
                    "gamma_transverse": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Transverse field strength (quantum mixing). 0 = classical."
                    },
                    "epsilon": {
                        "type": "number",
                        "default": 0.01,
                        "description": "Scaling exponent for Mertens penalty: x^(1/2+epsilon)"
                    }
                },
                "required": ["N"]
            }
        ),
        types.Tool(
            name="run_mertens_exact",
            description="Exact diagonalization of the Mertens Hamiltonian via sparse eigensolver. Returns eigenvalues, energy gap, ground state analysis, and frustration index. Limited to N<=20.",
            inputSchema={
                "type": "object",
                "properties": {
                    "N": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 20,
                        "description": "Maximum integer (number of qubits). Max 20 for exact diag."
                    },
                    "num_eigenvalues": {
                        "type": "integer",
                        "default": 6,
                        "minimum": 2,
                        "maximum": 20,
                        "description": "Number of lowest eigenvalues to compute"
                    },
                    "J_coupling": {"type": "number", "default": 1.0},
                    "lambda_penalty": {"type": "number", "default": 0.5},
                    "gamma_transverse": {"type": "number", "default": 0.5},
                    "epsilon": {"type": "number", "default": 0.01}
                },
                "required": ["N"]
            }
        ),
        types.Tool(
            name="run_mertens_qaoa",
            description="Run QAOA optimization on the Mertens Hamiltonian using QAOAAnsatz. Compares against exact diag for small N.",
            inputSchema={
                "type": "object",
                "properties": {
                    "N": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 30,
                        "description": "Maximum integer (number of qubits)"
                    },
                    "num_layers": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 10,
                        "description": "QAOA layers (p parameter)"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 100,
                        "description": "Maximum optimizer iterations"
                    },
                    "optimizer": {
                        "type": "string",
                        "enum": ["COBYLA", "SPSA"],
                        "default": "SPSA",
                        "description": "Optimizer: SPSA (robust, stochastic) or COBYLA (derivative-free)"
                    },
                    "J_coupling": {"type": "number", "default": 1.0},
                    "lambda_penalty": {"type": "number", "default": 0.5},
                    "gamma_transverse": {"type": "number", "default": 0.5},
                    "epsilon": {"type": "number", "default": 0.01}
                },
                "required": ["N"]
            }
        ),
        types.Tool(
            name="sweep_mertens_phase",
            description="Sweep (lambda, gamma) parameter space to map the phase diagram of the Mertens spin glass. Uses exact diag at each grid point. Generates heatmap plots.",
            inputSchema={
                "type": "object",
                "properties": {
                    "N": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 18,
                        "description": "Max integer. Keep <=12 for interactive use, <=18 for longer runs."
                    },
                    "lambda_min": {"type": "number", "default": 0.0},
                    "lambda_max": {"type": "number", "default": 2.0},
                    "gamma_min": {"type": "number", "default": 0.0},
                    "gamma_max": {"type": "number", "default": 2.0},
                    "grid_points": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 3,
                        "maximum": 30,
                        "description": "Points per axis (total runs = grid_points^2)"
                    },
                    "J_coupling": {"type": "number", "default": 1.0},
                    "epsilon": {"type": "number", "default": 0.01}
                },
                "required": ["N"]
            }
        ),
        types.Tool(
            name="get_mertens_plot",
            description="Generate plots from the most recent Mertens spin glass results: spectrum, phase diagram, or Mobius structure visualization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "enum": ["spectrum", "phase_diagram", "mobius_structure", "convergence"],
                        "description": "Type of plot to generate"
                    },
                    "return_base64": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["plot_type"]
            }
        ),
    ]


MERTENS_TOOL_NAMES = {
    "get_mertens_info",
    "build_mertens_hamiltonian",
    "run_mertens_exact",
    "run_mertens_qaoa",
    "sweep_mertens_phase",
    "get_mertens_plot",
}


class MertensHandlersMixin:
    """Mixin providing Mertens spin glass tool handlers for the MCP server.

    The host class must set self.output_dir before using these handlers.
    Call init_mertens_state() from __init__.
    """

    def init_mertens_state(self):
        self.mertens_hamiltonian = None
        self.mertens_metadata = None
        self.mertens_exact_results = None
        self.mertens_convergence_history = []
        self.mertens_sweep_results = None

    async def dispatch_mertens_tool(self, name: str, arguments: Dict[str, Any]) -> Optional[List[types.TextContent]]:
        """Dispatch a Mertens tool call. Returns None if name is not a Mertens tool."""
        dispatch = {
            "get_mertens_info": self.get_mertens_info,
            "build_mertens_hamiltonian": self.build_mertens_hamiltonian_tool,
            "run_mertens_exact": self.run_mertens_exact,
            "run_mertens_qaoa": self.run_mertens_qaoa,
            "sweep_mertens_phase": self.sweep_mertens_phase,
            "get_mertens_plot": self.get_mertens_plot,
        }
        handler = dispatch.get(name)
        if handler is None:
            return None
        return await handler(arguments)

    async def get_mertens_info(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            N = args["N"]
            table = mertens_table(N)

            max_dev = max(abs(v) for v in table["mertens"].values())
            sqrt_N = np.sqrt(N)
            M_N = table["mertens"][N]

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "N": N,
                    "encoding": "square-free (Option C)",
                    "mobius_values": table["mobius"],
                    "mertens_values": table["mertens"],
                    "squarefree_integers": table["squarefree"],
                    "num_qubits": table["num_qubits"],
                    "total_integers": N,
                    "qubit_savings_pct": round((1 - table["num_qubits"] / N) * 100, 1),
                    "primes": table["primes"],
                    "prime_edges": table["prime_edges_sqfree"],
                    "num_prime_edges": len(table["prime_edges_sqfree"]),
                    "M_N": M_N,
                    "max_mertens_deviation": max_dev,
                    "sqrt_N": float(sqrt_N),
                    "deviation_ratio": float(max_dev / sqrt_N),
                    "note": "deviation_ratio < 1 means |M(x)| < sqrt(x) holds in this range"
                }, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def build_mertens_hamiltonian_tool(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            N = args["N"]
            J = args.get("J_coupling", 1.0)
            lam = args.get("lambda_penalty", 0.5)
            gamma = args.get("gamma_transverse", 0.5)
            eps = args.get("epsilon", 0.01)

            H, meta = build_mertens_hamiltonian(N, J, lam, gamma, eps)
            self.mertens_hamiltonian = H
            self.mertens_metadata = meta

            h_list = H.to_list()
            preview = [
                {"pauli": str(p), "coeff": float(np.real(c))}
                for p, c in h_list[:20]
            ]

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "encoding": meta["encoding"],
                    "num_qubits": meta["num_qubits"],
                    "active_qubits": meta["active_qubits"],
                    "term_counts": meta["term_counts"],
                    "parameters": meta["parameters"],
                    "petri_net": {
                        "num_places": meta["petri_net"]["num_places"],
                        "num_squarefree": meta["petri_net"]["num_squarefree"],
                        "num_transitions": meta["petri_net"]["num_transitions"],
                        "primes": meta["petri_net"]["primes"],
                        "prime_edges": meta["petri_net"]["prime_edges"],
                    },
                    "hamiltonian_preview": preview,
                    "is_quantum": gamma > 0,
                    "cached": True,
                }, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def run_mertens_exact(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            N = args["N"]
            k = args.get("num_eigenvalues", 6)
            J = args.get("J_coupling", 1.0)
            lam = args.get("lambda_penalty", 0.5)
            gamma = args.get("gamma_transverse", 0.5)
            eps = args.get("epsilon", 0.01)

            if N > 20:
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "error",
                    "message": f"N={N} too large for exact diag. Max N=20. Use run_mertens_qaoa instead."
                }))]

            H, meta = build_mertens_hamiltonian(N, J, lam, gamma, eps)
            self.mertens_hamiltonian = H
            self.mertens_metadata = meta

            mat = H.to_matrix(sparse=True)
            max_k = min(k, mat.shape[0] - 2)
            vals, vecs = eigsh(mat, k=max_k, which='SA')

            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]

            ground_info = extract_ground_state_info(vecs[:, 0], N)
            frust = compute_frustration_index(vecs[:, 0], meta["petri_net"]["prime_edges"], N)

            self.mertens_exact_results = {
                "eigenvalues": vals.tolist(),
                "eigenvectors": vecs,
                "N": N,
                "parameters": meta["parameters"],
            }

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "N": N,
                    "num_qubits": N,
                    "active_qubits": meta["active_qubits"],
                    "eigenvalues": vals.tolist(),
                    "energy_gap": float(vals[1] - vals[0]),
                    "ground_state_energy": float(vals[0]),
                    "ground_state": ground_info,
                    "frustration_index": frust,
                    "parameters": meta["parameters"],
                    "matrix_dimension": mat.shape[0],
                }, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e), "type": type(e).__name__}))]

    async def run_mertens_qaoa(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            from qiskit.circuit.library import QAOAAnsatz

            N = args["N"]
            num_layers = args.get("num_layers", 2)
            max_iter = args.get("max_iterations", 100)
            optimizer_name = args.get("optimizer", "SPSA")
            J = args.get("J_coupling", 1.0)
            lam = args.get("lambda_penalty", 0.5)
            gamma = args.get("gamma_transverse", 0.5)
            eps = args.get("epsilon", 0.01)

            H_full, meta = build_mertens_hamiltonian(N, J, lam, gamma, eps)
            self.mertens_hamiltonian = H_full
            self.mertens_metadata = meta

            cost_op = build_cost_operator(N, J, lam, eps)

            ansatz_raw = QAOAAnsatz(cost_operator=cost_op, reps=num_layers)
            # Decompose evolved-operator gates to native gates for fast statevector sim
            ansatz = ansatz_raw.decompose().decompose().decompose()

            estimator = StatevectorEstimator()
            self.mertens_convergence_history = []

            def cost_func(params):
                qc = ansatz.assign_parameters(params)
                job = estimator.run([(qc, H_full)])
                energy = float(job.result()[0].data.evs)
                self.mertens_convergence_history.append({
                    "iteration": len(self.mertens_convergence_history),
                    "energy": energy,
                })
                return energy

            np.random.seed(42)
            initial_params = np.random.random(ansatz.num_parameters) * 2 * np.pi

            if optimizer_name == "SPSA":
                from qiskit_algorithms.optimizers import SPSA
                spsa = SPSA(maxiter=max_iter)
                spsa_result = spsa.minimize(cost_func, initial_params)
                qaoa_energy = float(spsa_result.fun)
                optimal_params = spsa_result.x
            else:
                result = minimize(
                    cost_func, initial_params,
                    method='COBYLA',
                    options={'maxiter': max_iter},
                )
                qaoa_energy = float(result.fun)
                optimal_params = result.x

            exact_energy = None
            if N <= 20:
                mat = H_full.to_matrix(sparse=True)
                exact_vals, _ = eigsh(mat, k=2, which='SA')
                exact_energy = float(min(exact_vals))

            response = {
                "status": "success",
                "N": N,
                "qaoa_energy": qaoa_energy,
                "exact_energy": exact_energy,
                "energy_error": abs(qaoa_energy - exact_energy) if exact_energy is not None else None,
                "num_layers": num_layers,
                "num_parameters": ansatz.num_parameters,
                "num_iterations": len(self.mertens_convergence_history),
                "optimizer": optimizer_name,
                "parameters": meta["parameters"],
            }

            return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e), "type": type(e).__name__}))]

    async def sweep_mertens_phase(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            N = args["N"]
            lam_min = args.get("lambda_min", 0.0)
            lam_max = args.get("lambda_max", 2.0)
            gam_min = args.get("gamma_min", 0.0)
            gam_max = args.get("gamma_max", 2.0)
            grid = args.get("grid_points", 10)
            J = args.get("J_coupling", 1.0)
            eps = args.get("epsilon", 0.01)

            if N > 18:
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "error", "message": f"N={N} too large for sweep. Max 18."
                }))]

            lambdas = np.linspace(lam_min, lam_max, grid)
            gammas = np.linspace(gam_min, gam_max, grid)

            energies = np.zeros((grid, grid))
            gaps = np.zeros((grid, grid))
            deviations = np.zeros((grid, grid))
            frustrations = np.zeros((grid, grid))

            total = grid * grid
            for i, lam in enumerate(lambdas):
                for j, gam in enumerate(gammas):
                    H, meta = build_mertens_hamiltonian(N, J, lam, gam, eps)
                    mat = H.to_matrix(sparse=True)
                    dim = mat.shape[0]
                    k = min(2, dim - 2)
                    vals, vecs = eigsh(mat, k=k, which='SA')
                    order = np.argsort(vals)
                    vals = vals[order]
                    vecs = vecs[:, order]

                    energies[i, j] = vals[0]
                    gaps[i, j] = vals[1] - vals[0] if len(vals) > 1 else 0

                    info = extract_ground_state_info(vecs[:, 0], N)
                    if info["top_configurations"]:
                        deviations[i, j] = info["top_configurations"][0]["max_mertens_deviation"]

                    frustrations[i, j] = compute_frustration_index(
                        vecs[:, 0], meta["petri_net"]["prime_edges"], N
                    )

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Mertens Spin Glass Phase Diagram (N={N})', fontsize=16)

            data_labels = [
                (energies, 'Ground State Energy', 'RdYlBu_r'),
                (gaps, 'Energy Gap (E1 - E0)', 'viridis'),
                (deviations, 'Max |M(x)| Deviation', 'hot'),
                (frustrations, 'Frustration Index', 'coolwarm'),
            ]

            for ax, (data, label, cmap) in zip(axes.flat, data_labels):
                im = ax.imshow(
                    data, origin='lower', aspect='auto', cmap=cmap,
                    extent=[gam_min, gam_max, lam_min, lam_max],
                )
                ax.set_xlabel('Gamma (transverse field)', fontsize=11)
                ax.set_ylabel('Lambda (Mertens penalty)', fontsize=11)
                ax.set_title(label, fontsize=12)
                plt.colorbar(im, ax=ax)

            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.output_dir, f"mertens_phase_N{N}_{timestamp}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.mertens_sweep_results = {
                "N": N, "lambdas": lambdas.tolist(), "gammas": gammas.tolist(),
                "energies": energies.tolist(), "gaps": gaps.tolist(),
                "deviations": deviations.tolist(), "frustrations": frustrations.tolist(),
                "plot_file": plot_path,
            }

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "N": N,
                    "grid_size": f"{grid}x{grid}",
                    "total_runs": total,
                    "lambda_range": [lam_min, lam_max],
                    "gamma_range": [gam_min, gam_max],
                    "energy_range": [float(energies.min()), float(energies.max())],
                    "gap_range": [float(gaps.min()), float(gaps.max())],
                    "deviation_range": [float(deviations.min()), float(deviations.max())],
                    "frustration_range": [float(frustrations.min()), float(frustrations.max())],
                    "plot_file": plot_path,
                }, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e), "type": type(e).__name__}))]

    async def get_mertens_plot(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            plot_type = args["plot_type"]
            return_b64 = args.get("return_base64", False)
            fig = None

            if plot_type == "spectrum":
                if not self.mertens_exact_results:
                    return [types.TextContent(type="text", text=json.dumps({
                        "status": "error", "message": "No exact diag results. Run run_mertens_exact first."
                    }))]
                vals = self.mertens_exact_results["eigenvalues"]
                N = self.mertens_exact_results["N"]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(vals)), vals, color='steelblue', alpha=0.8)
                ax.set_xlabel('Eigenvalue Index', fontsize=12)
                ax.set_ylabel('Energy', fontsize=12)
                ax.set_title(f'Mertens Spin Glass Eigenspectrum (N={N})', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                if len(vals) > 1:
                    ax.axhline(y=vals[0], color='red', linestyle='--', alpha=0.5, label=f'E0 = {vals[0]:.4f}')
                    ax.legend()

            elif plot_type == "phase_diagram":
                if not self.mertens_sweep_results:
                    return [types.TextContent(type="text", text=json.dumps({
                        "status": "error", "message": "No sweep results. Run sweep_mertens_phase first."
                    }))]
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "success",
                    "plot_file": self.mertens_sweep_results["plot_file"],
                    "plot_type": "phase_diagram",
                }))]

            elif plot_type == "convergence":
                if not self.mertens_convergence_history:
                    return [types.TextContent(type="text", text=json.dumps({
                        "status": "error", "message": "No QAOA convergence data. Run run_mertens_qaoa first."
                    }))]
                iters = [d["iteration"] for d in self.mertens_convergence_history]
                energies = [d["energy"] for d in self.mertens_convergence_history]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(iters, energies, 'b-', linewidth=2)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Energy', fontsize=12)
                ax.set_title('Mertens QAOA Convergence', fontsize=14)
                ax.grid(True, alpha=0.3)

            elif plot_type == "mobius_structure":
                if not self.mertens_metadata:
                    return [types.TextContent(type="text", text=json.dumps({
                        "status": "error", "message": "No Hamiltonian built. Run build_mertens_hamiltonian first."
                    }))]
                meta = self.mertens_metadata
                table = meta["table"]
                N = meta["parameters"]["N"]
                edges = meta["petri_net"]["prime_edges"]
                sqfree = table["squarefree"]

                fig, ax = plt.subplots(figsize=(12, 8))

                cols = int(np.ceil(np.sqrt(len(sqfree))))
                positions = {}
                for i, n in enumerate(sqfree):
                    row, col = divmod(i, cols)
                    positions[n] = (col, -row)

                prime_colors = {}
                cmap = plt.cm.Set1
                for i, p in enumerate(table["primes"]):
                    prime_colors[p] = cmap(i / max(len(table["primes"]), 1))

                for n, np_val in edges:
                    p = np_val // n
                    color = prime_colors.get(p, 'gray')
                    x = [positions[n][0], positions[np_val][0]]
                    y = [positions[n][1], positions[np_val][1]]
                    ax.plot(x, y, '-', color=color, alpha=0.3, linewidth=1)

                for n in sqfree:
                    mu = table["mobius"][n]
                    color = 'blue' if mu == 1 else 'red'
                    ax.scatter(*positions[n], c=color, s=200, zorder=5, edgecolors='black', linewidth=0.5)
                    ax.annotate(str(n), positions[n], ha='center', va='center', fontsize=7, fontweight='bold',
                                color='white')

                ax.set_title(f'Mobius Structure / Petri Net (N={N}, {len(sqfree)} square-free qubits)', fontsize=14)
                ax.set_aspect('equal')
                ax.axis('off')

                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='mu=+1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='mu=-1'),
                ]
                for p in table["primes"][:6]:
                    legend_elements.append(
                        Line2D([0], [0], color=prime_colors[p], linewidth=2, label=f'x{p}')
                    )
                ax.legend(handles=legend_elements, loc='upper right')

            else:
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "error", "message": f"Unknown plot type: {plot_type}"
                }))]

            if fig is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(self.output_dir, f"mertens_{plot_type}_{timestamp}.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')

                response = {"status": "success", "plot_file": plot_path, "plot_type": plot_type}

                if return_b64:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    response["plot"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

                plt.close(fig)
                return [types.TextContent(type="text", text=json.dumps(response))]

        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e), "type": type(e).__name__}))]
