#!/usr/bin/env python3
"""
Local VQE MCP Server for Qiskit
Simulation-only version that doesn't require IBM Quantum credentials

This server provides VQE calculations using local statevector simulation.
For quantum hardware access, use mcp_vqe_server_quantum.py
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.server.models import InitializationOptions
import os
from datetime import datetime
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Molecular Hamiltonians
MOLECULAR_HAMILTONIANS = {
    "H2": {
        "description": "Hydrogen molecule (H2) at equilibrium bond length",
        "hamiltonian": SparsePauliOp.from_list([
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156)
        ]),
        "reference_energy": -1.855,
        "num_qubits": 2
    },
    "HeH+": {
        "description": "Helium Hydride ion (HeH+)",
        "hamiltonian": SparsePauliOp.from_list([
            ("II", -1.467),
            ("ZI", 0.681),
            ("IZ", -0.681),
            ("ZZ", 0.112),
            ("XX", 0.427)
        ]),
        "reference_energy": -2.85,
        "num_qubits": 2
    }
}

class LocalVQEServer:
    def __init__(self):
        self.server = Server("qiskit-vqe-local")
        self.setup_handlers()
        self.convergence_history = []
        
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), "vqe_plots")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="list_molecules",
                    description="List available molecular systems with their properties",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_hamiltonian",
                    description="Get the Hamiltonian for a specific molecule",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "molecule": {
                                "type": "string",
                                "enum": list(MOLECULAR_HAMILTONIANS.keys()),
                                "description": "Molecule name"
                            }
                        },
                        "required": ["molecule"]
                    }
                ),
                types.Tool(
                    name="create_ghz_state",
                    description="Create an n-qubit GHZ state circuit",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n_qubits": {
                                "type": "integer",
                                "minimum": 2,
                                "maximum": 127,
                                "description": "Number of qubits for GHZ state"
                            }
                        },
                        "required": ["n_qubits"]
                    }
                ),
                types.Tool(
                    name="run_vqe",
                    description="Run VQE simulation to find ground state energy of a molecule",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "molecule": {
                                "type": "string",
                                "enum": list(MOLECULAR_HAMILTONIANS.keys()),
                                "description": "Molecule to solve"
                            },
                            "ansatz": {
                                "type": "string",
                                "enum": ["RealAmplitudes", "EfficientSU2"],
                                "default": "RealAmplitudes",
                                "description": "Variational ansatz to use"
                            },
                            "max_iterations": {
                                "type": "integer",
                                "default": 100,
                                "description": "Maximum optimizer iterations"
                            }
                        },
                        "required": ["molecule"]
                    }
                ),
                types.Tool(
                    name="get_convergence_plot",
                    description="Get convergence plot of the last VQE run",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "return_base64": {
                                "type": "boolean",
                                "default": False,
                                "description": "Return base64 encoded image (may be large)"
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            if name == "list_molecules":
                return await self.list_molecules()
            elif name == "get_hamiltonian":
                return await self.get_hamiltonian(arguments)
            elif name == "create_ghz_state":
                return await self.create_ghz_state(arguments)
            elif name == "run_vqe":
                return await self.run_vqe(arguments)
            elif name == "get_convergence_plot":
                return await self.get_convergence_plot(arguments)
            else:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def list_molecules(self) -> List[types.TextContent]:
        molecules_info = []
        for name, data in MOLECULAR_HAMILTONIANS.items():
            molecules_info.append({
                "name": name,
                "description": data["description"],
                "num_qubits": data["num_qubits"],
                "reference_energy": data["reference_energy"],
                "num_terms": len(data["hamiltonian"])
            })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "molecules": molecules_info,
                "backend": "local_statevector_simulator"
            }, indent=2)
        )]
    
    async def get_hamiltonian(self, args: Dict[str, Any]) -> List[types.TextContent]:
        molecule = args["molecule"]
        mol_data = MOLECULAR_HAMILTONIANS[molecule]
        
        # Extract Pauli terms
        terms = []
        for pauli, coeff in mol_data["hamiltonian"].to_list():
            terms.append({
                "pauli_string": pauli,
                "coefficient": float(np.real(coeff))
            })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "molecule": molecule,
                "description": mol_data["description"],
                "num_qubits": mol_data["num_qubits"],
                "reference_energy": mol_data["reference_energy"],
                "hamiltonian_terms": terms
            }, indent=2)
        )]
    
    async def create_ghz_state(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Create an n-qubit GHZ state circuit"""
        n = args["n_qubits"]
        
        if n < 2:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "GHZ state requires at least 2 qubits"
                })
            )]
        
        # Create GHZ circuit
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "circuit": {
                    "num_qubits": n,
                    "depth": qc.depth(),
                    "gates": len(qc),
                    "description": f"{n}-qubit GHZ state"
                }
            }, indent=2)
        )]
    
    async def run_vqe(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            molecule = args["molecule"]
            ansatz_type = args.get("ansatz", "RealAmplitudes")
            max_iterations = args.get("max_iterations", 100)
            
            # Get molecular data
            mol_data = MOLECULAR_HAMILTONIANS[molecule]
            hamiltonian = mol_data["hamiltonian"]
            num_qubits = mol_data["num_qubits"]
            
            # Create ansatz
            if ansatz_type == "RealAmplitudes":
                ansatz = RealAmplitudes(num_qubits, reps=3)
            else:  # EfficientSU2
                ansatz = EfficientSU2(num_qubits, reps=3)
            
            # Reset convergence tracking
            self.convergence_history = []
            
            # Create estimator
            estimator = StatevectorEstimator()
            
            # Define cost function
            def cost_func(params):
                # Create the ansatz circuit with parameters
                qc = ansatz.assign_parameters(params)
                
                # Evaluate expectation value
                job = estimator.run([(qc, hamiltonian)])
                result = job.result()
                energy = result[0].data.evs
                
                # Track convergence
                self.convergence_history.append({
                    "iteration": len(self.convergence_history),
                    "energy": float(energy)
                })
                
                return energy
            
            # Initial parameters
            np.random.seed(42)
            initial_params = np.random.random(ansatz.num_parameters) * 2 * np.pi
            
            # Run optimization
            result = minimize(
                cost_func,
                initial_params,
                method='COBYLA',
                options={'maxiter': max_iterations}
            )
            
            # Get final energy
            ground_state_energy = float(result.fun)
            reference_energy = mol_data["reference_energy"]
            error = abs(ground_state_energy - reference_energy)
            relative_error = error / abs(reference_energy) * 100
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "molecule": molecule,
                    "ansatz": ansatz_type,
                    "backend": "local_statevector_simulator",
                    "ground_state_energy": ground_state_energy,
                    "reference_energy": reference_energy,
                    "absolute_error": error,
                    "relative_error_percent": relative_error,
                    "num_parameters": ansatz.num_parameters,
                    "num_iterations": len(self.convergence_history),
                    "optimal_parameters": list(result.x),
                    "optimization_success": bool(result.success),
                    "chemical_accuracy": error < 0.0016  # 1 kcal/mol ≈ 0.0016 Hartree
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__
                })
            )]
    
    async def get_convergence_plot(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            if not self.convergence_history:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": "No VQE run data available. Run VQE first."
                    })
                )]
            
            # Create convergence plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            iterations = [d["iteration"] for d in self.convergence_history]
            energies = [d["energy"] for d in self.convergence_history]
            
            ax.plot(iterations, energies, 'b-', linewidth=2, label='VQE Energy')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Energy (Hartree)', fontsize=12)
            ax.set_title('VQE Convergence (Local Simulation)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"vqe_convergence_local_{timestamp}.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            response = {
                "status": "success",
                "plot_file": plot_path,
                "final_energy": energies[-1] if energies else None,
                "initial_energy": energies[0] if energies else None,
                "improvement": energies[0] - energies[-1] if len(energies) > 1 else 0
            }
            
            if args.get("return_base64", False):
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                response["plot"] = f"data:image/png;base64,{img_base64}"
            
            plt.close(fig)
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            )]
    
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            initialization_options = InitializationOptions(
                server_name="qiskit-vqe-local",
                server_version="1.0.0",
                capabilities={
                    "tools": {}
                }
            )
            await self.server.run(read_stream, write_stream, initialization_options)

def main():
    """Main entry point for the MCP server"""
    server = LocalVQEServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()