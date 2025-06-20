#!/usr/bin/env python3
"""
Quantum Hardware VQE MCP Server for Qiskit
Requires IBM Quantum credentials to be configured

This server provides VQE calculations on real quantum hardware.
For local simulation, use mcp_vqe_server_local.py

Environment variables:
- IBM_QUANTUM_TOKEN: Your IBM Quantum API token (REQUIRED)
- IBM_QUANTUM_CHANNEL: Channel type (default: ibm_cloud)
- IBM_QUANTUM_INSTANCE: Instance string/CRN (REQUIRED)
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
from dotenv import load_dotenv
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required IBM Quantum imports
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService, 
        EstimatorV2 as RuntimeEstimator,
        EstimatorOptions,
        Session
    )
except ImportError:
    logger.error("qiskit-ibm-runtime is required. Install with: pip install qiskit-ibm-runtime")
    sys.exit(1)

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

class QuantumVQEServer:
    def __init__(self):
        self.server = Server("qiskit-vqe-quantum")
        self.convergence_history = []
        self.job_history = {}
        
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), "vqe_plots")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate and setup IBM Quantum - exit if not configured
        self.ibm_service = self._setup_ibm_quantum()
        if self.ibm_service:
            self.setup_handlers()
        else:
            logger.error("IBM Quantum service initialization failed. Server will not start.")
            sys.exit(1)
    
    def _setup_ibm_quantum(self):
        """Setup IBM Quantum service - returns None if not properly configured"""
        token = os.environ.get('IBM_QUANTUM_TOKEN')
        if not token:
            logger.error("IBM_QUANTUM_TOKEN not found. Please set your IBM Quantum API token.")
            return None
        
        instance = os.environ.get('IBM_QUANTUM_INSTANCE')
        if not instance:
            logger.error("IBM_QUANTUM_INSTANCE not found. Please set your IBM Quantum instance/CRN.")
            return None
        
        try:
            channel = os.environ.get('IBM_QUANTUM_CHANNEL', 'ibm_cloud')
            
            # Initialize service
            service_kwargs = {
                "channel": channel,
                "token": token,
                "instance": instance
            }
            
            service = QiskitRuntimeService(**service_kwargs)
            logger.info(f"IBM Quantum service initialized successfully")
            logger.info(f"Channel: {channel}")
            logger.info(f"Instance: {instance[:20]}..." if len(instance) > 20 else f"Instance: {instance}")
            
            # Test connection by listing backends
            backends = service.backends()
            logger.info(f"Found {len(backends)} available backends")
            
            return service
            
        except Exception as e:
            logger.error(f"Failed to initialize IBM Quantum service: {e}")
            return None
    
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
                    name="run_vqe_quantum",
                    description="Run VQE on IBM Quantum hardware",
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
                                "default": 20,
                                "description": "Maximum optimizer iterations (limited for cost)"
                            },
                            "shots": {
                                "type": "integer",
                                "default": 1024,
                                "description": "Number of shots"
                            },
                            "optimization_level": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 3,
                                "default": 1,
                                "description": "Transpiler optimization level"
                            },
                            "resilience_level": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 2,
                                "default": 1,
                                "description": "Error mitigation level"
                            },
                            "dynamical_decoupling": {
                                "type": "boolean",
                                "default": True,
                                "description": "Enable dynamical decoupling"
                            },
                            "backend_name": {
                                "type": "string",
                                "description": "Specific backend name (optional, uses least busy if not specified)"
                            },
                            "use_session": {
                                "type": "boolean",
                                "default": False,
                                "description": "Use session mode (requires paid plan)"
                            }
                        },
                        "required": ["molecule"]
                    }
                ),
                types.Tool(
                    name="list_quantum_backends",
                    description="List available IBM Quantum backends",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "min_qubits": {
                                "type": "integer",
                                "default": 2,
                                "description": "Minimum number of qubits required"
                            },
                            "simulator": {
                                "type": "boolean",
                                "default": False,
                                "description": "Include simulators"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_backend_info",
                    description="Get detailed information about a quantum backend",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "backend_name": {
                                "type": "string",
                                "description": "Name of the backend"
                            }
                        },
                        "required": ["backend_name"]
                    }
                ),
                types.Tool(
                    name="get_job_result",
                    description="Retrieve results from a previously submitted job",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "Job ID to retrieve"
                            }
                        },
                        "required": ["job_id"]
                    }
                ),
                types.Tool(
                    name="list_jobs",
                    description="List recent quantum jobs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "default": 10,
                                "description": "Maximum number of jobs to return"
                            },
                            "pending": {
                                "type": "boolean",
                                "default": False,
                                "description": "Only show pending jobs"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_job_status",
                    description="Get detailed status of a specific job",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "Job ID to check"
                            }
                        },
                        "required": ["job_id"]
                    }
                ),
                types.Tool(
                    name="get_convergence_plot",
                    description="Get convergence plot of the last VQE run",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            if name == "list_molecules":
                return await self.list_molecules()
            elif name == "get_hamiltonian":
                return await self.get_hamiltonian(arguments)
            elif name == "run_vqe_quantum":
                return await self.run_vqe_quantum(arguments)
            elif name == "list_quantum_backends":
                return await self.list_quantum_backends(arguments)
            elif name == "get_backend_info":
                return await self.get_backend_info(arguments)
            elif name == "get_job_result":
                return await self.get_job_result(arguments)
            elif name == "list_jobs":
                return await self.list_jobs(arguments)
            elif name == "get_job_status":
                return await self.get_job_status(arguments)
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
                "backend": "IBM Quantum Hardware"
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
    
    async def run_vqe_quantum(self, args: Dict[str, Any]) -> List[types.TextContent]:
        try:
            molecule = args["molecule"]
            ansatz_type = args.get("ansatz", "RealAmplitudes")
            max_iterations = args.get("max_iterations", 20)
            shots = args.get("shots", 1024)
            optimization_level = args.get("optimization_level", 1)
            resilience_level = args.get("resilience_level", 1)
            dynamical_decoupling = args.get("dynamical_decoupling", True)
            backend_name = args.get("backend_name")
            use_session = args.get("use_session", False)
            
            # Get molecular data
            mol_data = MOLECULAR_HAMILTONIANS[molecule]
            hamiltonian = mol_data["hamiltonian"]
            num_qubits = mol_data["num_qubits"]
            
            # Create ansatz
            if ansatz_type == "RealAmplitudes":
                ansatz = RealAmplitudes(num_qubits, reps=3)
            else:  # EfficientSU2
                ansatz = EfficientSU2(num_qubits, reps=3)
            
            # Get backend
            if backend_name:
                backend = self.ibm_service.backend(backend_name)
            else:
                backend = self.ibm_service.least_busy(
                    operational=True, 
                    simulator=False,
                    min_num_qubits=num_qubits
                )
            backend_name = backend.name
            
            logger.info(f"Using backend: {backend_name}")
            
            # Transpile circuit
            pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
            isa_circuit = pm.run(ansatz)
            isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
            
            # Configure error mitigation
            options = EstimatorOptions()
            options.resilience_level = resilience_level
            options.default_shots = shots
            
            if dynamical_decoupling and resilience_level > 0:
                options.dynamical_decoupling.enable = True
                options.dynamical_decoupling.sequence_type = "XY4"
            
            # Reset convergence tracking
            self.convergence_history = []
            
            # Define cost function
            def cost_func(params, estimator):
                qc = isa_circuit.assign_parameters(params)
                job = estimator.run([(qc, isa_hamiltonian)])
                result = job.result()
                
                energy = result[0].data.evs
                if hasattr(energy, '__len__'):
                    energy = energy[0]
                
                self.convergence_history.append({
                    "iteration": len(self.convergence_history),
                    "energy": float(energy),
                    "backend": backend_name
                })
                
                logger.info(f"Iteration {len(self.convergence_history)}: Energy = {energy}")
                
                return float(energy)
            
            # Initial parameters
            np.random.seed(42)
            initial_params = np.random.random(ansatz.num_parameters) * 2 * np.pi
            
            # Run optimization with or without session
            if use_session:
                logger.info("Using session mode (requires paid plan)")
                with Session(backend=backend) as session:
                    estimator = RuntimeEstimator(session=session, options=options)
                    result = minimize(
                        lambda params: cost_func(params, estimator),
                        initial_params,
                        method='COBYLA',
                        options={'maxiter': max_iterations}
                    )
            else:
                logger.info("Using sessionless mode (free tier)")
                estimator = RuntimeEstimator(mode=backend, options=options)
                result = minimize(
                    lambda params: cost_func(params, estimator),
                    initial_params,
                    method='COBYLA',
                    options={'maxiter': max_iterations}
                )
            
            # Calculate results
            ground_state_energy = float(result.fun)
            reference_energy = mol_data["reference_energy"]
            error = abs(ground_state_energy - reference_energy)
            relative_error = error / abs(reference_energy) * 100
            
            # Generate job ID
            job_id = f"vqe_{backend_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            response = {
                "status": "success",
                "job_id": job_id,
                "molecule": molecule,
                "ansatz": ansatz_type,
                "backend": backend_name,
                "ground_state_energy": ground_state_energy,
                "reference_energy": reference_energy,
                "absolute_error": error,
                "relative_error_percent": relative_error,
                "num_parameters": ansatz.num_parameters,
                "num_iterations": len(self.convergence_history),
                "optimal_parameters": list(result.x),
                "optimization_success": bool(result.success),
                "chemical_accuracy": error < 0.0016,
                "shots": shots,
                "resilience_level": resilience_level,
                "transpilation_info": {
                    "original_gates": len(ansatz),
                    "transpiled_gates": len(isa_circuit),
                    "optimization_level": optimization_level
                }
            }
            
            # Store for later retrieval
            self.job_history[job_id] = response
            
            return [types.TextContent(
                type="text",
                text=json.dumps(response, indent=2)
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
    
    async def list_quantum_backends(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """List available IBM Quantum backends"""
        try:
            min_qubits = args.get("min_qubits", 2)
            include_simulator = args.get("simulator", False)
            
            backends = self.ibm_service.backends(
                filters=lambda x: (
                    x.configuration().n_qubits >= min_qubits and 
                    x.status().operational and
                    (include_simulator or not x.configuration().simulator)
                )
            )
            
            backend_info = []
            for backend in backends:
                config = backend.configuration()
                status = backend.status()
                backend_info.append({
                    "name": backend.name,
                    "n_qubits": config.n_qubits,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "simulator": config.simulator,
                    "basis_gates": config.basis_gates[:5] + ["..."],
                    "max_shots": config.max_shots
                })
            
            # Sort by pending jobs (least busy first)
            backend_info.sort(key=lambda x: x["pending_jobs"])
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "backends": backend_info,
                    "recommended": backend_info[0]["name"] if backend_info else None,
                    "total_backends": len(backend_info)
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            )]
    
    async def get_backend_info(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get detailed information about a specific backend"""
        try:
            backend_name = args["backend_name"]
            backend = self.ibm_service.backend(backend_name)
            
            config = backend.configuration()
            status = backend.status()
            
            info = {
                "name": backend.name,
                "version": config.backend_version,
                "n_qubits": config.n_qubits,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "simulator": config.simulator,
                "basis_gates": config.basis_gates,
                "max_shots": config.max_shots,
                "max_experiments": config.max_experiments,
                "description": config.description
            }
            
            if hasattr(config, 'coupling_map') and config.coupling_map:
                info["coupling_map_size"] = len(config.coupling_map)
                info["coupling_map_sample"] = config.coupling_map[:10]
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "backend_info": info
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            )]
    
    async def get_job_result(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Retrieve job results from IBM Quantum"""
        job_id = args["job_id"]
        
        # First check local history
        if job_id in self.job_history:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "source": "local_cache",
                    "job_result": self.job_history[job_id]
                }, indent=2)
            )]
        
        # Otherwise query IBM Quantum
        try:
            job = self.ibm_service.job(job_id)
            
            # Get job info
            job_info = {
                "job_id": job.job_id(),
                "status": str(job.status()),
                "creation_date": job.creation_date.isoformat() if job.creation_date else None,
                "backend": job.backend().name if job.backend() else None,
            }
            
            # Get queue info if pending
            if str(job.status()) in ['INITIALIZING', 'QUEUED', 'VALIDATING']:
                try:
                    queue_info = job.queue_info()
                    if queue_info:
                        job_info["queue_position"] = queue_info.position
                        job_info["estimated_completion"] = queue_info.estimated_complete_time.isoformat() if hasattr(queue_info, 'estimated_complete_time') else None
                except:
                    pass
            
            # Get results if completed
            if job.done():
                try:
                    result = job.result()
                    # Extract energy values from EstimatorV2 result
                    # EstimatorV2 returns PrimitiveResult with indexed access
                    if hasattr(result, '__getitem__'):
                        # For EstimatorV2, results are accessed via result[0]
                        pub_result = result[0]
                        if hasattr(pub_result, 'data'):
                            data = pub_result.data
                            job_info["result"] = {
                                "energy": float(data.evs) if hasattr(data, 'evs') else None,
                                "std_error": float(data.stds) if hasattr(data, 'stds') else None,
                                "ensemble_std_error": float(data.ensemble_standard_error) if hasattr(data, 'ensemble_standard_error') else None,
                                "metadata": pub_result.metadata if hasattr(pub_result, 'metadata') else None
                            }
                            # Add quantum-specific info
                            if job_info["result"]["metadata"]:
                                job_info["shots"] = job_info["result"]["metadata"].get("shots")
                                job_info["num_randomizations"] = job_info["result"]["metadata"].get("num_randomizations")
                    elif hasattr(result, 'values'):
                        # Legacy format
                        job_info["result"] = {
                            "values": list(result.values),
                            "metadata": result.metadata if hasattr(result, 'metadata') else None
                        }
                    else:
                        # Try to extract any available data
                        job_info["result"] = {
                            "raw_type": str(type(result)),
                            "attributes": [attr for attr in dir(result) if not attr.startswith('_')]
                        }
                except Exception as e:
                    job_info["result_error"] = str(e)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "source": "ibm_quantum",
                    "job_info": job_info
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            )]
    
    async def list_jobs(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """List recent jobs from IBM Quantum"""
        try:
            limit = args.get("limit", 10)
            pending_only = args.get("pending", False)
            
            # Get jobs from IBM Quantum
            jobs = self.ibm_service.jobs(limit=limit)
            
            job_list = []
            for job in jobs:
                try:
                    job_info = {
                        "job_id": job.job_id(),
                        "status": str(job.status()),
                        "creation_date": job.creation_date.isoformat() if job.creation_date else None,
                        "backend": job.backend().name if job.backend() else "Unknown",
                        "program_id": job.primitive_id if hasattr(job, 'primitive_id') else None
                    }
                    
                    # Filter by pending if requested
                    if pending_only and job_info["status"] not in ['INITIALIZING', 'QUEUED', 'VALIDATING']:
                        continue
                        
                    job_list.append(job_info)
                except Exception as e:
                    logger.warning(f"Error processing job: {e}")
                    continue
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "total_jobs": len(job_list),
                    "jobs": job_list
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            )]
    
    async def get_job_status(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get detailed status of a specific job"""
        try:
            job_id = args["job_id"]
            job = self.ibm_service.job(job_id)
            
            # Comprehensive job status
            status_info = {
                "job_id": job.job_id(),
                "status": str(job.status()),
                "creation_date": job.creation_date.isoformat() if job.creation_date else None,
                "backend": job.backend().name if job.backend() else None,
                "program": job.primitive_id if hasattr(job, 'primitive_id') else None,
            }
            
            # Add detailed status info
            if hasattr(job, 'error_message') and job.error_message():
                status_info["error_message"] = job.error_message()
            
            # Queue information
            if str(job.status()) in ['INITIALIZING', 'QUEUED', 'VALIDATING']:
                try:
                    queue_info = job.queue_info()
                    if queue_info:
                        status_info["queue"] = {
                            "position": queue_info.position if hasattr(queue_info, 'position') else None,
                            "estimated_start_time": queue_info.estimated_start_time.isoformat() if hasattr(queue_info, 'estimated_start_time') and queue_info.estimated_start_time else None,
                            "estimated_completion_time": queue_info.estimated_complete_time.isoformat() if hasattr(queue_info, 'estimated_complete_time') and queue_info.estimated_complete_time else None
                        }
                except Exception as e:
                    logger.debug(f"Could not get queue info: {e}")
            
            # Usage information
            if hasattr(job, 'usage'):
                try:
                    usage = job.usage()
                    if usage:
                        status_info["usage"] = {
                            "quantum_seconds": usage.get('quantum_seconds', 0),
                            "quantum_shots": usage.get('quantum_shots', 0)
                        }
                except:
                    pass
            
            # Check if results are available
            if job.done():
                status_info["results_available"] = True
                try:
                    result = job.result()
                    status_info["result_preview"] = "Results available - use get_job_result for full data"
                except Exception as e:
                    status_info["result_error"] = str(e)
            else:
                status_info["results_available"] = False
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "job_status": status_info
                }, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": str(e)
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
            backend_name = self.convergence_history[0].get('backend', 'Unknown')
            
            ax.plot(iterations, energies, 'b-', linewidth=2, label='VQE Energy')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Energy (Hartree)', fontsize=12)
            ax.set_title(f'VQE Convergence on {backend_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"vqe_convergence_quantum_{timestamp}.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            plt.close(fig)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "plot_file": plot_path,
                    "backend": backend_name,
                    "final_energy": energies[-1] if energies else None,
                    "initial_energy": energies[0] if energies else None,
                    "improvement": energies[0] - energies[-1] if len(energies) > 1 else 0
                })
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
                server_name="qiskit-vqe-quantum",
                server_version="1.0.0",
                capabilities={
                    "tools": {}
                }
            )
            await self.server.run(read_stream, write_stream, initialization_options)

def main():
    """Main entry point for the MCP server"""
    server = QuantumVQEServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()