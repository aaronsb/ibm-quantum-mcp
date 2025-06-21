# Qiskit VQE MCP Servers

MCP (Model Context Protocol) servers that enable AI assistants like Claude to perform quantum computing experiments using IBM Quantum hardware and local simulators.

## Overview

This proof-of-concept demonstrates how to make quantum computing accessible through conversational AI. The project provides two specialized MCP servers:

1. **Local Simulation Server** (`mcp_vqe_server_local.py`) - Fast, always-available quantum simulations
2. **Quantum Hardware Server** (`mcp_vqe_server_quantum.py`) - Access to real IBM Quantum processors

## Current Status: Proof of Concept

This is an early-stage implementation where the VQE algorithm code is embedded directly within each MCP server. While functional, this architecture has limitations for scaling. The simulation and hardware servers duplicate significant code, making maintenance and feature additions challenging.

### Architectural Considerations

For production use or expansion beyond VQE, consider:
- **Shared quantum algorithm library** - Extract VQE and other algorithms into a separate module
- **Abstract quantum backend interface** - Allow switching between simulators and hardware without code duplication
- **Plugin architecture** - Enable adding new algorithms without modifying server code
- **Result caching layer** - Store expensive quantum computations for reuse

## Features

- 🧪 Run Variational Quantum Eigensolver (VQE) for molecular energy calculations
- 🔬 Support for H₂ and HeH⁺ molecules (easily extensible)
- 📊 Real-time convergence tracking with matplotlib visualizations
- 🖥️ Access to IBM's latest Heron R2 quantum processors (156 qubits)
- 🛡️ Built-in error mitigation and transpilation optimization
- 📈 Job tracking and result retrieval from IBM Quantum

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install uv
   uv sync
   ```

2. **Set up IBM Quantum credentials (for hardware access):**
   ```bash
   cp .env.example .env
   # Edit .env with your IBM Quantum API key
   ```

3. **Register with Claude:**
   ```bash
   # Local simulation (no credentials needed)
   claude mcp add qiskit-vqe-local "uv" "--directory" "$(pwd)" "run" "python" "mcp_vqe_server_local.py"
   
   # Quantum hardware (requires IBM credentials)
   claude mcp add qiskit-vqe-quantum "uv" "--directory" "$(pwd)" "run" "python" "mcp_vqe_server_quantum.py"
   ```

## Project Structure

```
qiskit/
├── mcp_vqe_server_local.py    # Local quantum simulator
├── mcp_vqe_server_quantum.py  # IBM Quantum hardware interface
├── CLAUDE.md                  # Project context for AI assistants
├── .env.example               # Template for IBM credentials
├── pyproject.toml             # Python dependencies
└── vqe_plots/                 # Generated convergence plots
```

## Example Results

### Local Simulation (Ideal)
- H₂ Energy: -1.857 Hartree
- Error: 0.002 Hartree
- Time: <1 second

### IBM Quantum Hardware (Real)
- H₂ Energy: -1.728 Hartree
- Error: 0.127 Hartree
- Backend: IBM Torino (156 qubits)
- Time: ~30 seconds + queue

## Documentation

For the full story of this project, including technical details and the journey of building it, see the comprehensive report in the TeXFlow project: `quantum_journey_report.pdf`

## Future Enhancements: Quantum Circuit Lego Blocks

The next evolution of this MCP server would be to expose IBM's full suite of quantum primitives, circuits, and operators as modular "Lego blocks" that can be assembled through conversation:

### Proposed Additions

1. **Standard Circuit Library**
   - Quantum Fourier Transform (QFT)
   - Grover's Algorithm components
   - Quantum Phase Estimation
   - QAOA mixers and cost operators
   - Quantum arithmetic circuits

2. **Advanced Primitives**
   - Sampler for probability distributions
   - Extended Estimator options
   - Statevector access for debugging
   - Noise model simulation

3. **Quantum Operators**
   - Pauli operators and combinations
   - Fermionic operators
   - Spin operators
   - Custom Hamiltonian builders

4. **Circuit Building Tools**
   - Gate sequence optimization
   - Entanglement generation patterns
   - State preparation routines
   - Measurement optimization

5. **Utility Functions**
   - Circuit depth analysis
   - Gate count optimization
   - Connectivity mapping
   - Error rate estimation

### Example Future Usage

```
User: "Create a 4-qubit GHZ state and measure correlations"
Claude: [Assembles GHZ circuit, runs on quantum hardware, analyzes results]

User: "Build a quantum adder for 3-bit numbers"
Claude: [Constructs arithmetic circuit using standard components]

User: "Implement Grover's algorithm to search for |101⟩"
Claude: [Builds oracle and diffusion operators, executes search]
```

This modular approach would make quantum computing as accessible as building with blocks, allowing rapid prototyping of quantum algorithms through natural conversation.

## License

MIT License - Feel free to experiment and extend!