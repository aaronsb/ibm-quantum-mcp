# Qiskit VQE MCP Servers

MCP (Model Context Protocol) servers that enable AI assistants like Claude to perform quantum computing experiments using IBM Quantum hardware and local simulators.

## Overview

This project provides a conversational interface to quantum computing through two specialized MCP servers:

1. **Local Simulation Server** (`mcp_vqe_server_local.py`) - Fast, always-available quantum simulations
2. **Quantum Hardware Server** (`mcp_vqe_server_quantum.py`) - Access to real IBM Quantum processors

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

## License

MIT License - Feel free to experiment and extend!