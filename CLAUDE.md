# Qiskit VQE MCP Server Project

## Project Overview

This project implements Model Context Protocol (MCP) servers for Variational Quantum Eigensolver (VQE) calculations using Qiskit. The architecture uses **two sibling MCP servers** to separate local simulation from quantum hardware execution, ensuring reliability and proper credential management.

## Architecture Decision: Two Separate MCP Servers

### 1. **Local Simulation Server** (`mcp_vqe_server_local.py`)
- Always works without credentials
- Uses Qiskit's StatevectorEstimator for exact simulation
- Perfect for development, testing, and educational use
- No API keys or configuration required

### 2. **Quantum Hardware Server** (`mcp_vqe_server_quantum.py`)
- Requires IBM Quantum credentials
- Validates configuration on startup (fails gracefully if not configured)
- Provides access to real quantum devices
- Includes error mitigation and transpilation optimization

## Key Learnings & Best Practices

### MCP Server Development

1. **JSON Schema Validation**
   - Use Python boolean (`True/False`) not JSON (`true/false`) in inputSchema
   - This was a critical bug that prevented tool discovery

2. **Environment Configuration**
   - Use `python-dotenv` for loading `.env` files
   - Handle optional parameters gracefully (e.g., IBM instance)
   - Validate credentials early and fail fast with clear error messages

3. **Tool Naming Convention**
   - Local server: `run_vqe` (simple, indicates simulation)
   - Quantum server: `run_vqe_quantum` (explicit about hardware usage)

### IBM Quantum Platform Integration

1. **Authentication Variants**
   - **IBM Cloud**: Uses CRN format with `channel="ibm_cloud"`
   - **IBM Quantum Platform**: Uses hub/group/project format with `channel="ibm_quantum"`
   - API keys are channel-specific

2. **Instance Configuration**
   ```python
   # Only add instance if provided (some accounts don't have instances)
   service_kwargs = {
       "channel": channel,
       "token": token
   }
   if instance:
       service_kwargs["instance"] = instance
   ```

3. **Error Handling**
   - Check for "No instances associated with this account" error
   - Validate service initialization before exposing tools
   - Provide clear error messages for missing credentials

### Qiskit 2.x Compatibility

1. **API Changes**
   - Use `EstimatorV2` instead of legacy estimators
   - Results accessed via `result[0].data.evs` (not indexed)
   - `qiskit_algorithms` package has compatibility issues with Qiskit 2.x

2. **Transpilation Best Practice**
   ```python
   pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
   isa_circuit = pm.run(circuit)
   isa_observable = observable.apply_layout(isa_circuit.layout)
   ```

3. **Primitive Unified Blocks (PUBs)**
   ```python
   # Correct format for EstimatorV2
   job = estimator.run([(circuit, observable)])
   ```

## Configuration

### Environment Variables (.env)
```bash
# IBM Quantum Configuration
IBM_QUANTUM_TOKEN=your_api_key_here
IBM_QUANTUM_CHANNEL=ibm_cloud  # or ibm_quantum
IBM_QUANTUM_INSTANCE=crn:v1:bluemix:public:...  # or hub/group/project
```

### MCP Registration
```bash
# Register local simulation server
claude mcp add qiskit-vqe-local "uv" "--" "--directory" "/path/to/project" "run" "python" "mcp_vqe_server_local.py"

# Register quantum hardware server (only if credentials configured)
claude mcp add qiskit-vqe-quantum "uv" "--" "--directory" "/path/to/project" "run" "python" "mcp_vqe_server_quantum.py"
```

## Working Features

### Local Simulation
- ✅ VQE optimization with COBYLA
- ✅ RealAmplitudes and EfficientSU2 ansätze
- ✅ Convergence tracking and plotting
- ✅ H2 and HeH+ molecules
- ✅ GHZ state preparation (up to 127 qubits)

### Quantum Hardware
- ✅ Backend discovery and selection
- ✅ Automatic transpilation to ISA
- ✅ Error mitigation (resilience levels 0-2)
- ✅ Dynamical decoupling
- ✅ Session-based execution
- ✅ Job history tracking

## Known Issues & Solutions

1. **Reference Energy Values**
   - Current H2 reference (-1.855 Hartree) is from an effective Hamiltonian
   - Not the standard ab initio value (-1.174 Hartree)
   - Document Hamiltonian sources for production use

2. **Cost Management**
   - Quantum hardware runs limited to 20 iterations by default
   - Use least_busy backend selection
   - Session-based execution minimizes queuing overhead

3. **Error Messages**
   - "No instances associated" → Create instance on IBM Quantum dashboard
   - "Connection closed" → Check JSON boolean syntax (True not true)
   - "Module not found" → Ensure all dependencies in pyproject.toml

## Future Enhancements

1. **Additional Molecules**
   - LiH, H2O, BeH2 with proper reference energies
   - Support for custom molecular geometries
   - Integration with PySCF for Hamiltonian generation

2. **Advanced Features**
   - Gradient-based optimizers
   - Adaptive VQE variants
   - Excited state calculations
   - Noise model simulation

3. **Production Readiness**
   - Comprehensive error recovery
   - Cost estimation before execution
   - Result caching and persistence
   - Batch job submission

## Development Workflow

1. **Testing New Features**
   - Always test on local simulator first
   - Verify convergence behavior
   - Check transpilation metrics

2. **Quantum Execution**
   - Start with minimal iterations (5-10)
   - Monitor backend queue times
   - Verify error mitigation settings

3. **Debugging MCP Issues**
   - Check logs in `~/.cache/claude-cli-nodejs/`
   - Verify tool discovery with minimal server
   - Test credential loading separately

## Summary

This dual-server architecture provides a robust foundation for quantum-classical hybrid computing via MCP. The separation ensures that:
- Development and testing can proceed without quantum access
- Production quantum resources are protected by proper validation
- Users have clear understanding of simulation vs. hardware execution
- Credentials are managed securely and validated appropriately

The project demonstrates best practices for MCP server development, Qiskit 2.x compatibility, and IBM Quantum Platform integration.