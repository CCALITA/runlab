# Development Progress

## Status
- Project scaffolded with CMake and headers
- L1: sender utilities and sample kernels implemented
- L2: runtime graph, blackboard, scheduler, thread pool implemented
- L3: pybind11 bindings with basic ops implemented
- Docs and Python example added
- Runtime now uses stdexec (P2300) with static_thread_pool scheduling
- Runtime refactored to compose DAG execution with stdexec senders (split/when_all/sync_wait)
- stdexec (P2300) reference located at thirdparty/stdexec/examples/hello_world.cpp
- Unit test added for runtime DAG execution and cycle detection
- Expanded tests for error propagation and concurrent scheduling
- Build + ctest run completed after stdexec DAG refactor
- Python example currently requires NumPy and module path setup (e.g. PYTHONPATH=build)
- Dataflow runtime now binds kernels at compile-time: `KernelDef::bind(config)` produces a pure
  `KernelFn(inputs, resources) -> AnyValueSender`, and `GraphBuilder::compile` installs it on each
  node (no registry lookup during execution).

## Files
- CMakeLists.txt
- include/runlab/sender.hpp
- include/runlab/kernels.hpp
- include/runlab/runtime.hpp
- include/runlab/dataflow.hpp
- bindings/pybind_module.cpp
- README.md
- examples/example.py
- tests/test_runtime.cpp

## Next: stdexec-native runtime API (dataflow)

Goals:
- Build DAGs at runtime, but run as pure stdexec dataflow (no shared blackboard for data).
- Restrict orchestration to `kernel_id + config + input edges`; kernel is the compute unit.
- Manage resources monad-style via receiver env (no explicit “context” object).
- Snapshot `Resources` from the receiver env at graph entry; pass into kernels while binding
  to avoid env lookups inside the compute path.
- Added dataflow tests covering bias injection and fan-out with resource usage.

Proposed public surface (C++):
```cpp
namespace runlab::dataflow {
struct KernelDef {
  std::string id;
  size_t arity;
  // bind(config) -> pure kernel function; runtime injects config/DSL at bind time.
  std::function<KernelFn(const std::any& config)> bind;
};

class KernelRegistry;   // kernel_id -> KernelDef
class GraphBuilder;     // add_input/add_node/set_outputs
class CompiledGraph {   // immutable; validated topo order
 public:
  auto sender(InputMap inputs) const; // reads scheduler from env
};
} // namespace runlab::dataflow
```

Resource injection pattern:
- Define forwarding env queries (e.g. `get_resources`) and attach with
  `stdexec::write_env(sender, exec::with(get_resources, Resources{...}))`.
- Current implementation reads resources at the graph boundary (via
  `exec::read_with_default`) and passes a `Resources` snapshot to kernels. We keep
  `exec::any_sender` as the type-erasure boundary but aim to avoid additional
  vtables; each node runs a single concrete sender type produced at bind time.

### Next steps
- Tighten DSL/path: keep node spec to `kernel_id + config + inputs`; reject lambdas
  in orchestration.
- Explore sum-type sender box (fat-pointer) if we need runtime dispatch without
  virtual calls, while preserving env queries.
- Extend tests to cover multiple compiled DAGs running in parallel and resource
  snapshots per graph.
- Document the kernel binding contract (config decoded once; kernels pure) and
  migration path for the Python DSL.
