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

Proposed public surface (C++):
```cpp
namespace runlab::dataflow {
struct KernelDef {
  std::string id;
  size_t arity;
  // Prototype: runtime snapshots env resources once and passes them explicitly.
  AnyValueSender (*invoke)(
    const std::any& cfg,
    std::vector<Value> inputs,
    const Resources& resources);
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
  `exec::read_with_default`) and passes a `Resources` snapshot to kernels. Fully
  env-driven kernels will require a different type-erasure strategy (the current
  `exec::any_sender` path erases most receiver env queries).

### Next spike: env-preserving type erasure (stdexec-native)

Goal: let kernels query resources directly from the receiver env without an
explicit `Resources` argument while keeping runtime dynamic dispatch.

Plan:
- Introduce `EnvAnySender<Sigs, Queries...>` that preserves a chosen set of env
  queries through type erasure. Avoid `exec::any_sender` because it filters
  custom queries when receivers are erased.
- Storage: move-only box holding the concrete sender and a vtable of
  `{connect, get_env, destroy}`; `get_env` returns a wrapper that forwards all
  `Queries...` to the underlying sender env.
- Kernel signature becomes `AnyValueEnvSender invoke(const std::any&, std::vector<Value>)`.
- Graph wiring still uses `stdexec::split/when_all/starts_on`; value channels
  stay unchanged.
- Migration path: keep the current `Resources` snapshot code as a fallback while
  adding the new box; add tests that inject `get_resources` and assert kernels
  can `exec::read_with_default(get_resources, ...)` inside their returned sender.
