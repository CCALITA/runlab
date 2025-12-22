# Runlab Hybrid DAG (Minimal Reference)

This is a minimal C++20 + pybind11 implementation of the hybrid static/dynamic DAG
engine described in `read.md`. It focuses on the three-layer split:

- L1: static sender-based kernels in `include/runlab/kernels.hpp`
- L2: dynamic graph runtime, type-erased tasks, and a blackboard in
  `include/runlab/runtime.hpp`
- L3: Python DSL bindings in `bindings/pybind_module.cpp`

## Build

Pybind11 must already be available on your system (no FetchContent).

```bash
cmake -S . -B build
cmake --build build
```

The extension is built as `runlab_py` (module name `runlab_py`).

## Python usage

Requires NumPy. Inputs for `"input"` nodes must be 1-D, C-contiguous `float32`
arrays (use `np.ascontiguousarray(x, dtype=np.float32)` to prepare data).

```python
import numpy as np
import runlab_py as rl

engine = rl.Engine()
engine.add_node("input", "input", {"data": np.ascontiguousarray([1.0, 2.0, 3.0], dtype=np.float32)})
engine.add_node("scaled", "scale", {"input": "input", "factor": 2.5})
engine.add_node("embed", "embedding", {"input": "scaled"})
engine.add_node("total", "sum", {"input": "embed"})

engine.run()

print(engine.get_vector("embed"))
print(engine.get_float("total"))
```

## Multiple graphs

One `Engine` can host multiple named DAGs. Each graph has its own `GraphContext`
(blackboard + node statuses).

```python
engine.add_node("input", "input", {"data": x}, graph="g1")
engine.add_node("total", "sum", {"input": "input"}, graph="g1")
engine.run(graph="g1")
print(engine.get_float("total", graph="g1"))
```

## Pure C++ (control/data split)

Build/validate a `runlab::Graph` on a control thread, compile it once, then run the
immutable `runlab::CompiledGraph` on an `Engine` (optionally from another thread).

```cpp
runlab::Graph g;
// g.add_node(...); g.add_edge(...);
auto compiled = g.compile();
runlab::GraphContext ctx;
runlab::Engine engine(4);
engine.run(compiled, ctx);
```

For background graph building / hot-swap, install a compiled snapshot and run it:

```cpp
engine.install_graph("prod", g.compile());   // atomic publish point
engine.run_installed("prod");                // runs stable snapshot
```

For stdexec-native composition, prefer the sender-returning APIs:

```cpp
auto snd = engine.start_installed("prod");
stdexec::sync_wait(std::move(snd));
```

## Notes

- This is intentionally small and focuses on the architecture; kernels are simple.
- The Python binding supports zero-copy inputs for 1-D, C-contiguous `float32`
  NumPy arrays by storing a buffer-backed view in the runtime blackboard.
- Importing the built extension typically requires adding `build/` to your module
  search path (the provided `examples/example.py` does `sys.path.insert(0, "build")`).
- `engine.validate()` returns a topological order and raises on missing deps/cycles.
- `engine.compile()` builds and installs an immutable compiled snapshot; `engine.run_compiled()`
  runs the installed snapshot for that graph.
- `engine.node_status(id)` / `engine.node_statuses()` help debug failed runs.
- Sender/receiver is based on C++26 P2300 (`std::execution` / stdexec). A local
  reference copy exists at `thirdparty/stdexec/examples/hello_world.cpp`.

## Direction: stdexec-native dataflow runtime (WIP)

The current runtime (`include/runlab/runtime.hpp`) supports orchestration via a
shared blackboard. The next iteration removes shared data contexts and models the
DAG as pure stdexec dataflow:

- Data moves as sender values along edges (no `GraphContext` for data transport).
- Nodes are restricted to `kernel_id + config + input edges` (no per-node lambdas).
- Kernels acquire resources via the receiver environment (allocator/device/etc.).
- Running a graph returns a sender; `sync_wait` is optional at the boundary.
