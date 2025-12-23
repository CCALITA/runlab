# Repository Guidelines
a c++20 modern DAG framework
see design in `target.md`

## Project Structure & Module Map
- `include/runlab/`: C++20 headers. `runtime.hpp` hosts the thread-pool DAG engine; `dataflow.hpp` is the stdexec-native runtime (pure value channels, kernel registry binding). `kernels.hpp` holds sample implementation of basic computation unit; `sender.hpp` has helper adapters.
- `bindings/pybind_module.cpp`: pybind11 DSL layer base on the cpp runtime.
- `tests/`: C++ test executables (see `tests/test_runtime.cpp` for end-to-end DAG checks).
- `examples/`: quick usage samples (Python and C++).
- `thirdparty/stdexec/`: vendored stdexec. Treat as upstream-only.

## Runtime & API Expectations
- Kernels are pure operator that can be register and instantiation by `kernel_id`  at runtime
- Graph orchestration is restricted to `kernel_id + config + inputs`; `GraphBuilder::compile` binds `KernelDef::bind(config)` once and runs a single concrete sender type per node. Inject DSL/config at the binding layer, not inside kernels.
- Resources (e.g. global/dag-shared) are expected monad based implementation.
- Multiple DAGs can be installed and run independently; keep graph IDs unique and contexts isolated.

## Build, Test, and Dev Commands
- Configure: `cmake -S . -B build` (add `-DCMAKE_BUILD_TYPE=Debug` as needed).
- Build: `cmake --build build`.
- Tests: `ctest --test-dir build` or `./build/runlab_tests`.
- Python example (after building with bindings on): `python examples/example.py`.

## Coding Style & Naming
- C++20 modern and elegant framework library
- Prefer Functional Programming, monad style
- Free to refract interface without concerning about compatibility
- Framework which acquire high performance implementation
- Types `PascalCase`, functions `snake_case`, files `lower_snake_case`.
- Prefer sender/receiver P2300 model
- Prefer stdexec-native composition; avoid new type erasure layers unless aligned with `exec::any_sender`.

## Testing Guidelines
- Keep tests direct. Mirror patterns in `tests/test_runtime.cpp`: deterministic inputs, explicit failure messages, non-zero exit on failure.
- When adding kernels, cover both success and error propagation; verify fan-out/fan-in if applicable.

## Commit & PR Guidelines
- Use short, imperative subjects (e.g., `Add bias kernel binding`). Separate logical changes.
- PRs should state behavior changes, key commands run (`cmake`, `ctest`), and any API notes (kernel signatures, env queries). Include repro snippets if bindings or DSL change.
