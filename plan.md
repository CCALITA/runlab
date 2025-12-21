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
- bindings/pybind_module.cpp
- README.md
- examples/example.py
- tests/test_runtime.cpp
