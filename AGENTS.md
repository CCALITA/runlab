# Repository Guidelines

## Project Structure

- `include/runlab/`: C++20 public headers (core runtime + kernels; see
  `include/runlab/dataflow.hpp` for the stdexec DAG runtime that snapshots `Resources`
  from the receiver env).
- `bindings/`: pybind11 module (`bindings/pybind_module.cpp`) exposing a small Python DSL.
- `tests/`: C++ executable tests (currently `tests/test_runtime.cpp`).
- `examples/`: small usage snippets (e.g., `examples/example.py`).
- `thirdparty/stdexec/`: vendored stdexec headers and docs; treat as upstream code.

## Build, Test, and Development Commands

This repo uses CMake (see `CMakeLists.txt`).

- Configure + build:
  - `cmake -S . -B build`
  - `cmake --build build`
- Run tests (after building):
  - `ctest --test-dir build` (runs `runlab_tests`)
  - or `./build/runlab_tests`
- Common options:
  - `-DCMAKE_BUILD_TYPE=Debug`
  - `-DRUNLAB_BUILD_PYTHON=OFF` (skip pybind11 module)
  - `-DRUNLAB_BUILD_TESTS=OFF` (skip tests)

Python usage after building (module lands in `build/`):
- `python examples/example.py`

## Coding Style & Naming Conventions

- C++: match existing formatting (2-space indentation, braces on the same line).
- Naming: types in `PascalCase` (e.g., `GraphContext`), functions/methods in
  `snake_case` (e.g., `add_node`), files in `lower_snake_case` (e.g., `runtime.hpp`).
- Keep headers self-contained and prefer small, targeted changes.
- No repo-wide formatter config is enforced; keep diffs consistent with nearby code.

## Testing Guidelines

- Tests are plain C++ programs (no external framework). Follow the pattern in
  `tests/test_runtime.cpp`: return non-zero on failure and print a clear message.
- Add new tests under `tests/` and wire them up in `CMakeLists.txt` if you create
  additional executables.

## Commit & Pull Request Guidelines

- Commit messages in history are short, imperative summaries (e.g., “Vendor stdexec”).
  Prefer `Verb object` subjects and keep them scoped.
- PRs should include: what changed, how to reproduce, commands run (`cmake`, `ctest`),
  and any API/behavior notes. Include a minimal Python snippet if bindings change.
- Avoid modifying `thirdparty/stdexec/` unless you are intentionally updating the vendor
  drop; keep such changes isolated.
