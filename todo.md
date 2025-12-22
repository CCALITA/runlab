# Feature TODO

## Core runtime
- Typed any_sender_of per op (value channels) for stronger compile-time checks
- Optional priority/queue policies for node scheduling
- Resource-aware scheduling hooks (CPU/GPU, IO)
- New stdexec-native dataflow runtime (no shared context)
- Kernel registry + restricted node specs (kernel + config + inputs)
- Graph outputs as sender values (no blackboard reads)
- Env-based resource injection (allocator/device/logging)

## Python layer
- Richer DSL helpers (node builders, graph validation)
- Async run / cancellation support
- Migrate DSL to kernel registry + edge wiring (no `context.get/put`)

## Kernels
- More realistic ops (vector search, normalization, metrics)
- Pluggable kernel registry for user-defined ops

## Testing/Tooling
- Minimal benchmarking harness

## Done
- Expand tests for error propagation and concurrent scheduling
- Refactor runtime to compose DAG execution with stdexec senders
- Zero-copy NumPy input buffers (1-D, C-contiguous float32)
- Document NumPy requirements and `build/` module path
- Track per-node status and error causes
- Add `Engine.validate()` for graph validation/toposort
- Multi-graph support (named DAGs with per-graph contexts)
- Control/data split via `Graph::compile()` -> `CompiledGraph` (pure C++ run)
- Stdexec-native entrypoints (`Graph::sender`, `Engine::start_*`)
