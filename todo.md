# Feature TODO

## Core runtime
- Typed any_sender_of per op (value channels) for stronger compile-time checks
- Optional priority/queue policies for node scheduling
- Resource-aware scheduling hooks (CPU/GPU, IO)
- Structured error aggregation and per-node status

## Python layer
- Zero-copy buffer support for NumPy arrays
- Richer DSL helpers (node builders, graph validation)
- Async run / cancellation support

## Kernels
- More realistic ops (vector search, normalization, metrics)
- Pluggable kernel registry for user-defined ops

## Testing/Tooling
- Document NumPy dependency and runlab_py module path (or add requirements/install step)
- Minimal benchmarking harness

## Done
- Expand tests for error propagation and concurrent scheduling
- Refactor runtime to compose DAG execution with stdexec senders
