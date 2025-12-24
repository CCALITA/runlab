#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
CONFIG="${CONFIG:-}"

cmake -S . -B "${BUILD_DIR}" ${CONFIG:+-DCMAKE_BUILD_TYPE=${CONFIG}}
cmake --build "${BUILD_DIR}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
