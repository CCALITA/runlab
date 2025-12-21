cmake_minimum_required(VERSION 3.20)

project(runlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(runlab INTERFACE)
target_include_directories(runlab INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stdexec/include
)

find_package(Threads REQUIRED)
target_link_libraries(runlab INTERFACE Threads::Threads)

option(RUNLAB_BUILD_PYTHON "Build Python bindings" ON)
if(RUNLAB_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(runlab_py bindings/pybind_module.cpp)
  target_link_libraries(runlab_py PRIVATE runlab)
endif()

option(RUNLAB_BUILD_TESTS "Build tests" ON)
if(RUNLAB_BUILD_TESTS)
  enable_testing()
  add_executable(runlab_tests tests/test_runtime.cpp)
  target_link_libraries(runlab_tests PRIVATE runlab)
  add_test(NAME runlab_tests COMMAND runlab_tests)
endif()
