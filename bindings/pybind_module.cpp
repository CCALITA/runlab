#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "runlab/kernels.hpp"
#include "runlab/runtime.hpp"

namespace py = pybind11;

namespace {

std::vector<float> ToVector(const py::object& obj) {
  py::array_t<float, py::array::c_style | py::array::forcecast> array(obj);
  if (array.ndim() != 1) {
    throw std::runtime_error("input data must be 1-D");
  }
  const auto info = array.request();
  const auto* data = static_cast<const float*>(info.ptr);
  return std::vector<float>(data, data + info.size);
}

py::array_t<float> ToArray(const std::vector<float>& data) {
  py::array_t<float> array(data.size());
  std::memcpy(array.mutable_data(), data.data(), data.size() * sizeof(float));
  return array;
}

runlab::FloatSpan ToFloatSpanZeroCopy(const py::object& obj) {
  if (!py::isinstance<py::array>(obj)) {
    throw std::runtime_error(
        "data must be a NumPy array (1-D float32, C-contiguous)");
  }

  py::array array = py::reinterpret_borrow<py::array>(obj);
  if (!array.dtype().is(py::dtype::of<float>())) {
    throw std::runtime_error(
        "data must be float32; try np.ascontiguousarray(x, dtype=np.float32)");
  }
  if ((array.flags() & py::array::c_style) == 0) {
    throw std::runtime_error(
        "data must be C-contiguous; try np.ascontiguousarray(x, dtype=np.float32)");
  }

  const py::buffer_info info = array.request();
  if (info.ndim != 1) {
    throw std::runtime_error("data must be 1-D");
  }

  auto owner = std::shared_ptr<py::object>(
      new py::object(array),
      [](py::object* p) {
        py::gil_scoped_acquire acquire;
        delete p;
      });
  return runlab::FloatSpan{
      .owner = std::move(owner),
      .data = static_cast<const float*>(info.ptr),
      .size = static_cast<size_t>(info.size),
  };
}

py::array_t<float> ToArrayView(const runlab::FloatSpan& span) {
  auto* owner = new std::shared_ptr<void>(span.owner);
  py::capsule base(owner, [](void* p) {
    delete static_cast<std::shared_ptr<void>*>(p);
  });

  return py::array_t<float>(
      {static_cast<py::ssize_t>(span.size)},
      {static_cast<py::ssize_t>(sizeof(float))},
      const_cast<float*>(span.data),
      std::move(base));
}

std::string ExceptionToString(std::exception_ptr err) {
  if (!err) {
    return {};
  }
  try {
    std::rethrow_exception(err);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "unknown exception";
  }
}

std::string RequireString(const py::dict& params, const char* key) {
  if (!params.contains(key)) {
    throw std::runtime_error(std::string("missing param: ") + key);
  }
  return py::cast<std::string>(params[key]);
}

float RequireFloat(const py::dict& params, const char* key) {
  if (!params.contains(key)) {
    throw std::runtime_error(std::string("missing param: ") + key);
  }
  return py::cast<float>(params[key]);
}

}  // namespace

PYBIND11_MODULE(runlab_py, m) {
  m.doc() = "Hybrid static/dynamic DAG engine (minimal reference implementation)";

  py::class_<runlab::Engine>(m, "Engine")
      .def(py::init<size_t>(), py::arg("threads") = 0)
      .def(
          "add_node",
          [](runlab::Engine& engine, const std::string& id,
             const std::string& op_type, const py::dict& params,
             const std::string& graph_name) {
            if (op_type == "input") {
              if (!params.contains("data")) {
                throw std::runtime_error("input node requires data");
              }
              engine.add_value_source_to(
                graph_name, id, ToFloatSpanZeroCopy(params["data"]));
              return;
            }

            if (op_type == "scale") {
              const std::string input = RequireString(params, "input");
              const float factor = RequireFloat(params, "factor");
              engine.add_value_node_to(
                graph_name,
                id,
                {input},
                [input, factor](runlab::GraphContext& ctx) {
                  return runlab::kernels::scale(ctx.get_span(input), factor);
                });
              return;
            }

            if (op_type == "add") {
              const std::string left = RequireString(params, "left");
              const std::string right = RequireString(params, "right");
              engine.add_value_node_to(
                graph_name,
                id,
                {left, right},
                [left, right](runlab::GraphContext& ctx) {
                  return runlab::kernels::add(ctx.get_span(left), ctx.get_span(right));
                });
              return;
            }

            if (op_type == "sum") {
              const std::string input = RequireString(params, "input");
              engine.add_value_node_to(
                graph_name,
                id,
                {input},
                [input](runlab::GraphContext& ctx) {
                  return runlab::kernels::sum(ctx.get_span(input));
                });
              return;
            }

            if (op_type == "embedding") {
              const std::string input = RequireString(params, "input");
              engine.add_value_node_to(
                graph_name,
                id,
                {input},
                [input](runlab::GraphContext& ctx) {
                  return runlab::kernels::compute_embedding(ctx.get_span(input));
                });
              return;
            }

            throw std::runtime_error("unknown op_type: " + op_type);
          },
          py::arg("id"), py::arg("op_type"), py::arg("params") = py::dict(),
          py::arg("graph") = "default")
      .def(
          "add_edge",
          [](runlab::Engine& engine, const std::string& from,
             const std::string& to, const std::string& graph) {
            engine.add_edge(graph, from, to);
          },
          py::arg("from"), py::arg("to"), py::arg("graph") = "default")
      .def(
          "clear",
          [](runlab::Engine& engine, const std::string& graph) {
            engine.clear_graph(graph);
          },
          py::arg("graph") = "default")
      .def(
          "validate",
          [](runlab::Engine& engine, const std::string& graph) {
            return engine.validate(graph);
          },
          py::arg("graph") = "default")
      .def(
          "compile",
          [](runlab::Engine& engine, const std::string& graph) {
            return engine.compile_and_install(graph);
          },
          py::arg("graph") = "default")
      .def(
          "run",
          [](runlab::Engine& engine, const std::string& graph) {
            py::gil_scoped_release release;
            engine.run_graph(graph);
          },
          py::arg("graph") = "default")
      .def(
          "run_compiled",
          [](runlab::Engine& engine, const std::string& graph) {
            py::gil_scoped_release release;
            engine.run_installed(graph);
          },
          py::arg("graph") = "default")
      .def("get_vector",
           [](runlab::Engine& engine, const std::string& key,
              const std::string& graph) {
             runlab::FloatSpan span;
             auto& ctx = engine.context(graph);
             if (ctx.try_get<runlab::FloatSpan>(key, &span)) {
               return ToArrayView(span);
             }
             auto values = ctx.get<std::vector<float>>(key);
             return ToArray(values);
           },
           py::arg("key"), py::arg("graph") = "default")
      .def(
          "get_float",
          [](runlab::Engine& engine, const std::string& key,
             const std::string& graph) {
            return engine.context(graph).get<float>(key);
          },
          py::arg("key"), py::arg("graph") = "default")
      .def("node_status",
           [](runlab::Engine& engine, const std::string& id,
              const std::string& graph) {
             const auto status = engine.context(graph).node_status(id);
             return std::string(runlab::ToString(status));
           },
           py::arg("id"), py::arg("graph") = "default")
      .def("node_error",
           [](runlab::Engine& engine, const std::string& id,
              const std::string& graph) -> py::object {
             auto err = engine.context(graph).node_error(id);
             if (!err) {
               return py::none();
             }
             return py::str(ExceptionToString(std::move(err)));
           },
           py::arg("id"), py::arg("graph") = "default")
      .def(
          "node_statuses",
          [](runlab::Engine& engine, const std::string& graph) {
            py::dict out;
            const auto states = engine.context(graph).node_states_snapshot();
            for (const auto& [id, state] : states) {
              out[py::str(id)] = py::str(runlab::ToString(state.status));
            }
            return out;
          },
          py::arg("graph") = "default");
}
