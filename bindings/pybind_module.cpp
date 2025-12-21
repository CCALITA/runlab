#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexec/execution.hpp>

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
             const std::string& op_type, const py::dict& params) {
            if (op_type == "input") {
              if (!params.contains("data")) {
                throw std::runtime_error("input node requires data");
              }
              auto data = std::make_shared<std::vector<float>>(
                  ToVector(params["data"]));
              engine.add_node(
                  id, [id, data](runlab::GraphContext& ctx) {
                    return stdexec::then(stdexec::just(), [id, data, &ctx]() {
                      ctx.put(id, *data);
                    });
                  });
              return;
            }

            if (op_type == "scale") {
              const std::string input = RequireString(params, "input");
              const float factor = RequireFloat(params, "factor");
              engine.add_node(
                  id, {input},
                  [id, input, factor](runlab::GraphContext& ctx) {
                    auto values = ctx.get<std::vector<float>>(input);
                    auto sender =
                      runlab::kernels::scale(std::move(values), factor);
                    return stdexec::then(
                      std::move(sender),
                      [&ctx, id](std::vector<float> output) {
                        ctx.put(id, std::move(output));
                      });
                  });
              return;
            }

            if (op_type == "add") {
              const std::string left = RequireString(params, "left");
              const std::string right = RequireString(params, "right");
              engine.add_node(
                  id, {left, right},
                  [id, left, right](runlab::GraphContext& ctx) {
                    auto a = ctx.get<std::vector<float>>(left);
                    auto b = ctx.get<std::vector<float>>(right);
                    auto sender =
                      runlab::kernels::add(std::move(a), std::move(b));
                    return stdexec::then(
                      std::move(sender),
                      [&ctx, id](std::vector<float> output) {
                        ctx.put(id, std::move(output));
                      });
                  });
              return;
            }

            if (op_type == "sum") {
              const std::string input = RequireString(params, "input");
              engine.add_node(
                  id, {input},
                  [id, input](runlab::GraphContext& ctx) {
                    auto values = ctx.get<std::vector<float>>(input);
                    auto sender = runlab::kernels::sum(std::move(values));
                    return stdexec::then(
                      std::move(sender),
                      [&ctx, id](float total) { ctx.put(id, total); });
                  });
              return;
            }

            if (op_type == "embedding") {
              const std::string input = RequireString(params, "input");
              engine.add_node(
                  id, {input},
                  [id, input](runlab::GraphContext& ctx) {
                    auto values = ctx.get<std::vector<float>>(input);
                    auto sender =
                      runlab::kernels::compute_embedding(std::move(values));
                    return stdexec::then(
                      std::move(sender),
                      [&ctx, id](std::vector<float> output) {
                        ctx.put(id, std::move(output));
                      });
                  });
              return;
            }

            throw std::runtime_error("unknown op_type: " + op_type);
          },
          py::arg("id"), py::arg("op_type"),
          py::arg("params") = py::dict())
      .def("add_edge", &runlab::Engine::add_edge)
      .def("clear", &runlab::Engine::clear)
      .def("run", [](runlab::Engine& engine) {
        py::gil_scoped_release release;
        engine.run();
      })
      .def("get_vector",
           [](runlab::Engine& engine, const std::string& key) {
             auto values = engine.context().get<std::vector<float>>(key);
             return ToArray(values);
           })
      .def("get_float", [](runlab::Engine& engine, const std::string& key) {
        return engine.context().get<float>(key);
      });
}
