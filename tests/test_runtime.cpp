#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <stdexec/execution.hpp>

#include "runlab/runtime/dataflow.hpp"
#include "runlab/kernel/kernels.hpp"
#include "runlab/engine/engine.hpp"

namespace {

int Fail(const char* message) {
  std::cerr << "Test failure: " << message << "\n";
  return 1;
}

struct ConcurrencyProbe {
  std::mutex mu;
  std::condition_variable cv;
  int arrived = 0;
  int active = 0;
  int max_active = 0;
  bool release = false;
  bool timed_out = false;

  void enter_and_wait(int target) {
    std::unique_lock<std::mutex> lock(mu);
    ++arrived;
    ++active;
    if (active > max_active) {
      max_active = active;
    }
    if (arrived >= target) {
      release = true;
      cv.notify_all();
      return;
    }
    if (!cv.wait_for(lock, std::chrono::seconds(2), [&]() { return release; })) {
      timed_out = true;
      release = true;
      cv.notify_all();
    }
  }

  void leave() {
    std::lock_guard<std::mutex> lock(mu);
    --active;
  }
};

std::shared_ptr<runlab::dataflow::KernelRegistry> MakePassthroughRegistry() {
  using namespace runlab::dataflow;

  auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "id",
      .arity = 1,
      .bind =
        [](const std::any&) -> KernelFn {
        return [](std::vector<Value> inputs, const RuntimeConfig*) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("id expects 1 input");
          }
          return AnyValueSender(stdexec::just(std::move(inputs[0])));
        };
    },
  });
  return registry;
}

}  // namespace

int main() {
  {
    runlab::Engine engine(2);
    std::vector<float> input{1.0f, 2.0f, 3.0f};

    engine.add_node("input", [input](runlab::GraphContext& ctx) {
      return stdexec::then(stdexec::just(), [input, &ctx]() { ctx.put("input", input); });
    });
    engine.add_node("scaled", {"input"}, [](runlab::GraphContext& ctx) {
      auto values = ctx.get<std::vector<float>>("input");
      auto sender = stdexec::just(runlab::kernels::scale(std::move(values), 2.0f));
      return stdexec::then(std::move(sender), [&ctx](std::vector<float> output) {
        ctx.put("scaled", std::move(output));
      });
    });
    engine.add_node("total", {"scaled"}, [](runlab::GraphContext& ctx) {
      auto values = ctx.get<std::vector<float>>("scaled");
      auto sender = stdexec::just(runlab::kernels::sum(std::move(values)));
      return stdexec::then(std::move(sender), [&ctx](float total) { ctx.put("total", total); });
    });

    engine.run();

    const float total = engine.context().get<float>("total");
    if (std::fabs(total - 12.0f) > 0.001f) {
      return Fail("unexpected sum result");
    }
  }

  {
    runlab::Engine engine(1);
    engine.add_node("a", [](runlab::GraphContext&) { return stdexec::just(); });
    engine.add_node("b", {"a"}, [](runlab::GraphContext&) { return stdexec::just(); });
    engine.add_edge("b", "a");

    try {
      engine.run();
      return Fail("cycle detection did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    runlab::Engine engine(2);
    auto probe = std::make_shared<ConcurrencyProbe>();

    engine.add_node("left", [probe](runlab::GraphContext&) {
      return stdexec::then(stdexec::just(), [probe]() {
        probe->enter_and_wait(2);
        probe->leave();
      });
    });
    engine.add_node("right", [probe](runlab::GraphContext&) {
      return stdexec::then(stdexec::just(), [probe]() {
        probe->enter_and_wait(2);
        probe->leave();
      });
    });

    engine.run();

    if (probe->timed_out) {
      return Fail("concurrency barrier timed out");
    }
    if (probe->max_active < 2) {
      return Fail("concurrent scheduling did not overlap");
    }
  }

  {
    runlab::Engine engine(2);
    std::atomic<bool> after_ran{false};

    engine.add_node("ok", [](runlab::GraphContext&) { return stdexec::just(); });
    engine.add_node("bad", {"ok"}, [](runlab::GraphContext&) {
      return stdexec::then(stdexec::just(), []() { throw std::runtime_error("boom"); });
    });
    engine.add_node("after", {"bad"}, [&after_ran](runlab::GraphContext&) {
      return stdexec::then(stdexec::just(), [&after_ran]() { after_ran.store(true); });
    });

    try {
      engine.run();
      return Fail("error propagation did not throw");
    } catch (const std::exception&) {
    }

    if (after_ran.load()) {
      return Fail("downstream node executed after error");
    }
    if (engine.context().node_status("ok") != runlab::NodeStatus::kSuccess) {
      return Fail("ok node status was not success");
    }
    if (engine.context().node_status("bad") != runlab::NodeStatus::kError) {
      return Fail("bad node status was not error");
    }
    if (!engine.context().node_error("bad")) {
      return Fail("bad node error was not recorded");
    }
    if (engine.context().node_status("after") != runlab::NodeStatus::kBlocked) {
      return Fail("after node status was not blocked");
    }
  }

  {
    runlab::Engine engine(2);

    engine.add_node_to("g1", "input", [](runlab::GraphContext& ctx) {
      std::vector<float> values{1.0f, 2.0f};
      return stdexec::then(stdexec::just(), [&ctx, values]() { ctx.put("v", values); });
    });
    engine.add_node_to("g1", "total", {"input"}, [](runlab::GraphContext& ctx) {
      const auto values = ctx.get<std::vector<float>>("v");
      return stdexec::then(stdexec::just(), [&ctx, values]() {
        ctx.put("sum", values[0] + values[1]);
      });
    });

    engine.add_node_to("g2", "input", [](runlab::GraphContext& ctx) {
      std::vector<float> values{10.0f};
      return stdexec::then(stdexec::just(), [&ctx, values]() { ctx.put("v", values); });
    });
    engine.add_node_to("g2", "total", {"input"}, [](runlab::GraphContext& ctx) {
      const auto values = ctx.get<std::vector<float>>("v");
      return stdexec::then(stdexec::just(), [&ctx, values]() { ctx.put("sum", values[0]); });
    });

    engine.run_graph("g1");
    engine.run_graph("g2");

    const float g1_sum = engine.context("g1").get<float>("sum");
    const float g2_sum = engine.context("g2").get<float>("sum");
    if (std::fabs(g1_sum - 3.0f) > 0.001f) {
      return Fail("multi-graph run produced wrong g1 result");
    }
    if (std::fabs(g2_sum - 10.0f) > 0.001f) {
      return Fail("multi-graph run produced wrong g2 result");
    }
  }

  {
    runlab::Graph graph;
    graph.add_node("x", [](runlab::GraphContext& ctx) {
      return stdexec::then(stdexec::just(), [&ctx]() { ctx.put("x", 42); });
    });
    auto compiled = graph.compile();

    runlab::Engine engine(1);
    runlab::GraphContext ctx;
    engine.run(compiled, ctx);

    if (ctx.get<int>("x") != 42) {
      return Fail("compiled graph did not run in pure C++");
    }
  }

  {
    runlab::Engine engine(1);
    engine.graph("bg").add_node("x", [](runlab::GraphContext& ctx) {
      return stdexec::then(stdexec::just(), [&ctx]() { ctx.put("x", 1); });
    });
    engine.compile_and_install("bg");
    engine.run_installed("bg");
    if (engine.context("bg").get<int>("x") != 1) {
      return Fail("installed graph did not run");
    }

    runlab::Graph g2;
    g2.add_node("x", [](runlab::GraphContext& ctx) {
      return stdexec::then(stdexec::just(), [&ctx]() { ctx.put("x", 2); });
    });
    engine.install_graph("bg", g2.compile());
    engine.context("bg").clear();
    engine.run_installed("bg");
    if (engine.context("bg").get<int>("x") != 2) {
      return Fail("installed graph swap did not take effect");
    }
  }

  {
    runlab::Engine engine(1);
    engine.graph("s").add_node("x", [](runlab::GraphContext& ctx) {
      return stdexec::then(stdexec::just(), [&ctx]() { ctx.put("x", 7); });
    });

    auto snd = engine.start_graph("s");
    if (engine.context("s").contains("x")) {
      return Fail("sender creation had side effects");
    }
    stdexec::sync_wait(std::move(snd));
    if (engine.context("s").get<int>("x") != 7) {
      return Fail("start_graph sender did not run");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "add_f",
      .arity = 2,
      .bind =
        [](const std::any&) -> KernelFn {
        return [](std::vector<Value> inputs, const RuntimeConfig*) -> AnyValueSender {
          if (inputs.size() != 2) {
            throw std::runtime_error("add_f expects 2 inputs");
          }
          const auto* a = std::get_if<float>(&inputs[0]);
          const auto* b = std::get_if<float>(&inputs[1]);
          if (!a || !b) {
            throw std::runtime_error("add_f expects float inputs");
          }
          return AnyValueSender(stdexec::just(Value{*a + *b}));
        };
      },
    });

    GraphBuilder g;
    g.add_input("a");
    g.add_input("b");
    g.add_node("sum", "add_f", std::any{}, {"a", "b"});
    g.set_outputs({"sum"});

    auto compiled = g.compile(registry);
    auto out_opt = stdexec::sync_wait(
      compiled.sender(InputMap{{"a", Value{1.5f}}, {"b", Value{2.5f}}}));
    if (!out_opt) {
      return Fail("dataflow graph was stopped");
    }
    auto out = std::get<0>(*out_opt);
    auto it = out.find("sum");
    if (it == out.end()) {
      return Fail("dataflow output missing");
    }
    const auto* sum = std::get_if<float>(&it->second);
    if (!sum || std::fabs(*sum - 4.0f) > 0.001f) {
      return Fail("dataflow output unexpected");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "add_bias",
      .arity = 1,
      .bind =
        [](const std::any& cfg) -> KernelFn {
        const float bias = std::any_cast<float>(cfg);
        return [bias](std::vector<Value> inputs, const RuntimeConfig*) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("add_bias expects 1 input");
          }
          const auto* x = std::get_if<float>(&inputs[0]);
          if (!x) {
            throw std::runtime_error("add_bias expects float input");
          }
          return AnyValueSender(stdexec::just(Value{*x + bias}));
        };
      },
    });

    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "add_bias", std::any{2.0f}, {"x"});
    g.set_outputs({"y"});
    auto compiled = g.compile(registry);

    auto snd = compiled.sender(InputMap{{"x", Value{10.0f}}});

    auto out_opt = stdexec::sync_wait(std::move(snd));
    if (!out_opt) {
      return Fail("dataflow add_bias was stopped");
    }
    const auto out = std::get<0>(*out_opt);
    const auto it = out.find("y");
    if (it == out.end()) {
      return Fail("dataflow add_bias output missing");
    }
    const auto* y = std::get_if<float>(&it->second);
    if (!y || std::fabs(*y - 12.0f) > 0.001f) {
      return Fail("dataflow add_bias output unexpected");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "add_bias",
      .arity = 1,
      .bind =
        [](const std::any& cfg) -> KernelFn {
        const float bias = std::any_cast<float>(cfg);
        return [bias](std::vector<Value> inputs, const RuntimeConfig*) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("add_bias expects 1 input");
          }
          const auto* x = std::get_if<float>(&inputs[0]);
          if (!x) {
            throw std::runtime_error("add_bias expects float input");
          }
          return AnyValueSender(stdexec::just(Value{*x + bias}));
        };
      },
    });

    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "add_bias", std::any{-2.0f}, {"x"});
    g.add_node("z", "add_bias", std::any{-2.0f}, {"x"});
    g.set_outputs({"y", "z"});

    auto compiled = g.compile(registry);
    auto snd = compiled.sender(InputMap{{"x", Value{5.0f}}});

    auto out_opt = stdexec::sync_wait(std::move(snd));
    if (!out_opt) {
      return Fail("dataflow fan-out add_bias was stopped");
    }
    const auto out = std::get<0>(*out_opt);
    const auto it_y = out.find("y");
    const auto it_z = out.find("z");
    if (it_y == out.end() || it_z == out.end()) {
      return Fail("dataflow fan-out outputs missing");
    }
    const auto* y = std::get_if<float>(&it_y->second);
    const auto* z = std::get_if<float>(&it_z->second);
    if (!y || !z || std::fabs(*y - 3.0f) > 0.001f || std::fabs(*z - 3.0f) > 0.001f) {
      return Fail("dataflow fan-out bias incorrect");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("x", "id", std::any{}, {"x"});
    g.set_outputs({"x"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow duplicate node id did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("y", "id", std::any{}, {"missing"});
    g.set_outputs({"y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow missing dependency did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "id", std::any{}, {"x", "x"});
    g.set_outputs({"y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow arity mismatch did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "id", std::any{}, {"x"});
    g.set_outputs({"missing"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow unknown output did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("a", "id", std::any{}, {"b"});
    g.add_node("b", "id", std::any{}, {"a"});
    g.set_outputs({"a"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow cycle detection did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "id", std::any{}, {"x"});
    g.set_outputs({"y", "y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow duplicate outputs did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");

    try {
      (void)g.compile(registry);
      return Fail("dataflow missing outputs did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("in", std::string(kInputKernelId), std::any{}, {"x"});
    g.set_outputs({"in"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow input with deps did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("in", std::string(kInputKernelId), std::any{1}, {});
    g.set_outputs({"in"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow input with config did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("y", "unknown_kernel", std::any{}, {});
    g.set_outputs({"y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow unknown kernel did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_node("y", "", std::any{}, {});
    g.set_outputs({"y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow empty kernel id did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    KernelDef def{
      .id = "dup",
      .arity = 0,
      .bind =
        [](const std::any&) -> KernelFn {
        return [](std::vector<Value>, const RuntimeConfig*) -> AnyValueSender {
          return AnyValueSender(stdexec::just(Value{0.0f}));
        };
      },
    };
    registry->register_kernel(def);
    try {
      registry->register_kernel(def);
      return Fail("dataflow duplicate kernel registration did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    KernelDef def{
      .id = std::string(kInputKernelId),
      .arity = 0,
      .bind =
        [](const std::any&) -> KernelFn {
        return [](std::vector<Value>, const RuntimeConfig*) -> AnyValueSender {
          return AnyValueSender(stdexec::just(Value{0.0f}));
        };
      },
    };

    try {
      registry->register_kernel(std::move(def));
      return Fail("dataflow reserved kernel id registration did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "id", std::any{}, {"x"});
    g.set_outputs({"y"});
    auto compiled = g.compile(registry);

    try {
      (void)stdexec::sync_wait(compiled.sender(InputMap{}));
      return Fail("dataflow missing runtime input did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = MakePassthroughRegistry();
    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "id", std::any{}, {"x"});
    g.set_outputs({"y"});
    auto compiled = g.compile(registry);

    try {
      (void)stdexec::sync_wait(
        compiled.sender(InputMap{{"x", Value{1.0f}}, {"extra", Value{2.0f}}}));
      return Fail("dataflow unknown runtime input did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "needs_float",
      .arity = 0,
      .validate_config =
        [](const std::any& cfg) { (void)RequireConfig<float>(cfg, "needs_float", "float config"); },
      .bind =
        [](const std::any& cfg) -> KernelFn {
        const float value = RequireConfig<float>(cfg, "needs_float", "float config");
        return [value](std::vector<Value>, const RuntimeConfig*) -> AnyValueSender {
          return AnyValueSender(stdexec::just(Value{value}));
        };
      },
    });

    GraphBuilder g;
    g.add_node("y", "needs_float", std::any{std::string("bad")}, {});
    g.set_outputs({"y"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow config validation did not throw");
    } catch (const std::exception&) {
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    registry->register_kernel(KernelDef{
      .id = "needs_float",
      .arity = 0,
      .validate_config =
        [](const std::any& cfg) { (void)RequireConfig<float>(cfg, "needs_float", "float config"); },
      .bind =
        [](const std::any& cfg) -> KernelFn {
        const float value = RequireConfig<float>(cfg, "needs_float", "float config");
        return [value](std::vector<Value>, const RuntimeConfig*) -> AnyValueSender {
          return AnyValueSender(stdexec::just(Value{value}));
        };
      },
    });

    GraphBuilder g;
    g.add_node("y", "needs_float", std::any{2.5f}, {});
    g.set_outputs({"y"});
    auto compiled = g.compile(registry);
    auto out_opt = stdexec::sync_wait(compiled.sender(InputMap{}));
    if (!out_opt) {
      return Fail("dataflow config validation success path stopped");
    }
    const auto out = std::get<0>(*out_opt);
    const auto it = out.find("y");
    if (it == out.end()) {
      return Fail("dataflow config validation success output missing");
    }
    const auto* val = std::get_if<float>(&it->second);
    if (!val || std::fabs(*val - 2.5f) > 0.001f) {
      return Fail("dataflow config validation success output unexpected");
    }
  }

  {
    using namespace runlab::dataflow;

    try {
      (void)RequireConfig<int>(std::any{std::string("bad")}, "cfg_helper", "int config");
      return Fail("RequireConfig accepted wrong type");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      if (msg.find("cfg_helper") == std::string::npos ||
          msg.find("int config") == std::string::npos) {
        return Fail("RequireConfig error message missing context");
      }
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    RegisterDefaultKernels(registry);

    registry->register_kernel(MakeKernel<1>(
      "add_bias_env",
      [](std::monostate) -> KernelFn {
        return [](std::vector<Value> inputs, const RuntimeConfig* rcfg) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("add_bias_env expects 1 input");
          }
          const auto* x = std::get_if<float>(&inputs[0]);
          if (!x) {
            throw std::runtime_error("add_bias_env expects float input");
          }
          if (!rcfg) {
            throw std::runtime_error("add_bias_env missing runtime config");
          }
          return AnyValueSender(stdexec::just(Value{*x + rcfg->bias}));
        };
      }));

    RuntimeConfig cfg{.bias = 3.0f};

    GraphBuilder g;
    g.add_input("x");
    g.add_node("y", "add_bias_env", std::any{}, {"x"});
    g.set_outputs({"y"});
    auto compiled = g.compile(registry);

    auto snd = stdexec::write_env(
      compiled.sender(InputMap{{"x", Value{2.0f}}}),
      exec::with(get_runtime_config, &cfg));

    auto out_opt = stdexec::sync_wait(std::move(snd));
    if (!out_opt) {
      return Fail("dataflow add_bias_env was stopped");
    }
    const auto out = std::get<0>(*out_opt);
    const auto it = out.find("y");
    if (it == out.end()) {
      return Fail("dataflow add_bias_env output missing");
    }
    const auto* y = std::get_if<float>(&it->second);
    if (!y || std::fabs(*y - 5.0f) > 0.001f) {
      return Fail("dataflow add_bias_env output unexpected");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    RegisterDefaultKernels(registry);

    GraphBuilder g;
    g.add_input("x");
    g.add_input("y");
    g.add_node("scaled", "scale_f", std::any{2.0f}, {"x"});
    g.add_node("sum", "add_f", std::any{}, {"scaled", "y"});
    g.set_outputs({"sum"});

    auto compiled = g.compile(registry);
    auto out_opt =
      stdexec::sync_wait(compiled.sender(InputMap{{"x", Value{1.5f}}, {"y", Value{4.0f}}}));
    if (!out_opt) {
      return Fail("dataflow default kernels were stopped");
    }
    const auto out = std::get<0>(*out_opt);
    const auto it = out.find("sum");
    if (it == out.end()) {
      return Fail("dataflow default kernels missing output");
    }
    const auto* sum = std::get_if<float>(&it->second);
    if (!sum || std::fabs(*sum - 7.0f) > 0.001f) {
      return Fail("dataflow default kernels produced wrong sum");
    }
  }

  {
    using namespace runlab::dataflow;

    auto registry = std::make_shared<KernelRegistry>();
    RegisterDefaultKernels(registry);
    GraphBuilder g;
    g.add_input("x");
    g.add_node("scaled", "scale_f", std::any{std::string("bad")}, {"x"});
    g.set_outputs({"scaled"});

    try {
      (void)g.compile(registry);
      return Fail("dataflow default kernel config validation did not throw");
    } catch (const std::exception&) {
    }
  }

  std::cout << "All tests passed.\n";
  return 0;
}
