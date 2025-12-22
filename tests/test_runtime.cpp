#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include <stdexec/execution.hpp>

#include "runlab/kernels.hpp"
#include "runlab/runtime.hpp"

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
      auto sender = runlab::kernels::scale(std::move(values), 2.0f);
      return stdexec::then(std::move(sender), [&ctx](std::vector<float> output) {
        ctx.put("scaled", std::move(output));
      });
    });
    engine.add_node("total", {"scaled"}, [](runlab::GraphContext& ctx) {
      auto values = ctx.get<std::vector<float>>("scaled");
      auto sender = runlab::kernels::sum(std::move(values));
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

  std::cout << "All tests passed.\n";
  return 0;
}
