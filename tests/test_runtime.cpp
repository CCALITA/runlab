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
  }

  std::cout << "All tests passed.\n";
  return 0;
}
