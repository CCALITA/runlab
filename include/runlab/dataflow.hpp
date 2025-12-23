#pragma once

#include <any>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <exec/any_sender_of.hpp>
#include <exec/env.hpp>
#include <stdexec/execution.hpp>

namespace runlab::dataflow {

struct Resources {
  float bias = 0.0f;
};

struct get_resources_t : stdexec::__query<get_resources_t> {
  static consteval auto query(stdexec::forwarding_query_t) noexcept -> bool {
    return true;
  }
};
inline constexpr get_resources_t get_resources{};

using Value = std::variant<float, std::vector<float>>;
using InputMap = std::unordered_map<std::string, Value>;
using OutputMap = std::unordered_map<std::string, Value>;

using VoidSignatures = stdexec::completion_signatures<
  stdexec::set_value_t(),
  stdexec::set_error_t(std::exception_ptr),
  stdexec::set_stopped_t()>;

using ValueSignatures = stdexec::completion_signatures<
  stdexec::set_value_t(Value),
  stdexec::set_error_t(std::exception_ptr),
  stdexec::set_stopped_t()>;

using OutputSignatures = stdexec::completion_signatures<
  stdexec::set_value_t(OutputMap),
  stdexec::set_error_t(std::exception_ptr),
  stdexec::set_stopped_t()>;

using AnyVoidSender = exec::any_receiver_ref<
  VoidSignatures>::template any_sender<>;

using AnyValueSender = exec::any_receiver_ref<
  ValueSignatures>::template any_sender<>;

using AnyOutputSender = exec::any_receiver_ref<
  OutputSignatures>::template any_sender<>;

class SharedVoidSender {
 public:
  using sender_concept = stdexec::sender_t;
  using completion_signatures = VoidSignatures;

  SharedVoidSender() : factory_([]() { return AnyVoidSender(stdexec::just()); }) {}

  template <typename Sender>
  explicit SharedVoidSender(Sender sender) {
    static_assert(std::is_copy_constructible_v<Sender>,
                  "SharedVoidSender requires copyable senders");
    factory_ = [sender = std::move(sender)]() -> AnyVoidSender {
      auto copy = sender;
      return AnyVoidSender(std::move(copy));
    };
  }

  template <stdexec::receiver_of<completion_signatures> Receiver>
  auto connect(Receiver receiver) const {
    auto sender = factory_();
    return stdexec::connect(std::move(sender), std::move(receiver));
  }

  auto get_env() const noexcept {
    auto sender = factory_();
    return stdexec::get_env(sender);
  }

 private:
  std::function<AnyVoidSender()> factory_;
};

class SharedValueSender {
 public:
  using sender_concept = stdexec::sender_t;
  using completion_signatures = ValueSignatures;

  SharedValueSender() : factory_([]() { return AnyValueSender(stdexec::just(Value{0.0f})); }) {}

  template <typename Sender>
  explicit SharedValueSender(Sender sender) {
    static_assert(std::is_copy_constructible_v<Sender>,
                  "SharedValueSender requires copyable senders");
    factory_ = [sender = std::move(sender)]() -> AnyValueSender {
      auto copy = sender;
      return AnyValueSender(std::move(copy));
    };
  }

  template <stdexec::receiver_of<completion_signatures> Receiver>
  auto connect(Receiver receiver) const {
    auto sender = factory_();
    return stdexec::connect(std::move(sender), std::move(receiver));
  }

  auto get_env() const noexcept {
    auto sender = factory_();
    return stdexec::get_env(sender);
  }

 private:
  std::function<AnyValueSender()> factory_;
};

inline SharedVoidSender CombineAll(std::vector<SharedVoidSender> tasks) {
  if (tasks.empty()) {
    return SharedVoidSender(stdexec::just());
  }

  SharedVoidSender combined = std::move(tasks.front());
  for (size_t i = 1; i < tasks.size(); ++i) {
    auto joined =
      stdexec::when_all(std::move(combined), std::move(tasks[i]));
    auto flattened = stdexec::then(std::move(joined), [](auto&&...) {});
    combined = SharedVoidSender(std::move(flattened));
  }
  return combined;
}

struct KernelDef {
  std::string id;
  size_t arity = 0;
  std::function<AnyValueSender(const std::any& config,
                               std::vector<Value> inputs,
                               const Resources& resources)>
    invoke;
};

class KernelRegistry {
 public:
  void register_kernel(KernelDef def) {
    if (!def.invoke) {
      throw std::runtime_error("KernelDef.invoke must be set");
    }
    if (def.id.empty()) {
      throw std::runtime_error("KernelDef.id must be set");
    }
    kernels_[def.id] = std::move(def);
  }

  const KernelDef& get(const std::string& id) const {
    auto it = kernels_.find(id);
    if (it == kernels_.end()) {
      throw std::runtime_error("Unknown kernel: " + id);
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, KernelDef> kernels_;
};

struct NodeSpec {
  std::string id;
  std::string kernel_id;
  std::any config;
  std::vector<std::string> inputs;
};

class CompiledGraph {
 public:
  auto sender(InputMap inputs) const {
    return stdexec::let_value(
      stdexec::when_all(
        stdexec::get_scheduler(),
        exec::read_with_default(get_resources, Resources{})),
      [this, inputs = std::move(inputs)](auto sched, Resources resources) mutable {
        return build_sender(std::move(inputs), std::move(sched), std::move(resources));
      });
  }

  const std::vector<std::string>& order() const { return order_; }

 private:
  friend class GraphBuilder;

  CompiledGraph(std::unordered_map<std::string, NodeSpec> nodes,
                std::vector<std::string> order,
                std::vector<std::string> outputs,
                std::shared_ptr<const KernelRegistry> registry)
      : nodes_(std::move(nodes)),
        order_(std::move(order)),
        outputs_(std::move(outputs)),
        registry_(std::move(registry)) {}

  template <class Scheduler>
    requires stdexec::scheduler<Scheduler>
  AnyOutputSender build_sender(InputMap inputs, Scheduler sched, Resources resources) const {
    auto inputs_ptr = std::make_shared<InputMap>(std::move(inputs));
    auto resources_ptr = std::make_shared<Resources>(std::move(resources));

    std::unordered_map<std::string, SharedValueSender> tasks;
    tasks.reserve(nodes_.size());

    for (const auto& id : order_) {
      const auto& node = nodes_.at(id);

      if (node.kernel_id == "__input__") {
        auto s = stdexec::then(stdexec::just(), [inputs_ptr, id]() -> Value {
          auto it = inputs_ptr->find(id);
        if (it == inputs_ptr->end()) {
          throw std::runtime_error("Missing graph input: " + id);
        }
        return it->second;
      });
        auto started = stdexec::starts_on(sched, std::move(s));
        tasks.emplace(id, SharedValueSender(stdexec::split(std::move(started))));
        continue;
      }

      const auto& kernel = registry_->get(node.kernel_id);
      if (kernel.arity != node.inputs.size()) {
        throw std::runtime_error("Kernel arity mismatch for node: " + id);
      }

      auto values = std::make_shared<std::vector<Value>>(node.inputs.size());
      std::vector<SharedVoidSender> dep_writes;
      dep_writes.reserve(node.inputs.size());
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const auto& dep_id = node.inputs[i];
        auto dep_it = tasks.find(dep_id);
        if (dep_it == tasks.end()) {
          throw std::runtime_error("Missing dependency sender: " + dep_id);
        }
        auto write =
          stdexec::then(dep_it->second, [values, i](Value v) {
            (*values)[i] = std::move(v);
          });
        dep_writes.emplace_back(SharedVoidSender(stdexec::split(std::move(write))));
      }

      auto deps_barrier = CombineAll(std::move(dep_writes));
      auto node_sender = stdexec::let_value(
        std::move(deps_barrier),
        [kernel, config = node.config, values, resources_ptr]() mutable {
          return kernel.invoke(config, std::move(*values), *resources_ptr);
        });
      auto started = stdexec::starts_on(sched, std::move(node_sender));
      tasks.emplace(id, SharedValueSender(stdexec::split(std::move(started))));
    }

    auto outputs = std::make_shared<OutputMap>();
    outputs->reserve(outputs_.size());

    std::vector<SharedVoidSender> write_out;
    write_out.reserve(outputs_.size());
    for (const auto& id : outputs_) {
      auto it = tasks.find(id);
      if (it == tasks.end()) {
        throw std::runtime_error("Unknown output node: " + id);
      }
      auto write =
        stdexec::then(it->second, [outputs, id](Value v) { (*outputs)[id] = std::move(v); });
      write_out.emplace_back(SharedVoidSender(stdexec::split(std::move(write))));
    }

    auto out_barrier = CombineAll(std::move(write_out));
    auto out_sender =
      stdexec::then(std::move(out_barrier), [outputs]() mutable { return std::move(*outputs); });

    return AnyOutputSender(std::move(out_sender));
  }

  std::unordered_map<std::string, NodeSpec> nodes_;
  std::vector<std::string> order_;
  std::vector<std::string> outputs_;
  std::shared_ptr<const KernelRegistry> registry_;
};

class GraphBuilder {
 public:
  void add_input(std::string id) {
    add_node(NodeSpec{.id = std::move(id), .kernel_id = "__input__"});
  }

  void add_node(std::string id,
                std::string kernel_id,
                std::any config,
                std::vector<std::string> inputs) {
    add_node(NodeSpec{
      .id = std::move(id),
      .kernel_id = std::move(kernel_id),
      .config = std::move(config),
      .inputs = std::move(inputs),
    });
  }

  void set_outputs(std::vector<std::string> ids) { outputs_ = std::move(ids); }

  CompiledGraph compile(std::shared_ptr<const KernelRegistry> registry) const {
    if (!registry) {
      throw std::runtime_error("compile requires non-null registry");
    }

    std::unordered_map<std::string, NodeSpec> nodes;
    nodes.reserve(nodes_.size());
    for (const auto& n : nodes_) {
      if (n.id.empty()) {
        throw std::runtime_error("NodeSpec.id must be set");
      }
      nodes[n.id] = n;
    }

    if (outputs_.empty()) {
      throw std::runtime_error("GraphBuilder requires explicit outputs");
    }

    std::unordered_map<std::string, int> indeg;
    std::unordered_map<std::string, std::vector<std::string>> adj;
    indeg.reserve(nodes.size());
    adj.reserve(nodes.size());

    for (const auto& [id, node] : nodes) {
      if (node.kernel_id != "__input__") {
        registry->get(node.kernel_id);
      }

      indeg[id] = static_cast<int>(node.inputs.size());
      for (const auto& dep : node.inputs) {
        if (nodes.find(dep) == nodes.end()) {
          throw std::runtime_error("Missing input dependency: " + dep);
        }
        adj[dep].push_back(id);
      }
    }

    std::vector<std::string> order;
    order.reserve(nodes.size());
    std::vector<std::string> ready;
    ready.reserve(nodes.size());
    for (const auto& [id, d] : indeg) {
      if (d == 0) {
        ready.push_back(id);
      }
    }

    while (!ready.empty()) {
      auto id = std::move(ready.back());
      ready.pop_back();
      order.push_back(id);
      auto it = adj.find(id);
      if (it == adj.end()) {
        continue;
      }
      for (const auto& next : it->second) {
        if (--indeg[next] == 0) {
          ready.push_back(next);
        }
      }
    }

    if (order.size() != nodes.size()) {
      throw std::runtime_error("Graph has a cycle");
    }

    for (const auto& out : outputs_) {
      if (nodes.find(out) == nodes.end()) {
        throw std::runtime_error("Unknown output node: " + out);
      }
    }

    return CompiledGraph(std::move(nodes), std::move(order), outputs_, std::move(registry));
  }

 private:
  void add_node(NodeSpec node) { nodes_.push_back(std::move(node)); }

  std::vector<NodeSpec> nodes_;
  std::vector<std::string> outputs_;
};

}  // namespace runlab::dataflow
