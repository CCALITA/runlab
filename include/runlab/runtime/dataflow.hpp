#pragma once

#include <any>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include <vector>

#include <exec/any_sender_of.hpp>
#include <exec/env.hpp>
#include <stdexec/execution.hpp>

namespace runlab::dataflow {

inline constexpr std::string_view kInputKernelId = "__input__";

using Value = std::variant<float, std::vector<float>>;

// TODO constrain
template <typename T>
outputable = std::copyable<T> && std::default_initializable<T>;

template <typename T>
inputable = std::copyable<T> && std::default_initializable<T>;
using InputMap = std::unordered_map<std::string, Value>;
using OutputMap = std::unordered_map<std::string, Value>;

using VoidSignatures =
    stdexec::completion_signatures<stdexec::set_value_t(),
                                   stdexec::set_error_t(std::exception_ptr),
                                   stdexec::set_stopped_t()>;

using ValueSignatures =
    stdexec::completion_signatures<stdexec::set_value_t(Value),
                                   stdexec::set_error_t(std::exception_ptr),
                                   stdexec::set_stopped_t()>;

using OutputSignatures =
    stdexec::completion_signatures<stdexec::set_value_t(OutputMap),
                                   stdexec::set_error_t(std::exception_ptr),
                                   stdexec::set_stopped_t()>;

template <class Signatures>
using AnySenderOf =
    typename exec::any_receiver_ref<Signatures>::template any_sender<>;

using AnyVoidSender = AnySenderOf<VoidSignatures>;
using AnyValueSender = AnySenderOf<ValueSignatures>;
using AnyOutputSender = AnySenderOf<OutputSignatures>;

struct RuntimeConfig {
  float bias = 0.0f;
};

struct get_runtime_config_t : stdexec::__query<get_runtime_config_t> {
  static consteval auto query(stdexec::forwarding_query_t) noexcept -> bool {
    return true;
  }
};
inline constexpr get_runtime_config_t get_runtime_config{};

using KernelFn = std::function<AnyValueSender(
    std::vector<Value> inputs, const RuntimeConfig *runtime_cfg)>;

inline std::string ConfigError(std::string_view kernel_id,
                               std::string_view expectation) {
  return std::string("Kernel ") + std::string(kernel_id) +
         " config invalid: expected " + std::string(expectation);
}

template <typename T>
const T &RequireConfig(const std::any &config, std::string_view kernel_id,
                       std::string_view expectation = typeid(T).name()) {
  const auto *ptr = std::any_cast<T>(&config);
  if (!ptr) {
    throw std::runtime_error(ConfigError(kernel_id, expectation));
  }
  return *ptr;
}

inline void RequireEmptyConfig(const std::any &config,
                               std::string_view kernel_id) {
  if (config.has_value()) {
    throw std::runtime_error(ConfigError(kernel_id, "no config"));
  }
}

class SharedVoidSender {
public:
  using sender_concept = stdexec::sender_t;
  using completion_signatures = VoidSignatures;

  SharedVoidSender()
      : factory_([]() { return AnyVoidSender(stdexec::just()); }) {}

  template <typename Sender> explicit SharedVoidSender(Sender sender) {
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

  SharedValueSender()
      : factory_([]() { return AnyValueSender(stdexec::just(Value{0.0f})); }) {}

  template <typename Sender> explicit SharedValueSender(Sender sender) {
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
    auto joined = stdexec::when_all(std::move(combined), std::move(tasks[i]));
    auto flattened = stdexec::then(std::move(joined), [](auto &&...) {});
    combined = SharedVoidSender(std::move(flattened));
  }
  return combined;
}

struct KernelDef {
  std::string id;
  size_t arity = 0;
  std::function<void(const std::any &config)> validate_config;
  std::function<KernelFn(const std::any &config)> bind;
};

template <size_t Arity, class Config = std::monostate, class Factory>
KernelDef MakeKernel(std::string id, Factory factory,
                     std::string_view expectation = {}) {
  static_assert(Arity < static_cast<size_t>(std::numeric_limits<int>::max()),
                "Kernel arity must fit in int");
  using ConfigArg = std::conditional_t<std::is_same_v<Config, std::monostate>,
                                       Config, const Config &>;
  static_assert(std::is_invocable_r_v<KernelFn, Factory, ConfigArg>,
                "MakeKernel factory must return KernelFn from config");

  std::string name = id;
  std::string expected = expectation.empty()
                             ? (std::is_same_v<Config, std::monostate>
                                    ? std::string("no config")
                                    : std::string(typeid(Config).name()))
                             : std::string(expectation);

  auto validate = [name, expected](const std::any &cfg) {
    if constexpr (std::is_same_v<Config, std::monostate>) {
      RequireEmptyConfig(cfg, name);
    } else {
      (void)RequireConfig<Config>(cfg, name, expected);
    }
  };

  auto binder = [name, expected, factory = std::move(factory)](
                    const std::any &cfg) -> KernelFn {
    if constexpr (std::is_same_v<Config, std::monostate>) {
      RequireEmptyConfig(cfg, name);
      return factory(Config{});
    } else {
      const Config &typed = RequireConfig<Config>(cfg, name, expected);
      return factory(typed);
    }
  };

  return KernelDef{
      .id = std::move(id),
      .arity = Arity,
      .validate_config = std::move(validate),
      .bind = std::move(binder),
  };
}

class KernelRegistry {
public:
  void register_kernel(KernelDef def) {
    if (!def.bind) {
      throw std::runtime_error("KernelDef.bind must be set");
    }
    if (def.id.empty()) {
      throw std::runtime_error("KernelDef.id must be set");
    }
    if (def.id == kInputKernelId) {
      throw std::runtime_error("Kernel id is reserved for graph inputs: " +
                               def.id);
    }
    if (kernels_.find(def.id) != kernels_.end()) {
      throw std::runtime_error("Kernel already registered: " + def.id);
    }
    auto id = def.id;
    kernels_.emplace(std::move(id), std::move(def));
  }

  const KernelDef &get(const std::string &id) const {
    auto it = kernels_.find(id);
    if (it == kernels_.end()) {
      throw std::runtime_error("Unknown kernel: " + id);
    }
    return it->second;
  }

private:
  std::unordered_map<std::string, KernelDef> kernels_;
};

inline void
RegisterDefaultKernels(const std::shared_ptr<KernelRegistry> &registry) {
  if (!registry) {
    throw std::runtime_error(
        "RegisterDefaultKernels requires non-null registry");
  }
  std::vector<KernelDef> defs;
  defs.reserve(3);

  defs.push_back(MakeKernel<1>(
      "id",
      [](std::monostate) -> KernelFn {
        return [](std::vector<Value> inputs,
                  const RuntimeConfig *) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("id expects 1 input");
          }
          return AnyValueSender(stdexec::just(std::move(inputs[0])));
        };
      },
      "no config"));

  defs.push_back(MakeKernel<2>(
      "add_f",
      [](std::monostate) -> KernelFn {
        return [](std::vector<Value> inputs,
                  const RuntimeConfig *) -> AnyValueSender {
          if (inputs.size() != 2) {
            throw std::runtime_error("add_f expects 2 inputs");
          }
          const auto *a = std::get_if<float>(&inputs[0]);
          const auto *b = std::get_if<float>(&inputs[1]);
          if (!a || !b) {
            throw std::runtime_error("add_f expects float inputs");
          }
          return AnyValueSender(stdexec::just(Value{*a + *b}));
        };
      },
      "no config"));

  defs.push_back(MakeKernel<1, float>(
      "scale_f",
      [](float factor) -> KernelFn {
        return [factor](std::vector<Value> inputs,
                        const RuntimeConfig *) -> AnyValueSender {
          if (inputs.size() != 1) {
            throw std::runtime_error("scale_f expects 1 input");
          }
          const auto *x = std::get_if<float>(&inputs[0]);
          if (!x) {
            throw std::runtime_error("scale_f expects float input");
          }
          return AnyValueSender(stdexec::just(Value{*x * factor}));
        };
      },
      "float factor"));

  for (auto &def : defs) {
    registry->register_kernel(std::move(def));
  }
}

struct NodeSpec {
  std::string id;
  std::string kernel_id;
  std::any config;
  std::vector<std::string> inputs;
  KernelFn kernel;
};

class CompiledGraph {
public:
  auto sender(InputMap inputs) const {
    return stdexec::let_value(
        stdexec::when_all(stdexec::get_scheduler(),
                          exec::read_with_default(
                              get_runtime_config,
                              static_cast<const RuntimeConfig *>(nullptr))),
        [this, inputs = std::move(inputs)](
            auto sched, const RuntimeConfig *runtime_cfg) mutable {
          validate_inputs(inputs);
          return build_sender(std::move(inputs), std::move(sched), runtime_cfg);
        });
  }

  const std::vector<std::string> &order() const { return order_; }

private:
  friend class GraphBuilder;

  CompiledGraph(std::unordered_map<std::string, NodeSpec> nodes,
                std::vector<std::string> order, std::vector<std::string> inputs,
                std::unordered_set<std::string> input_set,
                std::vector<std::string> outputs)
      : nodes_(std::move(nodes)), order_(std::move(order)),
        inputs_(std::move(inputs)), input_set_(std::move(input_set)),
        outputs_(std::move(outputs)) {}

  void validate_inputs(const InputMap &provided) const {
    for (const auto &expected : inputs_) {
      if (provided.find(expected) == provided.end()) {
        throw std::runtime_error("Missing graph input: " + expected);
      }
    }

    if (provided.size() != input_set_.size()) {
      for (const auto &[key, _] : provided) {
        if (input_set_.find(key) == input_set_.end()) {
          throw std::runtime_error("Unknown graph input: " + key);
        }
      }
    }
  }

  template <class Scheduler>
    requires stdexec::scheduler<Scheduler>
  AnyOutputSender build_sender(InputMap inputs, Scheduler sched,
                               const RuntimeConfig *runtime_cfg) const {
    auto inputs_ptr = std::make_shared<InputMap>(std::move(inputs));

    std::unordered_map<std::string, SharedValueSender> tasks;
    tasks.reserve(nodes_.size());

    for (const auto &id : order_) {
      const auto &node = nodes_.at(id);

      if (node.kernel_id == kInputKernelId) {
        auto s = stdexec::then(stdexec::just(), [inputs_ptr, id]() -> Value {
          auto it = inputs_ptr->find(id);
          if (it == inputs_ptr->end()) {
            throw std::runtime_error("Missing graph input: " + id);
          }
          return it->second;
        });
        auto started = stdexec::starts_on(sched, std::move(s));
        tasks.emplace(id,
                      SharedValueSender(stdexec::split(std::move(started))));
        continue;
      }

      if (!node.kernel) {
        throw std::runtime_error("Kernel not bound for node: " + id);
      }
      if (node.inputs.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many inputs for node: " + id);
      }

      auto values = std::make_shared<std::vector<Value>>(node.inputs.size());
      std::vector<SharedVoidSender> dep_writes;
      dep_writes.reserve(node.inputs.size());
      for (size_t i = 0; i < node.inputs.size(); ++i) {
        const auto &dep_id = node.inputs[i];
        auto dep_it = tasks.find(dep_id);
        if (dep_it == tasks.end()) {
          throw std::runtime_error("Missing dependency sender: " + dep_id);
        }
        auto write = stdexec::then(dep_it->second, [values, i](Value v) {
          (*values)[i] = std::move(v);
        });
        dep_writes.emplace_back(
            SharedVoidSender(stdexec::split(std::move(write))));
      }

      auto deps_barrier = CombineAll(std::move(dep_writes));
      auto node_sender =
          stdexec::let_value(std::move(deps_barrier),
                             [fn = node.kernel, values, runtime_cfg]() mutable {
                               return fn(std::move(*values), runtime_cfg);
                             });
      auto started = stdexec::starts_on(sched, std::move(node_sender));
      tasks.emplace(id, SharedValueSender(stdexec::split(std::move(started))));
    }

    auto outputs = std::make_shared<OutputMap>();
    outputs->reserve(outputs_.size());

    std::vector<SharedVoidSender> write_out;
    write_out.reserve(outputs_.size());
    for (const auto &id : outputs_) {
      auto it = tasks.find(id);
      if (it == tasks.end()) {
        throw std::runtime_error("Unknown output node: " + id);
      }
      auto write = stdexec::then(it->second, [outputs, id](Value v) {
        (*outputs)[id] = std::move(v);
      });
      write_out.emplace_back(
          SharedVoidSender(stdexec::split(std::move(write))));
    }

    auto out_barrier = CombineAll(std::move(write_out));
    auto out_sender =
        stdexec::then(std::move(out_barrier),
                      [outputs]() mutable { return std::move(*outputs); });

    return AnyOutputSender(std::move(out_sender));
  }

  std::unordered_map<std::string, NodeSpec> nodes_;
  std::vector<std::string> order_;
  std::vector<std::string> inputs_;
  std::unordered_set<std::string> input_set_;
  std::vector<std::string> outputs_;
};

class GraphBuilder {
public:
  void add_input(std::string id) {
    add_node(NodeSpec{.id = std::move(id),
                      .kernel_id = std::string(kInputKernelId)});
  }

  void add_node(std::string id, std::string kernel_id, std::any config,
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
    std::vector<std::string> inputs;
    std::unordered_set<std::string> input_set;
    inputs.reserve(nodes_.size());
    input_set.reserve(nodes_.size());
    for (const auto &n : nodes_) {
      if (n.id.empty()) {
        throw std::runtime_error("NodeSpec.id must be set");
      }
      const bool is_input = n.kernel_id == kInputKernelId;
      if (!is_input && n.kernel_id.empty()) {
        throw std::runtime_error("NodeSpec.kernel_id must be set");
      }
      if (is_input && !n.inputs.empty()) {
        throw std::runtime_error("Graph input cannot have dependencies: " +
                                 n.id);
      }
      if (is_input && n.config.has_value()) {
        throw std::runtime_error("Graph input cannot have config: " + n.id);
      }
      if (is_input) {
        inputs.push_back(n.id);
        input_set.insert(n.id);
      }
      const auto [it, inserted] = nodes.emplace(n.id, n);
      if (!inserted) {
        throw std::runtime_error("Duplicate node id: " + n.id);
      }
    }

    if (outputs_.empty()) {
      throw std::runtime_error("GraphBuilder requires explicit outputs");
    }

    std::unordered_map<std::string, int> indeg;
    std::unordered_map<std::string, std::vector<std::string>> adj;
    indeg.reserve(nodes.size());
    adj.reserve(nodes.size());

    for (auto &[id, node] : nodes) {
      if (node.kernel_id != kInputKernelId) {
        const auto &def = registry->get(node.kernel_id);
        if (def.arity != node.inputs.size()) {
          throw std::runtime_error("Kernel arity mismatch for node: " + id);
        }
        if (def.validate_config) {
          def.validate_config(node.config);
        }
        auto bound = def.bind(node.config);
        if (!bound) {
          throw std::runtime_error(
              "Kernel binding returned empty function for node: " + id);
        }
        node.kernel = std::move(bound);
      }

      indeg[id] = static_cast<int>(node.inputs.size());
      for (const auto &dep : node.inputs) {
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
    for (const auto &[id, d] : indeg) {
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
      for (const auto &next : it->second) {
        if (--indeg[next] == 0) {
          ready.push_back(next);
        }
      }
    }

    if (order.size() != nodes.size()) {
      throw std::runtime_error("Graph has a cycle");
    }

    std::unordered_set<std::string> seen_outputs;
    seen_outputs.reserve(outputs_.size());
    for (const auto &out : outputs_) {
      if (nodes.find(out) == nodes.end()) {
        throw std::runtime_error("Unknown output node: " + out);
      }
      if (!seen_outputs.insert(out).second) {
        throw std::runtime_error("Duplicate output node: " + out);
      }
    }

    return CompiledGraph(std::move(nodes), std::move(order), std::move(inputs),
                         std::move(input_set), outputs_);
  }

private:
  void add_node(NodeSpec node) { nodes_.push_back(std::move(node)); }

  std::vector<NodeSpec> nodes_;
  std::vector<std::string> outputs_;
};

} // namespace runlab::dataflow
