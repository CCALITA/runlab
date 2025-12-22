#pragma once

#include <any>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <exec/any_sender_of.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace runlab {

struct FloatSpan {
  std::shared_ptr<void> owner;
  const float* data = nullptr;
  size_t size = 0;
};

enum class NodeStatus {
  kPending,
  kRunning,
  kSuccess,
  kError,
  kBlocked,
};

inline auto ToString(NodeStatus status) -> const char* {
  switch (status) {
    case NodeStatus::kPending:
      return "pending";
    case NodeStatus::kRunning:
      return "running";
    case NodeStatus::kSuccess:
      return "success";
    case NodeStatus::kError:
      return "error";
    case NodeStatus::kBlocked:
      return "blocked";
  }
  return "unknown";
}

struct NodeState {
  NodeStatus status = NodeStatus::kPending;
  std::exception_ptr error;
};

class GraphContext {
 public:
  void reset_run_state() {
    std::lock_guard<std::mutex> lock(mu_);
    error_ = nullptr;
    for (auto& [_, state] : node_states_) {
      state.status = NodeStatus::kPending;
      state.error = nullptr;
    }
  }

  void init_nodes(const std::vector<std::string>& node_ids) {
    std::lock_guard<std::mutex> lock(mu_);
    node_states_.clear();
    node_states_.reserve(node_ids.size());
    for (const auto& id : node_ids) {
      node_states_.emplace(id, NodeState{});
    }
  }

  void set_node_status(std::string_view id,
                       NodeStatus status,
                       std::exception_ptr err = nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& state = node_states_[std::string(id)];
    state.status = status;
    state.error = std::move(err);
  }

  NodeStatus node_status(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = node_states_.find(id);
    if (it == node_states_.end()) {
      return NodeStatus::kPending;
    }
    return it->second.status;
  }

  std::exception_ptr node_error(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = node_states_.find(id);
    if (it == node_states_.end()) {
      return nullptr;
    }
    return it->second.error;
  }

  std::unordered_map<std::string, NodeState> node_states_snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return node_states_;
  }

  template <typename T>
  void put(std::string key, T value) {
    std::lock_guard<std::mutex> lock(mu_);
    blackboard_[std::move(key)] = std::move(value);
  }

  template <typename T>
  T get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blackboard_.find(key);
    if (it == blackboard_.end()) {
      throw std::runtime_error("Missing key: " + key);
    }
    return std::any_cast<T>(it->second);
  }

  template <typename T>
  const T& get_ref(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blackboard_.find(key);
    if (it == blackboard_.end()) {
      throw std::runtime_error("Missing key: " + key);
    }
    return std::any_cast<const T&>(it->second);
  }

  template <typename T>
  bool try_get(const std::string& key, T* out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blackboard_.find(key);
    if (it == blackboard_.end()) {
      return false;
    }
    auto* value = std::any_cast<T>(&it->second);
    if (!value) {
      return false;
    }
    if (out) {
      *out = *value;
    }
    return true;
  }

  std::span<const float> get_span(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = blackboard_.find(key);
    if (it == blackboard_.end()) {
      throw std::runtime_error("Missing key: " + key);
    }

    if (auto* vec = std::any_cast<std::vector<float>>(&it->second)) {
      return std::span<const float>(vec->data(), vec->size());
    }
    if (auto* vec_ptr =
          std::any_cast<std::shared_ptr<std::vector<float>>>(&it->second)) {
      return std::span<const float>((*vec_ptr)->data(), (*vec_ptr)->size());
    }
    if (auto* span = std::any_cast<FloatSpan>(&it->second)) {
      return std::span<const float>(span->data, span->size);
    }

    throw std::runtime_error("Key has unsupported value type: " + key);
  }

  bool contains(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    return blackboard_.find(key) != blackboard_.end();
  }

  void set_error(std::exception_ptr err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!error_) {
      error_ = std::move(err);
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mu_);
    blackboard_.clear();
    error_ = nullptr;
    node_states_.clear();
  }

  bool has_error() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<bool>(error_);
  }

  void rethrow_if_error() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::any> blackboard_;
  std::unordered_map<std::string, NodeState> node_states_;
  std::exception_ptr error_;
};

using TaskSignatures = stdexec::completion_signatures<
  stdexec::set_value_t(),
  stdexec::set_error_t(std::exception_ptr),
  stdexec::set_stopped_t()>;

using DynTask = exec::any_receiver_ref<TaskSignatures>::template any_sender<>;
using DynTaskFactory = std::function<DynTask(GraphContext&)>;
using AnyScheduler = DynTask::template any_scheduler<>;

struct Node {
  std::string id;
  std::vector<std::string> deps;
  DynTaskFactory make_task;
};

class SharedTaskSender {
 public:
  using sender_concept = stdexec::sender_t;
  using completion_signatures = TaskSignatures;

  SharedTaskSender()
      : factory_([]() { return DynTask(stdexec::just()); }) {}

  template <typename Sender>
  explicit SharedTaskSender(Sender sender) {
    static_assert(std::is_copy_constructible_v<Sender>,
                  "SharedTaskSender requires copyable senders");
    factory_ = [sender = std::move(sender)]() -> DynTask {
      auto copy = sender;
      return DynTask(std::move(copy));
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
  std::function<DynTask()> factory_;
};

inline SharedTaskSender CombineAll(std::vector<SharedTaskSender> tasks) {
  if (tasks.empty()) {
    return SharedTaskSender(stdexec::just());
  }

  SharedTaskSender combined = std::move(tasks.front());
  for (size_t i = 1; i < tasks.size(); ++i) {
    auto joined =
      stdexec::when_all(std::move(combined), std::move(tasks[i]));
    auto flattened = stdexec::then(std::move(joined), [](auto&&...) {});
    combined = SharedTaskSender(std::move(flattened));
  }
  return combined;
}

template <typename Factory>
DynTaskFactory MakeTaskFactory(Factory factory) {
  return [factory = std::move(factory)](GraphContext& ctx) mutable -> DynTask {
    return DynTask(factory(ctx));
  };
}

class CompiledGraph {
 public:
  CompiledGraph() = default;

  CompiledGraph(std::unordered_map<std::string, Node> nodes,
                std::vector<std::string> order,
                std::vector<std::string> sinks)
      : nodes_(std::move(nodes)),
        order_(std::move(order)),
        sinks_(std::move(sinks)) {}

  const std::vector<std::string>& order() const { return order_; }

  SharedTaskSender sender(GraphContext& ctx, AnyScheduler sched) const {
    if (nodes_.empty()) {
      return SharedTaskSender(stdexec::just());
    }

    auto init = stdexec::then(stdexec::just(), [&ctx, order = order_]() mutable {
      ctx.reset_run_state();
      ctx.init_nodes(order);
    });
    auto init_shared = SharedTaskSender(stdexec::split(std::move(init)));

    std::unordered_map<std::string, SharedTaskSender> tasks;
    tasks.reserve(nodes_.size());

    for (const auto& id : order_) {
      const auto& node = nodes_.at(id);
      std::vector<SharedTaskSender> deps;
      deps.reserve(node.deps.size() + 1);
      deps.push_back(init_shared);
      for (const auto& dep : node.deps) {
        deps.push_back(tasks.at(dep));
      }

      auto deps_barrier = CombineAll(std::move(deps));
      auto deps_or_blocked = stdexec::let_error(
        std::move(deps_barrier),
        [&ctx, id](std::exception_ptr err) {
          ctx.set_node_status(id, NodeStatus::kBlocked, err);
          return stdexec::just_error(std::move(err));
        });

      auto task_sender = stdexec::let_value(
        std::move(deps_or_blocked),
        [make_task = node.make_task, &ctx, sched, id]() mutable {
          ctx.set_node_status(id, NodeStatus::kRunning);
          auto started = stdexec::starts_on(sched, make_task(ctx));
          auto success =
            stdexec::then(std::move(started),
                          [&ctx, id]() { ctx.set_node_status(id, NodeStatus::kSuccess); });
          auto tracked =
            stdexec::let_error(
              std::move(success),
              [&ctx, id](std::exception_ptr err) {
                ctx.set_node_status(id, NodeStatus::kError, err);
                ctx.set_error(err);
                return stdexec::just_error(std::move(err));
              });
          return tracked;
        });

      auto shared_sender = stdexec::split(std::move(task_sender));
      tasks.emplace(id, SharedTaskSender(std::move(shared_sender)));
    }

    std::vector<SharedTaskSender> sink_tasks;
    sink_tasks.reserve(sinks_.size());
    for (const auto& id : sinks_) {
      sink_tasks.push_back(tasks.at(id));
    }

    return CombineAll(std::move(sink_tasks));
  }

  void run(GraphContext& ctx, AnyScheduler sched) const {
    auto graph_sender = sender(ctx, std::move(sched));
    stdexec::sync_wait(std::move(graph_sender));
    ctx.rethrow_if_error();
  }

 private:
  std::unordered_map<std::string, Node> nodes_;
  std::vector<std::string> order_;
  std::vector<std::string> sinks_;
};

class Graph {
 public:
  template <typename Factory>
  void add_node(std::string id, Factory factory) {
    add_node(std::move(id), {}, MakeTaskFactory(std::move(factory)));
  }

  template <typename Factory>
  void add_node(std::string id, std::vector<std::string> deps, Factory factory) {
    add_node(std::move(id), std::move(deps), MakeTaskFactory(std::move(factory)));
  }

  void add_node(Node node) {
    nodes_[node.id] = std::move(node);
  }

  void add_node(std::string id,
                std::vector<std::string> deps,
                DynTaskFactory make_task) {
    add_node(Node{std::move(id), std::move(deps), std::move(make_task)});
  }

  void add_edge(const std::string& from, const std::string& to) {
    auto it = nodes_.find(to);
    if (it == nodes_.end()) {
      throw std::runtime_error("Unknown node: " + to);
    }
    it->second.deps.push_back(from);
  }

  void clear() { nodes_.clear(); }

  SharedTaskSender sender(GraphContext& ctx, AnyScheduler sched) const {
    auto compiled = compile();
    return compiled.sender(ctx, std::move(sched));
  }

  CompiledGraph compile() const {
    if (nodes_.empty()) {
      return CompiledGraph();
    }

    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int> dep_counts;
    std::unordered_map<std::string, int> out_counts;
    adj.reserve(nodes_.size());
    dep_counts.reserve(nodes_.size());
    out_counts.reserve(nodes_.size());

    for (const auto& [id, node] : nodes_) {
      dep_counts[id] = static_cast<int>(node.deps.size());
      out_counts.emplace(id, 0);
      for (const auto& dep : node.deps) {
        if (nodes_.find(dep) == nodes_.end()) {
          throw std::runtime_error("Missing dependency: " + dep);
        }
        adj[dep].push_back(id);
        ++out_counts[dep];
      }
    }

    auto order = TopologicalOrder(dep_counts, adj);
    if (order.size() != nodes_.size()) {
      throw std::runtime_error("Graph has a cycle");
    }

    std::vector<std::string> sinks;
    sinks.reserve(nodes_.size());
    for (const auto& [id, count] : out_counts) {
      if (count == 0) {
        sinks.push_back(id);
      }
    }

    return CompiledGraph(nodes_, std::move(order), std::move(sinks));
  }

  std::vector<std::string> validate() const {
    if (nodes_.empty()) {
      return {};
    }

    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int> dep_counts;
    adj.reserve(nodes_.size());
    dep_counts.reserve(nodes_.size());

    for (const auto& [id, node] : nodes_) {
      dep_counts[id] = static_cast<int>(node.deps.size());
      for (const auto& dep : node.deps) {
        if (nodes_.find(dep) == nodes_.end()) {
          throw std::runtime_error("Missing dependency: " + dep);
        }
        adj[dep].push_back(id);
      }
    }

    auto order = TopologicalOrder(dep_counts, adj);
    if (order.size() != nodes_.size()) {
      throw std::runtime_error("Graph has a cycle");
    }
    return order;
  }

  void run(GraphContext& ctx, exec::static_thread_pool& pool) const {
    auto compiled = compile();
    compiled.run(ctx, AnyScheduler(pool.get_scheduler()));
  }

 private:
  static std::vector<std::string> TopologicalOrder(
      const std::unordered_map<std::string, int>& dep_counts,
      const std::unordered_map<std::string, std::vector<std::string>>& adj)
      {
    std::queue<std::string> ready;
    std::unordered_map<std::string, int> temp = dep_counts;
    std::vector<std::string> order;
    order.reserve(dep_counts.size());
    for (const auto& [id, count] : temp) {
      if (count == 0) {
        ready.push(id);
      }
    }

    while (!ready.empty()) {
      std::string id = std::move(ready.front());
      ready.pop();
      order.push_back(id);
      auto it = adj.find(id);
      if (it == adj.end()) {
        continue;
      }
      for (const auto& next : it->second) {
        auto next_it = temp.find(next);
        if (next_it == temp.end()) {
          continue;
        }
        if (--next_it->second == 0) {
          ready.push(next);
        }
      }
    }
    return order;
  }

  std::unordered_map<std::string, Node> nodes_;
};

class Engine {
 public:
  explicit Engine(size_t thread_count = std::thread::hardware_concurrency())
      : pool_(thread_count == 0 ? 1 : thread_count) {
    ensure_graph("default");
  }

  Graph& graph() { return graph("default"); }
  const Graph& graph() const { return graph("default"); }

  Graph& graph(const std::string& name) { return ensure_graph(name); }
  const Graph& graph(const std::string& name) const {
    auto it = graphs_.find(name);
    if (it == graphs_.end()) {
      throw std::runtime_error("Unknown graph: " + name);
    }
    return it->second;
  }

  GraphContext& context() { return context("default"); }
  const GraphContext& context() const { return context("default"); }

  GraphContext& context(const std::string& graph_name) {
    ensure_graph(graph_name);
    auto& ptr = contexts_[graph_name];
    if (!ptr) {
      ptr = std::make_unique<GraphContext>();
    }
    return *ptr;
  }

  const GraphContext& context(const std::string& graph_name) const {
    auto it = contexts_.find(graph_name);
    if (it == contexts_.end() || !it->second) {
      throw std::runtime_error("Unknown graph context: " + graph_name);
    }
    return *it->second;
  }

  template <typename Factory>
  void add_node(std::string id, Factory factory) {
    graph("default").add_node(std::move(id), std::move(factory));
  }

  template <typename Factory>
  void add_node(std::string id,
                std::vector<std::string> deps,
                Factory factory) {
    graph("default").add_node(std::move(id), std::move(deps), std::move(factory));
  }

  void add_edge(const std::string& from, const std::string& to) {
    add_edge("default", from, to);
  }

  void add_edge(const std::string& graph_name,
                const std::string& from,
                const std::string& to) {
    ensure_graph(graph_name).add_edge(from, to);
  }

  template <typename Factory>
  void add_node_to(const std::string& graph_name, std::string id, Factory factory) {
    graph(graph_name).add_node(std::move(id), std::move(factory));
  }

  template <typename Factory>
  void add_node_to(const std::string& graph_name,
                   std::string id,
                   std::vector<std::string> deps,
                   Factory factory) {
    graph(graph_name).add_node(std::move(id), std::move(deps), std::move(factory));
  }

  void clear() {
    clear_graph("default");
  }

  void clear_graph(const std::string& graph_name) {
    auto it = graphs_.find(graph_name);
    if (it != graphs_.end()) {
      it->second.clear();
    }
    auto ctx_it = contexts_.find(graph_name);
    if (ctx_it != contexts_.end() && ctx_it->second) {
      ctx_it->second->clear();
    }
  }

  void run() { run_graph("default"); }

  void run_graph(const std::string& graph_name) {
    auto compiled = compile_graph(graph_name);
    compiled.run(context(graph_name), scheduler());
  }

  AnyScheduler scheduler() { return AnyScheduler(pool_.get_scheduler()); }

  SharedTaskSender start_graph(const std::string& graph_name) {
    auto compiled = compile_graph(graph_name);
    return compiled.sender(context(graph_name), scheduler());
  }

  SharedTaskSender start(const CompiledGraph& graph, GraphContext& ctx) {
    return graph.sender(ctx, scheduler());
  }

  void install_graph(const std::string& graph_name, CompiledGraph graph) {
    install_graph(graph_name, std::make_shared<CompiledGraph>(std::move(graph)));
  }

  void install_graph(const std::string& graph_name,
                     std::shared_ptr<const CompiledGraph> graph) {
    if (!graph) {
      throw std::runtime_error("install_graph requires non-null graph");
    }
    ensure_graph(graph_name);
    std::lock_guard<std::mutex> lock(installed_mu_);
    installed_[graph_name] = std::move(graph);
  }

  std::shared_ptr<const CompiledGraph> installed_graph(
      const std::string& graph_name) const {
    std::lock_guard<std::mutex> lock(installed_mu_);
    auto it = installed_.find(graph_name);
    if (it == installed_.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::vector<std::string> compile_and_install(const std::string& graph_name) {
    auto compiled = compile_graph(graph_name);
    std::vector<std::string> order = compiled.order();
    install_graph(graph_name, std::move(compiled));
    return order;
  }

  void run_installed(const std::string& graph_name) {
    auto graph = installed_graph(graph_name);
    if (!graph) {
      throw std::runtime_error("No installed graph: " + graph_name);
    }
    run(*graph, context(graph_name));
  }

  void run_installed(const std::string& graph_name, GraphContext& ctx) {
    auto graph = installed_graph(graph_name);
    if (!graph) {
      throw std::runtime_error("No installed graph: " + graph_name);
    }
    run(*graph, ctx);
  }

  SharedTaskSender start_installed(const std::string& graph_name) {
    auto graph = installed_graph(graph_name);
    if (!graph) {
      throw std::runtime_error("No installed graph: " + graph_name);
    }
    return graph->sender(context(graph_name), scheduler());
  }

  SharedTaskSender start_installed(const std::string& graph_name,
                                   GraphContext& ctx) {
    auto graph = installed_graph(graph_name);
    if (!graph) {
      throw std::runtime_error("No installed graph: " + graph_name);
    }
    return graph->sender(ctx, scheduler());
  }

  void run(const CompiledGraph& graph, GraphContext& ctx) {
    graph.run(ctx, scheduler());
  }

  std::vector<std::string> validate() const { return validate("default"); }

  std::vector<std::string> validate(const std::string& graph_name) const {
    return graph(graph_name).validate();
  }

  CompiledGraph compile_graph(const std::string& graph_name) const {
    return graph(graph_name).compile();
  }

 private:
  Graph& ensure_graph(const std::string& name) {
    auto [it, inserted] = graphs_.try_emplace(name);
    if (inserted) {
      contexts_[name] = std::make_unique<GraphContext>();
    }
    return it->second;
  }

  std::unordered_map<std::string, Graph> graphs_;
  std::unordered_map<std::string, std::unique_ptr<GraphContext>> contexts_;
  mutable std::mutex installed_mu_;
  std::unordered_map<std::string, std::shared_ptr<const CompiledGraph>> installed_;
  exec::static_thread_pool pool_;
};

}  // namespace runlab
