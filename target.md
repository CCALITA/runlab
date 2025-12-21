# 项目目标

混合态 DAG 引擎架构设计文档 (Hybrid Static/Dynamic DAG Engine)
版本: 1.0 核心技术栈: C++20 , Sender/Receiver(P2300) , pybind11, Python 3.x 设计目标: 兼具 Python 的动态构图灵活性与 C++ 的裸机执行性能。

1. 架构总览 (Architecture Overview)
系统自下而上分为三层。数据流在层间通过类型擦除 (Type Erasure) 进行转换，控制流由 stdexec 的 Sender/Receiver 模型驱动。

层级 名称 职责范围 多态机制 关键组件
L3 Python DSL Layer 图定义、配置管理、业务逻辑编排 动态 (Duck Typing) pybind11, Python Interpreter
L2 Graph Runtime Layer 依赖解析、并发调度、资源分配 运行时多态 (VTable/Type Erasure) any_sender, TaskNode, Blackboard
L1 Static Kernel Layer 核心算法、硬件加速、I/O 操作 编译期多态 (Templates/Concepts) Sender, Receiver, OperationState

2. 详细层级设计
L1: 静态内核层 (Static Kernel Layer)
定位: 系统的“发动机”。完全由 C++ 模板编写，不涉及任何 Python 对象或虚函数开销。

设计原则:

Everything is a Sender: 所有算子函数不直接执行计算，而是返回一个描述计算的 Sender 对象。

No Virtual Functions: 禁止使用虚函数，依赖编译器内联 (Inlining) 和模板实例化 (Monomorphization)。

Concepts 约束: 使用 C++20 Concepts 定义接口协变。

代码示例 (Kernel Definition):

C++

namespace kernels {
    // 定义一个 Concept，约束输入数据必须是连续内存
    template<typename T>
    concept DenseVector = requires(T t) { std::span(t); };

    // 静态工厂函数：返回一个复杂的模板类型 Sender
    template<DenseVector Data>
    auto compute_embedding(Data&& data) {
        return stdexec::just(std::forward<Data>(data))
             | stdexec::then([](auto&& span) {
                 // SIMD 指令集优化区域
                 return do_avx512_calc(span);
             });
    }
}
L2: 动态图运行时层 (Graph Runtime Layer)
定位: 系统的“调度中心”。负责将 L1 产生的异构 Sender 统一化，管理任务依赖与生命周期。

设计原则:

Type Erasure (any_sender): 这是本层核心。将 L1 返回的 Sender<T> 包装进 any_sender<void>。此时丢失了具体的返回值类型信息，统一变为“可执行的任务”。

Blackboard Pattern (黑板模式): 由于 Sender 类型被擦除，返回值无法直接通过 C++ 强类型传递。因此，算子执行的副作用 (Side Effect) 是将结果写入一个共享的 Context (黑板)。

Thread Pool Scheduling: 使用 stdexec::static_thread_pool 或自定义的 io_uring_context。

关键数据结构:

C++

// 统一的任务句柄：不返回具体值，只抛出异常
using DynTask = stdexec::any_sender_of<void, std::exception_ptr>;

struct GraphContext {
    // 线程安全的黑板，用于存储算子中间结果
    // Key: NodeID, Value: Any Data
    tbb::concurrent_hash_map<std::string, std::any> black_board;
};

class Node {
    DynTask _task; // 类型擦除后的任务
    std::vector<std::string>_deps; // 依赖的 Node ID

public:
    // 泛型 Setter：在此处发生 "Static -> Dynamic" 的转换
    template<typename StaticSender>
    void set_logic(StaticSender&& s) {
        _task = stdexec::ensure_started(std::forward<StaticSender>(s));
    }
};
L3: Python 绑定层 (Python DSL Layer)
定位: 系统的“控制台”。暴露给最终用户，用于定义图的拓扑结构。

设计原则:

GIL Management: 在 C++ 开始执行图调度 (sync_wait) 前，必须释放 Python GIL，允许 Python 侧的多线程或 IO 操作，同时防止死锁。

Zero-Copy Config: 使用 py::buffer 协议传递 Numpy 数组指针，避免在大规模向量数据传输时发生内存拷贝。

交互流程:

Python 创建 Engine 实例。

Python 调用 engine.add_node("node_A", op_type="vector_search", params=...)。

C++ 层根据 op_type 选取对应的 L1 模板函数，生成 Sender。

C++ 层将 Sender 擦除类型存入 L2 Node。

Python 调用 engine.run() -> C++ 释放 GIL -> C++ 线程池狂奔 -> 结束获取 GIL -> 返回。

3. 核心机制说明
3.1 数据流转：黑板模式 (The Blackboard Pattern)
由于 any_sender 擦除了类型，我们也切断了 Node A -> return int -> Node B 的直接编译期链路。我们采用**“发布/订阅”**式的黑板模式：

Producer (Node A): 内部逻辑是 fetch(input) -> calc -> context.put("A_result", res)。

Consumer (Node B): 内部逻辑是 res = context.get<int>("A_result") -> calc。

优势: 解耦了节点的编译期依赖，允许 Python 动态改变图结构。 劣势: 引入了 std::any_cast 的运行时检查开销（可忽略不计）。

3.2 异常处理 (Error Propagation)
stdexec 提供了专门的 set_error 通道。

L1: 算子内部抛出异常，或者调用 set_error。

L2: any_sender 捕获异常，传递给 stdexec::sync_wait 的接收器。

L3: C++ 捕获 std::exception_ptr，转换为 Python 的 RuntimeError 并抛回给 Python 调用栈。

4. 性能损耗分析 (Performance Analysis)
在追求极致性能的场景下（如向量检索），我们需要明确开销在哪里：

阶段 机制 开销评估 影响
算子内部 (L1) Template Monomorphization Zero Overhead 核心计算路径完全内联，与手写 C 相同。
算子构建 (L3->L2) any_sender Allocation Low (Heap Alloc) 仅在构图时发生一次堆内存分配。
算子调度 (L2) Virtual Function Call Very Low (ns级) 每个 Node 执行前需查虚表。相比 Node 内部 ms 级的计算耗时，占比 < 0.01%。
数据传递 std::any_cast Negligible 简单的类型 ID 检查。

5. 扩展性指南 (Extensibility)
如何添加一个新的 C++ 算子？
无需修改 Runtime: 不需要修改 L2 的调度逻辑。

编写 L1 Kernel: 在 kernels 命名空间下增加一个新的模板函数，返回 Sender。

注册 L3 Binding: 在 pybind11 模块中添加一行代码，将 Python 参数转化为调用该 L1 Kernel 并存入 L2 Node 的逻辑。

如何支持 GPU？
在 L1 层，将 Sender 的执行策略从 on(cpu_pool) 改为 on(cuda_scheduler)。L2 层看到的仍然是一个通用的 DynTask，完全不需要感知硬件细节。

6. 总结 (Conclusion)
本方案通过 stdexec 的 Sender/Receiver 模型解决了 C++ 静态性能与 Python 动态灵活性的矛盾。

对于 C++ 开发者: 你在编写高性能的模板元编程代码。

对于 Python 用户: 你在使用一个灵活的 DAG 库。

对于系统架构: 利用类型擦除作为防火墙，隔离了业务逻辑的复杂性和底层计算的高效性。

7. Progress Notes
- Added runtime tests for error propagation and concurrent scheduling.
- Build + ctest run completed; Python example blocked by missing NumPy.
- Refactored runtime to build DAG execution with stdexec senders (split/when_all/sync_wait).
