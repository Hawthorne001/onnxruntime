// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

#if !defined(ORT_MINIMAL_BUILD)
struct PriorityNodeCompare {
  inline bool IsHighPri(const Node* n) const {
    // local statics so we can compare std::strings in the checks
    static constexpr std::string_view shape_op("Shape");
    static constexpr std::string_view size_op("Size");

    const auto& op_type = n->OpType();
    return op_type == shape_op || op_type == size_op;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const Node* n1, const Node* n2) const {
    // nodes in global high priority list will be output first
    const bool isN1HighPri = IsHighPri(n1);
    const bool isN2HighPri = IsHighPri(n2);
    if (isN1HighPri != isN2HighPri) {
      return isN2HighPri;
    }

    // nodes with lower priority value will be output first
    const auto n1_priority = n1->Priority();
    const auto n2_priority = n2->Priority();
    if (n1_priority != n2_priority) {
      return n1_priority > n2_priority;
    }

    // otherwise, nodes with lower index will be output first
    return n1->Index() > n2->Index();
  }
};
#endif

GraphViewer::GraphViewer(const Graph& graph)
    : GraphViewer(graph, nullptr) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph& filter_info)
    : GraphViewer(graph, &filter_info) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph* filter_info)
    : graph_{&graph},
      // we can setup the filter here if needed. filtered_node_indices_ will have been populated by the time it's used
      graph_nodes_{graph_->FilteredNodes(
          filter_info ? [this](NodeIndex idx) { return filtered_node_indices_.count(idx) == 0; }
                      : ConstGraphNodes::NodeFilterFunc(nullptr))},
      filter_info_{filter_info} {
  std::vector<const Node*> leaf_nodes;

#ifdef ENABLE_TRAINING
  const Node* yield_node = nullptr;

  // Keep the info of shape and size nodes and their parents so that after topological sort, we can move them
  // right after their parents. This is to make sure the shape and size nodes are executed right after their parents
  // so it's possible the input tensor memory can be released as soon as possible. This is especially important
  // for non-CPU devices or for training case where some gradient graphs use only shape/size of tensors from forward.
  InlinedHashSet<NodeIndex> shape_size_nodes;
  InlinedHashMap<NodeIndex, InlinedVector<NodeIndex>> shape_size_parents;
#endif
  for (auto& node : graph_->Nodes()) {
    // This is a leaf node (without any output node)
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      leaf_nodes.push_back(&node);
    }
    // This is a root node (without any input node)
    if (node.InputEdgesBegin() == node.InputEdgesEnd()) {
      root_nodes_.push_back(node.Index());
    }
#ifdef ENABLE_TRAINING
    if ((node.OpType() == "Shape" || node.OpType() == "Size") && node.InputEdgesBegin() != node.InputEdgesEnd()) {
      shape_size_nodes.insert(node.Index());
      NodeIndex parent = node.InputNodesBegin()->Index();
      if (shape_size_parents.find(parent) == shape_size_parents.end()) {
        shape_size_parents[parent] = InlinedVector<NodeIndex>{node.Index()};
      } else {
        shape_size_parents[parent].push_back(node.Index());
      }
    }

    if (node.OpType() == "YieldOp") {
      yield_node = &node;
    }
#endif
  }

  graph.ReverseDFSFrom(
      leaf_nodes,
      nullptr,
      [this](const Node* n) {
        nodes_in_topological_order_.push_back(n->Index());
      },
      NodeCompare());

#ifdef ENABLE_TRAINING
  auto original = std::move(nodes_in_topological_order_);
  nodes_in_topological_order_.reserve(original.size());
  InlinedHashSet<NodeIndex> visited;
  for (auto& node : original) {
    if (visited.find(node) != visited.end()) {
      continue;
    }
    nodes_in_topological_order_.push_back(node);
    visited.insert(node);
    if (shape_size_parents.find(node) != shape_size_parents.end()) {
      for (auto& following_node : shape_size_parents[node]) {
        nodes_in_topological_order_.push_back(following_node);
        visited.insert(following_node);
      }
    }
  }

#endif

#if !defined(ORT_MINIMAL_BUILD)
  graph_->KahnsTopologicalSort(
      [this](const Node* n) {
        nodes_in_topological_order_with_priority_.push_back(n->Index());
      },
      PriorityNodeCompare());
#endif

#ifdef ENABLE_TRAINING
  if (yield_node != nullptr) {
    std::vector<NodeIndex> node_orders;
    const size_t num_of_nodes = NumberOfNodes();
    node_orders.reserve(num_of_nodes);
    graph_->MemoryEfficientTopologicalSort(
        yield_node,
        shape_size_parents,
        node_orders);

    ORT_ENFORCE(node_orders.size() == num_of_nodes,
                "Topological sort failed.", node_orders.size(), "!=", num_of_nodes);
    nodes_in_mem_efficient_topological_order_ = std::move(node_orders);
  } else {
    nodes_in_mem_efficient_topological_order_ = nodes_in_topological_order_;
  }
#endif

  if (filter_info_) {
    // validate. if something is off here it's a bug in our code
    for (NodeIndex idx : filter_info->nodes) {
      ORT_ENFORCE(graph_->GetNode(idx) != nullptr, "IndexedSubGraph contains values not present in the Graph");
    }

    // create set of node indexes as we need quick lookups and don't care about the order
    filtered_node_indices_ = FilteredNodeSet(filter_info->nodes.cbegin(),
                                             filter_info->nodes.cend());

    const auto& metadef = filter_info->GetMetaDef();

    filtered_node_inputs_.reserve(metadef->inputs.size());
    filtered_node_inputs_including_initializers_.reserve(metadef->inputs.size());

    for (const auto& input : metadef->inputs) {
      // NodeArgs from the current scope or any outer scopes should be handled correctly.
      //
      // There is an edge case where the model consists of a graph with subgraphs nested across three levels.
      // In this scenario, a third-layer subgraph consumes an input from the first-layer graph (not an initializer).
      // When constructing a new GraphViewer for the second- and third-layer subgraphs,
      // the second-layer graph may not have the corresponding value_info for that first-layer input,
      // because the second-layer graph itself doesn't consume it.
      // Therefore, when working within the second-layer graph, we need to search outer scopes for the missing value_info.
      const auto* nodearg = graph.GetNodeArgIncludingParentGraphs(input);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Input not found:", input);
      filtered_node_inputs_including_initializers_.push_back(nodearg);
      if (!graph.IsInitializedTensor(input)) {
        filtered_node_inputs_.push_back(nodearg);
      }
    }

    for (const auto& output : metadef->outputs) {
      const auto* nodearg = graph.GetNodeArgIncludingParentGraphs(output);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Output not found:", output);
      filtered_node_outputs_.push_back(nodearg);
    }

    // filter nodes in topo order to just the nodes in filter_info_
    auto orig_order = std::move(nodes_in_topological_order_);
    nodes_in_topological_order_.reserve(filter_info->nodes.size());
    std::copy_if(orig_order.cbegin(), orig_order.cend(), std::back_inserter(nodes_in_topological_order_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });

    // Filter the initializers also
    // Get the names of all the inputs and implicit inputs of all the nodes in this subgraph
    for (const auto node_idx : filtered_node_indices_) {
      const auto* node = GetNode(node_idx);
      ORT_ENFORCE(node, "Mismatch between Graph and IndexedSubGraph. Node not found: ", node_idx);
      const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
      for (const auto* node_input : node->InputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }

      // The implicit inputs for subgraphs (if any)
      for (const auto* node_input : node->ImplicitInputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }
    }

#if !defined(ORT_MINIMAL_BUILD)
    auto orig_priority_order = std::move(nodes_in_topological_order_with_priority_);
    nodes_in_topological_order_with_priority_.reserve(filter_info_->nodes.size());
    std::copy_if(orig_priority_order.cbegin(), orig_priority_order.cend(),
                 std::back_inserter(nodes_in_topological_order_with_priority_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });
#endif

#ifdef ENABLE_TRAINING
    auto orig_mem_efficient_order = std::move(nodes_in_mem_efficient_topological_order_);
    nodes_in_mem_efficient_topological_order_.reserve(filter_info_->nodes.size());
    std::copy_if(orig_mem_efficient_order.cbegin(), orig_mem_efficient_order.cend(),
                 std::back_inserter(nodes_in_mem_efficient_topological_order_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });
#endif
  }
}

// Graph name.
const std::string& GraphViewer::Name() const noexcept {
  return (filter_info_ == nullptr) ? graph_->Name()
                                   : filter_info_->GetMetaDef()->name;
}

const std::string& GraphViewer::Description() const noexcept {
  // filter_info_ doesn't have description so return 'name' instead of nothing
  // and to disambiguate between the full graph's description
  return (filter_info_ == nullptr) ? graph_->Description()
                                   : filter_info_->GetMetaDef()->name;
}

bool GraphViewer::GetInitializedTensor(const std::string& tensor_name,
                                       const ONNX_NAMESPACE::TensorProto*& value) const {
  value = nullptr;

  // if we are using filtered subgraph, the initializer has to be part of the subgraph
  if (filter_info_ != nullptr && filtered_initializers_.find(tensor_name) == filtered_initializers_.cend())
    return false;

  return graph_->GetInitializedTensor(tensor_name, value);
}

bool GraphViewer::CanOverrideInitializer() const noexcept {
  return graph_->CanOverrideInitializer();
}

// Graph inputs excluding initializers.
const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputs()
                                   : filtered_node_inputs_;
}
// Graph inputs including initializers. Contains no nullptr values.
// This will match the number and order of inputs from the GraphProto.
const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputsIncludingInitializers()
                                   : filtered_node_inputs_including_initializers_;
}

// Graph outputs. Should have no nullptr values.
const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetOutputs()
                                   : filtered_node_outputs_;
}

bool GraphViewer::NodeProducesGraphOutput(const Node& node) const {
  const auto& outputs = GetOutputs();
  auto end_outputs = outputs.cend();
  for (auto output_def : node.OutputDefs()) {
    if (std::find(outputs.cbegin(), end_outputs, output_def) != end_outputs) {
      return true;
    }
  }
  return false;
}

// Get graph value infos.
const std::unordered_set<const NodeArg*>& GraphViewer::GetValueInfo() const noexcept {
  return graph_->GetValueInfo();
}

// Get const Node given specific node index. May return nullptr if node as been freed.
const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  if (filter_info_ && filtered_node_indices_.count(node_index) == 0) {
    return nullptr;
  }

  return graph_->GetNode(node_index);
}

const ConstGraphNodes& GraphViewer::Nodes() const noexcept {
  return graph_nodes_;
}

int GraphViewer::NumberOfNodes() const noexcept {
  return (filter_info_ == nullptr) ? graph_->NumberOfNodes()
                                   : gsl::narrow_cast<int>(filter_info_->nodes.size());
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return graph_->MaxNodeIndex();
}

const std::vector<NodeIndex>& GraphViewer::GetNodesInTopologicalOrder(ExecutionOrder order) const {
  switch (order) {
    case ExecutionOrder::DEFAULT:
      return nodes_in_topological_order_;
    case ExecutionOrder::PRIORITY_BASED:
#if !defined(ORT_MINIMAL_BUILD)
      return nodes_in_topological_order_with_priority_;
#else
      ORT_THROW("Priority based topological order is not enabled for ORT minimal build.");
#endif
    case ExecutionOrder::MEMORY_EFFICIENT:
#ifdef ENABLE_TRAINING
      return nodes_in_mem_efficient_topological_order_;
#else
      ORT_THROW("Memory efficient topological order is not enabled for non-training build.");
#endif
    default:
      ORT_THROW("Invalid ExecutionOrder");
  }
}

const std::vector<NodeIndex>& GraphViewer::GetRootNodes() const {
  // TODO(somebody): See if we need to calculate the root_nodes_ of the filtered graph.
  // GetRootNodes is only used by parallel executor currently, and isn't relevant to the usage of a filtered graph.
  ORT_ENFORCE(filter_info_ == nullptr, "Not supported with filtered graph.");

  return root_nodes_;
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return (filter_info_ == nullptr)
             ? graph_->GetAllInitializedTensors()
             : filtered_initializers_;
}

const NodeArg* GraphViewer::GetNodeArg(const std::string& name) const {
  return graph_->GetNodeArg(name);
}

bool GraphViewer::IsSubgraph() const {
  return graph_->IsSubgraph();
}

bool GraphViewer::IsConstantInitializer(const std::string& name, bool check_outer_scope) const {
  return GetConstantInitializer(name, check_outer_scope) != nullptr;
}

bool GraphViewer::IsInitializedTensor(const std::string& name) const {
  return graph_->IsInitializedTensor(name);
}

const ONNX_NAMESPACE::TensorProto* GraphViewer::GetConstantInitializer(const std::string& initializer_name,
                                                                       bool check_outer_scope) const {
  return graph_->GetConstantInitializer(initializer_name, check_outer_scope);
}

#if !defined(ORT_MINIMAL_BUILD)
const std::unordered_set<std::string>& GraphViewer::GetOuterScopeNodeArgNames() const noexcept {
  return graph_->GetOuterScopeNodeArgNames();
}
#endif

}  // namespace onnxruntime
