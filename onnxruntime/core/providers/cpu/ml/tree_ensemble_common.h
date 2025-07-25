// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include "core/platform/threadpool.h"
#include "tree_ensemble_helper.h"
#include "tree_ensemble_attribute.h"
#include "tree_ensemble_aggregator.h"

namespace onnxruntime {
namespace ml {
namespace detail {

// This class checks if a value is in a set of categories. It is used to handle categorical features.
// It internally uses an unordered_set for O(1) average time complexity lookups.
// TCat represents the type of values stored in the set (typically int32_t)
// TInput represents the type of input values to test (can be converted to TCat)
// This implementation could be optimized for different kinds of integer sets,
// e.g., using a bitmap representation for small sets of consecutive integers.
template <typename TCat, typename TInput>
class TreeCategorySet {
 public:
  TreeCategorySet(const std::vector<TCat>& values) : set_(values.begin(), values.end()) {}
  inline bool isIn(const TInput& val) const {
    return set_.find(static_cast<TCat>(val)) != set_.end();
  }

 private:
  std::unordered_set<TCat> set_;
};

/**
 * These attributes are the kernel attributes. They are different from the onnx operator attributes
 * to improve the computation efficiency. The initialization consists in moving the onnx attributes
 * into the kernel attributes.
 */
class TreeEnsembleCommonAttributes {
 public:
  int64_t get_target_or_class_count() const { return this->n_targets_or_classes_; }
  virtual Status Init(const OpKernelInfo&) = 0;
  virtual Status compute(OpKernelContext*, const Tensor*, Tensor*, Tensor*) const = 0;
  virtual ~TreeEnsembleCommonAttributes() {}

  // Default constructor with reasonable defaults for all members
  TreeEnsembleCommonAttributes()
      : n_targets_or_classes_(1),
        post_transform_(POST_EVAL_TRANSFORM::NONE),
        aggregate_function_(AGGREGATE_FUNCTION::SUM),
        n_nodes_(0),
        max_tree_depth_(1),
        max_feature_id_(0),
        n_trees_(0),
        same_mode_(true),
        has_missing_tracks_(false),
        parallel_tree_(80),
        parallel_tree_N_(128),
        parallel_N_(50) {}

 protected:
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t n_nodes_;
  int64_t max_tree_depth_;
  int64_t max_feature_id_;
  int64_t n_trees_;
  bool same_mode_;
  bool has_missing_tracks_;
  int parallel_tree_;    // starts parallelizing the computing by trees if n_tree >= parallel_tree_
  int parallel_tree_N_;  // batch size if parallelizing by trees
  int parallel_N_;       // starts parallelizing the computing by rows if n_rows <= parallel_N_
};

// TI: input type
// TH: tree type (types of the node values and targets)
// TO: output type, usually float
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommon : public TreeEnsembleCommonAttributes {
 protected:
  std::vector<ThresholdType> base_values_;
  std::vector<TreeNodeElement<ThresholdType>> nodes_;
  // Type of weights should be a vector of OutputType. Onnx specifications says it must be float.
  // Lightgbm requires a double to do the summation of all trees predictions. That's why
  // `ThresholdType` is used as well for output type (double as well for lightgbm) and not `OutputType`.
  std::vector<SparseValue<ThresholdType>> weights_;
  std::vector<TreeNodeElement<ThresholdType>*> roots_;
  std::vector<TreeCategorySet<int32_t, InputType>> category_sets_;

 public:
  TreeEnsembleCommon() {}

  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_tree_N,
              int parallel_N,
              const TreeEnsembleAttributesV3<ThresholdType>& attributes);

 protected:
  TreeNodeElement<ThresholdType>* ProcessTreeNodeLeave(TreeNodeElement<ThresholdType>* root,
                                                       const InputType* x_data) const;

  template <typename AGG>
  void ComputeAgg(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Y, Tensor* label, const AGG& agg) const;

  inline const TreeCategorySet<int32_t, InputType>& GetCategorySet(const ThresholdType& set_id) const {
    return category_sets_[static_cast<size_t>(set_id)];
  }

 private:
  bool CheckIfSubtreesAreEqual(const size_t left_id, const size_t right_id, const int64_t tree_id, const InlinedVector<NODE_MODE_ONNX>& cmodes,
                               const InlinedVector<size_t>& truenode_ids, const InlinedVector<size_t>& falsenode_ids, gsl::span<const int64_t> nodes_featureids,
                               gsl::span<const ThresholdType> nodes_values_as_tensor, gsl::span<const float> node_values,
                               gsl::span<const float> target_class_weights, gsl::span<const ThresholdType> target_class_weights_as_tensor,
                               const InlinedVector<TreeNodeElementId>& node_tree_ids, InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices);
  size_t AddNodes(const size_t i, const InlinedVector<NODE_MODE_ONNX>& cmodes, const InlinedVector<size_t>& truenode_ids,
                  const InlinedVector<size_t>& falsenode_ids, gsl::span<const int64_t> nodes_featureids,
                  gsl::span<const ThresholdType> nodes_values_as_tensor, gsl::span<const float> node_values,
                  gsl::span<const int64_t> nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping,
                  int64_t tree_id, const InlinedVector<TreeNodeElementId>& node_tree_ids, gsl::span<const float> target_class_weights,
                  gsl::span<const ThresholdType> target_class_weights_as_tensor, InlinedVector<std::pair<TreeNodeElementId, uint32_t>>& indices);
};

// Below is simple implementation of `bit_cast` as it is supported from c++20 and the current supported version is c++17
// Remove it when that is not the case
template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) &&
        std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
    // constexpr support needs compiler magic
    static bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible_v<To>,
                "This implementation additionally requires "
                "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t> bit_cast_int(T val) {
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    return bit_cast<uint32_t>(val);
  } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
    return bit_cast<uint64_t>(val);
  }
  static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  TreeEnsembleAttributesV3<ThresholdType> attributes(info, false);
  return Init(80, 128, 50, attributes);
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
    int parallel_N,
    const TreeEnsembleAttributesV3<ThresholdType>& attributes) {
  parallel_tree_ = parallel_tree;
  parallel_tree_N_ = parallel_tree_N;
  parallel_N_ = parallel_N;

  aggregate_function_ = MakeAggregateFunction(attributes.aggregate_function);
  post_transform_ = MakeTransform(attributes.post_transform);
  if (!attributes.base_values_as_tensor.empty()) {
    ORT_ENFORCE(attributes.base_values.empty());
    base_values_ = attributes.base_values_as_tensor;
  } else {
    base_values_.reserve(attributes.base_values.size());
    for (size_t i = 0, limit = attributes.base_values.size(); i < limit; ++i) {
      base_values_.push_back(static_cast<ThresholdType>(attributes.base_values[i]));
    }
  }
  n_targets_or_classes_ = attributes.n_targets_or_classes;
  max_tree_depth_ = 1000;

  // Additional members
  size_t limit;
  uint32_t i;  // this variable is used in different loops with different types for the upper bound
  InlinedVector<NODE_MODE_ONNX> cmodes;
  cmodes.reserve(attributes.nodes_modes.size());
  same_mode_ = true;
  bool check_same_mode_again = false;
  int fpos = -1;
  for (i = 0, limit = attributes.nodes_modes.size(); i < limit; ++i) {
    cmodes.push_back(attributes.nodes_modes[i]);
    if (cmodes[i] == NODE_MODE_ONNX::LEAF) continue;
    // The struture may be compressed if it contains only BRANCH_EQ
    // and if ai.onnx.ml == 3, it changes the branch mode into BRANCH_MEMBER.
    // same_mode_ needs to be recomputed again.
    check_same_mode_again |= cmodes[i] == NODE_MODE_ONNX::BRANCH_EQ;
    if (fpos == -1) {
      fpos = static_cast<int>(i);
      continue;
    }
    if (cmodes[i] != cmodes[fpos]) same_mode_ = false;
  }

  n_nodes_ = attributes.nodes_treeids.size();
  limit = static_cast<size_t>(n_nodes_);
  InlinedVector<TreeNodeElementId> node_tree_ids;
  node_tree_ids.reserve(limit);
  nodes_.clear();
  nodes_.reserve(limit);
  roots_.clear();
  std::unordered_map<TreeNodeElementId, size_t, TreeNodeElementId::hash_fn> node_tree_ids_map;
  node_tree_ids_map.reserve(limit);

  InlinedVector<size_t> truenode_ids, falsenode_ids;
  truenode_ids.reserve(limit);
  falsenode_ids.reserve(limit);
  max_feature_id_ = 0;

  // Build node_tree_ids and node_tree_ids_map and truenode_ids and falsenode_ids
  for (i = 0; i < limit; ++i) {
    TreeNodeElementId node_tree_id{static_cast<int>(attributes.nodes_treeids[i]), static_cast<int>(attributes.nodes_nodeids[i])};
    auto p = node_tree_ids_map.insert(std::pair<TreeNodeElementId, size_t>(node_tree_id, i));
    if (!p.second) {
      ORT_THROW("Node ", node_tree_id.node_id, " in tree ", node_tree_id.tree_id, " is already there.");
    }
    node_tree_ids.emplace_back(node_tree_id);
  }

  TreeNodeElementId coor;
  for (i = 0; i < limit; ++i) {
    if (cmodes[i] == NODE_MODE_ONNX::LEAF) {
      truenode_ids.push_back(0);
      falsenode_ids.push_back(0);
    } else {
      TreeNodeElementId& node_tree_id = node_tree_ids[i];
      coor.tree_id = node_tree_id.tree_id;
      coor.node_id = static_cast<int>(attributes.nodes_truenodeids[i]);
      ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));

      auto found = node_tree_ids_map.find(coor);
      if (found == node_tree_ids_map.end()) {
        ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (truenode).");
      }
      if (found->second == truenode_ids.size()) {
        ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (truenode).");
      }
      truenode_ids.emplace_back(found->second);

      coor.node_id = static_cast<int>(attributes.nodes_falsenodeids[i]);
      ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));
      found = node_tree_ids_map.find(coor);
      if (found == node_tree_ids_map.end()) {
        ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (falsenode).");
      }
      if (found->second == falsenode_ids.size()) {
        ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (falsenode).");
      }
      falsenode_ids.emplace_back(found->second);
      // We could also check that truenode_ids[truenode_ids.size() - 1] != falsenode_ids[falsenode_ids.size() - 1]).
      // It is valid but no training algorithm would produce a tree where left and right nodes are the same.
    }
  }

  // Sort targets
  InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices;
  indices.reserve(attributes.target_class_nodeids.size());
  for (i = 0, limit = attributes.target_class_nodeids.size(); i < limit; i++) {
    indices.emplace_back(
        TreeNodeElementId{attributes.target_class_treeids[i], attributes.target_class_nodeids[i]}, i);
  }

  std::sort(indices.begin(), indices.end());

  // Let's construct nodes_ such that the false branch is always the next element in nodes_.
  // updated_mapping will translates the old position of each node to the new node position in nodes_.
  std::vector<size_t> updated_mapping(attributes.nodes_treeids.size(), 0);
  int64_t previous_tree_id = -1;
  for (i = 0; i < n_nodes_; ++i) {
    if (previous_tree_id == -1 || (previous_tree_id != node_tree_ids[i].tree_id)) {
      // New tree.
      int64_t tree_id = node_tree_ids[i].tree_id;
      size_t root_position =
          AddNodes(i, cmodes, truenode_ids, falsenode_ids, attributes.nodes_featureids, attributes.nodes_values_as_tensor, attributes.nodes_values,
                   attributes.nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids,
                   attributes.target_class_weights, attributes.target_class_weights_as_tensor, indices);
      roots_.push_back(&nodes_[root_position]);
      previous_tree_id = tree_id;
    }
  }
  n_trees_ = roots_.size();

  TreeNodeElementId ind;
  SparseValue<ThresholdType> w;
  size_t indi;
  for (indi = 0, limit = attributes.target_class_nodeids.size(); indi < limit; ++indi) {
    ind = indices[indi].first;
    i = indices[indi].second;
    auto found = node_tree_ids_map.find(ind);
    if (found == node_tree_ids_map.end()) {
      ORT_THROW("Unable to find node ", ind.tree_id, "-", ind.node_id, " (weights).");
    }

    TreeNodeElement<ThresholdType>& leaf = nodes_[updated_mapping[found->second]];
    if (leaf.is_not_leaf()) {
      // An exception should be raised in that case. But this case may happen in
      // models converted with an old version of onnxmltools. These weights are ignored.
      // ORT_THROW("Node ", ind.tree_id, "-", ind.node_id, " is not a leaf.");
      continue;
    }
    w.i = attributes.target_class_ids[i];
    w.value = attributes.target_class_weights_as_tensor.empty()
                  ? static_cast<ThresholdType>(attributes.target_class_weights[i])
                  : attributes.target_class_weights_as_tensor[i];
    if (leaf.truenode_or_weight.weight_data.n_weights == 0) {
      leaf.truenode_or_weight.weight_data.weight = static_cast<int32_t>(weights_.size());
      leaf.value_or_unique_weight = w.value;
    }
    ++leaf.truenode_or_weight.weight_data.n_weights;
    weights_.push_back(w);
  }

  has_missing_tracks_ = false;
  for (auto itm = attributes.nodes_missing_value_tracks_true.begin(); itm != attributes.nodes_missing_value_tracks_true.end(); ++itm) {
    if (*itm) {
      has_missing_tracks_ = true;
      break;
    }
  }

  if (same_mode_ && check_same_mode_again) {
    // A node BRANCH_EQ may have been changed into BRANCH_MEMBER
    // to compress the structure. same_mode_ needs to evaluated again.
    same_mode_ = true;
    auto mode = nodes_[0].mode();
    for (auto& node : nodes_) {
      if (node.is_not_leaf()) {
        if (node.mode() != mode) {
          same_mode_ = false;
          break;
        }
      }
    }
  }

  // Handling bigsets
  if (!attributes.bigsets.empty()) {
    category_sets_.reserve(attributes.bigsets.size());
    for (auto bigset : attributes.bigsets) {
      // We check every value in the bigset is a valid value (an integer).
      std::vector<ThresholdType> wrong_values;
      std::vector<int32_t> bigset_int32;
      bigset_int32.reserve(bigset.size());
      for (auto value : bigset) {
        if (static_cast<ThresholdType>(static_cast<int32_t>(value)) != value) {
          wrong_values.push_back(value);
        }
        bigset_int32.push_back(static_cast<int32_t>(value));
      }
      ORT_ENFORCE(
          wrong_values.empty(),
          "MemberShip values is only implemented for integers, ", wrong_values.size(), "values cannot be cast into int32_t.");
      TreeCategorySet<int32_t, InputType> catset(bigset_int32);
      category_sets_.emplace_back(std::move(catset));
    }
  }

#if defined(_TREE_DEBUG)
  std::cout << "TreeEnsemble:same_mode_=" << (same_mode_ ? 1 : 0) << "\n";
  for (auto& node : nodes_) {
    std::cout << node.str() << "\n";
  }
#endif
  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
bool TreeEnsembleCommon<InputType, ThresholdType, OutputType>::CheckIfSubtreesAreEqual(
    const size_t left_id, const size_t right_id, const int64_t tree_id, const InlinedVector<NODE_MODE_ONNX>& cmodes,
    const InlinedVector<size_t>& truenode_ids, const InlinedVector<size_t>& falsenode_ids, gsl::span<const int64_t> nodes_featureids,
    gsl::span<const ThresholdType> nodes_values_as_tensor, gsl::span<const float> node_values,
    gsl::span<const float> target_class_weights, gsl::span<const ThresholdType> target_class_weights_as_tensor,
    const InlinedVector<TreeNodeElementId>& node_tree_ids, InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices) {
  // Leaves have values set at 0
  if (cmodes[left_id] != cmodes[right_id] || nodes_featureids[left_id] != nodes_featureids[right_id] ||
      (!nodes_values_as_tensor.empty() && nodes_values_as_tensor[left_id] != nodes_values_as_tensor[right_id]) ||
      (nodes_values_as_tensor.empty() && node_values[left_id] != node_values[right_id])) {
    return false;
  }

  if (cmodes[left_id] == NODE_MODE_ONNX::LEAF) {
    const auto left_target_node = std::lower_bound(indices.begin(), indices.end(), std::make_pair(node_tree_ids[left_id], uint32_t(0)))->second;
    const auto right_target_node = std::lower_bound(indices.begin(), indices.end(), std::make_pair(node_tree_ids[right_id], uint32_t(0)))->second;

    if (target_class_weights_as_tensor.empty()) {
      return target_class_weights[left_target_node] == target_class_weights[right_target_node];
    } else {
      return target_class_weights_as_tensor[left_target_node] == target_class_weights_as_tensor[right_target_node];
    }
  }

  return CheckIfSubtreesAreEqual(falsenode_ids[left_id], falsenode_ids[right_id], tree_id, cmodes, truenode_ids, falsenode_ids, nodes_featureids,
                                 nodes_values_as_tensor, node_values, target_class_weights, target_class_weights_as_tensor, node_tree_ids, indices) &&
         CheckIfSubtreesAreEqual(truenode_ids[left_id], truenode_ids[right_id], tree_id, cmodes, truenode_ids, falsenode_ids, nodes_featureids,
                                 nodes_values_as_tensor, node_values, target_class_weights, target_class_weights_as_tensor, node_tree_ids, indices);
}

inline void UpdateThreshold(double val, double& mask) {
  uint64_t new_mask = bit_cast<uint64_t>(mask) | (1ll << (static_cast<uint32_t>(val) - 1));
  mask = bit_cast<double>(new_mask);
}

inline void UpdateThreshold(float val, float& mask) {
  uint32_t new_mask = bit_cast<uint32_t>(mask) | (1 << (static_cast<uint32_t>(val) - 1));
  mask = bit_cast<float>(new_mask);
}

#define BITCOUNT(T) int64_t(sizeof(T) * 8)
#define CANMASK(v, T) (v >= 1 && v <= BITCOUNT(T)) && v == std::floor(v)

template <typename InputType, typename ThresholdType, typename OutputType>
size_t TreeEnsembleCommon<InputType, ThresholdType, OutputType>::AddNodes(
    const size_t i, const InlinedVector<NODE_MODE_ONNX>& cmodes, const InlinedVector<size_t>& truenode_ids,
    const InlinedVector<size_t>& falsenode_ids, gsl::span<const int64_t> nodes_featureids,
    gsl::span<const ThresholdType> nodes_values_as_tensor, gsl::span<const float> node_values,
    gsl::span<const int64_t> nodes_missing_value_tracks_true, std::vector<size_t>& updated_mapping, int64_t tree_id,
    const InlinedVector<TreeNodeElementId>& node_tree_ids, gsl::span<const float> target_class_weights,
    gsl::span<const ThresholdType> target_class_weights_as_tensor, InlinedVector<std::pair<TreeNodeElementId, uint32_t>>& indices) {
  // Validate this index maps to the same tree_id as the one we should be building.
  if (node_tree_ids[i].tree_id != tree_id) {
    ORT_THROW("Tree id mismatch. Expected ", tree_id, " but got ", node_tree_ids[i].tree_id, " at position ", i);
  }

  if (updated_mapping[i] != 0) {
    // In theory we should not accept any cycles, however in practice LGBM conversion implements set membership via a
    // series of "Equals" nodes, with the true branches directed at the same child node (a cycle).
    // We may instead seek to formalize set membership in the future.
    return updated_mapping[i];
  }

  size_t node_pos = nodes_.size();
  updated_mapping[i] = node_pos;

  TreeNodeElement<ThresholdType> node;
  node.flags = Convert_NODE_MODE_ONNX_to_ORT(cmodes[i]);
  node.feature_id = static_cast<int>(nodes_featureids[i]);
  if (node.feature_id > max_feature_id_) {
    max_feature_id_ = node.feature_id;
  }

  node.value_or_unique_weight = 0;
  const ThresholdType node_threshold = nodes_values_as_tensor.empty() ? static_cast<ThresholdType>(node_values[i]) : nodes_values_as_tensor[i];
  if (node.flags == NODE_MODE_ORT::BRANCH_EQ && CANMASK(node_threshold, ThresholdType)) {
    UpdateThreshold(node_threshold, node.value_or_unique_weight);
    node.flags = NODE_MODE_ORT::BRANCH_MEMBER;
  } else {
    node.value_or_unique_weight = node_threshold;
  }

  if (i < static_cast<size_t>(nodes_missing_value_tracks_true.size()) && nodes_missing_value_tracks_true[i] == 1) {
    node.flags = static_cast<NODE_MODE_ORT>(static_cast<uint8_t>(node.flags) | static_cast<uint8_t>(MissingTrack::kTrue));
  }
  nodes_.push_back(std::move(node));
  if (nodes_[node_pos].is_not_leaf()) {
    size_t falsenode_id = falsenode_ids[i];

    // Categoricals are represented as a chain of `EQ` nodes where the subtree for the true child is identical for all nodes in the chain
    // Below we are folding together these nodes into one of mode `BRANCH_MEMBER`
    // The threshold of this node should be interpreted as a bitmask showing which categoricals values were found in the chain
    // Afterwards, when looking whether a feature is included we can do an `and` with the mask of the node
    // and the one of the feature (the mask has only one bit set on the place for its value)
    // Beware that if a category is bigger than the threshold type, the node stays as `EQ` and no combination is done
    if (nodes_[node_pos].flags == NODE_MODE_ORT::BRANCH_MEMBER) {
      ThresholdType falsenode_threshold = nodes_values_as_tensor.empty() ? static_cast<ThresholdType>(node_values[falsenode_id]) : nodes_values_as_tensor[falsenode_id];

      while (cmodes[falsenode_id] == NODE_MODE_ONNX::BRANCH_EQ && nodes_[node_pos].feature_id == nodes_featureids[falsenode_id] &&
             CANMASK(falsenode_threshold, ThresholdType) &&
             CheckIfSubtreesAreEqual(truenode_ids[i], truenode_ids[falsenode_id], tree_id, cmodes, truenode_ids, falsenode_ids,
                                     nodes_featureids, nodes_values_as_tensor, node_values, target_class_weights, target_class_weights_as_tensor, node_tree_ids, indices)) {
        UpdateThreshold(falsenode_threshold, nodes_[node_pos].value_or_unique_weight);
        falsenode_id = falsenode_ids[falsenode_id];
        falsenode_threshold = nodes_values_as_tensor.empty() ? static_cast<ThresholdType>(node_values[falsenode_id]) : nodes_values_as_tensor[falsenode_id];
      }
    }

    size_t false_branch =
        AddNodes(falsenode_id, cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor,
                 node_values, nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids,
                 target_class_weights, target_class_weights_as_tensor, indices);
    if (false_branch != node_pos + 1) {
      ORT_THROW("False node must always be the next node, but it isn't at index ", node_pos, " with flags ",
                static_cast<int>(nodes_[node_pos].flags));
    }
    size_t true_branch =
        AddNodes(truenode_ids[i], cmodes, truenode_ids, falsenode_ids, nodes_featureids, nodes_values_as_tensor,
                 node_values, nodes_missing_value_tracks_true, updated_mapping, tree_id, node_tree_ids,
                 target_class_weights, target_class_weights_as_tensor, indices);
    // We don't need to store the false branch pointer since we know it is always in the immediate next entry in nodes_.
    // nodes_[node_pos].falsenode_inc_or_n_weights.ptr = &nodes_[false_branch];
    nodes_[node_pos].truenode_or_weight.ptr = &nodes_[true_branch];
  } else {
    nodes_[node_pos].truenode_or_weight.weight_data.weight = 0;
    nodes_[node_pos].truenode_or_weight.weight_data.n_weights = 0;
  }
  return node_pos;
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                         const Tensor* X,
                                                                         Tensor* Y,
                                                                         Tensor* label) const {
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorAverage<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::SUM:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorSum<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MIN:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMin<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MAX:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMax<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    default:
      ORT_THROW("Unknown aggregation function in TreeEnsemble.");
  }
}

template <typename InputType, typename ThresholdType, typename OutputType>
template <typename AGG>
void TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ComputeAgg(concurrency::ThreadPool* ttp,
                                                                          const Tensor* X, Tensor* Z,
                                                                          Tensor* label, const AGG& agg) const {
  if (X->Shape().NumDimensions() > 2) {
    ORT_THROW("TreeEnsemble only works on 1D, 2D tensors.");
  }
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  int64_t C = X->Shape().NumDimensions() == 2 ? X->Shape()[1] : 1;
  if (max_feature_id_ >= C) {
    ORT_THROW("One path in the graph requests feature ", max_feature_id_, " but input tensor has ", C, " features.");
  }
  OutputType* z_data = Z->MutableData<OutputType>();

  const InputType* x_data = X->Data<InputType>();
  int64_t* label_data = label == nullptr ? nullptr : label->MutableData<int64_t>();
  auto max_num_threads = concurrency::ThreadPool::DegreeOfParallelism(ttp);

  if (n_targets_or_classes_ == 1) {
    if (N == 1) {
      ScoreValue<ThresholdType> score = {0, 0};
      if (n_trees_ <= parallel_tree_ || max_num_threads == 1) { /* section A: 1 output, 1 row and not enough trees to parallelize */
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[onnxruntime::narrow<size_t>(j)], x_data));
        }
      } else { /* section B: 1 output, 1 row and enough trees to parallelize */
        std::vector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_trees_), {0, 0});
        concurrency::ThreadPool::TryBatchParallelFor(
            ttp,
            SafeInt<int32_t>(n_trees_),
            [this, &scores, &agg, x_data](ptrdiff_t j) {
              agg.ProcessTreeNodePrediction1(scores[j], *ProcessTreeNodeLeave(roots_[j], x_data));
            },
            max_num_threads);

        for (auto it = scores.cbegin(); it != scores.cend(); ++it) {
          agg.MergePrediction1(score, *it);
        }
      }
      agg.FinalizeScores1(z_data, score, label_data);
    } else if (N <= parallel_N_ || max_num_threads == 1) { /* section C: 1 output, 2+ rows but not enough rows to parallelize */
      // Not enough data to parallelize but the computation is split into batches of 128 rows,
      // and then loop on trees to evaluate every tree on this batch.
      // This change was introduced by PR: https://github.com/microsoft/onnxruntime/pull/13835.
      // The input tensor (2D) is stored in a contiguous array. Therefore, it is faster
      // to loop on tree first and inside that loop evaluate a tree on the input tensor (inner loop).
      // The processor is faster when it has to move chunks of a contiguous array (branching).
      // However, if the input tensor is too big, the data does not hold on caches (L1, L2, L3).
      // In that case, looping first on tree or on data is almost the same. That's why the first loop
      // split into batch so that every batch holds on caches, then loop on trees and finally loop
      // on the batch rows.
      std::vector<ScoreValue<ThresholdType>> scores(parallel_tree_N_);
      size_t j;
      int64_t i, batch, batch_end;

      for (batch = 0; batch < N; batch += parallel_tree_N_) {
        batch_end = std::min(N, batch + parallel_tree_N_);
        for (i = batch; i < batch_end; ++i) {
          scores[SafeInt<ptrdiff_t>(i - batch)] = {0, 0};
        }
        for (j = 0; j < static_cast<size_t>(n_trees_); ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction1(scores[SafeInt<ptrdiff_t>(i - batch)], *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores1(z_data + i, scores[SafeInt<ptrdiff_t>(i - batch)],
                              label_data == nullptr ? nullptr : (label_data + i));
        }
      }
    } else if (n_trees_ > max_num_threads) { /* section D: 1 output, 2+ rows and enough trees to parallelize */
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<ScoreValue<ThresholdType>> scores(SafeInt<size_t>(num_threads) * N);
      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_N_);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data, N, begin_n, end_n, stride](ptrdiff_t batch_num) {
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(this->n_trees_));
              for (int64_t i = begin_n; i < end_n; ++i) {
                scores[batch_num * SafeInt<ptrdiff_t>(N) + i] = {0, 0};
              }
              for (auto j = work.start; j < work.end; ++j) {
                for (int64_t i = begin_n; i < end_n; ++i) {
                  agg.ProcessTreeNodePrediction1(scores[batch_num * SafeInt<ptrdiff_t>(N) + i],
                                                 *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
                }
              }
            });
        begin_n = end_n;
      }
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [&agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(N));
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction1(scores[i], scores[j * SafeInt<ptrdiff_t>(N) + i]);
              }
              agg.FinalizeScores1(z_data + i, scores[i],
                                  label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    } else { /* section E: 1 output, 2+ rows, parallelization by rows */
      concurrency::ThreadPool::TryBatchParallelFor(
          ttp,
          SafeInt<int32_t>(N),
          [this, &agg, x_data, z_data, stride, label_data](ptrdiff_t i) {
            ScoreValue<ThresholdType> score = {0, 0};
            for (size_t j = 0; j < static_cast<size_t>(n_trees_); ++j) {
              agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
            }

            agg.FinalizeScores1(z_data + i, score,
                                label_data == nullptr ? nullptr : (label_data + i));
          },
          max_num_threads);
    }
  } else {
    if (N == 1) {                                               /* section A2: 2+ outputs, 1 row, not enough trees to parallelize */
      if (n_trees_ <= parallel_tree_ || max_num_threads == 1) { /* section A2 */
        InlinedVector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[onnxruntime::narrow<size_t>(j)], x_data), weights_);
        }
        agg.FinalizeScores(scores, z_data, -1, label_data);
      } else { /* section B2: 2+ outputs, 1 row, enough trees to parallelize */
        auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
        std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(num_threads);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data](ptrdiff_t batch_num) {
              scores[batch_num].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(n_trees_));
              for (auto j = work.start; j < work.end; ++j) {
                agg.ProcessTreeNodePrediction(scores[batch_num], *ProcessTreeNodeLeave(roots_[j], x_data), weights_);
              }
            });
        for (size_t i = 1, limit = scores.size(); i < limit; ++i) {
          agg.MergePrediction(scores[0], scores[i]);
        }
        agg.FinalizeScores(scores[0], z_data, -1, label_data);
      }
    } else if (N <= parallel_N_ || max_num_threads == 1) { /* section C2: 2+ outputs, 2+ rows, not enough rows to parallelize */
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(parallel_tree_N_);
      size_t j, limit;
      int64_t i, batch, batch_end;
      batch_end = std::min(N, static_cast<int64_t>(parallel_tree_N_));
      for (i = 0; i < batch_end; ++i) {
        scores[SafeInt<ptrdiff_t>(i)].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_));
      }
      for (batch = 0; batch < N; batch += parallel_tree_N_) {
        batch_end = std::min(N, batch + parallel_tree_N_);
        for (i = batch; i < batch_end; ++i) {
          std::fill(scores[SafeInt<ptrdiff_t>(i - batch)].begin(), scores[SafeInt<ptrdiff_t>(i - batch)].end(), ScoreValue<ThresholdType>({0, 0}));
        }
        for (j = 0, limit = roots_.size(); j < limit; ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction(scores[SafeInt<ptrdiff_t>(i - batch)], *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores(scores[SafeInt<ptrdiff_t>(i - batch)], z_data + i * n_targets_or_classes_, -1,
                             label_data == nullptr ? nullptr : (label_data + i));
        }
      }

    } else if (n_trees_ >= max_num_threads) { /* section: D2: 2+ outputs, 2+ rows, enough trees to parallelize*/
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(SafeInt<size_t>(num_threads) * N);
      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_N_);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data, N, stride, begin_n, end_n](ptrdiff_t batch_num) {
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(this->n_trees_));
              for (int64_t i = begin_n; i < end_n; ++i) {
                scores[batch_num * SafeInt<ptrdiff_t>(N) + i].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
              }
              for (auto j = work.start; j < work.end; ++j) {
                for (int64_t i = begin_n; i < end_n; ++i) {
                  agg.ProcessTreeNodePrediction(scores[batch_num * SafeInt<ptrdiff_t>(N) + i],
                                                *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
                }
              }
            });
        begin_n = end_n;
      }
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, onnxruntime::narrow<ptrdiff_t>(num_threads), onnxruntime::narrow<ptrdiff_t>(N));
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction(scores[i], scores[j * SafeInt<ptrdiff_t>(N) + i]);
              }
              agg.FinalizeScores(scores[i], z_data + i * this->n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    } else { /* section E2: 2+ outputs, 2+ rows, parallelization by rows */
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(N));
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, num_threads, x_data, z_data, label_data, N, stride](ptrdiff_t batch_num) {
            size_t j, limit;
            InlinedVector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_targets_or_classes_));
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, onnxruntime::narrow<ptrdiff_t>(num_threads), onnxruntime::narrow<ptrdiff_t>(N));

            for (auto i = work.start; i < work.end; ++i) {
              std::fill(scores.begin(), scores.end(), ScoreValue<ThresholdType>({0, 0}));
              for (j = 0, limit = roots_.size(); j < limit; ++j) {
                agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
              }

              agg.FinalizeScores(scores,
                                 z_data + i * n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    }
  }
}  // namespace detail

#define TREE_FIND_VALUE(CMP)                                                                           \
  if (has_missing_tracks_) {                                                                           \
    while (root->is_not_leaf()) {                                                                      \
      val = x_data[root->feature_id];                                                                  \
      root = (val CMP root->value_or_unique_weight || (root->is_missing_track_true() && _isnan_(val))) \
                 ? root->truenode_or_weight.ptr                                                        \
                 : root + 1;                                                                           \
    }                                                                                                  \
  } else {                                                                                             \
    while (root->is_not_leaf()) {                                                                      \
      val = x_data[root->feature_id];                                                                  \
      root = val CMP root->value_or_unique_weight ? root->truenode_or_weight.ptr : root + 1;           \
    }                                                                                                  \
  }

// Check whether the feature value is set true in the mask
template <typename T1, typename T2>
inline bool SetMembershipCheck(T1 val, T2 mask) {
  const int64_t val_as_int = static_cast<int64_t>(val);
  return CANMASK(val, T2) && (((1ll << (val_as_int - 1)) & bit_cast_int(mask)) != 0);
}

template <typename InputType, typename ThresholdType, typename OutputType>
TreeNodeElement<ThresholdType>*
TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ProcessTreeNodeLeave(
    TreeNodeElement<ThresholdType>* root, const InputType* x_data) const {
  InputType val;
  if (same_mode_) {
    switch (root->mode()) {
      case NODE_MODE_ORT::BRANCH_LEQ:
        if (has_missing_tracks_) {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = (val <= root->value_or_unique_weight || (root->is_missing_track_true() && _isnan_(val)))
                       ? root->truenode_or_weight.ptr
                       : root + 1;
          }
        } else {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = val <= root->value_or_unique_weight ? root->truenode_or_weight.ptr : root + 1;
          }
        }
        break;
      case NODE_MODE_ORT::BRANCH_LT:
        TREE_FIND_VALUE(<)
        break;
      case NODE_MODE_ORT::BRANCH_GTE:
        TREE_FIND_VALUE(>=)
        break;
      case NODE_MODE_ORT::BRANCH_GT:
        TREE_FIND_VALUE(>)
        break;
      case NODE_MODE_ORT::BRANCH_EQ:
        TREE_FIND_VALUE(==)
        break;
      case NODE_MODE_ORT::BRANCH_NEQ:
        TREE_FIND_VALUE(!=)
        break;
      case NODE_MODE_ORT::BRANCH_MEMBER:
        if (has_missing_tracks_) {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = (SetMembershipCheck(val, root->value_or_unique_weight) || (root->is_missing_track_true() && _isnan_(val)))
                       ? root->truenode_or_weight.ptr
                       : root + 1;
          }
        } else {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = SetMembershipCheck(val, root->value_or_unique_weight) ? root->truenode_or_weight.ptr : root + 1;
          }
        }
        break;
      case NODE_MODE_ORT::BRANCH_MEMBER_BIGSET:
        // The threshold or node value (stored in value_or_unique_weight) in an index of a set in `bigsets`.
        // val is the feature value, method isIn checks whether the value is in the set.
        if (has_missing_tracks_) {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = (GetCategorySet(root->value_or_unique_weight).isIn(val) || (root->is_missing_track_true() && _isnan_(val)))
                       ? root->truenode_or_weight.ptr
                       : root + 1;
          }
        } else {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root = GetCategorySet(root->value_or_unique_weight).isIn(val) ? root->truenode_or_weight.ptr : root + 1;
          }
        }
        break;
      case NODE_MODE_ORT::LEAF:
        break;
      default:
        ORT_THROW("Unknown node mode in TreeEnsembleCommon::ProcessTreeNodeLeave: ", static_cast<int>(root->mode()));
    }
  } else {  // Different rules to compare to node thresholds.
    ThresholdType threshold;
    while (1) {
      val = x_data[root->feature_id];
      threshold = root->value_or_unique_weight;
      switch (root->mode()) {
        case NODE_MODE_ORT::BRANCH_LEQ:
          root = val <= threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_LT:
          root = val < threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                    : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_GTE:
          root = val >= threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_GT:
          root = val > threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                    : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_EQ:
          root = val == threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_NEQ:
          root = val != threshold || (root->is_missing_track_true() && _isnan_(val)) ? root->truenode_or_weight.ptr
                                                                                     : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_MEMBER:
          root = (SetMembershipCheck(val, root->value_or_unique_weight) || (root->is_missing_track_true() && _isnan_(val)))
                     ? root->truenode_or_weight.ptr
                     : root + 1;
          break;
        case NODE_MODE_ORT::BRANCH_MEMBER_BIGSET:
          // The threshold or node value (stored in value_or_unique_weight) in an index of a set in `bigsets`.
          // val is the feature value, method isIn checks whether the value is in the set.
          root = (GetCategorySet(root->value_or_unique_weight).isIn(val) || (root->is_missing_track_true() && _isnan_(val)))
                     ? root->truenode_or_weight.ptr
                     : root + 1;
          break;
        case NODE_MODE_ORT::LEAF:
          return root;
        default:
          ORT_THROW("Unknown node mode in TreeEnsembleCommon::ProcessTreeNodeLeave: ", static_cast<int>(root->mode()));
      }
    }
  }
  return root;
}

// TI: input type
// TH: threshold type, double if T==double, float otherwise
// TO: output type
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommonClassifier : public TreeEnsembleCommon<InputType, ThresholdType, OutputType> {
 private:
  bool weights_are_all_positive_;
  bool binary_case_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<int64_t> class_labels_;

 public:
  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Z, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_tree_N,
              int parallel_N,
              const TreeEnsembleAttributesV3<ThresholdType>& attributes);
};

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  TreeEnsembleAttributesV3<ThresholdType> attributes(info, true);
  return Init(80, 128, 50, attributes);
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
    int parallel_N,
    const TreeEnsembleAttributesV3<ThresholdType>& attributes) {
  auto status = TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(parallel_tree, parallel_tree_N, parallel_N, attributes);
  ORT_RETURN_IF_ERROR(status);

  classlabels_strings_ = attributes.classlabels_strings;
  classlabels_int64s_ = attributes.classlabels_int64s;

  InlinedHashSet<int64_t> weights_classes;
  weights_classes.reserve(attributes.target_class_ids.size());
  weights_are_all_positive_ = true;
  for (size_t i = 0, end = attributes.target_class_ids.size(); i < end; ++i) {
    weights_classes.insert(attributes.target_class_ids[i]);
    if (weights_are_all_positive_ && (!attributes.target_class_weights.empty() ? attributes.target_class_weights[i]
                                                                               : attributes.target_class_weights_as_tensor[i]) < 0)
      weights_are_all_positive_ = false;
  }
  binary_case_ = this->n_targets_or_classes_ == 2 && weights_classes.size() == 1;
  if (!classlabels_strings_.empty()) {
    class_labels_.reserve(classlabels_strings_.size());
    for (size_t i = 0, end = classlabels_strings_.size(); i < end; ++i)
      class_labels_.push_back(i);
  }
  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                                   const Tensor* X,
                                                                                   Tensor* Z,
                                                                                   Tensor* label) const {
  if (classlabels_strings_.empty()) {
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, label,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            classlabels_int64s_, binary_case_,
            weights_are_all_positive_));
  } else {
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    AllocatorPtr alloc;
    ORT_THROW_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    Tensor label_int64(DataTypeImpl::GetType<int64_t>(), TensorShape({N}), std::move(alloc));
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, &label_int64,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            class_labels_, binary_case_,
            weights_are_all_positive_));
    const int64_t* plabel = label_int64.Data<int64_t>();
    std::string* labels = label->MutableData<std::string>();
    for (size_t i = 0; i < (size_t)N; ++i)
      labels[i] = classlabels_strings_[onnxruntime::narrow<size_t>(plabel[i])];
  }
  return Status::OK();
}

template <typename IOType, typename ThresholdType>
class TreeEnsembleCommonV5 : public TreeEnsembleCommon<IOType, ThresholdType, IOType> {
 public:
  virtual Status Init(const OpKernelInfo& info);

  Status Init(int parallel_tree,
              int parallel_tree_N,
              int parallel_N,
              const TreeEnsembleAttributesV5<ThresholdType>& attributes);
};

template <typename IOType, typename ThresholdType>
Status TreeEnsembleCommonV5<IOType, ThresholdType>::Init(const OpKernelInfo& info) {
  TreeEnsembleAttributesV5<ThresholdType> attributes(info);
  return Init(80, 128, 50, attributes);
}

template <typename IOType, typename ThresholdType>
Status TreeEnsembleCommonV5<IOType, ThresholdType>::Init(
    int parallel_tree,
    int parallel_tree_N,
    int parallel_N,
    const TreeEnsembleAttributesV5<ThresholdType>& attributes) {
  TreeEnsembleAttributesV3<ThresholdType> attributes_v3;
  attributes.convert_to_v3(attributes_v3);

  attributes_v3.base_values.clear();
  attributes_v3.base_values_as_tensor.clear();
  attributes_v3.nodes_hitrates.clear();
  attributes_v3.nodes_values.clear();
  attributes_v3.target_class_weights.clear();

  auto status = TreeEnsembleCommon<IOType, ThresholdType, IOType>::Init(parallel_tree, parallel_tree_N, parallel_N, attributes_v3);
  ORT_RETURN_IF_ERROR(status);
  return Status::OK();
}

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
