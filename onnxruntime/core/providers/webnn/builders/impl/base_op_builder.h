// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webnn/builders/op_builder.h"

namespace onnxruntime {
namespace webnn {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  // Add operator related.
 public:
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */, const Node& /* node */) const override {}
  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                           const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  explicit BaseOpBuilder(bool allow_empty_tensor_as_input = false)
      : allow_empty_tensor_as_input_(allow_empty_tensor_as_input) {
  }
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                       const logging::Logger& logger) const ORT_MUST_USE_RESULT = 0;

  // Operator support related.
 public:
  bool IsOpSupported(const GraphViewer& graph_viewer, const Node& node,
                     const WebnnDeviceType device_type, const emscripten::val& wnn_limits,
                     const logging::Logger& logger) const override;

 protected:
  virtual bool IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const Node& /* node */,
                                 const WebnnDeviceType /* device_type */, const logging::Logger& /* logger */) const {
    return true;
  }

  virtual bool HasSupportedInputsImpl(const GraphViewer&, const Node& node, const emscripten::val& wnn_limits,
                                      const logging::Logger& logger) const;
  virtual bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                       const logging::Logger& logger) const;

  // ONNX Runtime only *guarantees* support for models stamped
  // with opset version 7 or above for opset domain 'ai.onnx'.
  // WebNN EP ignores node support for opset less than 7 by
  // default as which will be fallback earlier by ONNX Runtime.
  // We still set the minimal supported opset to 1 as we couldn't
  // get the model opset version at this stage.
  virtual int GetMinSupportedOpSet(const Node& /* node */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /* node */) const { return 23; }

 private:
  bool HasSupportedOpSet(const Node& node, const logging::Logger& logger) const;
  bool HasSupportedInputs(const GraphViewer&, const Node& node, const emscripten::val& wnn_limits, const logging::Logger& logger) const;
  bool HasSupportedOutputs(const Node& node, const emscripten::val& wnn_limits, const logging::Logger& logger) const;

  const bool allow_empty_tensor_as_input_;  // Some operators can handle ignoring an empty tensor as input.
};

}  // namespace webnn
}  // namespace onnxruntime
