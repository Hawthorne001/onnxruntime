// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <fstream>
#include <sstream>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

GlobalContext& BackendManager::GetGlobalContext() {
  return global_context_;
}

BackendManager::BackendManager(const GlobalContext& global_context,
                               const onnxruntime::Node& fused_node,
                               const onnxruntime::GraphViewer& subgraph,
                               const logging::Logger& logger,
                               EPCtxHandler& ctx_handle) {
  global_context_ = global_context;
  ep_ctx_handle_ = ctx_handle;

  openvino_sdk_version_ = std::to_string(global_context_.OpenVINO_Version.at(0)) + "." +
                          std::to_string(global_context_.OpenVINO_Version.at(1));
  if (ep_ctx_handle_.CheckForOVEPCtxNode(subgraph, openvino_sdk_version_)) {
    if (ep_ctx_handle_.ImportBlobFromEPCtxModel(subgraph) != Status::OK())
      ORT_THROW("Import blob from model failed");
  }

  auto prec_str = GetGlobalContext().precision_str;

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  auto node_input_defs = fused_node.InputDefs();
  int i = 0;
  for (auto idef : node_input_defs) {
    subgraph_context_.input_names.insert({idef->Name(), i});
    i++;
  }

  auto graph_inputs = subgraph.GetInputs();
  for (auto input : graph_inputs) {
    auto it = subgraph_context_.input_names.find(input->Name());
    if (it == subgraph_context_.input_names.end()) {
      ORT_THROW("Input not found in the input defs list");
    }
    int index = it->second;
    subgraph_context_.input_indexes.push_back(index);
  }

  auto graph_outputs_defs = fused_node.OutputDefs();
  i = 0;
  for (auto output_def : graph_outputs_defs) {
    subgraph_context_.output_names.insert({output_def->Name(), i});
    i++;
  }
  subgraph_context_.subgraph_name = fused_node.Name();
  model_proto_ = GetModelProtoFromFusedNode(fused_node, subgraph, logger);
  std::string device_type = openvino_ep::BackendManager::GetGlobalContext().device_type;

  if (ModelHasSymbolicInputDims(subgraph)) {
    subgraph_context_.has_dynamic_input_shape = true;
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has symbolic input dims";
    if (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
        GetGlobalContext().device_type.find("GPU") != std::string::npos) {
      if (!GetGlobalContext().disable_dynamic_shapes) {
        LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Starting backend initialization. "
                           << "Creating backend Dynamic Shapes";
        try {
          concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
        } catch (std::string const& msg) {
          ORT_THROW(msg);
        }
        LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                           << "Backend created for graph " << subgraph_context_.subgraph_name;
      }
    }
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has concrete input dims. "
                       << "Initializing backend for graph "
                       << subgraph_context_.subgraph_name;

    subgraph_context_.has_dynamic_input_shape = false;

    // OV NPU plugin is supported with fallback to OV CPU upon compilation failures.
    try {
      concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                      GetGlobalContext(),
                                                      subgraph_context_,
                                                      ep_ctx_handle_);
    } catch (const OnnxRuntimeException& ex) {
      if (device_type.find("NPU") != std::string::npos) {
        LOGS_DEFAULT(WARNING) << ex.what();
        LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                              << "Falling back to OV CPU for execution";
        GetGlobalContext().device_type = "CPU";
        GetGlobalContext().precision_str = "FP32";
        try {
          concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
        } catch (std::string const& msg) {
          ORT_THROW(msg);
        }
      } else {
        ORT_THROW(ex.what());
      }
    }
  }
}

// Call EPContext model exporter here if the provider option for exporting
// precompiled blob is set. If that's the case:
// By default, create model in embed mode where the blob stream is exported as data within
// the EPContext node.
Status BackendManager::ExportCompiledBlobAsEPCtxNode(const onnxruntime::GraphViewer& graph_body_viewer,
                                                     const logging::Logger& logger) {
  std::string model_blob_str;
  auto compiled_model = concrete_backend_->GetOVCompiledModel();
  auto graph_name = global_context_.onnx_model_path_name;
  // Remove extension so we can append suffix to form the complete name of output graph
  graph_name = [&]() {
    size_t dot = graph_name.find_last_of(".");
    if (dot == std::string::npos) return graph_name;
    return graph_name.substr(0, dot);
  }();
  // If embed_mode, then pass on the serialized blob
  // If not embed_mode, dump the blob here and only pass on the path to the blob
  if (global_context_.ep_context_embed_mode) {
    std::ostringstream model_blob_stream;
    compiled_model.export_model(model_blob_stream);
    model_blob_str = model_blob_stream.str();
    ORT_ENFORCE(model_blob_str.size() != 0);
  } else {
    std::ofstream f(graph_name + ".blob", std::ios::out | std::ios::trunc | std::ios::binary);
    compiled_model.export_model(f);
    model_blob_str = graph_name + ".blob";
  }

  ORT_RETURN_IF_ERROR(ep_ctx_handle_.ExportEPCtxModel(graph_body_viewer,
                                                      graph_name,
                                                      logger,
                                                      global_context_.ep_context_embed_mode,
                                                      model_blob_str,
                                                      openvino_sdk_version_,
                                                      GetGlobalContext().device_type));

  return Status::OK();
}

bool BackendManager::ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_batched_inputs = true;

  for (int i = 0; i < static_cast<int>(subgraph_context_.input_indexes.size()); i++) {
    auto& input = model_proto.graph().input(subgraph_context_.input_indexes[i]);

    // Batch-process only raw image inputs (NCHW or NHWC layouts)
    auto& shape = input.type().tensor_type().shape();
    if (shape.dim_size() != 4) {
      has_batched_inputs = false;
      break;
    }

    if (shape.dim(0).value_case() == shape.dim(0).kDimValue) {
      has_batched_inputs = false;
      break;
    }

    for (int index = 1; index < 4; index++) {
      if (shape.dim(index).value_case() != shape.dim(0).kDimValue) {
        has_batched_inputs = false;
        break;
      }
    }
    if (!has_batched_inputs) {
      break;
    }
  }
  return has_batched_inputs;
}

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const {
  bool has_sym_dims = false;
  auto graph_inputs = subgraph.GetInputs();
  for (auto input : graph_inputs) {
    if (input->Shape() == nullptr) {
      has_sym_dims = true;
      break;
    }
    for (auto& dim : input->Shape()->dim()) {
      if (dim.value_case() != dim.kDimValue) {
        has_sym_dims = true;
        break;
      }
    }
    if (has_sym_dims) {
      break;
    }
  }
  return has_sym_dims;
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node& fused_node,
                                           const onnxruntime::GraphViewer& subgraph,
                                           const logging::Logger& logger) const {
  auto model = subgraph.CreateModel(logger);

  auto model_proto = model->ToProto();
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  subgraph.ToProto(*model_proto->mutable_graph(), true, true);

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    const std::string& name = fused_node.Name();
    std::fstream dump(name + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }
#else
  ORT_UNUSED_PARAMETER(fused_node);
#endif

  return model_proto;
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(const Ort::KernelContext& context) {
  const auto input_count = context.GetInputCount();
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto input_tensor = context.GetInput(i);
    auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    input_shapes.push_back(std::move(tensor_shape));
  }
  return input_shapes;
}

std::string MakeMapKeyString(const std::vector<std::vector<int64_t>>& shapes,
                             const std::string& device_type) {
  std::string key;
  key += device_type;
  key += "|";  // separator
  for (auto shape : shapes) {
    for (auto dim : shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }
  return key;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                                      const std::vector<std::vector<int64_t>>& input_shapes) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0, limit = input_shapes.size(); i < limit; i++) {
    auto g_in_shape = graph_proto->mutable_input(static_cast<int>(i))
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
    g_in_shape->clear_dim();
    const auto& shape = input_shapes[i];
    for (size_t dim = 0, end = shape.size(); dim < end; dim++) {
      g_in_shape->add_dim()->set_dim_value(shape[dim]);
    }
  }
  return model_copy;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (int i = 0; i < graph_proto->input_size(); i++) {
    ONNX_NAMESPACE::TensorShapeProto* g_in_shape =
        graph_proto->mutable_input(static_cast<int>(i))
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape();
    g_in_shape->mutable_dim(0)->clear_dim_value();
    g_in_shape->mutable_dim(0)->set_dim_value(1);
  }
  return model_copy;
}

void BackendManager::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  std::chrono::high_resolution_clock::time_point start_compute, end_compute;
#ifdef OPENVINO_FIL_ENABLED
  static bool fil_enabled = true;
  if (fil_enabled) {
    start_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "Start Compute";
  }
#endif
  // OV NPU doesn't support dynamic shaped model inference.
  // if disable_dynamic_shapes is set to true then execution of dynamic model is done
  // by rewriting the model to static shaped model at runtime based on input shape.
  // disable_dynamic_shapes is always set to true for OV NPU plugin.
  bool use_dynamic_backend = true;
  if (subgraph_context_.has_dynamic_input_shape &&
      !GetGlobalContext().disable_dynamic_shapes &&
      (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
       GetGlobalContext().device_type.find("GPU") != std::string::npos)) {
    concrete_backend_->Infer(context);
    use_dynamic_backend = false;
  } else if (use_dynamic_backend && subgraph_context_.has_dynamic_input_shape) {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(ctx);
    auto key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);
    std::shared_ptr<IBackend> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating dynamic backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(*model_proto_, tensor_shapes);
      try {
        dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                      GetGlobalContext(),
                                                      subgraph_context_,
                                                      ep_ctx_handle_);
      } catch (const OnnxRuntimeException& ex) {
        if (GetGlobalContext().device_type.find("NPU") != std::string::npos) {
          LOGS_DEFAULT(WARNING) << ex.what();
          LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                                << "Falling back to OV CPU for execution";
          GetGlobalContext().device_type = "CPU";
          GetGlobalContext().precision_str = "FP32";
          key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);
          try {
            dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
          } catch (std::string const& msg) {
            ORT_THROW(msg);
          }
        }
      }
      backend_map_.insert({key, dynamic_backend});
    } else {
      dynamic_backend = search->second;
    }

    dynamic_backend->Infer(context);
  } else {
    concrete_backend_->Infer(context);
  }
#ifdef OPENVINO_FIL_ENABLED
  if (fil_enabled) {
    end_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "End Compute";
    std::chrono::duration<double> compute_time = end_compute - start_compute;
    std::cout << "Compute Time: " << compute_time.count() << " s" << std::endl;
    fil_enabled = false;  // calculating compute time for first run only
  }
#endif
}

void BackendManager::ShutdownBackendManager() {
}

}  // namespace openvino_ep
}  // namespace onnxruntime
