// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vaip/global_api.h"

#include <atomic>
#include <iostream>
#include <codecvt>
#include <fstream>
#ifdef _WIN32
#include <Windows.h>
#endif
#include "./vai_assert.h"

#include "core/common/exceptions.h"
#include "core/framework/error_code_helper.h"
#include "core/providers/shared/common.h"

#include "vaip/dll_safe.h"
#include "vaip/vaip_ort_api.h"
#include "vaip/graph.h"
#include "vaip/node.h"
#include "vaip/node_arg.h"

#include "./tensor_proto.h"
#include "./attr_proto.h"
#include "./register_xir_ops.h"

#include "onnxruntime_config.h"
#include "version_info.h"  // version_info.hpp.in

using namespace onnxruntime;
// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

/// @brief Gets the path of directory containing the dynamic library that contains the address.
/// @param address An address of a function or variable in the dynamic library.
/// @return The path of the directory containing the dynamic library, or an empty string if the path cannot be determined.
static onnxruntime::PathString GetDynamicLibraryLocationByAddress(const void* address) {
#ifdef _WIN32
  HMODULE moduleHandle;
  if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &moduleHandle)) {
    return {};
  }
  std::wstring buffer;
  for (std::uint32_t size{70}; size < 4096; size *= 2) {
    buffer.resize(size, L'\0');
    const std::uint32_t requiredSize = ::GetModuleFileNameW(moduleHandle, buffer.data(), size);
    if (requiredSize == 0) {
      break;
    }
    if (requiredSize == size) {
      continue;
    }
    buffer.resize(requiredSize);
    return {std::move(buffer)};
  }
#else
  std::ignore = address;
#endif
  return {};
}

vaip_core::OrtApiForVaip* create_org_api_hook();
struct OrtVitisAIEpAPI {
  void (*initialize_onnxruntime_vitisai_ep)(vaip_core::OrtApiForVaip* api, std::vector<OrtCustomOpDomain*>& ret_domain);
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_with_options)(
      const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options);
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_vitisai_ep_with_error_handling)(
      const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options, void* status, vaip_core::error_report_func func);
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_vitisai_ep_v3)(
      const std::filesystem::path& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options, void* status, vaip_core::error_report_func func);
  uint32_t (*vaip_get_version)();
  void (*create_ep_context_nodes)(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
      vaip_core::DllSafe<std::vector<Node*>>* ret_value) = nullptr;
  int (*vitisai_ep_on_run_start)(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps, const void* state,
      vaip_core::DllSafe<std::string> (*get_config_entry)(const void* state, const char* entry_name)) = nullptr;
  int (*vitisai_ep_set_ep_dynamic_options)(
      const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
      const char* const* keys,
      const char* const* values, size_t kv_len) = nullptr;
  void (*profiler_collect)(
      std::vector<EventInfo>& api_events,
      std::vector<EventInfo>& kernel_events);
  void (*deinitialize_onnxruntime_vitisai_ep)();
  void Ensure() {
    if (handle_)
      return;
    auto& env = Provider_GetHost()->Env__Default();
#ifdef _WIN32
    // this dll is already linked to the executable, normally a test program
    handle_ = reinterpret_cast<void*>(GetModuleHandle(TEXT("onnxruntime_vitisai_ep.dll")));
    if (!handle_) {
      // First try loading with full path
      auto library_filename = PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_vitisai_ep") LIBRARY_EXTENSION);
      auto full_path = env.GetRuntimePath() + PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_vitisai_ep") LIBRARY_EXTENSION);
      if (std::filesystem::exists(full_path)) {
        ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(full_path, true, &handle_));
      } else {
        // Identify the path of the current dynamic library, and expect that onnxruntime_vitisai_ep is in the same directory.
        PathString current_path = GetDynamicLibraryLocationByAddress(reinterpret_cast<const void*>(create_org_api_hook));
        if (!current_path.empty()) {
          const std::filesystem::path parent_path = std::filesystem::path{std::move(current_path)}.parent_path();
          PathString module_relative_full_path = PathString(parent_path / library_filename);
          ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(module_relative_full_path, true, &handle_));
        }
      }
    }
#else
    auto full_path = env.GetRuntimePath() + PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_vitisai_ep") LIBRARY_EXTENSION);
    ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(full_path, true, &handle_));
#endif
    ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(handle_, "initialize_onnxruntime_vitisai_ep", (void**)&initialize_onnxruntime_vitisai_ep));
    auto status1 = env.GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep_with_error_handling", (void**)&compile_onnx_model_vitisai_ep_with_error_handling);
    auto status2 = env.GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep_with_options", (void**)&compile_onnx_model_with_options);
    auto status3 = env.GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep_v3", (void**)&compile_onnx_model_vitisai_ep_v3);
    if ((!status1.IsOK()) && (!status2.IsOK()) && (!status3.IsOK())) {
      ::onnxruntime::LogRuntimeError(0, status2, __FILE__, static_cast<const char*>(__FUNCTION__), __LINE__);
      ORT_THROW(status2);
    }
    std::ignore = env.GetSymbolFromLibrary(handle_, "vaip_get_version",
                                           (void**)&vaip_get_version);
    std::ignore = env.GetSymbolFromLibrary(handle_, "profiler_collect", (void**)&profiler_collect);
    ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(handle_, "create_ep_context_nodes", (void**)&create_ep_context_nodes));
    ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(handle_, "vitisai_ep_on_run_start", (void**)&vitisai_ep_on_run_start));
    ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(handle_, "vitisai_ep_set_ep_dynamic_options", (void**)&vitisai_ep_set_ep_dynamic_options));
    std::ignore = env.GetSymbolFromLibrary(handle_, "deinitialize_onnxruntime_vitisai_ep", (void**)&deinitialize_onnxruntime_vitisai_ep);
  }
  void Clear() {
    if (handle_) {
      auto& env = Provider_GetHost()->Env__Default();
      auto status = env.UnloadDynamicLibrary(handle_);
      vai_assert(status.IsOK(), status.ErrorMessage());
      handle_ = nullptr;
    }
  }

 private:
  void* handle_{};
};

static OrtVitisAIEpAPI s_library_vitisaiep;
static std::shared_ptr<KernelRegistry> s_kernel_registry_vitisaiep;
static std::vector<OrtCustomOpDomain*> s_domains_vitisaiep;
static vaip_core::OrtApiForVaip the_global_api;
std::shared_ptr<KernelRegistry> get_kernel_registry_vitisaiep() { return s_kernel_registry_vitisaiep; }
const std::vector<OrtCustomOpDomain*>& get_domains_vitisaiep() { return s_domains_vitisaiep; }

void profiler_collect(
    std::vector<EventInfo>& api_events,
    std::vector<EventInfo>& kernel_events) {
  if (s_library_vitisaiep.profiler_collect) {
    s_library_vitisaiep.profiler_collect(api_events, kernel_events);
  }
}

void change_status_with_error(void* status_ptr, int error_code, const char* error_msg) {
  auto status = reinterpret_cast<Status*>(status_ptr);
  *status = Status(onnxruntime::common::ONNXRUNTIME, error_code, error_msg);
}

vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(
    const onnxruntime::GraphViewer& graph_viewer, const onnxruntime::logging::Logger& logger, const onnxruntime::ProviderOptions& options) {
  auto model_path = graph_viewer.ModelPath();
  if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_v3) {
    Status status = Status::OK();
    auto status_ptr = reinterpret_cast<void*>(&status);
    auto ret = vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_vitisai_ep_v3(model_path, graph_viewer.GetGraph(), options, status_ptr, change_status_with_error));
    if (!status.IsOK()) {
      ORT_THROW(status);
    }
    return ret;
  } else if (s_library_vitisaiep.compile_onnx_model_vitisai_ep_with_error_handling) {
    Status status = Status::OK();
    auto status_ptr = reinterpret_cast<void*>(&status);
    auto ret = vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_vitisai_ep_with_error_handling(model_path.u8string(), graph_viewer.GetGraph(), options, status_ptr, change_status_with_error));
    if (!status.IsOK()) {
      ORT_THROW(status);
    }
    return ret;
  } else {
    return vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_with_options(model_path.u8string(), graph_viewer.GetGraph(), options));
  }
}

std::optional<std::vector<Node*>> create_ep_context_nodes(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps) {
  if (s_library_vitisaiep.create_ep_context_nodes) {
    vaip_core::DllSafe<std::vector<Node*>> nodes;
    s_library_vitisaiep.create_ep_context_nodes(eps, &nodes);
    if (nodes.get()) {
      auto ret = std::vector<Node*>(*nodes);
      return ret;
    }
  }
  return std::nullopt;
}

int vitisai_ep_on_run_start(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps, const void* state,
    vaip_core::DllSafe<std::string> (*get_config_entry)(const void* state, const char* entry_name)) {
  if (s_library_vitisaiep.vitisai_ep_on_run_start) {
    return s_library_vitisaiep.vitisai_ep_on_run_start(eps, state, get_config_entry);
  }
  return 100;
}

int vitisai_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps, const char* const* keys,
    const char* const* values, size_t kv_len) {
  if (s_library_vitisaiep.vitisai_ep_set_ep_dynamic_options) {
    return s_library_vitisaiep.vitisai_ep_set_ep_dynamic_options(eps, keys, values, kv_len);
  }
  return 100;
}

struct MyCustomOpKernel : OpKernel {
  MyCustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    op_kernel_ =
        op_.CreateKernel(&op_, Ort::Global<void>::api_, reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~MyCustomOpKernel() override { op_.KernelDestroy(op_kernel_); }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyCustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

void create_kernel_registry(const std::vector<OrtCustomOpDomain*>& domains) {
  s_kernel_registry_vitisaiep = KernelRegistry::Create();
  for (const auto& domain : domains) {
    for (const auto* op : domain->custom_ops_) {
      const size_t input_count = op->GetInputTypeCount(op);
      const size_t output_count = op->GetOutputTypeCount(op);
      auto def_builder = KernelDefBuilder::Create();
      def_builder->SetName(op->GetName(op));
      def_builder->SetDomain(domain->domain_.c_str());
      def_builder->SinceVersion(op->GetStartVersion(op), op->GetEndVersion(op));
      if (op->version > 12) {
        for (auto i = 0u; i < input_count; i++) {
          def_builder->InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }
      def_builder->Provider(op->GetExecutionProviderType(op));

      auto schema = Provider_GetHost()->GetSchema(op->GetName(op), op->GetStartVersion(op), domain->domain_);
      for (size_t i = 0; i < input_count; i++) {
        const auto input_type = op->GetInputType(op, i);
        auto input_name = schema->inputs__GetName(i);
        if (schema->typeConstraintMap().count(schema->inputs__GetTypeStr(i))) {
          input_name = schema->inputs__GetTypeStr(i);
        }
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          def_builder->TypeConstraint(input_name.c_str(), DataTypeImpl::AllTensorTypes());
        } else {
          def_builder->TypeConstraint(input_name.c_str(), DataTypeImpl::GetTensorTypeFromOnnxType(input_type));
        }
      }
      for (size_t i = 0; i < output_count; i++) {
        const auto output_type = op->GetOutputType(op, i);
        auto output_name = schema->outputs__GetName(i);
        if (schema != nullptr) {
          if (schema->typeConstraintMap().count(schema->outputs__GetTypeStr(i))) {
            output_name = schema->outputs__GetTypeStr(i);
          }
        }
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          def_builder->TypeConstraint(output_name.c_str(), DataTypeImpl::AllTensorTypes());
        } else {
          def_builder->TypeConstraint(output_name.c_str(), DataTypeImpl::GetTensorTypeFromOnnxType(output_type));
        }
      }
      KernelCreateFn kernel_create_fn =
          [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<MyCustomOpKernel>(info, *op);
        return Status::OK();
      };
      std::ignore = s_kernel_registry_vitisaiep->Register(KernelCreateInfo(def_builder->Build(), kernel_create_fn));
    }
  }
}

void initialize_vitisai_ep() {
  s_library_vitisaiep.Ensure();
  s_domains_vitisaiep.reserve(100);
  s_library_vitisaiep.initialize_onnxruntime_vitisai_ep(create_org_api_hook(), s_domains_vitisaiep);
  vaip::register_xir_ops(s_domains_vitisaiep);
  create_kernel_registry(s_domains_vitisaiep);
}

void deinitialize_vitisai_ep() {
  vaip::deregister_xir_ops(s_domains_vitisaiep);
  if (s_library_vitisaiep.deinitialize_onnxruntime_vitisai_ep) {
    s_library_vitisaiep.deinitialize_onnxruntime_vitisai_ep();
  }
  // kernel registry would be repopulated, no need to delete kernel registry
  s_domains_vitisaiep.clear();

  s_library_vitisaiep.Clear();
  s_kernel_registry_vitisaiep.reset();
}

static void set_version_info(vaip_core::OrtApiForVaip& api) {
  const char* magic = "VAIP";
  std::memcpy(reinterpret_cast<char*>(&api.magic), magic, sizeof(api.magic));
  api.major = VAIP_ORT_API_MAJOR;
  api.minor = VAIP_ORT_API_MINOR;
  api.patch = VAIP_ORT_API_PATCH;
}

vaip_core::OrtApiForVaip* create_org_api_hook() {
  InitProviderOrtApi();
  set_version_info(the_global_api);
  the_global_api.host_ = Provider_GetHost();
  assert(Ort::Global<void>::api_ != nullptr);
  the_global_api.ort_api_ = Ort::Global<void>::api_;
  the_global_api.model_load = [](const std::string& filename) -> Model* {
    auto model_proto = ONNX_NAMESPACE::ModelProto::Create();
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto file_path = ToPathString(filename);
    auto status = Model::Load(file_path, *model_proto);
    vai_assert(status.IsOK(), "load model proto error");
    auto model = Model::Create(std::move(*model_proto), file_path, nullptr, logger);
    return model.release();
  };
  the_global_api.model_delete = [](Model* model) { delete model; };

  the_global_api.model_clone = vaip::model_clone;
  the_global_api.model_set_meta_data = [](Model& model, const std::string& key, const std::string& value) {
    const_cast<ModelMetaData&>(model.MetaData())[key] = value;
  };
  the_global_api.model_get_meta_data =
      [](const Model& model, const std::string& key) -> vaip_core::DllSafe<std::string> {
    if (model.MetaData().count(key)) {
      return vaip_core::DllSafe(model.MetaData().at(key));
    }
    return vaip_core::DllSafe(std::string());
  };
  the_global_api.model_has_meta_data = [](const Model& model, const std::string& key) -> int {
    return int(model.MetaData().count(key));
  };
  the_global_api.model_main_graph = [](Model& model) -> Graph& { return model.MainGraph(); };
  the_global_api.graph_get_model = [](const Graph& graph) -> const Model& { return graph.GetModel(); };
  the_global_api.graph_get_inputs_unsafe = [](const Graph& graph) -> auto {
    return vaip_core::DllSafe(graph.GetInputs());
  };
  the_global_api.graph_get_outputs_unsafe = [](const Graph& graph) -> auto {
    return vaip_core::DllSafe(graph.GetOutputs());
  };
  the_global_api.graph_set_outputs = [](Graph& graph, gsl::span<const NodeArg* const> outputs) {
    graph.SetOutputs(outputs);
  };
  the_global_api.graph_get_node_arg = [](const Graph& graph, const std::string& name) -> const NodeArg* {
    return graph.GetNodeArg(name);
  };
  the_global_api.graph_producer_node = [](const Graph& graph, const std::string& name) -> const Node* {
    return graph.GetProducerNode(name);
  };
  the_global_api.graph_get_node = [](const Graph& graph, size_t index) -> const Node* {
    return graph.GetNode(index);
  };
  the_global_api.graph_save = vaip::graph_save;
  the_global_api.graph_fuse = vaip::graph_fuse;
  the_global_api.graph_remove_node = vaip::graph_remove_node;
  the_global_api.graph_add_node = vaip::graph_add_node;
  the_global_api.graph_get_all_initialized_tensors = [](const Graph& graph) -> const InitializedTensorSet& {
    return graph.GetAllInitializedTensors();
  };
  the_global_api.graph_resolve = [](Graph& graph, bool force) {
    if (force) {
      graph.SetGraphResolveNeeded();
    }
    auto status = graph.Resolve();
    if (!status.IsOK()) {
      std::cerr << "graph resolve error:" << status.ErrorMessage() << std::endl;
    }
    return status.Code();
  };
  the_global_api.graph_get_consumer_nodes_unsafe = [](const Graph& graph, const std::string& node_arg_name) -> auto {
    return vaip_core::DllSafe(graph.GetConsumerNodes(node_arg_name));
  };
  the_global_api.graph_nodes_unsafe = [](const Graph& graph) -> auto { return vaip_core::DllSafe(graph.Nodes()); };
  the_global_api.graph_get_name = [](const Graph& graph) -> const std::string& { return graph.Name(); };
  the_global_api.graph_set_name = [](Graph& graph, const char* name) -> void { return graph.SetName(std::string(name)); };
  the_global_api.graph_reverse_dfs_from = [](const Graph& graph, gsl::span<const Node* const> from,
                                             const auto& enter, const auto& leave, const auto& stop) {
    graph.ReverseDFSFrom(from, enter, leave, nullptr, stop);
  };

  the_global_api.graph_infer_shapes_from_filepath = [](const std::string& m, const std::string& save_path) -> auto { return Provider_GetHost()->InferShapes(m, save_path); };
  the_global_api.graph_to_graph_proto = [](const Graph& graph) -> ONNX_NAMESPACE::GraphProto* {
    return graph.ToGraphProto().release();
  };
  the_global_api.graph_proto_delete = [](ONNX_NAMESPACE::GraphProto* p) { delete p; };
  the_global_api.graph_infer_shapes = [](ONNX_NAMESPACE::ModelProto& m) -> auto { return Provider_GetHost()->InferShapes(m); };

  // node
  the_global_api.node_get_inputs_unsafe = vaip::node_get_inputs;
  the_global_api.node_get_output_node_args_unsafe = vaip::node_get_output_node_args;
  the_global_api.node_op_type = [](const Node& node) -> const std::string& { return node.OpType(); };
  the_global_api.node_op_domain = [](const Node& node) -> const std::string& { return node.Domain(); };
  the_global_api.node_get_index = [](const Node& node) -> size_t { return node.Index(); };
  the_global_api.node_get_name = [](const Node& node) -> const std::string& { return node.Name(); };
  the_global_api.node_description = [](const Node& node) -> const std::string& { return node.Description(); };
  the_global_api.node_get_attributes = [](Node& node) -> NodeAttributes& {
    return const_cast<NodeAttributes&>(node.GetAttributes());
  };
  the_global_api.node_type_is_fused = [](const Node& node) { return node.NodeType() == Node::Type::Fused; };
  the_global_api.node_get_function_body = [](const Node& node) -> const auto& {
    assert(node.GetFunctionBody() != nullptr);
    return node.GetFunctionBody()->Body();
  };

  // node_arg
  the_global_api.node_arg_get_name_unsafe =
      [](const NodeArg& node_arg) -> const std::string& { return node_arg.Name(); };
  the_global_api.node_arg_clone = vaip::node_arg_clone;
  the_global_api.node_arg_new = vaip::node_arg_new;
  the_global_api.node_arg_is_exists = [](const NodeArg& node_arg) { return node_arg.Exists(); };
  the_global_api.node_arg_is_constant = vaip::node_arg_is_constant;
  the_global_api.node_arg_get_shape_i64_unsafe = vaip::node_arg_get_shape_i64;
  the_global_api.node_arg_set_shape_i64 = vaip::node_arg_set_shape_i64;
  the_global_api.node_arg_get_denotation_unsafe = vaip::node_arg_get_denotation;

  the_global_api.node_arg_set_denotation = vaip::node_arg_set_denotation;
  the_global_api.node_arg_get_const_data_as_tensor = vaip::node_arg_get_const_data_as_tensor;

  the_global_api.node_arg_get_element_type = vaip::node_arg_get_element_type;
  the_global_api.node_arg_set_element_type = vaip::node_arg_set_element_type;
  /// attr proto
  the_global_api.attr_proto_delete = [](ONNX_NAMESPACE::AttributeProto* v) { delete v; };
  the_global_api.attr_proto_clone = [](const ONNX_NAMESPACE::AttributeProto& v) -> ONNX_NAMESPACE::AttributeProto* {
    auto ret = ONNX_NAMESPACE::AttributeProto::Create();
    *ret = v;
    return ret.release();
  };
  the_global_api.attr_proto_get_name = [](const auto& attr_proto) -> const std::string& { return attr_proto.name(); };
  the_global_api.attr_proto_set_name = [](auto* attr_proto, const auto& name) { attr_proto->set_name(name); };
  the_global_api.attr_proto_new_int = vaip::attr_proto_new_int;
  the_global_api.attr_proto_new_float = vaip::attr_proto_new_float;
  the_global_api.attr_proto_new_string = vaip::attr_proto_new_string;
  the_global_api.attr_proto_new_tensor = vaip::attr_proto_new_tensor;
  the_global_api.attr_proto_new_ints = vaip::attr_proto_new_ints;
  the_global_api.attr_proto_new_floats = vaip::attr_proto_new_floats;
  the_global_api.attr_proto_new_strings = vaip::attr_proto_new_strings;
  the_global_api.attr_proto_get_int = vaip::attr_proto_get_int;
  the_global_api.attr_proto_get_float = vaip::attr_proto_get_float;
  the_global_api.attr_proto_get_string = vaip::attr_proto_get_string;
  the_global_api.attr_proto_get_tensor = vaip::attr_proto_get_tensor;
  the_global_api.attr_proto_get_ints = vaip::attr_proto_get_ints;
  the_global_api.attr_proto_get_floats = vaip::attr_proto_get_floats;
  the_global_api.attr_proto_get_strings = vaip::attr_proto_get_strings;
  the_global_api.attr_proto_get_type = [](const ONNX_NAMESPACE::AttributeProto& attr) -> int { return attr.type(); };

  /// node attributes
  the_global_api.node_attributes_new = []() { return NodeAttributes::Create().release(); };
  the_global_api.node_attributes_add = [](NodeAttributes& p, ONNX_NAMESPACE::AttributeProto&& attr) {
    p.insert_or_assign(attr.name(), std::move(attr));
  };

  the_global_api.node_attributes_delete = [](NodeAttributes* p) { delete p; };
  the_global_api.node_attributes_get =
      [](const NodeAttributes& attr, const std::string& name) -> const ONNX_NAMESPACE::AttributeProto* {
    if (attr.count(name)) {
      return &attr.at(name);
    }
    return nullptr;
  };
  the_global_api.node_attributes_get_keys = [](NodeAttributes& attr) -> vaip_core::DllSafe<std::vector<std::string>> {
    auto ret = std::vector<std::string>();
    ret.reserve(attr.size());
    for (auto& it : attr) {
      ret.push_back(it.first);
    }
    return vaip_core::DllSafe(std::move(ret));
  };
  /// tensor proto
  the_global_api.tensor_proto_get_shape_unsafe = vaip::tensor_proto_get_shape;
  the_global_api.tensor_proto_data_type = [](const ONNX_NAMESPACE::TensorProto& t) -> int { return t.data_type(); };
  the_global_api.tensor_proto_delete = [](ONNX_NAMESPACE::TensorProto* tp) { delete tp; };
  the_global_api.tensor_proto_new_i4 = vaip::tensor_proto_new_i4;
  the_global_api.tensor_proto_new_i8 = vaip::tensor_proto_new_i8;
  the_global_api.tensor_proto_new_i16 = vaip::tensor_proto_new_i16;
  the_global_api.tensor_proto_new_i32 = vaip::tensor_proto_new_i32;
  the_global_api.tensor_proto_new_i64 = vaip::tensor_proto_new_i64;
  the_global_api.tensor_proto_new_u4 = vaip::tensor_proto_new_u4;
  the_global_api.tensor_proto_new_u8 = vaip::tensor_proto_new_u8;
  the_global_api.tensor_proto_new_u16 = vaip::tensor_proto_new_u16;
  the_global_api.tensor_proto_new_u32 = vaip::tensor_proto_new_u32;
  the_global_api.tensor_proto_new_u64 = vaip::tensor_proto_new_u64;
  the_global_api.tensor_proto_new_floats = vaip::tensor_proto_new_floats;
  the_global_api.tensor_proto_new_doubles = vaip::tensor_proto_new_doubles;
  the_global_api.tensor_proto_new_bf16 = vaip::tensor_proto_new_bf16;
  the_global_api.tensor_proto_new_fp16 = vaip::tensor_proto_new_fp16;
  the_global_api.tensor_proto_raw_data_size = [](const auto& tensor) { return tensor.raw_data().size(); };
  the_global_api.tensor_proto_as_raw = vaip::tensor_proto_as_raw;
  the_global_api.tensor_proto_get_name = [](const auto& tensor) -> const std::string& { return tensor.name(); };

  the_global_api.get_lib_name = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string("onnxruntime.") + std::string(ORT_VERSION));
  };

  the_global_api.get_lib_id = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string(GIT_COMMIT_ID));
  };

  the_global_api.graph_add_initialized_tensor = [](Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor) {
    graph.AddInitializedTensor(tensor);
  };

  the_global_api.get_model_path = [](const Graph& graph) -> const std::filesystem::path& {
    return graph.ModelPath();
  };

  the_global_api.create_empty_model = [](const std::filesystem::path& path, const std::vector<std::pair<std::string, int64_t>>& opset) -> Model* {
    auto model_proto = ONNX_NAMESPACE::ModelProto::Create();
    auto graph_proto = ONNX_NAMESPACE::GraphProto::Create();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    for (const auto& op : opset) {
      auto* opset_import = model_proto->add_opset_import();
      *(opset_import->mutable_domain()) = op.first;
      opset_import->set_version(op.second);
    }
    std::ignore = model_proto->mutable_graph();  // create a graph
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto model = Model::Create(std::move(*model_proto), path, nullptr, logger);
    return model.release();
  };

  the_global_api.graph_set_inputs = [](Graph& graph, gsl::span<const NodeArg* const> inputs) {
    graph.SetInputs(inputs);
  };
  the_global_api.session_option_configuration = [](
                                                    void* mmap, void* session_options, void (*push)(void* mmap, const char* name, const char* value)) {
    auto options = reinterpret_cast<OrtSessionOptions*>(session_options);
    auto option_list = options->GetConfigOptionsMap();
    // option_list.GetConfigEntry
    for (const auto& option : option_list) {
      push(mmap, option.first.c_str(), option.second.c_str());
    }
  };
  the_global_api.node_arg_external_location = vaip::node_arg_external_location;
  the_global_api.model_to_proto = [](onnxruntime::Model& model) { return model.ToProto().release(); };
  the_global_api.model_proto_serialize_as_string = [](ONNX_NAMESPACE::ModelProto& model_proto) {
    return vaip_core::DllSafe(model_proto.SerializeAsString());
  };
  the_global_api.model_proto_delete = [](ONNX_NAMESPACE::ModelProto* p) { delete p; };
  the_global_api.attr_proto_release_string = [](ONNX_NAMESPACE::AttributeProto* attr) -> vaip_core::DllSafe<std::string> {
    auto pstr = vaip::attr_proto_release_string(attr);
    std::string local_str = std::move(*pstr);
    pstr = nullptr;
    return vaip_core::DllSafe<std::string>(std::move(local_str));
  };

  the_global_api.is_profiling_enabled = [](void* session_options) {
    auto options = reinterpret_cast<OrtSessionOptions*>(session_options);
    return options->GetEnableProfiling();
  };
  the_global_api.graph_remove_initialized_tensor = [](Graph& graph, const std::string& tensor_name) {
    graph.RemoveInitializedTensor(tensor_name);
  };
  the_global_api.graph_reverse_dfs_from_preemp = vaip::graph_reverse_dfs_from;
  the_global_api.graph_save_string = vaip::graph_save_string;

  if (!s_library_vitisaiep.vaip_get_version) {
    return reinterpret_cast<vaip_core::OrtApiForVaip*>(&(the_global_api.host_));
  } else {
    return &the_global_api;
  }
}
