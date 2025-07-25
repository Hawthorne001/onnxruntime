// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
#include <fstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <set>
#include <filesystem>

#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/safeint.h"
#include "core/common/logging/severity.h"
#include "migraphx_execution_provider.h"
#include "migraphx_execution_provider_info.h"
#include "migraphx_execution_provider_utils.h"
#include "migraphx_allocator.h"
#include "gpu_data_transfer.h"
#include <hip/hip_version.h>
#include "migraphx_call.h"

#include "migraphx_stream_handle.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    const IDataTransfer* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
    if (!gpu_data_transfer)
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "gpu data transfer is missing in Migraphx EP.");
    // CopyTensorAsync could handle both pinned memory and non-pinned CPU memory.
    // For non-pinned CPU memory, the copy is synchronous.
    return gpu_data_transfer->CopyTensorAsync(*X, *Y, *(ctx->GetComputeStream()));
  }
};

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMIGraphXExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMIGraphXExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMIGraphXExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_THROW_IF_ERROR(s_kernel_registry->Register(function_table_entry()));
  }
}

void DeleteRegistry() {
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> MIGraphXExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

MIGraphXExecutionProvider::MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMIGraphXExecutionProvider,
                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD,
                                   info.device_id)},
      info_(info) {
  InitProviderOrtApi();
  get_flags_from_session_info(info);
  metadef_id_generator_ = ModelMetadefIdGenerator::Create();
  get_flags_from_env();
}

MIGraphXExecutionProvider::~MIGraphXExecutionProvider() {
}

void MIGraphXExecutionProvider::get_flags_from_session_info(const MIGraphXExecutionProviderInfo& info) {
  // Set GPU device to be used
  HIP_CALL_THROW(hipSetDevice(info_.device_id));
  t_ = migraphx::target(info.target_device.c_str());

  // Quantization
  fp16_enable_ = info.fp16_enable;

#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
  fp8_enable_ = info.fp8_enable;
#else
  LOGS_DEFAULT(WARNING) << "MIGraphX: FP8 Quantization requires ROCm 6.4 or greater";
  fp8_enable_ = false;
#endif
  int8_enable_ = info.int8_enable;

  if (int8_enable_ and fp8_enable_) {
    LOGS_DEFAULT(FATAL) << "MIGraphX: FP8 and INT8 Quantization Mutually exclusive. Ignoring both Quantization flags";
  }

  if (int8_enable_ xor fp8_enable_) {
    int8_calibration_cache_name_ = info.int8_calibration_table_name;
    int8_use_native_migraphx_calibration_table_ = info.int8_use_native_calibration_table;
  }

  if (int8_enable_ or fp8_enable_) {
    int8_calibration_cache_available_ = !info.int8_calibration_table_name.empty();
  }

  // Load INT8 calibration table
  std::unordered_map<std::string, float> dynamic_range_map;
  if ((int8_enable_ || fp8_enable_) && int8_calibration_cache_available_) {
    const std::string calibration_cache_path = GetCachePath(calibration_cache_path_, int8_calibration_cache_name_);
    if (!ReadDynamicRange(calibration_cache_path, int8_use_native_migraphx_calibration_table_, dynamic_range_map)) {
      throw std::runtime_error("Session Failed to read INT8 calibration table " + calibration_cache_path);
    }
  }

  // Save/load migraphx compiled models
  save_compiled_model_ = info.save_compiled_model;
  save_compiled_path_ = info.save_model_file;
  load_compiled_model_ = info.load_compiled_model;
  load_compiled_path_ = info.load_model_file;

  exhaustive_tune_ = info.exhaustive_tune;

  LOGS_DEFAULT(WARNING) << "[MIGraphX EP] MIGraphX provider Session Options:";
  print_migraphx_ep_flags();
}

void MIGraphXExecutionProvider::get_flags_from_env() {
  LOGS_DEFAULT(WARNING) << "\n[MIGraphX EP] MIGraphX ENV Override Variables Set:";
  // whether fp16 is enable
  const std::string fp16_enable_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kFP16Enable);
  if (!fp16_enable_env.empty()) {
    fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_FP16_ENABLE: " << fp16_enable_;
  }

  // whether fp8 quantization is enabled
  const std::string fp8_enable_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kFP8Enable);
  if (!fp8_enable_env.empty()) {
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
    fp8_enable_ = (std::stoi(fp8_enable_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_FP8_ENABLE: " << fp8_enable_;
#else
    LOGS_DEFAULT(WARNING) << "MIGraphX: FP8 Quantization requires ROCm 6.4 or greater";
    fp8_enable = false;
#endif
  }

  // whether int8 is enabled
  const std::string int8_enable_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kINT8Enable);
  if (!int8_enable_env.empty()) {
    int8_enable_ = (std::stoi(int8_enable_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_INT8_ENABLE: " << int8_enable_;
  }

  if (int8_enable_ and fp8_enable_) {
    LOGS_DEFAULT(FATAL) << "\nMIGraphX: FP8 and INT8 Quantization Mutually exclusive. Ignoring both Quantization flags";
  }

  if (int8_enable_ || fp8_enable_) {
    const std::string int8_calibration_cache_name_env =
        onnxruntime::GetEnvironmentVar(migraphx_env_vars::kINT8CalibrationTableName);
    if (!int8_calibration_cache_name_env.empty()) {
      int8_calibration_cache_name_ = int8_calibration_cache_name_env;
      LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_CALIBRATION_TABLE_NAME: " << int8_calibration_cache_name_;
    }

    const std::string cache_path = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kCachePath);
    if (!cache_path.empty()) {
      calibration_cache_path_ = cache_path;
      LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_CACHE_PATH: " << calibration_cache_path_;
    }

    const std::string int8_use_native_migraphx_calibration_table_env =
        onnxruntime::GetEnvironmentVar(migraphx_env_vars::kINT8UseNativeMIGraphXCalibrationTable);
    if (!int8_use_native_migraphx_calibration_table_env.empty()) {
      int8_use_native_migraphx_calibration_table_ =
          (std::stoi(int8_use_native_migraphx_calibration_table_env) == 0 ? false : true);
      LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE: "
                            << int8_use_native_migraphx_calibration_table_;
    }
  }

  if (int8_enable_ or fp8_enable_) {
    int8_calibration_cache_available_ = !int8_calibration_cache_name_.empty();
  }

  // Load INT8 calibration table
  std::unordered_map<std::string, float> dynamic_range_map;
  if ((int8_enable_ || fp8_enable_) && int8_calibration_cache_available_) {
    const std::string calibration_cache_path = GetCachePath(calibration_cache_path_, int8_calibration_cache_name_);
    if (!ReadDynamicRange(calibration_cache_path, int8_use_native_migraphx_calibration_table_, dynamic_range_map)) {
      throw std::runtime_error("ENV Failed to read calibration table " + calibration_cache_path);
    }
  }

  // Save/load migraphx compiled models
  const std::string save_comp_model_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kSaveCompiledModel);
  if (!save_comp_model_env.empty()) {
    save_compiled_model_ = (std::stoi(save_comp_model_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_SAVE_COMPILED_MODEL: " << save_compiled_model_;
  }

  const std::string save_model_path_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kSavedModelPath);
  if (save_compiled_model_ && !save_model_path_env.empty()) {
    save_compiled_path_ = save_model_path_env;
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_SAVE_COMPILED_PATH: " << save_compiled_path_;
  }

  const std::string load_comp_model_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kLoadCompiledModel);
  if (!load_comp_model_env.empty()) {
    load_compiled_model_ = (std::stoi(load_comp_model_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_LOAD_COMPILED_MODEL: " << load_compiled_model_;
  }

  const std::string load_model_path_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kLoadModelPath);
  if (load_compiled_model_ && !load_model_path_env.empty()) {
    load_compiled_path_ = load_model_path_env;
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_LOAD_COMPILED_PATH: " << load_compiled_path_;
  }

  // dump unsupported ops
  const std::string dump_model_ops_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::dumpModelOps);
  if (!dump_model_ops_env.empty()) {
    dump_model_ops_ = (std::stoi(dump_model_ops_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_DUMP_MODEL_OPS: " << dump_model_ops_;
  }

  // Allow for exhaustive tune during compile
  const std::string exhaustive_tune_env = onnxruntime::GetEnvironmentVar(migraphx_env_vars::kExhaustiveTune);
  if (!exhaustive_tune_env.empty()) {
    exhaustive_tune_ = (std::stoi(exhaustive_tune_env) == 0 ? false : true);
    LOGS_DEFAULT(WARNING) << "\nORT_MIGRAPHX_EXHAUSTIVE_TUNE_OPS: " << exhaustive_tune_;
  }
}

void MIGraphXExecutionProvider::print_migraphx_ep_flags() {
  LOGS_DEFAULT(WARNING) << "\n device_id: " << info_.device_id
                        << "\n migraphx_fp16_enable: " << fp16_enable_
                        << "\n migraphx_fp8_enable: " << fp8_enable_
                        << "\n migraphx_int8_enable: " << int8_enable_
                        << "\n dump_model_ops: " << dump_model_ops_
                        << "\n exhaustive_tune: " << exhaustive_tune_
                        << "\n migraphx_int8_calibration_cache_name: " << int8_calibration_cache_name_
                        << "\n int8_calibration_cache_available: " << int8_calibration_cache_available_
                        << "\n use_native_migraphx_calibration_table: " << int8_use_native_migraphx_calibration_table_
                        << "\n migraphx_save_compiled_model: " << save_compiled_model_
                        << "\n migraphx_save_compiled_model_path: " << save_compiled_path_
                        << "\n migraphx_load_compiled_model: " << load_compiled_model_
                        << "\n migraphx_load_compiled_model_path: " << load_compiled_path_;
}

AllocatorPtr MIGraphXExecutionProvider::CreateMIGraphXAllocator(OrtDevice::DeviceId device_id,
                                                                size_t migx_mem_limit,
                                                                ArenaExtendStrategy arena_extend_strategy,
                                                                MIGraphXExecutionProviderExternalAllocatorInfo
                                                                    external_allocator_info,
                                                                const OrtArenaCfg* default_memory_arena_cfg) {
  if (external_allocator_info.UseExternalAllocator()) {
    AllocatorCreationInfo default_memory_info(
        [external_allocator_info](OrtDevice::DeviceId id) {
          return std::make_unique<MIGraphXExternalAllocator>(id, HIP,
                                                             external_allocator_info.alloc,
                                                             external_allocator_info.free,
                                                             external_allocator_info.empty_cache);
        },
        device_id,
        false);

    return CreateAllocator(default_memory_info);
  } else {
    AllocatorCreationInfo default_memory_info(
        [](OrtDevice::DeviceId id) {
          return std::make_unique<MIGraphXAllocator>(id, HIP);
        },
        device_id,
        true,
        {default_memory_arena_cfg ? *default_memory_arena_cfg
                                  : OrtArenaCfg(migx_mem_limit, static_cast<int>(arena_extend_strategy),
                                                -1, -1, -1, -1L)},
        // make it stream aware
        true);

    // ROCM malloc/free is expensive so always use an arena
    return CreateAllocator(default_memory_info);
  }
}

std::vector<AllocatorPtr> MIGraphXExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo default_memory_info(
      [](OrtDevice::DeviceId device_id) { return std::make_unique<MIGraphXAllocator>(device_id, onnxruntime::CUDA); },
      info_.device_id);
  AllocatorCreationInfo pinned_allocator_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<MIGraphXPinnedAllocator>(device_id, onnxruntime::CUDA_PINNED);
      },
      info_.device_id);
  return std::vector<AllocatorPtr>{CreateAllocator(default_memory_info), CreateAllocator(pinned_allocator_info)};
}

std::unique_ptr<onnxruntime::IDataTransfer> MIGraphXExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::GPUDataTransfer>();
}

static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
      return true;
    default:
      return false;
  }
}

static bool getMIGraphXType(ONNXTensorElementDataType type,
                            migraphx_shape_datatype_t& mgx_type) {
  mgx_type = migraphx_shape_float_type;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      mgx_type = migraphx_shape_half_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      mgx_type = migraphx_shape_float_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      mgx_type = migraphx_shape_double_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      mgx_type = migraphx_shape_fp8e4m3fnuz_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      mgx_type = migraphx_shape_fp8e4m3fn_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      mgx_type = migraphx_shape_fp8e5m2_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      mgx_type = migraphx_shape_fp8e5m2fnuz_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      mgx_type = migraphx_shape_int8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      mgx_type = migraphx_shape_int8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      mgx_type = migraphx_shape_int16_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      mgx_type = migraphx_shape_int32_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      mgx_type = migraphx_shape_int64_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      mgx_type = migraphx_shape_uint8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      mgx_type = migraphx_shape_uint8_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      mgx_type = migraphx_shape_uint16_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      mgx_type = migraphx_shape_uint32_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      mgx_type = migraphx_shape_uint64_type;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      mgx_type = migraphx_shape_bool_type;
      break;
    default:
      LOGS_DEFAULT(WARNING) << "MiGraphx: unsupported data type " << type << ", fallback to CPU";
      LOGS_DEFAULT(WARNING) << "implementation";
      return false;
  }

  return true;
}

std::vector<int> toVector(const ONNX_NAMESPACE::int64s& nums) {
  std::vector<int> result;
  int num = nums.size();
  for (int i = 0; i < num; ++i) {
    result.push_back(static_cast<int>(nums[i]));
  }

  return result;
}

static bool IsUnsupportedOpMode(const onnxruntime::GraphViewer& graph_viewer, const Node* node) {
  std::vector<NodeIndex> input_nodes;
  const auto& optype = node->OpType();
  if (optype == "ArgMax" or optype == "ArgMin") {
    const auto& attributes = node->GetAttributes();
    // we do not support select_last_index = 1 for now
    auto sli_attr = attributes.find("select_last_index");
    if (sli_attr != attributes.end() && (*sli_attr).second.i() != 0) {
      return true;
    }
  } else if (optype == "ConstantOfShape") {
    if (!canEvalNodeArgument(graph_viewer, node, {0}, input_nodes)) {
      return true;
    }
  } else if (optype == "ConvInteger") {
    // only support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }

    if ((input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) and
        (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)) {
      return true;
    }
  } else if (optype == "Expand") {
    // MIGraphX only supports constant shape input values
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "MaxPool") {
    // MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    // ceil_mode and dilations attrs are not supported in MIGraphX
    const auto& attributes = node->GetAttributes();
    auto dila_attr = attributes.find("dilations");
    if (dila_attr != attributes.end()) {
      auto dilas = toVector((*dila_attr).second.ints());
      bool ret = std::all_of(dilas.begin(), dilas.end(), [](auto i) { return i == 1; });
      if (ret == false) {
        return true;
      }
    }

    // storage order 1 (column major format) is not supported
    auto storage_order_attr = attributes.find("storage_order");
    if (storage_order_attr != attributes.end() and (*storage_order_attr).second.i() != 0) {
      return true;
    }

    // do not support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }
    auto data_type = input_type->tensor_type().elem_type();
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 or
        data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
      return true;
    }
  } else if (optype == "MatMulInteger") {
    // only support int8 and uint8 type
    const auto& input_type = node->InputDefs()[0]->TypeAsProto();
    if (input_type == nullptr) {
      return true;
    }

    if ((input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) and
        (input_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)) {
      return true;
    }
  } else if (optype == "NonZero") {
    if (!canEvalNodeArgument(graph_viewer, node, {0}, input_nodes)) {
      return true;
    }
  } else if (optype == "OneHot") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "Pad") {
    const auto& args = node->InputDefs();
    // if pad size is not constant, migraphx cannot support
    if (args.size() >= 2) {
      if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return true;
      }
    }

    const auto& attributes = node->GetAttributes();
    // Pad only support constant mode
    auto mode_attr = attributes.find("mode");
    std::string mode = "constant";
    if (mode_attr != attributes.end()) {
      mode = (*mode_attr).second.s();
    }
    static const std::set<std::string> allowed_modes = {"constant", "reflect"};
    if (allowed_modes.count(mode) == 0) {
      return true;
    }

    // input value only applied to constant mode
    if (mode == "constant") {
      if (args.size() == 3) {
        if (!canEvalNodeArgument(graph_viewer, node, {2}, input_nodes)) {
          return true;
        }
      }
    }
  } else if (optype == "Range") {
    auto arg_num = node->InputDefs().size();
    std::vector<std::size_t> vec(arg_num);
    std::iota(vec.begin(), vec.end(), 0);
    if (!canEvalNodeArgument(graph_viewer, node, vec, input_nodes)) {
      return true;
    }
  } else if (optype == "Reshape") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Resize" or optype == "Upsample") {
    const auto& attributes = node->GetAttributes();
    auto ct_attr = attributes.find("coordinate_transformation_mode");
    if (ct_attr != attributes.end()) {
      auto ct = (*ct_attr).second.s();
      if (ct == "tf_crop_and_resize") {
        return true;
      }
    }

    auto mode_attr = attributes.find("mode");
    if (mode_attr != attributes.end()) {
      auto mode = (*mode_attr).second.s();
      if (mode == "cubic") {
        return true;
      }
    }
  } else if (optype == "ReduceSum") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Slice") {
    // MIGraphX does not properly handle the situation where any
    // value of the "starts" attribute is higher than a corresponding
    // value in the "ends"
    auto arg_num = node->InputDefs().size();
    std::vector<std::size_t> vec(arg_num);
    std::iota(vec.begin(), vec.end(), 0);
    vec.erase(vec.begin());
    if (!canEvalNodeArgument(graph_viewer, node, vec, input_nodes)) {
      return true;
    }

    const auto& attributes = node->GetAttributes();
    if (attributes.count("starts") > 0 and attributes.count("ends") > 0) {
      auto starts = toVector((*attributes.find("starts")).second.ints());
      auto ends = toVector((*attributes.find("ends")).second.ints());
      for (std::size_t i = 0; i < starts.size(); ++i) {
        if (starts.at(i) > ends.at(i)) {
          return true;
        }
      }
    }
  } else if (optype == "Split") {
    // cannot process input dim of 0 size
    const auto arg_s = node->InputDefs()[0]->Shape();
    if (arg_s != nullptr) {
      const auto& tensor_dims = arg_s->dim();
      std::vector<std::size_t> dims;
      for (auto&& dim : tensor_dims) {
        dims.emplace_back(dim.has_dim_value() ? dim.dim_value() : 0);
      }
      if (dims == std::vector<std::size_t>{0}) {
        return true;
      }
    }

    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  } else if (optype == "Tile") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "TopK") {
    if (!canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
      return true;
    }
  } else if (optype == "Unsqueeze" or optype == "Squeeze") {
    const auto& args = node->InputDefs();
    if (args.size() == 2) {
      if (canEvalNodeArgument(graph_viewer, node, {1}, input_nodes)) {
        return false;
      }
      return true;
    }
  }

  // Op doesn't fall into known any of unsupported modes.
  return false;
}

void SubgraphPostProcessing(const onnxruntime::GraphViewer& graph_viewer, std::vector<std::vector<NodeIndex>>& clusters,
                            [[maybe_unused]] const logging::Logger& logger) {
  // Then check whether a subgraph should fall back to CPU
  // 1. Check whether a subgraph contains a RNN operator
  std::unordered_set<std::string> rnn_names = {"RNN", "GRU", "LSTM"};
  std::unordered_set<std::string> op_names = {"AveragePool", "Conv", "Gemm", "LRN", "MatMul", "MaxPool"};

  auto it = std::remove_if(clusters.begin(), clusters.end(), [&](auto git) {
    for (auto index : git) {
      auto node = graph_viewer.GetNode(index);
      if (node->OpType() == "Reshape") {
        const auto& args = node->InputDefs();
        if (args.size() == 2) {
          std::vector<NodeIndex> node_inputs;
          if (canEvalNodeArgument(graph_viewer, node, {1}, node_inputs)) {
            return (not std::all_of(node_inputs.begin(), node_inputs.end(), [&](auto index) {
              return std::find(git.begin(), git.end(), index) != git.end();
            }));
          } else {
            return true;
          }
        }
      }
    }

    // rnn operators, run on GPU
    if (std::any_of(git.begin(), git.end(), [&](auto nid) {
          const auto& node = graph_viewer.GetNode(nid);
          const auto& op_type = node->OpType();
          return (rnn_names.count(op_type) > 0);
        })) {
      return false;
    }

    // check operators gemm, matmul, convolution, lrn.
    if (std::any_of(git.begin(), git.end(), [&](auto nid) {
          const auto& node = graph_viewer.GetNode(nid);
          const auto& op_type = node->OpType();
          if (op_names.count(op_type) > 0) {
            // check number of elements in input
            auto inputs = node->InputDefs();
            if (std::any_of(inputs.begin(), inputs.end(), [&](auto& arg) {
                  const auto& arg_s = arg->Shape();
                  if (arg_s == nullptr) return false;
                  const auto& tensor_dims = arg_s->dim();
                  std::vector<std::size_t> dims;
                  for (auto&& dim : tensor_dims) {
                    dims.emplace_back(dim.has_dim_value() ? dim.dim_value() : 1);
                  }
                  return (std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<std::size_t>{}) > 300);
                })) {
              return false;
            }

            return true;
          }

          return false;
        })) {
      return false;
    }

    return true;
  });

  clusters.erase(it, clusters.end());
}

static bool IsNodeSupported(const std::set<std::string>& op_set,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx,
                            [[maybe_unused]] const logging::Logger& logger) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();
  const auto& domain = node->Domain();

  // Three types of checking:
  // 1. Check input and output data types are supported.
  // 2. Check op_type is implemented in migraphx
  // 3. Check the mode is implemented in migraphx
  // if 3. failed, call the constant folding capability in migraphx
  // to see whether some input parameters can be calculated statically
  // check data type
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    are_types_supported &= IsTypeSupported(&node_arg);
  });

  if (!are_types_supported) {
    return false;
  }

  // whether an operator implemented in migraphx
  if (op_set.count(optype) == 0) {
    return false;
  }

  // check that some modes might not be supported in migraphx for some operators
  if (domain == kOnnxDomain && IsUnsupportedOpMode(graph_viewer, node)) {
    // not supported, then check the constant folding capability of migraphx
    // to see whether it is supported
    return false;
  }

  return true;
}

std::unique_ptr<IndexedSubGraph> MIGraphXExecutionProvider::GetSubGraph(const std::vector<std::size_t>& graph_nodes_index, const GraphViewer& graph) const {
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.size());
  for (const auto& index : graph_nodes_index) {
    node_set.insert(index);
  }

  // Get parent graph output names
  std::vector<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.push_back(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index) {
    sub_graph->Nodes().push_back(index);
    const auto& node = graph.GetNode(index);
    for (const auto& input : node->InputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    for (const auto& input : node->ImplicitInputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there are two special cases,
    // One is, if node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    // The other one is, if subgraph's node output is parent graph's output. the node output should
    // be also added to the subgraph's output list
    if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto& node_idx = it->GetNode().Index();
        const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];
        if (node_set.find(node_idx) != node_set.end()) {
          const auto& iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            if (std::find(graph_output_names.begin(),
                          graph_output_names.end(), output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }
      }
    } else {
      for (const auto& output : node->OutputDefs()) {
        const auto& it = fused_inputs.find(output);
        if (it != fused_inputs.end()) {
          fused_inputs.erase(it);
          erased.insert(output);
        }
        // Only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end()) {
          if (std::find(graph_output_names.begin(),
                        graph_output_names.end(), output->Name()) != graph_output_names.end()) {
            graph_outputs_to_add[output] = output_order;
          }
          fused_outputs[output] = output_order++;
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // It is possible that an output of an node is put bebind the output of an later
  // node in the graph output list. So we should sort the output name according
  // to the graph output names
  std::vector<std::string> output_names;
  std::unordered_set<std::string> graph_out_names;
  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      auto name = output.second->Name();
      if (std::find(graph_output_names.begin(), graph_output_names.end(), name) == graph_output_names.end()) {
        output_names.push_back(name);
      } else {
        graph_out_names.insert(name);
      }
    }
  }

  for (auto& name : graph_output_names) {
    if (std::find(graph_out_names.begin(), graph_out_names.end(), name) != graph_out_names.end())
      output_names.push_back(name);
  }

  // Generate unique kernel name for MIGraphX subgraph
  uint64_t model_hash = 0;
  int id = metadef_id_generator_->GenerateId(graph, model_hash);
  std::string subgraph_id = std::to_string(model_hash) + "_" + std::to_string(id);
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name() = "MGXKernel_" + graph_type + "_" + graph.Name() + "_" + subgraph_id;

  // Assign inputs and outputs to subgraph's meta_def
  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& output : output_names) {
    meta_def->outputs().push_back(output);
  }

  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

static std::vector<NodeIndex>
GetUnsupportedNodeIndices(const GraphViewer& graph_viewer,
                          /*out*/ std::unordered_set<std::string>& mgx_required_initializers,
                          const logging::Logger& logger) {
  static std::set<std::string> mgx_supported_ops = {"Abs",
                                                    "Acos",
                                                    "Acosh",
                                                    "Add",
                                                    "And",
                                                    "ArgMax",
                                                    "ArgMin",
                                                    "Asin",
                                                    "Asinh",
                                                    "Atan",
                                                    "Atanh",
                                                    "ATen",
                                                    "AveragePool",
                                                    "BatchNormalization",
                                                    "BiasGelu",
                                                    "Cast",
                                                    "Ceil",
                                                    "Celu",
                                                    "Clip",
                                                    "Concat",
                                                    "Constant",
                                                    "ConstantFill",
                                                    "ConstantOfShape",
                                                    "Conv",
                                                    "ConvInteger",
                                                    "ConvTranspose",
                                                    "Cos",
                                                    "Cosh",
                                                    "CumSum",
                                                    "DepthToSpace",
                                                    "DequantizeLinear",
                                                    "Div",
                                                    "Dropout",
                                                    "Einsum",
                                                    "Elu",
                                                    "Equal",
                                                    "Erf",
                                                    "Exp",
                                                    "Expand",
                                                    "EyeLike",
                                                    "FastGelu",
                                                    "Flatten",
                                                    "Floor",
                                                    "GRU",
                                                    "Gather",
                                                    "GatherElements",
                                                    "GatherND",
                                                    "Gelu",
                                                    "Gemm",
                                                    "GlobalAveragePool",
                                                    "GlobalMaxPool",
                                                    "Greater",
                                                    "GreaterOrEqual",
                                                    "GroupNormalization",
                                                    "GroupQueryAttention",
                                                    "HardSigmoid",
                                                    "HardSwish",
                                                    "Identity",
                                                    "If",
                                                    "ImageScaler",
                                                    "InstanceNormalization",
                                                    "IsNan",
                                                    "LayerNormalization",
                                                    "LeakyRelu",
                                                    "Less",
                                                    "LessOrEqual",
                                                    "Log",
                                                    "LogSoftmax",
                                                    "Loop",
                                                    "LpNormalization",
                                                    "LRN",
                                                    "LSTM",
                                                    "MatMul",
                                                    "MatMulInteger",
                                                    "MatMulNBits",
                                                    "Max",
                                                    "MaxPool",
                                                    "Mean",
                                                    "Min",
                                                    "Mod",
                                                    "Mul",
                                                    "Multinomial",
                                                    "MultiHeadAttention",
                                                    "Neg",
                                                    "NegativeLogLikelihoodLoss",
                                                    "NonMaxSuppression",
                                                    "NonZero",
                                                    "Not",
                                                    "OneHot",
                                                    "Or",
                                                    "Pad",
                                                    "Pow",
                                                    "PRelu",
                                                    "QLinearAdd",
                                                    "QLinearConv",
                                                    "QLinearMatMul",
                                                    "QuantizeLinear",
                                                    "QuickGelu",
                                                    "DynamicQuantizeLinear",
                                                    "RandomNormal",
                                                    "RandomNormalLike",
                                                    "RandomUniform",
                                                    "RandomUniformLike",
                                                    "Range",
                                                    "Reciprocal",
                                                    "ReduceL1",
                                                    "ReduceL2",
                                                    "ReduceLogSum",
                                                    "ReduceLogSumExp",
                                                    "ReduceMax",
                                                    "ReduceMean",
                                                    "ReduceMin",
                                                    "ReduceProd",
                                                    "ReduceSum",
                                                    "ReduceSumSquare",
                                                    "Relu",
                                                    "Reshape",
                                                    "Resize",
                                                    "ReverseSequence",
                                                    "RNN",
                                                    "Roialign",
                                                    "Round",
                                                    "Scatter",
                                                    "ScatterElements",
                                                    "ScatterND",
                                                    "Selu",
                                                    "Shape",
                                                    "Sigmoid",
                                                    "Sign",
                                                    "SimplifiedLayerNormalization",
                                                    "Sin",
                                                    "Sinh",
                                                    "SkipLayerNormalization",
                                                    "SkipSimplifiedLayerNormalization",
                                                    "Slice",
                                                    "Softmax",
                                                    "SoftmaxCrossEntropyLoss",
                                                    "Softplus",
                                                    "Softsign",
                                                    "SpaceToDepth",
                                                    "Split",
                                                    "Sqrt",
                                                    "Squeeze",
                                                    "Sub",
                                                    "Sum",
                                                    "Tan",
                                                    "Tanh",
                                                    "ThresholdedRelu",
                                                    "Tile",
                                                    "TopK",
                                                    "Transpose",
                                                    "Trilu",
                                                    "Unsqueeze",
                                                    "Upsample",
                                                    "Where",
                                                    "Xor"};
  std::vector<NodeIndex> unsupported_nodes_idx;
  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(mgx_supported_ops, graph_viewer, node_idx, logger)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&mgx_required_initializers,
                                                  &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                mgx_required_initializers.insert(node_arg.Name());
              } },
                                                 true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

// Returns a vector clusters(or node_idx). For each unsupported node, the graph
// is split into 3 parts. supported_cluster + (UNsupported_node + rest_of_the_graph).
// This functions returns vector of all supported_subgraphx by amdmigraphx
static std::vector<std::vector<NodeIndex>>
GetPartitionedSubgraphs(const std::vector<NodeIndex>& topological_order,
                        const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> mgx_subgraphx;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx)
    // and append it to return list.
    std::vector<NodeIndex> this_subgraph{prev, it};
    if (!this_subgraph.empty()) {
      mgx_subgraphx.push_back(std::move(this_subgraph));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_subgraph{prev, topological_order.end()};
  if (!this_subgraph.empty()) {
    mgx_subgraphx.push_back(std::move(this_subgraph));
  }

  return mgx_subgraphx;
}

std::vector<std::unique_ptr<ComputeCapability>>
MIGraphXExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/,
                                         const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                         IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  auto model = graph_viewer.CreateModel(*GetLogger());
  auto model_proto = model->ToProto();
  graph_viewer.ToProto(*model_proto->mutable_graph(), true, true);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string onnx_string_buffer;
  model_proto->SerializeToString(onnx_string_buffer);
  model_path_ = graph_viewer.ModelPath();

  // dump onnx file if environment var is set
  if (dump_model_ops_) {
    std::string model_name = graph_viewer.Name() + ".onnx";
    std::ofstream ofs(model_name);
    ofs.write(onnx_string_buffer.c_str(), onnx_string_buffer.size());
    ofs.close();
  }

  // This is a list of initializers that migraphx considers as constants.
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> mgx_required_initializers;
  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, mgx_required_initializers, *GetLogger());

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
    auto sub_graph = GetSubGraph(node_indices, graph_viewer);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  } else {  // unsupported_nodes_idx.empty()
    if (dump_model_ops_) {
      LOGS_DEFAULT(INFO) << "============= Unsupported nodes ====================";
      for (auto idx : unsupported_nodes) {
        LOGS_DEFAULT(INFO) << graph_viewer.GetNode(idx)->OpType();
      }
      LOGS_DEFAULT(INFO) << "************* Unsupported nodes ********************";
    }

    if (unsupported_nodes.size() > 10) {
      return result;
    }

    auto mgx_clusters = GetPartitionedSubgraphs(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    // check whether a subgrap should fallback to CPU
    SubgraphPostProcessing(graph_viewer, mgx_clusters, *GetLogger());

    for (const auto& this_cluster : mgx_clusters) {
      auto sub_graph = GetSubGraph(this_cluster, graph_viewer);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    }
  }

  return result;
}

bool get_input_output_names(const GraphViewer& graph,
                            std::vector<std::string>& input_names,
                            std::vector<std::string>& output_names) {
  input_names.clear();
  output_names.clear();
  const auto& input_args = graph.GetInputs();
  std::transform(input_args.begin(), input_args.end(), std::back_inserter(input_names), [](auto& arg) {
    return arg->Name();
  });

  bool no_input_shape = std::any_of(input_args.begin(), input_args.end(), [&](auto arg) {
    if (arg == nullptr)
      return true;

    auto sptr = arg->Shape();
    if (sptr == nullptr)
      return true;

    if (sptr->dim_size() == 0)
      return true;

    for (int i = 0; i < sptr->dim_size(); i++) {
      if (sptr->dim(i).has_dim_param())
        return true;
    }

    return false;
  });

  const auto& out_args = graph.GetOutputs();
  std::vector<std::string> tmp_out_names;
  std::transform(out_args.begin(),
                 out_args.end(),
                 std::back_inserter(tmp_out_names),
                 [](auto& arg) { return arg->Name(); });

  std::copy_if(
      tmp_out_names.begin(),
      tmp_out_names.end(),
      std::back_inserter(output_names),
      [&](const auto& name) { return !name.empty(); });

  return no_input_shape;
}

// Attempt to load a model and catch any exceptions on load fail.
// Useful to default to EP to trigger the compile if file doesn't exist or loading fails.
bool load_precompiled_model(migraphx::program& prog, bool load_enable, std::string path) {
  try {
    if (load_enable) {
      LOGS_DEFAULT(WARNING) << "Attempting to load model at:" << path;
      prog = migraphx::load(path.c_str());
      LOGS_DEFAULT(WARNING) << "load model : Success";
      return true;
    } else {
      return false;
    }
  } catch (...) {
    return false;
  }
  return false;
}

void save_compiled_model(migraphx::program& prog, bool save_enable, std::string out_path) {
  if (save_enable) {
    LOGS_DEFAULT(WARNING) << "Model Save at " << out_path << ": Begin";
    migraphx::file_options fo;
    fo.set_file_format("msgpack");
    migraphx::save(prog, out_path.c_str(), fo);
    LOGS_DEFAULT(WARNING) << "Model Save: Complete";
  }
}

// Order matters here especially if the program uses mixed quantization
// Calibrate on full precision for int8/fp8 and then quantize down to fp16
void calibrate_and_quantize(migraphx::program& prog,
                            const migraphx::target& t,
                            const migraphx::program_parameters quant_params,
                            bool fp16_enable,
                            bool int8_enable,
                            bool fp8_enable,
                            bool int8_calibration_cache_available,
                            std::unordered_map<std::string, float>& dynamic_range_map) {
  // Read in the calibration data and map it to an migraphx paramater map for the calibration ops
  if ((int8_enable xor fp8_enable) && int8_calibration_cache_available) {
    LOGS_DEFAULT(WARNING) << "Quantizing input program";

    auto param_shapes = prog.get_parameter_shapes();

    // Add all calibration data read in from int8 table
    for (auto& [cal_key, cal_val] : dynamic_range_map) {
      auto cal_val_shape = migraphx::shape(migraphx_shape_float_type);
      quant_params.add(cal_key.c_str(), migraphx::argument(cal_val_shape, static_cast<void*>(std::move(&cal_val))));
    }

    // perform static quantization on the programs
    if (int8_enable) {
      LOGS_DEFAULT(WARNING) << "Quantizing input program to int8";
      migraphx::quantize_int8_options quant_opts;
      quant_opts.add_calibration_data(quant_params);
      // specify thing we want to int8 quantize
      quant_opts.add_op_name("convolution");
      quant_opts.add_op_name("dot");
      migraphx::quantize_int8(prog, t, quant_opts);
      LOGS_DEFAULT(WARNING) << "Quantizing int8: Complete";
    } else if (fp8_enable) {
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4)
      LOGS_DEFAULT(WARNING) << "Quantizing input program to fp8";
      migraphx::quantize_fp8_options quant_opts;
      quant_opts.add_calibration_data(quant_params);
      migraphx::quantize_fp8(prog, t, quant_opts);
      LOGS_DEFAULT(WARNING) << "Quantizing fp8: Complete";
#endif
    }
  }

  if (fp16_enable) {
    LOGS_DEFAULT(WARNING) << "Quantizing input program to fp16";
    migraphx::quantize_fp16(prog);
    LOGS_DEFAULT(WARNING) << "Quantizing fp16: Complete";
  }
}

void compile_program(migraphx::program& prog,
                     const migraphx::target& t,
                     bool exhaustive_tune) {
  LOGS_DEFAULT(WARNING) << "Model Compile: Begin";
  migraphx::compile_options co;
  co.set_fast_math(false);
  co.set_exhaustive_tune_flag(exhaustive_tune);
  prog.compile(t, co);
  LOGS_DEFAULT(WARNING) << "Model Compile: Complete";
}

Status MIGraphXExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                          std::vector<NodeComputeInfo>& node_compute_funcs) {
  migraphx::onnx_options options;
  bool no_input_shape = false;
  for (const auto& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;
    // map parameter input name to index
    std::unordered_map<std::string, std::size_t> input_name_index;
    const auto& input_defs = fused_node.InputDefs();
    input_name_index.reserve(input_defs.size());
    for (std::size_t i = 0; i < input_defs.size(); ++i) {
      input_name_index[input_defs[i]->Name()] = i;
    }

    auto model = graph_body_viewer.CreateModel(*GetLogger());
    auto model_proto = model->ToProto();
    graph_body_viewer.ToProto(*model_proto->mutable_graph(), true, true);
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string onnx_string_buffer;
    model_proto->SerializeToString(onnx_string_buffer);

    if (dump_model_ops_) {
      std::string onnx_name = fused_node.Name() + ".onnx";
      std::ofstream ofs(onnx_name);
      ofs.write(onnx_string_buffer.data(), onnx_string_buffer.size());
      ofs.close();
    }

    std::vector<std::string> input_names, output_names;
    no_input_shape = no_input_shape || get_input_output_names(graph_body_viewer, input_names, output_names);

    // by parsing the model_proto, create a program corresponding to
    // the input fused_node
    migraphx::program prog;

    if (!no_input_shape) {
      if (!load_precompiled_model(prog, load_compiled_model_, std::string{load_compiled_path_})) {
        LOGS_DEFAULT(INFO) << "No input shapes detected quantizing model";
        prog = migraphx::parse_onnx_buffer(onnx_string_buffer, options);
        migraphx::program_parameters quant_params;

        calibrate_and_quantize(prog, t_, quant_params, fp16_enable_, int8_enable_,
                               fp8_enable_, int8_calibration_cache_available_, dynamic_range_map_);
        compile_program(prog, t_, exhaustive_tune_);
        save_compiled_model(prog, save_compiled_model_, save_compiled_path_);
      }

      auto prog_output_shapes = prog.get_output_shapes();
      for (std::size_t i = 0; i < output_names.size(); ++i) {
        auto out_len = prog_output_shapes[i].lengths();
        options.set_input_parameter_shape(output_names[i], out_len);
      }
    }

    // compile the program
    map_progs_[fused_node.Name()] = prog;

    map_onnx_string_[fused_node.Name()] = onnx_string_buffer;
    map_input_index_[fused_node.Name()] = input_name_index;
    map_no_input_shape_[fused_node.Name()] = no_input_shape;
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<MIGraphXFuncState> p = std::make_unique<MIGraphXFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, map_progs_[context->node_name],
            map_onnx_string_[context->node_name], options, t_, map_input_index_[context->node_name], &mgx_mu_,
            map_no_input_shape_[context->node_name], fp16_enable_, fp8_enable_, int8_enable_,
            int8_calibration_cache_available_, dynamic_range_map_,
            save_compiled_model_, save_compiled_path_,
            load_compiled_model_, load_compiled_path_, dump_model_ops_};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<MIGraphXFuncState*>(state);
    };

    compute_info.compute_func = [this](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      MIGraphXFuncState* mgx_state = reinterpret_cast<MIGraphXFuncState*>(state);

      std::unordered_map<std::string, std::size_t>& map_input_name_index = mgx_state->input_name_indexes;
      std::unordered_map<std::string, float>& map_dynamic_range = mgx_state->dynamic_range_map;
      migraphx::target t = mgx_state->t;
      migraphx::program& prog = mgx_state->prog;
      std::string& onnx_string = mgx_state->onnx_string;
      migraphx::onnx_options& cmp_options = mgx_state->options;
      bool& no_input_shape = mgx_state->no_input_shape;
      bool fp16_enable = mgx_state->fp16_enable;
      bool fp8_enable = mgx_state->fp8_enable;
      bool int8_enable = mgx_state->int8_enable;
      bool int8_calibration_cache_available = mgx_state->int8_calibration_cache_available;

      // mean no program at all, so need to get the input shape info
      // from input data
      bool input_shape_match = true;
      migraphx::program_parameter_shapes param_shapes;
      if (no_input_shape) {
        LOGS_DEFAULT(INFO) << "Missing input shape setting input parameters again";
        for (auto& it : map_input_name_index) {
          auto& name = it.first;
          auto& index = it.second;
          auto input_tensor = ctx.GetInput(index);
          auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
          const auto tensor_shape = tensor_info.GetShape();
          std::vector<std::size_t> ort_lens(tensor_shape.begin(), tensor_shape.end());
          cmp_options.set_input_parameter_shape(name, ort_lens);
          input_shape_match = false;
        }
      } else {
        LOGS_DEFAULT(INFO) << "Assigning inputs, and parameters from compiled model";
        param_shapes = prog.get_parameter_shapes();
        auto prog_output_shapes = prog.get_output_shapes();

        // check whether input shapes match with shapes of program inputs
        // migraphx::onnx_options cmp_options;
        if (param_shapes.size() > 0) {
          for (auto&& name : param_shapes.names()) {
            if (map_input_name_index.count(name) > 0) {
              auto input_tensor = ctx.GetInput(map_input_name_index[name]);
              auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
              const auto tensor_shape = tensor_info.GetShape();
              std::vector<std::size_t> ort_lens(tensor_shape.begin(), tensor_shape.end());

              auto mgx_s = param_shapes[name];
              auto mgx_lens = mgx_s.lengths();
              auto mgx_strides = mgx_s.strides();
              if (mgx_lens.size() == 1 and mgx_lens[0] == 1 and
                  mgx_strides.size() == 1 and mgx_strides[0] == 0) {
                mgx_lens.clear();
              }

              if (mgx_lens != ort_lens) {
                cmp_options.set_input_parameter_shape(name, ort_lens);
                input_shape_match = false;
              }
            }
          }
        }
      }

      // input shapes are different, needs to re-parse onnx and
      // re-compile the program
      if (!input_shape_match) {
        if (!load_precompiled_model(prog, load_compiled_model_, std::string{load_compiled_path_})) {
          LOGS_DEFAULT(VERBOSE) << "Input shape mismatch detected. Recompiling" << std::endl;
#ifndef ENABLE_TRAINING_CORE
#if HIP_VERSION_MAJOR > 6 || (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 2)
          cmp_options.set_external_data_path(model_path_.has_parent_path() ? model_path_.parent_path().string() : std::filesystem::current_path().string());
#endif
#endif
          prog = migraphx::parse_onnx_buffer(onnx_string, cmp_options);
          migraphx::program_parameters quant_params;

          if ((int8_enable xor fp8_enable) and int8_calibration_cache_available) {
            auto param_shapes = prog.get_parameter_shapes();
            // Add input parameter data and the values they're set to
            for (auto&& name : param_shapes.names()) {
              if (map_input_name_index.count(name) > 0) {
                auto input_tensor = ctx.GetInput(map_input_name_index[name]);
                auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
                const auto tensor_shape = tensor_info.GetShape();
                const auto tensor_type = tensor_info.GetElementType();

                migraphx_shape_datatype_t mgx_type;
                getMIGraphXType(tensor_type, mgx_type);
                auto mgx_s = param_shapes[name];

                if (mgx_type != mgx_s.type()) {
                  LOGS_DEFAULT(FATAL) << "MIGraphX: param type mismatch";
                }
                quant_params.add(name, migraphx::argument(param_shapes[name], const_cast<void*>(input_tensor.GetTensorRawData())));
              }
            }
          }
          calibrate_and_quantize(prog, t, quant_params, fp16_enable, int8_enable,
                                 fp8_enable, int8_calibration_cache_available, map_dynamic_range);
          compile_program(prog, t, exhaustive_tune_);
          save_compiled_model(prog, mgx_state->save_compiled_mode, mgx_state->save_compiled_path);
        }

        mgx_state->prog = prog;
        param_shapes = prog.get_parameter_shapes();
        no_input_shape = false;
      }

      migraphx::program_parameters m;
      auto prog_output_shapes = prog.get_output_shapes();
      std::vector<std::size_t> prog_output_indices;
      if (param_shapes.size() > 0) {
        for (auto&& name : param_shapes.names()) {
          if (map_input_name_index.count(name) > 0) {
            LOGS_DEFAULT(INFO) << "Setting parameters for:" << name;
            auto input_tensor = ctx.GetInput(map_input_name_index[name]);
            auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
            const auto tensor_shape = tensor_info.GetShape();
            const auto tensor_type = tensor_info.GetElementType();

            migraphx_shape_datatype_t mgx_type;
            getMIGraphXType(tensor_type, mgx_type);
            auto mgx_s = param_shapes[name];

            if (mgx_type != mgx_s.type()) {
              LOGS_DEFAULT(FATAL) << "MIGraphX: param type mismatch";
            }

            LOGS_DEFAULT(INFO) << "Writing Raw tensor data ";
            m.add(name, migraphx::argument(param_shapes[name],
                                           const_cast<void*>(input_tensor.GetTensorRawData())));
          }
          // It is a output argument
          else {
            auto compute_output_index = [](const std::string& name) -> int {
              std::string out_name_prefix = "#output_";
              auto pos = name.find(out_name_prefix);
              if (pos == std::string::npos) {
                return -1;
              }

              std::string index_str = name.substr(pos + out_name_prefix.length());
              return std::stoi(index_str);
            };

            int output_index = compute_output_index(name);
            if (output_index != -1) {
              prog_output_indices.push_back(output_index);
              auto mgx_output_shape = prog_output_shapes[output_index];
              auto lens = mgx_output_shape.lengths();
              std::vector<int64_t> ort_output_shape(lens.begin(), lens.end());
              auto output_tensor = ctx.GetOutput(output_index, ort_output_shape.data(), ort_output_shape.size());
              void* output_data = output_tensor.GetTensorMutableRawData();

              // argument shape
              auto mgx_arg_shape = param_shapes[name];
              m.add(name, migraphx::argument(mgx_arg_shape, output_data));
            }
          }
        }
      }

      {
        // lock to avoid race condition
        std::lock_guard<std::mutex> lock(*(mgx_state->mgx_mu_ptr));

        void* rocm_stream;
        Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &rocm_stream));
        auto prog_outputs = prog.run_async(m, static_cast<hipStream_t>(rocm_stream));

        // In case of input parameters are reused as output parameter call hipMemcpy
        auto output_num = prog_outputs.size();
        if (prog_output_indices.size() < output_num) {
          for (std::size_t i = 0; i < output_num; ++i) {
            if (std::find(prog_output_indices.begin(), prog_output_indices.end(), i) != prog_output_indices.end())
              continue;
            auto gpu_res = prog_outputs[i];
            migraphx::shape res_shape = gpu_res.get_shape();
            auto res_lens = res_shape.lengths();
            std::vector<int64_t> ort_shape{res_lens.begin(), res_lens.end()};
            auto output_tensor = ctx.GetOutput(i, ort_shape.data(), ort_shape.size());
            void* output_data = output_tensor.GetTensorMutableRawData();
            HIP_CALL_THROW(hipMemcpyWithStream(output_data,
                                               gpu_res.data(),
                                               res_shape.bytes(),
                                               hipMemcpyDeviceToDevice,
                                               static_cast<hipStream_t>(rocm_stream)));
          }
        }
      };

      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

void MIGraphXExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry,
                                                       AllocatorMap& allocators) const {
  auto allocator = allocators[GetOrtDeviceByMemType(OrtMemTypeCPU)];
  RegisterMIGraphXStreamHandles(stream_handle_registry, OrtDevice::GPU, allocator, true, stream_, false /*TODO:external_stream_*/);
}

OrtDevice MIGraphXExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeCPUInput)
    return OrtDevice();
  if (mem_type == OrtMemTypeCPUOutput)
    return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::AMD,
                     default_device_.Id());
  return default_device_;
}

Status MIGraphXExecutionProvider::Sync() const {
  HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(nullptr)));

  auto status = hipStreamQuery(stream_);
  if (status != hipSuccess) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::EP_FAIL);
  }
  return Status::OK();
}

Status MIGraphXExecutionProvider::OnRunStart(const onnxruntime::RunOptions& /*run_options*/) {
  return Status::OK();
}

Status MIGraphXExecutionProvider::OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& /*run_options*/) {
  auto status = hipStreamQuery(stream_);

  if (status != hipSuccess) {
    HIP_CALL_THROW(hipStreamSynchronize(stream_));
  }
  return Status::OK();
}

}  // namespace onnxruntime
