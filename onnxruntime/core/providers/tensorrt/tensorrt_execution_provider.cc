// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <fstream>
#include <list>
#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "tensorrt_execution_provider.h"
#include "tensorrt_execution_provider_utils.h"
#include "tensorrt_execution_provider_custom_ops.h"
#include "onnx_ctx_model_helper.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/session/allocator_adapters.h"
#include "cuda_runtime_api.h"
#include <gsl/gsl>
#include <unordered_map>
#include <utility>
#include <limits>
#include <map>
#include <memory>
#include <filesystem>
// TODO: find a better way to share this
#include "core/providers/cuda/cuda_stream_handle.h"

#ifdef _WIN32
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) LoadLibrary(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#else
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#endif

#define CUDA_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(CUDA_CALL(expr))

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
namespace {
// Check if cycle exists in the graph after partitioning
bool FindCycleHelper(size_t i, gsl::span<const InlinedVector<size_t>> adjacency_map, gsl::span<bool> visited, gsl::span<bool> st,
                     InlinedVector<size_t>& cycles) {
  if (!visited[i]) {
    visited[i] = true;
    st[i] = true;
    for (auto iter = adjacency_map[i].begin(); iter != adjacency_map[i].end(); ++iter) {
      if (!visited[*iter] && FindCycleHelper(*iter, adjacency_map, visited, st, cycles)) {
        cycles.push_back(*iter);
        return true;
      } else if (st[*iter]) {
        cycles.push_back(*iter);
        return true;
      }
    }
  }
  st[i] = false;
  return false;
}

bool SetDynamicRange(nvinfer1::INetworkDefinition& network, std::unordered_map<std::string, float>& dynamic_range_map) {
  // Set dynamic range for input tensors
  for (int i = 0; i < network.getNbInputs(); ++i) {
    const std::string tensor_name = network.getInput(i)->getName();
    auto dynamic_range_iter = dynamic_range_map.find(tensor_name);
    if (dynamic_range_iter != dynamic_range_map.end()) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      if (!network.getInput(i)->setDynamicRange(-dynamic_range_iter->second, dynamic_range_iter->second)) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        LOGS_DEFAULT(ERROR) << "Failed to set dynamic range for network input " << tensor_name;
        return false;
      }
    }
  }

  // Set dynamic range for activations and weights
  for (int i = 0; i < network.getNbLayers(); ++i) {
    auto trt_layer = network.getLayer(i);
    for (int j = 0, e = trt_layer->getNbOutputs(); j < e; ++j) {
      const std::string tensor_name = trt_layer->getOutput(j)->getName();
      auto dynamic_range_iter = dynamic_range_map.find(tensor_name);
      if (dynamic_range_iter != dynamic_range_map.end()) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        if (!trt_layer->getOutput(j)->setDynamicRange(-dynamic_range_iter->second, dynamic_range_iter->second)) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
          LOGS_DEFAULT(ERROR) << "Failed to set dynamic range for tensor " << tensor_name;
          return false;
        }
      } else if (trt_layer->getType() == nvinfer1::LayerType::kCONSTANT) {
        nvinfer1::IConstantLayer* const_layer = static_cast<nvinfer1::IConstantLayer*>(trt_layer);
        const std::string const_layer_name = const_layer->getName();
        auto trt_weights = const_layer->getWeights();
        double max_weight = std::numeric_limits<double>::min();
        for (int64_t k = 0, end = trt_weights.count; k < end; ++k) {
          double weight{};
          switch (trt_weights.type) {
            case nvinfer1::DataType::kFLOAT:
              weight = static_cast<const float*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kBOOL:
              weight = static_cast<const bool*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kINT8:
              weight = static_cast<const int8_t*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kHALF:
              weight = static_cast<const uint16_t*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kINT32:
              weight = static_cast<const int32_t*>(trt_weights.values)[k];
              break;
#if NV_TENSORRT_MAJOR >= 10
            case nvinfer1::DataType::kINT64:
              weight = static_cast<double>(static_cast<const int64_t*>(trt_weights.values)[k]);
              break;
#endif  // NV_TENSORRT_MAJOR >= 10
            default:
              LOGS_DEFAULT(ERROR) << "Found unsupported datatype for layer " << const_layer_name;
              return false;
          }
          max_weight = std::max(max_weight, std::abs(weight));
        }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        if (!trt_layer->getOutput(j)->setDynamicRange(static_cast<float>(-max_weight), static_cast<float>(max_weight))) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
          LOGS_DEFAULT(ERROR) << "Failed to set dynamic range for layer " << const_layer_name;
          return false;
        }
      }
    }
  }
  return true;
}

std::vector<std::string> SplitToStringVec(std::string const& s, char separator) {
  std::vector<std::string> splitted;

  for (size_t start = 0; start < s.length();) {
    size_t separatorIndex = s.find(separator, start);
    if (separatorIndex == std::string::npos) {
      separatorIndex = s.length();
    }
    splitted.emplace_back(s.substr(start, separatorIndex - start));
    start = separatorIndex + 1;
  }

  return splitted;
}

nvinfer1::TacticSources GetTacticSourceFromString(std::string& tactic_string) {
  nvinfer1::TacticSources disabledTactics = 0;
  nvinfer1::TacticSources enabledTactics = 0;
  std::vector<std::string> tacticList = SplitToStringVec(tactic_string, ',');
  for (auto& t : tacticList) {
    bool enable{false};
    if (t.front() == '+') {
      enable = true;
    } else if (t.front() != '-') {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Tactic source must be prefixed with + or - skipping: " << t;
    }
    t.erase(0, 1);

    const auto toUpper = [](std::string& sourceName) {
      std::transform(sourceName.begin(), sourceName.end(), sourceName.begin(),
                     [](char c) { return onnxruntime::narrow<char>(std::toupper(c)); });
      return sourceName;
    };

    nvinfer1::TacticSource source{};
    t = toUpper(t);
    if (t == "CUBLAS") {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Tactic kCUBLAS is deprecated in TensorRT 10.0";
#if NV_TENSORRT_MAJOR < 10
      source = nvinfer1::TacticSource::kCUBLAS;
#endif
    } else if (t == "CUBLASLT" || t == "CUBLAS_LT") {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Tactic kCUBLAS_LT is deprecated in TensorRT 9.0";
#if NV_TENSORRT_MAJOR < 9
      source = nvinfer1::TacticSource::kCUBLAS_LT;
#endif
    } else if (t == "CUDNN") {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Tactic kCUDNN is deprecated in TensorRT 10.0";
#if NV_TENSORRT_MAJOR < 10
      source = nvinfer1::TacticSource::kCUDNN;
#endif
    } else if (t == "EDGE_MASK_CONVOLUTIONS") {
      source = nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS;
    } else if (t == "JIT_CONVOLUTIONS") {
      source = nvinfer1::TacticSource::kJIT_CONVOLUTIONS;
    } else {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Tactic source was not found with name: " << t;
    }

    uint32_t sourceBit = 1U << static_cast<uint32_t>(source);

    if (enable) {
      enabledTactics |= sourceBit;
    } else {
      disabledTactics |= sourceBit;
    }
  }
  return enabledTactics & ~disabledTactics;
}

inline std::vector<char> loadTimingCacheFile(const std::string inFileName) {
  std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
  if (!iFile) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Could not read timing cache from: " << inFileName
                          << ". A new timing cache will be generated and written.";
    return std::vector<char>();
  }
  iFile.seekg(0, std::ifstream::end);
  size_t fsize = iFile.tellg();
  iFile.seekg(0, std::ifstream::beg);
  std::vector<char> content(fsize);
  iFile.read(content.data(), fsize);
  iFile.close();
  return content;
}

inline void saveTimingCacheFile(const std::string outFileName, const nvinfer1::IHostMemory* blob) {
  std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
  if (!oFile) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Could not write timing cache to: " << outFileName;
    return;
  }
  oFile.write((char*)blob->data(), blob->size());
  oFile.close();
}
}  // namespace

namespace google {
namespace protobuf {
void ShutdownProtobufLibrary();
}
}  // namespace google

struct ShutdownProtobuf {
  ~ShutdownProtobuf() {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
} g_protobuf;

namespace onnxruntime {
namespace cuda {
template <>
void Impl_Cast(
    cudaStream_t stream,
    const int64_t* input_data, int32_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}

template <>
void Impl_Cast(
    cudaStream_t stream,
    const int32_t* input_data, int64_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}

template <>
void Impl_Cast(
    cudaStream_t stream,
    const double* input_data, float* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}

template <>
void Impl_Cast(
    cudaStream_t stream,
    const float* input_data, double* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}
}  // namespace cuda

template <>
Status CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_false(retCode, exprString, libName, successCode, msg, file, line);
}

template <>
void CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_true(retCode, exprString, libName, successCode, msg, file, line);
}

#ifndef USE_CUDA_MINIMAL
template <>
Status CudaCall<cublasStatus_t, false>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_false(retCode, exprString, libName, successCode, msg, file, line);
}

template <>
void CudaCall<cublasStatus_t, true>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_true(retCode, exprString, libName, successCode, msg, file, line);
}

template <>
Status CudaCall<cudnnStatus_t, false>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_false(retCode, exprString, libName, successCode, msg, file, line);
}

template <>
void CudaCall<cudnnStatus_t, true>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line) {
  return g_host->CudaCall_true(retCode, exprString, libName, successCode, msg, file, line);
}
#endif

#if NV_TENSORRT_MAJOR >= 10
void* OutputAllocator::reallocateOutputAsync(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                             uint64_t /*alignment*/, cudaStream_t /*stream*/) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size) {
    cudaFree(outputPtr);
    outputPtr = nullptr;
    allocated_size = 0;
    if (cudaMalloc(&outputPtr, size) == cudaSuccess) {
      allocated_size = size;
    }
  }
  // if cudaMalloc fails, returns nullptr.
  return outputPtr;
}
#else
// Only override this method when TensorRT <= 8.6
void* OutputAllocator::reallocateOutput(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                        uint64_t /*alignment*/) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size) {
    cudaFree(outputPtr);
    outputPtr = nullptr;
    allocated_size = 0;
    if (cudaMalloc(&outputPtr, size) == cudaSuccess) {
      allocated_size = size;
    }
  }
  // if cudaMalloc fails, returns nullptr.
  return outputPtr;
}
#endif

void OutputAllocator::notifyShape(char const* /*tensorName*/, nvinfer1::Dims const& dims) noexcept {
  output_shapes.clear();
  output_shapes.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) {
    output_shapes.push_back(dims.d[i]);
  }
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
    if (!gpu_data_transfer)
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "gpu data transfer is missing in TRT EP.");
    if (!ctx->GetComputeStream())
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "Compute Stream is missing in TRT MemCpy kernel's context.");
    return gpu_data_transfer->CopyTensorAsync(*X, *Y, *(ctx->GetComputeStream()));
  }
};

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_THROW_IF_ERROR(s_kernel_registry->Register(function_table_entry()));
  }
}

void DeleteRegistry() {
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> TensorrtExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger(bool verbose_log) {
  const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
  static TensorrtLogger trt_logger(log_level);
  if (log_level != trt_logger.get_level()) {
    trt_logger.set_level(verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
  }
  return trt_logger;
}

std::unique_lock<std::mutex> TensorrtExecutionProvider::GetApiLock() const {
  static std::mutex singleton;
  return std::unique_lock<std::mutex>(singleton);
}

/*
 * Get the shape of "shape tensor" input
 */
template <typename T>
Status GetShapeOfShapeTensor(Ort::ConstValue& input_tensor,
                             void* shape_values,
                             int shape_size,
                             cudaStream_t stream) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shape_values,
                                       input_tensor.GetTensorData<T>(),
                                       shape_size * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  return Status::OK();
}

/*
 * Apply TensorRT optimization profile shapes from provider options.
 *
 * This function supports single/multiple profile(s).
 * (Note: An optimization profile describes a range of dimensions for each network input)
 *
 */
bool ApplyProfileShapesFromProviderOptions(std::vector<nvinfer1::IOptimizationProfile*>& trt_profiles,
                                           nvinfer1::ITensor* input,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes,
                                           ShapeRangesMap& input_explicit_shape_ranges) {
  if (trt_profiles.size() == 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Number of optimization profiles should be greater than 0, but it's 0.";
    return false;
  }

  const std::string& input_name = input->getName();
  if (profile_min_shapes.find(input_name) == profile_min_shapes.end()) {
    return false;
  }

  if (input_explicit_shape_ranges.find(input_name) == input_explicit_shape_ranges.end()) {
    std::unordered_map<size_t, std::vector<std::vector<int64_t>>> inner_map;
    input_explicit_shape_ranges[input_name] = inner_map;
  }

  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Begin to apply profile shapes ...";
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Input tensor name is '" << input_name << "', number of profiles found is " << trt_profiles.size();

  for (size_t i = 0; i < trt_profiles.size(); i++) {
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    auto trt_profile = trt_profiles[i];

    // Shape tensor
    if (input->isShapeTensor()) {
      int shape_size = nb_dims == 0 ? 1 : static_cast<int>(profile_min_shapes[input_name][i].size());
      std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);

      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] shape size of this shape tensor is " << shape_size;

      for (int j = 0; j < shape_size; j++) {
        auto min_value = profile_min_shapes[input_name][i][j];
        auto max_value = profile_max_shapes[input_name][i][j];
        auto opt_value = profile_opt_shapes[input_name][i][j];
        shapes_min[j] = static_cast<int32_t>(min_value);
        shapes_max[j] = static_cast<int32_t>(max_value);
        shapes_opt[j] = static_cast<int32_t>(opt_value);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] shapes_min.d[" << j << "] is " << shapes_min[j];
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] shapes_max.d[" << j << "] is " << shapes_max[j];
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] shapes_opt.d[" << j << "] is " << shapes_opt[j];

        if (input_explicit_shape_ranges[input_name].find(j) == input_explicit_shape_ranges[input_name].end()) {
          std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
          input_explicit_shape_ranges[input_name][j] = profile_vector;
        }
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
      }

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 10) || NV_TENSORRT_MAJOR > 10
      std::vector<int64_t> shapes_min_64(shapes_min.begin(), shapes_min.end());
      std::vector<int64_t> shapes_opt_64(shapes_opt.begin(), shapes_opt.end());
      std::vector<int64_t> shapes_max_64(shapes_max.begin(), shapes_max.end());
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min_64[0], shape_size);
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_opt_64[0], shape_size);
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_max_64[0], shape_size);
#else
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
#endif
    }
    // Execution tensor
    else {
      nvinfer1::Dims dims_min, dims_opt, dims_max;
      dims_min.nbDims = nb_dims;
      dims_max.nbDims = nb_dims;
      dims_opt.nbDims = nb_dims;

      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] number of dimension of this execution tensor is " << nb_dims;

      for (int j = 0; j < nb_dims; j++) {
        if (dims.d[j] == -1) {
          auto min_value = profile_min_shapes[input_name][i][j];
          auto max_value = profile_max_shapes[input_name][i][j];
          auto opt_value = profile_opt_shapes[input_name][i][j];
          dims_min.d[j] = static_cast<int32_t>(min_value);
          dims_max.d[j] = static_cast<int32_t>(max_value);
          dims_opt.d[j] = static_cast<int32_t>(opt_value);
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] dims_min.d[" << j << "] is " << dims_min.d[j];
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] dims_max.d[" << j << "] is " << dims_max.d[j];
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] dims_opt.d[" << j << "] is " << dims_opt.d[j];

          if (input_explicit_shape_ranges[input_name].find(j) == input_explicit_shape_ranges[input_name].end()) {
            std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
            input_explicit_shape_ranges[input_name][j] = profile_vector;
          }
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
        } else {
          dims_min.d[j] = dims.d[j];
          dims_max.d[j] = dims.d[j];
          dims_opt.d[j] = dims.d[j];
        }
      }

      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
    }
  }
  return true;
}

/*
 * Apply TensorRT optimization profile shapes from input tensor value.
 *
 * This function supports single/multiple profile(s).
 * (Note: An optimization profile describes a range of dimensions for each network input)
 *
 * @param shape_tensor_values holds "shape tensor -> shape values" for the INT32 shape tensor input across this inference run
 * @param shape_tensor_values_int64 holds "shape tensor -> shape values" for the INT64 shape tensor input across this inference run
 */
Status ApplyProfileShapesFromInputTensorValue(std::vector<nvinfer1::IOptimizationProfile*>& trt_profiles,
                                              Ort::KernelContext ctx,
                                              nvinfer1::ITensor* input,
                                              ShapeRangesMap& shape_ranges,
                                              const std::unordered_map<std::string, size_t>& input_indexes,
                                              std::unordered_map<std::string, std::vector<int32_t>>& shape_tensor_values,
                                              std::unordered_map<std::string, std::vector<int64_t>>& shape_tensor_values_int64,
                                              cudaStream_t stream,
                                              bool* engine_update) {
  for (size_t i = 0; i < trt_profiles.size(); i++) {
    const std::string& input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    size_t input_index = 0;
    const auto& iter = input_indexes.find(input_name);
    if (iter != input_indexes.end()) {
      input_index = iter->second;
    }

    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();
    auto& shape_ranges_per_input = shape_ranges[input_name];

    auto trt_profile = trt_profiles[i];

    // If there are multiple profiles, for second and rest of profiles, simply copy the min/max/opt profile values from the first profile.
    // Following "if statement" won't be executed since TRT EP currently only allows single profile for non-explicit profiles case.
    if (i > 0) {
      if (input->isShapeTensor()) {
        // shape tensor
        int shape_size = nb_dims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 10) || NV_TENSORRT_MAJOR > 10
        std::vector<int64_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
        for (int j = 0; j < shape_size; j++) {
          shapes_min[j] = *(trt_profiles[0]->getShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN));
          shapes_max[j] = *(trt_profiles[0]->getShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX));
          shapes_opt[j] = *(trt_profiles[0]->getShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT));
        }
        trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
        trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
        trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
#else
        std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
        for (int j = 0; j < shape_size; j++) {
          shapes_min[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN));
          shapes_max[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX));
          shapes_opt[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT));
        }
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
#endif
      } else {
        // execution tensor
        nvinfer1::Dims dims_min, dims_opt, dims_max;
        dims_min = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN);
        dims_max = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX);
        dims_opt = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
      }
      continue;
    }

    // Create shape profile
    if (input->isShapeTensor()) {
      // Get shape values for shape tensor input
      const auto tensor_type = tensor_info.GetElementType();
      // The shape of the "shape tensor" is either zero dimension (scalar) or 1-dimension
      int shape_size = dims.nbDims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
      // For setting TRT optimization profile. (Note: the min/opt/max profile values are still int32 even though int64 is supported after TRT 10)
      std::vector<int32_t> values(shape_size);

      switch (tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          auto buffer = std::make_unique<int32_t[]>(shape_size);
          auto status = GetShapeOfShapeTensor<int32_t>(input_tensor, buffer.get(), shape_size, stream);
          if (status != Status::OK()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
          }
          shape_tensor_values[input_name].resize(shape_size);
          for (int j = 0; j < shape_size; ++j) {
            shape_tensor_values[input_name][j] = buffer[j];
            values[j] = buffer[j];
          }
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          auto buffer = std::make_unique<int64_t[]>(shape_size);
          auto status = GetShapeOfShapeTensor<int64_t>(input_tensor, buffer.get(), shape_size, stream);
          if (status != Status::OK()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
          }
          shape_tensor_values_int64[input_name].resize(shape_size);
          for (int j = 0; j < shape_size; ++j) {
            shape_tensor_values_int64[input_name][j] = buffer[j];
            values[j] = static_cast<int32_t>(buffer[j]);
          }
          break;
        }
        default: {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT shape tensor data type: " + std::to_string(tensor_type) + " not supported.");
        }
      }

      // Update shape ranges
      std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
      int shape_range_size = static_cast<int>(shape_ranges_per_input.size());
      if (shape_size == shape_range_size) {
        // If shape size matches, check/update shape range
        for (int j = 0; j < shape_size; ++j) {
          auto& shape_range = shape_ranges_per_input[j][0];  // only has one profile
          shapes_min[j] = static_cast<int32_t>(shape_range[0]);
          shapes_max[j] = static_cast<int32_t>(shape_range[1]);
          shapes_opt[j] = static_cast<int32_t>(shape_range[2]);

          const auto& tensor_shape_value = values[j];
          // Update shape range lower bound
          if (tensor_shape_value < shape_range[0]) {
            shape_range[0] = tensor_shape_value;
            shapes_min[j] = tensor_shape_value;
            *engine_update = true;
          }
          // Update shape range upper bound
          if (tensor_shape_value > shape_range[1]) {
            shape_range[1] = tensor_shape_value;
            shape_range[2] = tensor_shape_value;
            shapes_max[j] = tensor_shape_value;
            shapes_opt[j] = tensor_shape_value;
            *engine_update = true;
          }
        }
      } else {
        // If shape size doesn't match, initialize shape_range with the new shape value
        shape_ranges_per_input.clear();
        for (int j = 0; j < shape_size; ++j) {
          const auto& tensor_shape_value = values[j];
          std::vector<std::vector<int64_t>> profile_vector;
          std::vector<int64_t> shape_vector{tensor_shape_value, tensor_shape_value, tensor_shape_value};
          profile_vector.push_back(shape_vector);  // only one profile needed
          shape_ranges_per_input[j] = profile_vector;
          shapes_min[j] = tensor_shape_value;
          shapes_opt[j] = tensor_shape_value;
          shapes_max[j] = tensor_shape_value;
        }
        *engine_update = true;
      }
#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 10) || NV_TENSORRT_MAJOR > 10
      std::vector<int64_t> shapes_min_64(shapes_min.begin(), shapes_min.end());
      std::vector<int64_t> shapes_opt_64(shapes_opt.begin(), shapes_opt.end());
      std::vector<int64_t> shapes_max_64(shapes_max.begin(), shapes_max.end());
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min_64[0], shape_size);
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_opt_64[0], shape_size);
      trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_max_64[0], shape_size);
#else
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
#endif
    } else {  // Execution tensor
      nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
      for (int j = 0, end = nb_dims; j < end; ++j) {
        const auto& tensor_shape = tensor_shapes[j];
        if (shape_ranges_per_input.find(j) != shape_ranges_per_input.end()) {
          auto& shape_range = shape_ranges_per_input[j][0];  // only has one profile
          dims_min.d[j] = static_cast<int32_t>(shape_range[0]);
          dims_max.d[j] = static_cast<int32_t>(shape_range[1]);
          dims_opt.d[j] = static_cast<int32_t>(shape_range[2]);

          // Update minimum dimension
          if (tensor_shape < shape_range[0]) {
            shape_range[0] = tensor_shape;
            dims_min.d[j] = static_cast<int32_t>(tensor_shape);
            *engine_update = true;
          }
          // Update maximum dimension
          if (tensor_shape > shape_range[1]) {
            shape_range[1] = tensor_shape;
            shape_range[2] = tensor_shape;
            dims_max.d[j] = static_cast<int32_t>(tensor_shape);
            dims_opt.d[j] = static_cast<int32_t>(tensor_shape);
            *engine_update = true;
          }
        }
      }

      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
    }
  }
  return Status::OK();
}

#define CASE_GET_INPUT_TENSOR(DATA_TYPE, SrcT)                                              \
  case DATA_TYPE: {                                                                         \
    auto input_tensor_ptr = input_tensor.GetTensorData<SrcT>();                             \
    if (input_tensor_ptr != nullptr && elem_cnt > 0) {                                      \
      data = const_cast<SrcT*>(input_tensor_ptr);                                           \
    } else {                                                                                \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
      data = scratch_buffers.back().get();                                                  \
    }                                                                                       \
    break;                                                                                  \
  }

#define CASE_GET_CAST_INPUT_TENSOR(DATA_TYPE, SrcT, DstT)                                                         \
  case DATA_TYPE: {                                                                                               \
    auto input_tensor_ptr = input_tensor.GetTensorData<SrcT>();                                                   \
    if (input_tensor_ptr != nullptr && elem_cnt > 0) {                                                            \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, elem_cnt * sizeof(DstT))); \
      data = scratch_buffers.back().get();                                                                        \
      cuda::Impl_Cast<SrcT, DstT>(stream, input_tensor_ptr, reinterpret_cast<DstT*>(data), elem_cnt);             \
    } else {                                                                                                      \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, 1));                       \
      data = scratch_buffers.back().get();                                                                        \
    }                                                                                                             \
    break;                                                                                                        \
  }

#define CASE_GET_OUTPUT_TENSOR(DATA_TYPE, SrcT)                                             \
  case DATA_TYPE: {                                                                         \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<SrcT>();                    \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                     \
      buffers[output_name] = output_tensor_ptr;                                             \
    } else {                                                                                \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
      buffers[output_name] = scratch_buffers.back().get();                                  \
    }                                                                                       \
    break;                                                                                  \
  }

#define CASE_GET_CAST_OUTPUT_TENSOR(DATA_TYPE, SrcT, DstT)                                                        \
  case DATA_TYPE: {                                                                                               \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<SrcT>();                                          \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                                           \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, elem_cnt * sizeof(DstT))); \
      buffers[output_name] = scratch_buffers.back().get();                                                        \
      output_dim_sizes[i] = static_cast<int>(elem_cnt);                                                           \
    } else {                                                                                                      \
      scratch_buffers.push_back(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, 1));                       \
      buffers[output_name] = scratch_buffers.back().get();                                                        \
      output_dim_sizes[i] = 1;                                                                                    \
    }                                                                                                             \
    break;                                                                                                        \
  }

#define CASE_COPY_TENSOR(DATA_TYPE, DstT)                                                                                                          \
  case DATA_TYPE: {                                                                                                                                \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<DstT>();                                                                           \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                                                                            \
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_tensor_ptr, allocator->getBuffer(), elem_cnt * sizeof(DstT), cudaMemcpyDeviceToDevice, stream)); \
    }                                                                                                                                              \
    break;                                                                                                                                         \
  }

#define CASE_CAST_TENSOR(DATA_TYPE, SrcT, DstT)                                                                                                   \
  case DATA_TYPE: {                                                                                                                               \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<DstT>();                                                                          \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                                                                           \
      cuda::Impl_Cast<SrcT, DstT>(stream, reinterpret_cast<SrcT*>(allocator->getBuffer()), reinterpret_cast<DstT*>(output_tensor_ptr), elem_cnt); \
    }                                                                                                                                             \
    break;                                                                                                                                        \
  }

/*
 * Set TensorRT execution context input.
 *
 * There are two types of input tensor: (1) shape tensor and (2) execution tensor.
 * The input buffer binding needs to be handled differently.
 *
 * @param shape_tensor_values holds "shape tensor -> shape values" for the INT32 shape tensor input across this inference run
 * @param shape_tensor_values_int64 holds "shape tensor -> shape values" for the INT64 shape tensor input across this inference run
 */
Status BindContextInput(Ort::KernelContext& ctx,
                        nvinfer1::ICudaEngine* trt_engine,
                        nvinfer1::IExecutionContext* trt_context,
                        const char* input_name,
                        size_t input_index,
                        std::unordered_map<std::string, std::vector<int32_t>>& shape_tensor_values,
                        std::unordered_map<std::string, std::vector<int64_t>>& shape_tensor_values_int64,
                        std::vector<IAllocatorUniquePtr<void>>& scratch_buffers,
                        OrtAllocator* alloc,
                        cudaStream_t stream) {
  auto input_tensor = ctx.GetInput(input_index);
  auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
  const auto tensor_shapes = tensor_info.GetShape();
  const auto tensor_type = tensor_info.GetElementType();
  /*
   * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
   * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
   *
   * Examples:<br>
   * [] = 1<br>
   * [1,3,4] = 12<br>
   * [2,0,4] = 0<br>
   * [-1,3,4] = -1<br>
   */
  const auto elem_cnt = tensor_info.GetElementCount();

  if (trt_engine->isShapeInferenceIO(input_name)) {
    // Bind "shape tensor" input buffer

    // The shape of the "shape tensor" is either zero dimension (scalar) or 1-dimension
    int shape_size = trt_engine->getTensorShape(input_name).nbDims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
    switch (tensor_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        // get shape tensor value if not present
        if (shape_tensor_values.find(input_name) == shape_tensor_values.end()) {
          auto input = std::make_unique<int32_t[]>(shape_size);
          auto status = GetShapeOfShapeTensor<int32_t>(input_tensor, input.get(), shape_size, stream);
          if (status != Status::OK()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
          }
          shape_tensor_values[input_name].resize(shape_size);
          for (int i = 0; i < shape_size; ++i) {
            shape_tensor_values[input_name][i] = input[i];
          }
        }

        if (!trt_context->setTensorAddress(input_name, &shape_tensor_values[input_name][0])) {
          std::string error_input_name = input_name;
          std::string error_msg =
              "TensorRT EP failed to call nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
              error_input_name + "'";
          ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, error_msg));
        }
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        // get shape tensor value if not present
        if (shape_tensor_values_int64.find(input_name) == shape_tensor_values_int64.end()) {
          auto input = std::make_unique<int64_t[]>(shape_size);
          auto status = GetShapeOfShapeTensor<int64_t>(input_tensor, input.get(), shape_size, stream);
          if (status != Status::OK()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
          }
          shape_tensor_values_int64[input_name].resize(shape_size);
          for (int i = 0; i < shape_size; ++i) {
            shape_tensor_values_int64[input_name][i] = input[i];
          }
        }

        if (!trt_context->setTensorAddress(input_name, &shape_tensor_values_int64[input_name][0])) {
          std::string error_input_name = input_name;
          std::string error_msg =
              "TensorRT EP failed to call nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
              error_input_name + "'";
          ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, error_msg));
        }
        break;
      }
      default: {
        std::string error_input_name = input_name;
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "The data type of shape tensor should be INT32 or INT64. Please check the data type of " + error_input_name);
      }
    }
  } else {
    // Set shape for input tensor which is execution tensor
    nvinfer1::Dims dims = trt_context->getTensorShape(input_name);
    int nb_dims = dims.nbDims;
    for (int j = 0, end = nb_dims; j < end; ++j) {
      dims.d[j] = static_cast<int32_t>(tensor_shapes[j]);
    }
    if (!trt_context->setInputShape(input_name, dims)) {
      std::string error_input_name = input_name;
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                         "TensorRT EP failed to call nvinfer1::IExecutionContext::setInputShape() for input '" + error_input_name + "'"));
    }

    // Bind "execution tensor" input buffer
    //
    // Note: If an engine binding is an empty tensor, it still needs a non-null memory address, and different tensors should have different addresses.
    //       Therefore, in the case of empty tensor, TRT EP always allocates a dummy byte.
    //       https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#empty-tensors
    void* data = nullptr;
    switch (tensor_type) {
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
      CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
      // Cast int64 input to int32 input because TensorRT < 10 doesn't support int64
      CASE_GET_CAST_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t, int32_t)
#endif
      // Cast double input to float because TensorRT doesn't support double
      CASE_GET_CAST_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, double, float)
      default: {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");
      }
    }
    trt_context->setTensorAddress(input_name, data);
  }

  return Status::OK();
}

/*
 * Bind TensorRT execution context output.
 *
 * Please note that the "data-depedent shape" output needs corresponding allocator provided.
 *
 *
 * param ctx - ORT kernel context
 * param trt_context - A pointer to TensorRT Execution context object
 * param output_name - Output tensor name
 * param output_index - The index of the output to the ORT kernel context
 * param output_type - Data type of the output
 * param i - Output iteration index
 * param output_tensors - Output iteration index to output's ORT value
 * param output_dim_sizes - Output iteration index to the multiplocation of its shape's dimensions
 * param dds_output_set - DDS output set
 * param dds_output_allocator_map - DDS output to its allocator
 * param scratch_buffer - The allocation buffer created by TRT EP
 * param allocator - ORT allocator
 * param buffers - It holds all the output values which are binding to TRT's execution context
 *
 */
Status BindContextOutput(Ort::KernelContext& ctx,
                         nvinfer1::IExecutionContext* trt_context,
                         const char* output_name,
                         size_t output_index,
                         size_t output_type,
                         size_t i,
                         std::unordered_map<size_t, Ort::UnownedValue>& output_tensors,
                         std::unordered_map<size_t, int>& output_dim_sizes,
                         DDSOutputAllocatorMap& dds_output_allocator_map,
                         std::vector<IAllocatorUniquePtr<void>>& scratch_buffers,
                         OrtAllocator* alloc,
                         std::unordered_map<char const*, void*>& buffers) {
  // Get output shape
  nvinfer1::Dims dims = trt_context->getTensorShape(output_name);
  int nb_dims = dims.nbDims;
  bool is_DDS = false;
  std::vector<int64_t> output_shapes(nb_dims);
  for (int j = 0, end = nb_dims; j < end; ++j) {
    // data-dependent shape
    if (dims.d[j] == -1) {
      is_DDS = true;
      break;
    }
    output_shapes[j] = dims.d[j];
  }

  auto known_DDS = dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end();

  // If the output tensor has data-dependent shape, TRT EP will provide an IOutputAllocator for enqueueV3 to dynamically allocate memory buffer.
  // Once enqueueV3 returns, TRT EP will then bind the output allocation to ORT kernel context output.
  // (Please note that we take strategy A mentioned in https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dynamic-shaped-output,
  //  which we defer allocation until the size is known and don't call IExecution::setTensorAddress)
  //
  // Otherwise, if the shape of the output tensor is known prior to the runtime, ORT will pre-allocate memory buffer for the output tensor for enqueueV3.
  if (is_DDS || known_DDS) {
    if (!known_DDS) {
      auto allocatorPtr = std::make_unique<OutputAllocator>();
      trt_context->setOutputAllocator(output_name, allocatorPtr.get());
      dds_output_allocator_map[output_name] = std::move(allocatorPtr);
    }
  } else {
    output_tensors[i] = ctx.GetOutput(output_index, output_shapes);
    auto& output_tensor = output_tensors[i];
    const auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    switch (output_type) {
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
      CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
      // Allocate int32 CUDA memory for int64 output type because TensorRT < 10 doesn't support int64
      CASE_GET_CAST_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t, int32_t)
#endif
      // Allocate float CUDA memory for double output type because TensorRT doesn't support double
      CASE_GET_CAST_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, double, float)
      default: {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.");
      }
    }
    trt_context->setTensorAddress(output_name, buffers[output_name]);
  }

  return Status::OK();
}

/*
 * Bind ORT kernel context Output.
 *
 * In the case of DDS (data-dependent shape) output, TRT requires a provided allocator to allocate memory during runtime.
 * Once the output has been put in the allocation buffer, ORT calls this function to bind the allocation to ORT kernel context output.
 *
 * Note: Current approach of setting the ORT kernel context output is copying the output data from allocation buffer to ORT context output address which is not optimal,
 * we are waiting for ORT core to support "assign" memory address to ORT context output. Some works need to be done in ORT memory planner to be aware of this memory support.
 */
Status BindKernelOutput(Ort::KernelContext& ctx,
                        OrtMemoryInfo* /*mem_info*/,
                        DDSOutputAllocatorMap& allocator_map,
                        char const* output_name,
                        size_t output_index,
                        size_t output_type,
                        cudaStream_t stream) {
  auto allocator = allocator_map[output_name].get();
  auto& shape = allocator->getOutputShape();
  auto output_tensor = ctx.GetOutput(output_index, shape);

  /*
   * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
   * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
   *
   * Examples:<br>
   * [] = 1<br>
   * [1,3,4] = 12<br>
   * [2,0,4] = 0<br>
   * [-1,3,4] = -1<br>
   */
  auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

  /*
   * Copy output data from allocation buffer to ORT kernel context output location or
   * cast (int32 or float) -> (int64 or double) to ORT kernel context output location.
   *
   * Note:
   * 1. If the output tensor is empty tensor (i.e. any of the dimension is 0) which means element count is 0,
   *    TRT EP does not perform cuda memory copy nor cuda cast to prevent overwriting other location that might belong to other tensors.
   * 2. The cudaMemcpyAsync() and cuda::Impl_Cast() (implemented as _UnaryElementWise() in cuda ep) are all async, but we
   *    don't need to explicitly call cudaStreamSynchronize() after those APIs due to CUDA EP and TRT EP uses same stream,
   *    and within the same stream, operations are guaranteed to be executed in order.
   */
  switch (output_type) {
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
    CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
    // The allocation buffer holds the int32 output data since TRT doesn't support int64. So, we need to cast the data (int32 -> int64) for ORT kernel output.
    CASE_CAST_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int32_t, int64_t)
#endif
    // The allocation buffer holds the float output data since TRT doesn't support double. So, we need to cast the data (float -> double) for ORT kernel output.
    CASE_CAST_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, float, double)
    default: {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.");
    }
  }
  return Status::OK();
}

TensorrtExecutionProvider::PerThreadContext::PerThreadContext(OrtDevice::DeviceId device_id, bool has_user_compute_stream, cudaStream_t stream) {
  if (has_user_compute_stream) {
    CUDA_CALL_THROW(cudaSetDevice(device_id));
#ifndef USE_CUDA_MINIMAL
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasCreate(&external_cublas_handle_)));
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasSetStream(external_cublas_handle_, stream)));
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnCreate(&external_cudnn_handle_)));
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnSetStream(external_cudnn_handle_, stream)));
#else
    (void)(stream);
#endif
  }
}

TensorrtExecutionProvider::PerThreadContext::~PerThreadContext() {
#ifndef USE_CUDA_MINIMAL
  if (external_cublas_handle_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasDestroy(external_cublas_handle_)));
  }
  if (external_cudnn_handle_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnDestroy(external_cudnn_handle_)));
  }
#endif
  trt_context_map_.clear();
}

/*
 * Returns true if the shape ranges maintained by the PerThreadContext is different from the shape ragnes maintained by TRT EP, meaning the
 * engine is being updated and the execution context maintained by the PerThreadContext should be updated as well. Otherwise, returns false.
 *
 */
bool TensorrtExecutionProvider::PerThreadContext::CompareProfileShapes(std::string fused_node, ShapeRangesMap& shape_ranges) {
  if (shape_ranges.size() > 0) {
    if (input_shape_ranges_[fused_node] != shape_ranges) {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] The shape ranges maintained by the PerThreadContext is different from the shape ranges maintained by TRT EP. \
                                This means the engine is updated and will need to update the execution context as well.";
      return true;
    }
  }
  return false;
}

/*
 * Updates the shape ranges maintained by the PerThreadContext.
 * As long as the execution context maintained by the PerThreadContext is updated, the associated shape ranges should be updated as well.
 *
 */
void TensorrtExecutionProvider::PerThreadContext::UpdateProfileShapes(std::string fused_node, ShapeRangesMap& shape_ranges) {
  input_shape_ranges_[fused_node] = shape_ranges;
}

void TensorrtExecutionProvider::PerThreadContext::ResetTensorRTContext(std::string fused_node) {
  auto it = trt_context_map_.find(fused_node);
  if (it != trt_context_map_.end()) {
    trt_context_map_[fused_node].reset();
  }
}

bool TensorrtExecutionProvider::PerThreadContext::UpdateTensorRTContext(std::string fused_node, std::unique_ptr<nvinfer1::IExecutionContext> context) {
  if (!context) {
    context = std::make_unique<nvinfer1::IExecutionContext>();
  }
  trt_context_map_[fused_node] = std::move(context);

  if (trt_context_map_[fused_node]) {
    return true;
  }
  return false;
}

bool TensorrtExecutionProvider::PerThreadContext::IsTensorRTContextInMap(std::string fused_node) {
  auto it = trt_context_map_.find(fused_node);
  if (it != trt_context_map_.end()) {
    return true;
  }
  return false;
}

nvinfer1::IExecutionContext& TensorrtExecutionProvider::PerThreadContext::GetTensorRTContext(std::string fused_node) {
  auto it = trt_context_map_.find(fused_node);
  if (it != trt_context_map_.end()) {
    return *(it->second);  // dereference shared pointer
  }
  auto context = std::make_unique<nvinfer1::IExecutionContext>();
  trt_context_map_[fused_node] = std::move(context);
  return *(trt_context_map_[fused_node]);  // dereference shared pointer
}

void TensorrtExecutionProvider::ReleasePerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
  auto cached_context = cached_context_it->second.lock();
  ORT_ENFORCE(cached_context);

  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    context_state_.active_contexts.erase(cached_context);
    context_state_.retired_context_pool.push_back(cached_context);
  }

  per_thread_context_cache->erase(cached_context_it);
}

TensorrtExecutionProvider::PerThreadContext& TensorrtExecutionProvider::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    ORT_ENFORCE(cached_context);
    return *cached_context;
  }

  // get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);

    // get or create a context
    if (context_state_.retired_context_pool.empty()) {
      context = std::make_shared<PerThreadContext>(narrow<OrtDevice::DeviceId>(info_.device_id),
                                                   info_.has_user_compute_stream, stream_);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // insert into active_contexts, should not already be present
    const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
    ORT_ENFORCE(active_contexts_insert_result.second);

    // insert into caches_to_update_on_destruction, may already be present
    ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

std::vector<nvinfer1::PreviewFeature> ParseTrtPreviewFeatures(const std::string& str) {
  std::vector<std::string> featureNames{SplitToStringVec(str, ',')};

  std::vector<nvinfer1::PreviewFeature> previewFeatures;
  previewFeatures.reserve(featureNames.size());
  for (auto featureName : featureNames) {
    if (featureName == "ALIASED_PLUGIN_IO_10_03") {
      previewFeatures.push_back(nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03);
    } else {
      throw std::invalid_argument(std::string("Unkown or unsupported preview feature: ") + featureName);
    }
  }

  return previewFeatures;
}

TensorrtExecutionProvider::TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTensorrtExecutionProvider,
                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                                   narrow<OrtDevice::DeviceId>(info.device_id))},
      info_(info),
      device_id_(info.device_id) {
  InitProviderOrtApi();

  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  cudaDeviceProp prop;
  CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id_));
  compute_capability_ = GetComputeCapacity(prop);
  if (info.has_user_compute_stream) {
    external_stream_ = true;
    stream_ = static_cast<cudaStream_t>(info.user_compute_stream);
#ifndef USE_CUDA_MINIMAL
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasCreate(&external_cublas_handle_)));
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasSetStream(external_cublas_handle_, stream_)));
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnCreate(&external_cudnn_handle_)));
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnSetStream(external_cudnn_handle_, stream_)));
#endif
  }

  std::string profile_min_shapes, profile_max_shapes, profile_opt_shapes;

  // incase the EP context is dumped the engine cache has to be enabled
  auto enable_engine_cache_for_ep_context_model = [this]() {
    if (dump_ep_context_model_ && ep_context_embed_mode_ == 0) {
      engine_cache_enable_ = true;
    }
  };

  // Get environment variables
  if (info.has_trt_options) {
    max_partition_iterations_ = info.max_partition_iterations;
    min_subgraph_size_ = info.min_subgraph_size;
    max_workspace_size_ = info.max_workspace_size;
    fp16_enable_ = info.fp16_enable;
    bf16_enable_ = info.bf16_enable;
    // BF16 support is primarily available on NVIDIA GPUs with the Ampere and later architectures with compute capability of 8.0 or higher.
    if (bf16_enable_ && prop.major < 8) {
      bf16_enable_ = false;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] trt_bf16_enable is set, but platform doesn't support bf16.";
    }
    int8_enable_ = info.int8_enable;
    if (int8_enable_) {
      int8_calibration_cache_name_ = info.int8_calibration_table_name;
      int8_use_native_tensorrt_calibration_table_ = info.int8_use_native_calibration_table;
    }
    if (fp16_enable_ || int8_enable_) {  // DLA can only be enabled with FP16 or INT8
      dla_enable_ = info.dla_enable;
      dla_core_ = info.dla_core;
    }
    dump_subgraphs_ = info.dump_subgraphs;
    engine_cache_enable_ = info.engine_cache_enable;
    weight_stripped_engine_enable_ = info.weight_stripped_engine_enable;
    onnx_model_folder_path_ = info.onnx_model_folder_path;
    onnx_model_bytestream_ = info.onnx_bytestream;
    onnx_model_bytestream_size_ = info.onnx_bytestream_size;
    if ((onnx_model_bytestream_ != nullptr && onnx_model_bytestream_size_ == 0) ||
        (onnx_model_bytestream_ == nullptr && onnx_model_bytestream_size_ != 0)) {
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                         "When providing either 'trt_onnx_bytestream_size' or "
                                         "'trt_onnx_bytestream' both have to be provided"));
    }
    onnx_external_data_bytestream_ = info.external_data_bytestream;
    onnx_external_data_bytestream_size_ = info.external_data_bytestream_size;
    if ((onnx_external_data_bytestream_ != nullptr && onnx_external_data_bytestream_size_ == 0) ||
        (onnx_external_data_bytestream_ == nullptr && onnx_external_data_bytestream_size_ != 0)) {
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                         "When providing either 'trt_external_data_bytestream_size' or "
                                         "'trt_external_data_bytestream' both have to be provided"));
    }
    timing_cache_enable_ = info.timing_cache_enable;
    force_timing_cache_match_ = info.force_timing_cache;
    detailed_build_log_ = info.detailed_build_log;
    dump_ep_context_model_ = info.dump_ep_context_model;
    ep_context_file_path_ = info.ep_context_file_path;
    ep_context_embed_mode_ = info.ep_context_embed_mode;
    enable_engine_cache_for_ep_context_model();
    if (engine_cache_enable_ || int8_enable_ || timing_cache_enable_) {
      cache_path_ = info.engine_cache_path;
      cache_prefix_ = info.engine_cache_prefix;
    }
    // use a more global cache if given
    if (timing_cache_enable_) {
      if (!info.timing_cache_path.empty()) {
        global_cache_path_ = info.timing_cache_path;
      } else {
        global_cache_path_ = cache_path_;
      }
    }
    engine_decryption_enable_ = info.engine_decryption_enable;
    if (engine_decryption_enable_) {
      engine_decryption_lib_path_ = info.engine_decryption_lib_path;
    }
    force_sequential_engine_build_ = info.force_sequential_engine_build;
    context_memory_sharing_enable_ = info.context_memory_sharing_enable;
    if (fp16_enable_ || bf16_enable_) {
      layer_norm_fp32_fallback_ = info.layer_norm_fp32_fallback;
    }
    build_heuristics_enable_ = info.build_heuristics_enable;
    sparsity_enable_ = info.sparsity_enable;
    builder_optimization_level_ = info.builder_optimization_level;
    auxiliary_streams_ = info.auxiliary_streams;
    tactic_sources_ = info.tactic_sources;
    profile_min_shapes = info.profile_min_shapes;
    profile_max_shapes = info.profile_max_shapes;
    profile_opt_shapes = info.profile_opt_shapes;
    cuda_graph_enable_ = info.cuda_graph_enable;
    engine_hw_compatible_ = info.engine_hw_compatible;
    op_types_to_exclude_ = info.op_types_to_exclude;
    preview_features_ = ParseTrtPreviewFeatures(info.preview_features);
    load_user_initializer_ = info.load_user_initializer;
  } else {
    try {
      const std::string max_partition_iterations_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxPartitionIterations);
      if (!max_partition_iterations_env.empty()) {
        max_partition_iterations_ = std::stoi(max_partition_iterations_env);
      }

      const std::string min_subgraph_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMinSubgraphSize);
      if (!min_subgraph_size_env.empty()) {
        min_subgraph_size_ = std::stoi(min_subgraph_size_env);
      }

      const std::string max_workspace_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxWorkspaceSize);
      if (!max_workspace_size_env.empty()) {
        max_workspace_size_ = std::stoull(max_workspace_size_env);
      }

      const std::string fp16_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kFP16Enable);
      if (!fp16_enable_env.empty()) {
        fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
      }

      const std::string bf16_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kBF16Enable);
      if (!bf16_enable_env.empty()) {
        bf16_enable_ = (std::stoi(bf16_enable_env) == 0 ? false : true);
      }

      const std::string int8_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8Enable);
      if (!int8_enable_env.empty()) {
        int8_enable_ = (std::stoi(int8_enable_env) == 0 ? false : true);
      }

      if (int8_enable_) {
        const std::string int8_calibration_cache_name_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8CalibrationTableName);
        if (!int8_calibration_cache_name_env.empty()) {
          int8_calibration_cache_name_ = int8_calibration_cache_name_env;
        }

        const std::string int8_use_native_tensorrt_calibration_table_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8UseNativeTensorrtCalibrationTable);
        if (!int8_use_native_tensorrt_calibration_table_env.empty()) {
          int8_use_native_tensorrt_calibration_table_ = (std::stoi(int8_use_native_tensorrt_calibration_table_env) == 0 ? false : true);
        }
      }

      if (fp16_enable_ || int8_enable_) {  // DLA can only be enabled with FP16 or INT8
        const std::string dla_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDLAEnable);
        if (!dla_enable_env.empty()) {
          dla_enable_ = (std::stoi(dla_enable_env) == 0 ? false : true);
        }

        if (dla_enable_) {
          const std::string dla_core_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDLACore);
          if (!dla_core_env.empty()) {
            dla_core_ = std::stoi(dla_core_env);
          }
        }
      }

      const std::string dump_subgraphs_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDumpSubgraphs);
      if (!dump_subgraphs_env.empty()) {
        dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
      }

      const std::string engine_cache_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCacheEnable);
      if (!engine_cache_enable_env.empty()) {
        engine_cache_enable_ = (std::stoi(engine_cache_enable_env) == 0 ? false : true);
      }

      const std::string weight_stripped_engine_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kWeightStrippedEngineEnable);
      if (!weight_stripped_engine_enable_env.empty()) {
        weight_stripped_engine_enable_ = std::stoi(weight_stripped_engine_enable_env) != 0;
      }

      const std::string onnx_model_folder_path_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kOnnxModelFolderPath);
      if (!onnx_model_folder_path_env.empty()) {
        onnx_model_folder_path_ = onnx_model_folder_path_env;
      }

      const std::string timing_cache_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kTimingCacheEnable);
      if (!timing_cache_enable_env.empty()) {
        timing_cache_enable_ = (std::stoi(timing_cache_enable_env) == 0 ? false : true);
      }

      const std::string detailed_build_log_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDetailedBuildLog);
      if (!detailed_build_log_env.empty()) {
        detailed_build_log_ = (std::stoi(detailed_build_log_env) == 0 ? false : true);
      }

      const std::string timing_force_match_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kForceTimingCache);
      if (!timing_force_match_env.empty()) {
        force_timing_cache_match_ = (std::stoi(timing_force_match_env) == 0 ? false : true);
      }

      const std::string dump_ep_context_model_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDumpEpContextModel);
      if (!dump_ep_context_model_env.empty()) {
        dump_ep_context_model_ = (std::stoi(dump_ep_context_model_env) == 0 ? false : true);
      }

      const std::string ep_context_file_path_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEpContextComputeCapabilityEnable);
      if (!ep_context_file_path_env.empty()) {
        ep_context_file_path_ = ep_context_file_path_env;
      }

      const std::string ep_context_embed_mode_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEpContextEmbedMode);
      if (!ep_context_embed_mode_env.empty()) {
        ep_context_embed_mode_ = std::stoi(ep_context_embed_mode_env);
      }
      // incase the EP context is dumped the engine cache has to be enabled
      if (dump_ep_context_model_ && ep_context_embed_mode_ == 0) {
        engine_cache_enable_ = true;
      }

      enable_engine_cache_for_ep_context_model();

      if (engine_cache_enable_ || int8_enable_ || timing_cache_enable_) {
        const std::string engine_cache_path = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCachePath);
        cache_path_ = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kCachePath);
        cache_prefix_ = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCachePrefix);
        if (!engine_cache_path.empty() && cache_path_.empty()) {
          cache_path_ = engine_cache_path;
          LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_ENGINE_CACHE_PATH is deprecated! Please use ORT_TENSORRT_CACHE_PATH to specify engine cache path";
        }
      }
      if (timing_cache_enable_) {
        std::string timing_cache_path = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kTimingCachePath);
        // use a more global cache if given
        if (!timing_cache_path.empty()) {
          global_cache_path_ = timing_cache_path;
        } else {
          global_cache_path_ = cache_path_;
        }
      }

      const std::string engine_decryption_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDecryptionEnable);
      if (!engine_decryption_enable_env.empty()) {
        engine_decryption_enable_ = (std::stoi(engine_decryption_enable_env) == 0 ? false : true);
      }

      if (engine_decryption_enable_) {
        engine_decryption_lib_path_ = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDecryptionLibPath);
      }

      const std::string force_sequential_engine_build_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kForceSequentialEngineBuild);
      if (!force_sequential_engine_build_env.empty()) {
        force_sequential_engine_build_ = (std::stoi(force_sequential_engine_build_env) == 0 ? false : true);
      }

      const std::string context_memory_sharing_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kContextMemorySharingEnable);
      if (!context_memory_sharing_enable_env.empty()) {
        context_memory_sharing_enable_ = (std::stoi(context_memory_sharing_enable_env) == 0 ? false : true);
      }

      const std::string layer_norm_fp32_fallback_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kLayerNormFP32Fallback);
      if (!layer_norm_fp32_fallback_env.empty()) {
        layer_norm_fp32_fallback_ = (std::stoi(layer_norm_fp32_fallback_env) == 0 ? false : true);
      }

      const std::string build_heuristics_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kBuildHeuristics);
      if (!build_heuristics_env.empty()) {
        build_heuristics_enable_ = (std::stoi(build_heuristics_env) == 0 ? false : true);
      }

      const std::string sparsity_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kSparsityEnable);
      if (!sparsity_enable_env.empty()) {
        sparsity_enable_ = (std::stoi(sparsity_enable_env) == 0 ? false : true);
      }

      const std::string builder_optimization_level_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kBuilderOptimizationLevel);
      if (!builder_optimization_level_env.empty()) {
        builder_optimization_level_ = std::stoi(builder_optimization_level_env);
      }

      const std::string auxiliary_streams_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kAuxiliaryStreams);
      if (!auxiliary_streams_env.empty()) {
        auxiliary_streams_ = std::stoi(auxiliary_streams_env);
      }

      const std::string tactic_sources_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kTacticSources);
      if (!tactic_sources_env.empty()) {
        tactic_sources_ = tactic_sources_env;
      }

      profile_min_shapes = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kProfilesMinShapes);
      profile_max_shapes = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kProfilesMaxShapes);
      profile_opt_shapes = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kProfilesOptShapes);

      const std::string cuda_graph_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kCudaGraphEnable);
      if (!cuda_graph_enable_env.empty()) {
        cuda_graph_enable_ = (std::stoi(cuda_graph_enable_env) == 0 ? false : true);
      }

      const std::string op_types_to_exclude_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kOpTypesToExclude);
      if (!op_types_to_exclude_env.empty()) {
        op_types_to_exclude_ = op_types_to_exclude_env;
      }

    } catch (const std::invalid_argument& ex) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Invalid Argument (from environment variables): " << ex.what();
    } catch (const std::out_of_range& ex) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Out Of Range Error (from environment variables): " << ex.what();
    } catch (...) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Unknown Exception (from environment variables)";
    }
  }

  // Validate setting
  if (max_partition_iterations_ <= 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] TensorRT option trt_max_partition_iterations must be a positive integer value. Set it to 1000";
    max_partition_iterations_ = 1000;
  }
  if (min_subgraph_size_ <= 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] TensorRT option trt_min_subgraph_size must be a positive integer value. Set it to 1";
    min_subgraph_size_ = 1;
  }
  if (dla_core_ < 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] TensorRT option trt_dla_core must be a non-negative integer value. Set it to 0";
    dla_core_ = 0;
  }

  // If ep_context_file_path_ is provided as a directory, create it if it's not existed
  if (dump_ep_context_model_ && !ep_context_file_path_.empty() && std::filesystem::path(ep_context_file_path_).extension().empty() && !std::filesystem::is_directory(ep_context_file_path_)) {
    if (!std::filesystem::create_directory(ep_context_file_path_)) {
      throw std::runtime_error("Failed to create directory " + ep_context_file_path_);
    }
  }

  // If dump_ep_context_model_ is enable, TRT EP forces cache_path_ to be the relative path of ep_context_file_path_.
  // For example,
  //    - original cache path = "engine_cache_dir" -> new cache path = "./context_model_dir/engine_cache_dir"
  //    - original cache path = ""                 -> new cache path = "./context_model_dir"
  // The new cache path will be saved as the "ep_cache_context" node attritue of the EP context node.
  // For security reason, it needs to make sure the engine cache is saved inside context model directory.
  if (dump_ep_context_model_ && engine_cache_enable_) {
    if (IsAbsolutePath(cache_path_)) {
      LOGS_DEFAULT(ERROR) << "In the case of dumping context model and for security purpose, the trt_engine_cache_path should be set with a relative path, but it is an absolute path:  " << cache_path_;
    }
    if (IsRelativePathToParentPath(cache_path_)) {
      LOGS_DEFAULT(ERROR) << "In the case of dumping context model and for security purpose, The trt_engine_cache_path has '..', it's not allowed to point outside the directory.";
    }

    // Engine cache relative path to context model directory.
    // It's used when dumping the "ep_cache_context" node attribute.
    engine_cache_relative_path_to_context_model_dir = cache_path_;

    // Make cache_path_ to be the relative path of ep_context_file_path_
    cache_path_ = GetPathOrParentPathOfCtxModel(ep_context_file_path_).append(cache_path_).string();
  }

  // Hardware compatibility: pre-check on environment
  if (engine_cache_enable_ && engine_hw_compatible_) {
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
    if (std::stoi(compute_capability_) < 80) {
      LOGS_DEFAULT(WARNING) << "Engine hardware compatibility cannot be enabled as GPU arch < 80. ";
      engine_hw_compatible_ = false;
    } else if (std::stoi(compute_capability_) == 87) {
      LOGS_DEFAULT(WARNING) << "Engine hardware compatibility cannot be enabled on Jetson Orin. ";
      engine_hw_compatible_ = false;
    }
#else
    LOGS_DEFAULT(WARNING) << "Engine hardware compatibility cannot be enabled as TRT < 8.6. ";
    engine_hw_compatible_ = false;
#endif
  }

  if (engine_cache_enable_ || int8_enable_ || timing_cache_enable_) {
    if (!cache_path_.empty() && !fs::is_directory(cache_path_)) {
      if (!fs::create_directory(cache_path_)) {
        throw std::runtime_error("Failed to create directory " + cache_path_);
      }
    }
    if (!global_cache_path_.empty() && !fs::is_directory(global_cache_path_)) {
      if (!fs::create_directory(global_cache_path_)) {
        throw std::runtime_error("Failed to create directory " + global_cache_path_);
      }
    }
  }

  if (engine_decryption_enable_) {
    LIBTYPE handle = OPENLIB(engine_decryption_lib_path_.c_str());
    if (handle == nullptr) {
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                         "TensorRT EP could not open shared library from " + engine_decryption_lib_path_));
    }
    engine_decryption_ = (int (*)(const char*, char*, size_t*))LIBFUNC(handle, "decrypt");
    engine_encryption_ = (int (*)(const char*, char*, size_t))LIBFUNC(handle, "encrypt");
    if (engine_decryption_ == nullptr) {
      ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                         "TensorRT EP could not find decryption function in shared library from " + engine_decryption_lib_path_));
    }
  }

  if (int8_enable_) {
    int8_calibration_cache_available_ = !int8_calibration_cache_name_.empty();
  }

  /*
   * Parse explicit min/max/opt profile shapes from provider options.
   *
   * The format of min/max/opt profile shapes is defined as below:
   * "input1:dim1xdim2...,input2:dim1xdim2...,...,input1:dim3xdim4...,input2:dim3xdim4...,..."
   *
   * (Note: if multiple shapes with same input name are specified, TRT EP will consider them as multiple profiles.
   *  Please refer to ParserProfileShapes() for more details)
   *
   */
  bool status = true;
  if (status) {
    status = ParseProfileShapes(profile_min_shapes, profile_min_shapes_);
    if (!status) {
      profile_min_shapes_.clear();
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The format of provider option 'trt_profile_min_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
    }
  }

  if (status) {
    status = ParseProfileShapes(profile_max_shapes, profile_max_shapes_);
    if (!status) {
      profile_max_shapes_.clear();
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The format of provider option 'trt_profile_max_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
    }
  }

  if (status) {
    status = ParseProfileShapes(profile_opt_shapes, profile_opt_shapes_);
    if (!status) {
      profile_opt_shapes_.clear();
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] The format of provider option 'trt_profile_opt_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
    }
  }

  if (status) {
    status = ValidateProfileShapes(profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_);
    if (!status) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Profile shapes validation failed. Make sure the provider options 'trt_profile_min_shapes', 'trt_profile_max_shapes' and 'trt_profile_opt_shapes' have same input name and number of profile.";
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] TRT EP will implicitly create optimization profiles based on input tensor for you.";
      profile_min_shapes_.clear();
      profile_max_shapes_.clear();
      profile_opt_shapes_.clear();
    }
  }

  // cuda graph:
  // cudaStreamSynchronize() is not allowed in cuda graph capture.
  //
  // external stream:
  // If user provides "external" cuda stream, only this cuda stream will be used even if multiple threads are running InferenceSession.Run() concurrently.
  // So, no need to synchronize different streams after enqueueV3.
  if (cuda_graph_enable_ || external_stream_) {
    sync_stream_after_enqueue_ = false;
  }

  {
    auto lock = GetApiLock();
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(GetTensorrtLogger(detailed_build_log_)));
  }

  trt_version_ = getInferLibVersion();
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&cuda_version_));

  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] TensorRT version is " << trt_version_;
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] CUDA version is " << cuda_version_;

  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] TensorRT provider options: "
                        << "device_id: " << device_id_
                        << ", trt_max_partition_iterations: " << max_partition_iterations_
                        << ", trt_min_subgraph_size: " << min_subgraph_size_
                        << ", trt_max_workspace_size: " << max_workspace_size_
                        << ", trt_fp16_enable: " << fp16_enable_
                        << ", trt_bf16_enable: " << bf16_enable_
                        << ", trt_int8_enable: " << int8_enable_
                        << ", trt_int8_calibration_cache_name: " << int8_calibration_cache_name_
                        << ", int8_calibration_cache_available: " << int8_calibration_cache_available_
                        << ", trt_int8_use_native_tensorrt_calibration_table: " << int8_use_native_tensorrt_calibration_table_
                        << ", trt_dla_enable: " << dla_enable_
                        << ", trt_dla_core: " << dla_core_
                        << ", trt_dump_subgraphs: " << dump_subgraphs_
                        << ", trt_engine_cache_enable: " << engine_cache_enable_
                        << ", trt_weight_stripped_engine_enable: " << weight_stripped_engine_enable_
                        << ", trt_onnx_model_folder_path: " << onnx_model_folder_path_
                        << ", trt_cache_path: " << cache_path_
                        << ", trt_global_cache_path: " << global_cache_path_
                        << ", trt_engine_decryption_enable: " << engine_decryption_enable_
                        << ", trt_engine_decryption_lib_path: " << engine_decryption_lib_path_
                        << ", trt_force_sequential_engine_build: " << force_sequential_engine_build_
                        << ", trt_context_memory_sharing_enable: " << context_memory_sharing_enable_
                        << ", trt_layer_norm_fp32_fallback: " << layer_norm_fp32_fallback_
                        << ", trt_build_heuristics_enable: " << build_heuristics_enable_
                        << ", trt_sparsity_enable: " << sparsity_enable_
                        << ", trt_builder_optimization_level: " << builder_optimization_level_
                        << ", trt_auxiliary_streams: " << auxiliary_streams_
                        << ", trt_tactic_sources: " << tactic_sources_
                        << ", trt_profile_min_shapes: " << profile_min_shapes
                        << ", trt_profile_max_shapes: " << profile_max_shapes
                        << ", trt_profile_opt_shapes: " << profile_opt_shapes
                        << ", trt_cuda_graph_enable: " << cuda_graph_enable_
                        << ", trt_dump_ep_context_model: " << dump_ep_context_model_
                        << ", trt_ep_context_file_path: " << ep_context_file_path_
                        << ", trt_ep_context_embed_mode: " << ep_context_embed_mode_
                        << ", trt_cache_prefix: " << cache_prefix_
                        << ", trt_engine_hw_compatible: " << engine_hw_compatible_
                        << ", trt_onnx_model_bytestream_size_: " << onnx_model_bytestream_size_
                        << ", trt_onnx_external_data_bytestream_size: " << onnx_external_data_bytestream_size_
                        << ", trt_op_types_to_exclude: " << op_types_to_exclude_
                        << ", trt_load_user_initializer: " << load_user_initializer_;
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {
  // clean up thread local context caches
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    for (const auto& cache_weak : context_state_.caches_to_update_on_destruction) {
      const auto cache = cache_weak.lock();
      if (!cache) continue;
      ORT_IGNORE_RETURN_VALUE(cache->erase(this));
    }
  }

  if (external_stream_) {
#ifndef USE_CUDA_MINIMAL
    ORT_IGNORE_RETURN_VALUE(CUBLAS_CALL(cublasDestroy(external_cublas_handle_)));
    ORT_IGNORE_RETURN_VALUE(CUDNN_CALL(cudnnDestroy(external_cudnn_handle_)));
#endif
  }

  if (!external_stream_ && stream_) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream_)));
  }
  ReleaseTensorRTCustomOpDomainList(info_.custom_op_domain_list);

  if (context_memory_) {
    context_memory_.reset();
  }

  if (alloc_ != nullptr) {
    // This code is same as OrtApis::ReleaseAllocator defined in allocator_adapters.cc.
    // We can't get api inside destructor so that's why we duplicate the code here.
    delete static_cast<OrtAllocatorImpl*>(alloc_);
  }
}

bool TensorrtExecutionProvider::IsGraphCaptureEnabled() const {
  return cuda_graph_enable_;
}

bool TensorrtExecutionProvider::IsGraphCaptureAllowed() const {
  return regular_run_count_before_graph_capture_ >= min_num_runs_before_cuda_graph_capture_;
}

void TensorrtExecutionProvider::CaptureBegin(int) {
  cuda_graph_.Reset();
  cuda_graph_.CaptureBegin(0);
}

void TensorrtExecutionProvider::CaptureEnd(int) {
  cuda_graph_.CaptureEnd(0);
  is_graph_captured_ = true;
}

bool TensorrtExecutionProvider::IsGraphCaptured(int) const {
  return is_graph_captured_;
}

Status TensorrtExecutionProvider::ReplayGraph(int) {
  ORT_ENFORCE(IsGraphCaptured(0));
  // Please note that CUDAGraph::Replay() is not thread safe.
  // ORT TRT calls ReplayGraph() in compute_func() where synchronization is enforced due to lock_guard(),
  // therefore calling CUDAGraph::Replay() here is guaranteed to be thread safe.
  return cuda_graph_.Replay(0);
}

void TensorrtExecutionProvider::IncrementRegularRunCountBeforeGraphCapture() {
  // Please note that this function is not thread safe.
  // ORT TRT calls this function in compute_func() where synchronization is enforced due to lock_guard(),
  // therefore following increment is guaranteed to be thread safe.
  ++regular_run_count_before_graph_capture_;
}

std::vector<AllocatorPtr> TensorrtExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo default_memory_info(
      [](OrtDevice::DeviceId device_id) { return CreateCUDAAllocator(device_id, onnxruntime::CUDA); },
      narrow<OrtDevice::DeviceId>(device_id_));

  AllocatorCreationInfo pinned_allocator_info(
      [](OrtDevice::DeviceId device_id) {
        return CreateCUDAPinnedAllocator(device_id, onnxruntime::CUDA_PINNED);
      },
      narrow<OrtDevice::DeviceId>(device_id_));

  return std::vector<AllocatorPtr>{CreateAllocator(default_memory_info), CreateAllocator(pinned_allocator_info)};
}

std::unique_ptr<IDataTransfer> TensorrtExecutionProvider::GetDataTransfer() const {
  return onnxruntime::CreateGPUDataTransfer();
}

Status TensorrtExecutionProvider::OnRunStart(const onnxruntime::RunOptions& /*run_options*/) {
  return Status::OK();
}

Status TensorrtExecutionProvider::OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& /*run_options*/) {
  if (sync_stream && external_stream_) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  }
  return Status::OK();
}

// Get the pointer to the IBuilder instance.
// Note: This function is not thread safe. Calls to this function from different threads must be serialized
// even though it doesn't make sense to have multiple threads initializing the same inference session.
nvinfer1::IBuilder* TensorrtExecutionProvider::GetBuilder(TensorrtLogger& trt_logger) const {
  if (!builder_) {
    {
      auto lock = GetApiLock();
      builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    }
  }
  return builder_.get();
}

void TensorrtExecutionProvider::GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) const {
  std::string extra_plugin_lib_paths{""};
  if (info_.has_trt_options) {
    if (!info_.extra_plugin_lib_paths.empty()) {
      extra_plugin_lib_paths = info_.extra_plugin_lib_paths;
    }
  } else {
    const std::string extra_plugin_lib_paths_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kExtraPluginLibPaths);
    if (!extra_plugin_lib_paths_env.empty()) {
      extra_plugin_lib_paths = extra_plugin_lib_paths_env;
    }
  }
  auto status = CreateTensorRTCustomOpDomainList(custom_op_domain_list, extra_plugin_lib_paths);
  if (status != Status::OK()) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Failed to get TRT plugins from TRT plugin registration.";
  }
}

// Check the graph is the subgraph of control flow op
bool TensorrtExecutionProvider::IsSubGraphOfControlFlowOp(const GraphViewer& graph) const {
  if (graph.IsSubgraph()) {
    const auto& node = graph.ParentNode();
    if (control_flow_op_set_.find(node->OpType()) != control_flow_op_set_.end()) {
      return true;
    }
  }
  return false;
}

// Check whether all the nodes of the graph are assigned to specific ep
bool TensorrtExecutionProvider::AllNodesAssignedToSpecificEP(const GraphViewer& graph, const std::string& provider_type) const {
  const int number_of_ort_nodes = graph.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
  for (const auto& index : nodes_vector) {
    const auto& node = graph.GetNode(node_index[index]);
    if (node->GetExecutionProviderType() != provider_type) {
      return false;
    }
  }

  return number_of_ort_nodes != 0;
}

// Check whether all the nodes of subgraph are supported
bool TensorrtExecutionProvider::IsSubGraphFullySupported(SubGraphCollection_t supported_nodes_vector, const int number_of_ort_nodes) const {
  int number_of_trt_nodes = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      number_of_trt_nodes += static_cast<int>(group.first.size());
    }
  }

  return number_of_trt_nodes == number_of_ort_nodes;
}

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, const GraphViewer& graph, const HashValue& model_hash, int subgraph_index) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
  std::unordered_map<const NodeArg*, int> original_inputs, fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  std::vector<std::string> initializers;
  for (const auto& index : graph_nodes_index.first) {
    sub_graph->Nodes().push_back(node_index[index]);
    const auto& node = graph.GetNode(node_index[index]);
    for (const auto& input : node->InputDefs()) {
      if (graph.IsConstantInitializer(input->Name(), true)) {
        initializers.push_back(input->Name());
        continue;
      }
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
      if (graph.IsConstantInitializer(input->Name(), true)) {
        initializers.push_back(input->Name());
        continue;
      }
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
        const onnxruntime::NodeArg* output;
        // The dst_arg_index from GetDstArgIndex() could be the index for explicit/implicit input defs of the node.
        // We need to get the correct input index accordingly. (See Graph::BuildConnections() in graph.cc for more details)
        if (it->GetDstArgIndex() < static_cast<int>(it->GetNode().InputDefs().size())) {
          output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];
        } else {
          output = (it->GetNode()).ImplicitInputDefs()[it->GetDstArgIndex() - static_cast<int>(it->GetNode().InputDefs().size())];
        }
        if (node_set.find(node_idx) != node_set.end()) {
          const auto& iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
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
          if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
            graph_outputs_to_add[output] = output_order;
          }
          fused_outputs[output] = output_order++;
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  std::multimap<int, const NodeArg*> inputs, outputs;

  // Get the input order of the original graph
  int order = 0;
  for (const auto* input : graph.GetInputs()) {
    original_inputs[input] = order++;
  }

  // input order needs to be consistent with original graph's input order
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    const auto& iter = original_inputs.find(it->first);
    if (iter != original_inputs.end()) {
      inputs.insert(std::pair<int, const NodeArg*>(iter->second, iter->first));
    } else {
      inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
    }
  }

  // Sort outputs by the order they were added
  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Generate unique kernel name for TRT subgraph
  std::string subgraph_id = std::to_string(model_hash) + "_" + std::to_string(subgraph_index);
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name() = "TRTKernel_" + graph_type + "_" + graph.Name() + "_" + subgraph_id;
  LOGS_DEFAULT(INFO) << "[TensorRT EP] TensorRT subgraph MetaDef name " + meta_def->name();

  // Assign inputs and outputs to subgraph's meta_def
  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& initializer : initializers) {
    meta_def->constant_initializers().push_back(initializer);
  }

  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      meta_def->outputs().push_back(output.second->Name());
    }
  }

  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                                                 const GraphViewer& graph, bool* early_termination) const {
  // Return if iterations are exceeding predefined number
  SubGraphCollection_t nodes_list_output;
  if (iterations > max_iterations) {
    *early_termination = true;
    return nodes_list_output;
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  iterations++;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
  for (const auto& group : nodes_vector_input) {
    // Construct subgraph
    if (!group.first.empty()) {
      if (group.second) {
        nodes_list_output.push_back(group);
      } else {
        auto model_build = graph.CreateModel(*GetLogger());
        auto& graph_build = model_build->MainGraph();
        bool has_control_flow_op = false;

        // Add node and node args
        // If node output is also parent graph output, the output will be added to the
        // subgraph's output list
        std::vector<std::string> subgraph_output_names;
        for (const auto& index : group.first) {
          // Initializers that refer to a memory location in OrtValue
          // can not be handled by TRT (unlike those that are on disk).
          // This prevents us from sharing the data and we have to make a copy here.
          constexpr const bool load_initializers_inline_true = true;
          const auto& node = graph.GetNode(node_index[index]);
          std::vector<onnxruntime::NodeArg*> inputs, outputs;
          for (auto input : node->InputDefs()) {
            auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
            graph_utils::MakeInitializerCopyIfNotExist(graph.GetGraph(), graph_build, input->Name(),
                                                       load_initializers_inline_true);
          }

          for (auto input : node->ImplicitInputDefs()) {
            graph_utils::MakeInitializerCopyIfNotExist(graph.GetGraph(), graph_build, input->Name(),
                                                       load_initializers_inline_true);
          }
          for (auto output : node->OutputDefs()) {
            auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
            const auto name = output->Name();
            if (graph_output_names.find(name) != graph_output_names.end()) {
              subgraph_output_names.push_back(name);
            }
          }

          if (control_flow_op_set_.find(node->OpType()) != control_flow_op_set_.end()) {
            has_control_flow_op = true;
          }

          // If the node has subgraph, it's possible that the ORT graph of that subgraph and the GraphProto in the node attributes are not in sync because of graph optimization.
          // Therefore, we need to force GraphProto attributes to be updated in order to get the valid GraphProto.
          if (node->GetAttributes().size() > 0) {
            auto node_proto = ONNX_NAMESPACE::NodeProto::Create();
            // we need to update any GraphProto attributes for subgraphs so that any changes made by things
            // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
            node->ToProto(*node_proto, /* update_subgraphs */ true);
            const int num_attributes = node_proto->attribute_size();
            auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
            node_attributes->reserve(num_attributes);

            for (int i = 0; i < num_attributes; ++i) {
              auto& attr = node_proto->attribute(i);
              node_attributes->emplace(attr.name(), attr);
            }

            // The GraphProto attributes are the updated ones.
            graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, node_attributes.get(), node->Domain());
          } else {
            // The GraphProto attributes are the original ones.
            graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
          }
        }

        // Only if the newly built graph has control flow op as well as it has parent node,
        // it needs to handle outer scope values before calling graph.Resolve().
        if (has_control_flow_op && graph.ParentNode()) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Handle outer scope values for the subgraph " << graph_build.Name();
          BuildSubGraphContext(graph_build);
          SetGraphOuterScopeValuesAndInputs(graph_build, graph.GetGraph());
          SetAllGraphInputs(graph_build);
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Add parent graph output to the subgraph
        int i = 0;
        std::vector<const NodeArg*> subgraph_outputs;
        subgraph_outputs.resize(subgraph_output_names.size());
        for (auto& name : subgraph_output_names) {
          auto output_arg = graph.GetNodeArg(name);
          auto& subgraph_output_arg = graph_build.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
          subgraph_outputs[i] = &subgraph_output_arg;
          ++i;
        }
        auto& graph_build_outputs = graph_build.GetOutputs();
        subgraph_outputs.insert(subgraph_outputs.begin(), graph_build_outputs.begin(), graph_build_outputs.end());
        graph_build.SetOutputs(graph_build_outputs);
        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Check if input tensors have shapes
        if (iterations > 1) {
          auto graph_inputs = graph_build.GetInputs();
          for (auto input_arg : graph_inputs) {
            bool has_dim_value_or_param = true;
            auto input_shape = input_arg->Shape();
            if (input_shape != nullptr) {
              auto dim_size = input_shape->dim_size();
              for (int i = 0; i < dim_size; ++i) {
                auto& dim = input_shape->dim(i);
                if (!dim.has_dim_value() && !dim.has_dim_param()) {
                  has_dim_value_or_param = false;
                  break;
                }
              }
            }

            if (input_shape == nullptr || !has_dim_value_or_param) {
              ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                                 "TensorRT input: " + input_arg->Name() + " has no shape specified. " +
                                                     "Please run shape inference on the onnx model first. Details can be found in " +
                                                     "https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#shape-inference-for-tensorrt-subgraphs"));
            }
          }
        }

        // Serialize modelproto to string
        auto graph_viewer = graph_build.CreateGraphViewer();
        auto model = graph_viewer->CreateModel(*GetLogger());
        auto model_proto = model->ToProto();

        // ORT's default topological sort is using reversed DFS.
        // When creating model proto from graph viewer, let ORT use priority-based topological sort based on node index.
        // The reason is, in some cases, for example ResNet50, using default topological sort will end up with generating
        // the model proto that has different node ordering compared to original onnx model.
        // Save Initializer Data.

        std::vector<TensorrtUserWeights> userWeights;

        // Keep inits in memory instead of writing to ModelProto.
        if (load_user_initializer_) {
          auto allInitializers = graph_viewer->GetAllInitializedTensors();

          for (auto& entry : allInitializers) {
            auto* tp = entry.second;
            if (tp->has_raw_data()) {
              userWeights.emplace_back(tp->name(), tp->raw_data());
            } else if (utils::HasExternalDataInMemory(*tp)) {
              std::unique_ptr<ONNX_NAMESPACE::TensorProto> full_init;
              ORT_THROW_IF_ERROR(utils::GetTensorProtoWithDataIfInMemory(*tp, full_init));
              userWeights.emplace_back(full_init->name(), full_init->raw_data());
            }
          }
        }

        graph_viewer->ToProto(*model_proto->mutable_graph(), true, true, 1 /*priority-based topological sort*/, !load_user_initializer_ /*include_initializer_data*/);
        model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

        std::string string_buf;
        model_proto->SerializeToString(string_buf);

        if (dump_subgraphs_) {
          // Dump TensorRT subgraph for debugging
          std::fstream dump("TensorrtExecutionProvider_TRT_Subgraph.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
          model_proto->SerializeToOstream(dump);
        }

        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log_);
        auto trt_builder = GetBuilder(trt_logger);
        auto network_flags = 0;
#if NV_TENSORRT_MAJOR > 8
        network_flags |= (fp16_enable_ || int8_enable_ || bf16_enable_) ? 0 : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
        network_flags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif

        auto trt_network = std::unique_ptr<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
        auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        bool is_model_supported = false;

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 1) || NV_TENSORRT_MAJOR > 10
#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
        if (load_user_initializer_) {
          trt_parser->loadModelProto(string_buf.data(), string_buf.size(), model_path_);
          for (auto const& userWeight : userWeights) {
            trt_parser->loadInitializer(userWeight.Name(), userWeight.Data(), userWeight.Size());
          }
          is_model_supported = trt_parser->parseModelProto();
        } else {
          is_model_supported = trt_parser->supportsModelV2(string_buf.data(), string_buf.size(), model_path_);
        }
#else
        is_model_supported = trt_parser->supportsModelV2(string_buf.data(), string_buf.size(), model_path_);
#endif  // (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10

        // Note: Calling getNbSubgraphs or getSubgraphNodes before calling supportsModelV2 results in undefined behavior.
        auto num_subgraphs = trt_parser->getNbSubgraphs();
        parser_nodes_list.reserve(num_subgraphs);

        for (int64_t i = 0; i < num_subgraphs; ++i) {
          int64_t subgraph_len = 0;
          int64_t* nodes = trt_parser->getSubgraphNodes(i, subgraph_len);
          parser_nodes_list.emplace_back();
          parser_nodes_list.back().first.reserve(subgraph_len);
          for (int64_t j = 0; j < subgraph_len; ++j) {
            parser_nodes_list.back().first.push_back(nodes[j]);
          }
          parser_nodes_list.back().second = is_model_supported ? true : false;
        }
#else
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list, model_path_);
#endif  // (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 1) || NV_TENSORRT_MAJOR > 10

        SubGraphCollection_t next_nodes_list;
        const std::vector<NodeIndex>& subgraph_node_index = graph_viewer->GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, *graph_viewer, early_termination);
        for (size_t i = 0, end = next_nodes_list.size(); i < end; ++i) {
          for (size_t j = 0, end = next_nodes_list[i].first.size(); j < end; ++j) {
            /*
             * Convert the supported node list returning from onnx-tensorrt parser to the node list recognized by ORT TRT.
             *
             * TRT EP reconstructs the graph based on the nodes in group.first and feeds this graph (converts to model proto and to string buffer) to onnx-tensorrt parser.
             * The node index in the list returning from onnx-tensorrt parser might not be the same as the node index in group.first. Therefore, TRT EP needs a node index mapping table here.
             *
             * The order of iterating the nodes in group.first and calling graph_build.AddNode() determines the node order in the newly constructed graph (see Graph::AllocateNode() in graph.cc),
             * however, once the graph is converted to model proto, the node proto order in model proto (ex: onnx-tensorrt calls model.graph().node() to iterate NodeProto in ModelProto) is decided by topo sort.
             *
             * The topo sort list (i.e. subgraph_node_index) acts as the node index mapping table:
             * subgraph_node_index[node index from onnx-tensorrt parser] = index in group.first
             *
             * In the past, TRT EP uses ORT's default reversed DFS topo sort which might end up with the sorting result not sequence of 0, 1, ... n-1, ex: the subgraph_node_index = [0,2,1,3,4].
             * With the change of using ORT's priority-based topo sort (node with lower node index outputs first) the sorting result is the sequence of 0, 1, ... n-1 for most of the cases,
             * therefore subgraph_node_index as a mapping table is not needed anymore.
             *
             * TODO: Remove the subgraph_node_index
             */
            next_nodes_list[i].first[j] = group.first[subgraph_node_index[next_nodes_list[i].first[j]]];
          }
          nodes_list_output.push_back(next_nodes_list[i]);
        }
      }
    }
  }
  return nodes_list_output;
}

// Detect and remove cycles from supported node list
bool TensorrtExecutionProvider::DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const GraphViewer& graph, const HashValue& model_hash, bool remove_cycles) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
  bool trt_cycle = true, cycle_detected = false;
  while (trt_cycle) {
    trt_cycle = false;
    std::unordered_map<std::string, size_t> node_to_index_map;
    std::unordered_map<size_t, std::string> index_to_node_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> input_to_nodes_map, node_to_outputs_map;
    std::unordered_set<size_t> non_trt_node_index(node_index.begin(), node_index.end());
    size_t id = 0;
    int subgraph_index = 0;
    for (const auto& group : supported_nodes_vector) {
      if (!group.first.empty()) {
        // Construct subgraph from node list
        std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, graph, model_hash, subgraph_index);

        // Create node to inputs/outputs/index maps
        const auto& meta_def = sub_graph->GetMetaDef();
        const std::string node_name = meta_def->name();
        if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
          index_to_node_map[id] = node_name;
          node_to_index_map[node_name] = id++;
        }

        if (meta_def != nullptr) {
          for (const auto& input : meta_def->inputs()) {
            input_to_nodes_map[input].insert(node_name);
          }
          for (const auto& output : meta_def->outputs()) {
            node_to_outputs_map[node_name].insert(output);
          }
        }

        // Remove TensorRT nodes from node index list
        for (const auto& index : group.first) {
          non_trt_node_index.erase(node_index[index]);
        }
        subgraph_index++;
      }
    }

    // Add non TensorRT nodes to the maps
    for (const auto& index : non_trt_node_index) {
      const auto& node = graph.GetNode(index);
      const std::string node_name = node->Name();
      if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
        index_to_node_map[id] = node_name;
        node_to_index_map[node_name] = id++;
      }

      for (const auto& input : node->InputDefs()) {
        input_to_nodes_map[input->Name()].insert(node_name);
      }

      for (const auto& input : node->ImplicitInputDefs()) {
        input_to_nodes_map[input->Name()].insert(node_name);
      }

      for (const auto& output : node->OutputDefs()) {
        node_to_outputs_map[node_name].insert(output->Name());
      }
    }

    // Create adjacency list
    size_t graph_size = node_to_index_map.size();
    std::vector<InlinedVector<size_t>> adjacency_map(graph_size);
    for (const auto& node : node_to_outputs_map) {
      for (auto iter = node.second.begin(); iter != node.second.end(); ++iter) {
        const auto& loc = input_to_nodes_map.find(*iter);
        if (loc != input_to_nodes_map.end()) {
          size_t parent_node_index = node_to_index_map.find(node.first)->second;
          for (auto child_node : loc->second) {
            size_t child_node_index = node_to_index_map.find(child_node)->second;
            adjacency_map[parent_node_index].push_back(child_node_index);
          }
        }
      }
    }

    // Check cycle in the graph
    InlinedVector<bool> visited(graph_size);
    InlinedVector<bool> st(graph_size);
    for (size_t i = 0; i < graph_size; ++i) {
      visited[i] = false;
      st[i] = false;
    }

    InlinedVector<size_t> cycles;
    bool has_cycle = false;
    for (size_t i = 0; i < graph_size; ++i) {
      if (FindCycleHelper(i, adjacency_map, visited, st, cycles)) {
        has_cycle = true;
        cycle_detected = true;
        break;
      }
    }

    // Remove TensorRT subgraph from the supported node list if it's part of the cycle
    if (has_cycle && remove_cycles) {
      for (size_t i = 0; i < cycles.size(); ++i) {
        auto loc = index_to_node_map.find(cycles[i]);
        if (loc != index_to_node_map.end() && loc->second.find("TRTKernel") != std::string::npos) {
          supported_nodes_vector.erase(supported_nodes_vector.begin() + cycles[i]);
          trt_cycle = true;
          break;
        }
      }
    }
  }
  return cycle_detected;
}

std::vector<std::unique_ptr<ComputeCapability>>
TensorrtExecutionProvider::GetCapability(const GraphViewer& graph,
                                         const IKernelLookup& /*kernel_lookup*/,
                                         const GraphOptimizerRegistry& graph_optimizer_registry,
                                         IResourceAccountant* /* resource_accountant */) const {
  // Construct subgraph capability from node list
  std::vector<std::unique_ptr<ComputeCapability>> result;
  // Get ModelPath
  const auto& path_string = graph.ModelPath().string();
#ifdef _WIN32
  strncpy_s(model_path_, path_string.c_str(), sizeof(model_path_) - 1);
#else
  strncpy(model_path_, path_string.c_str(), sizeof(model_path_) - 1);
#endif
  model_path_[sizeof(model_path_) - 1] = '\0';

  // If the model consists of only a single "EPContext" contrib op, it means TRT EP can fetch the precompiled engine info from the node and
  // load the engine directly without having to go through the processes of graph proto reconstruction, calling TRT parser and engine compilation.
  // So, simply return the ComputeCapability here.
  if (graph.NumberOfNodes() == 1 && GraphHasCtxNode(graph)) {
    SubGraph_t supported_node_vector = {{0}, true};
    std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(supported_node_vector, graph, TRTGenerateId(graph, std::to_string(trt_version_), std::to_string(cuda_version_)), 0);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    return result;
  }

  // Generate unique kernel name for TRT graph
  HashValue model_hash = TRTGenerateId(graph, std::to_string(trt_version_), std::to_string(cuda_version_));

  // Get supported node list from TensorRT parser
  const int number_of_ort_nodes = graph.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);

  auto get_exclude_ops_set = [&](std::string node_list_to_exclude) -> std::set<std::string> {
    std::set<std::string> set;
    if (!node_list_to_exclude.empty()) {
      std::stringstream node_list(node_list_to_exclude);
      std::string node;
      while (std::getline(node_list, node, ',')) {
        set.insert(node);
      }
    }
    return set;
  };

  auto exclude_ops_set = get_exclude_ops_set(op_types_to_exclude_);

  SubGraphCollection_t parser_nodes_vector, supported_nodes_vector;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
  bool new_subgraph = true;

  /* Iterate all the nodes and exclude the node if:
   *   1. It's a control flow op and its subgraph(s) is not fully TRT eligible.
   *   2. It's a DDS op.
   */
  for (const auto& index : nodes_vector) {
    const auto& node = graph.GetNode(node_index[index]);
    bool supported_node = true;

    /* If current node is control flow op, we take different approach based on following four cases:
     *
     * (1) control flow op is supported by TRT, and its subgraphs are all supported by TRT. Assign this node to TRT.
     * (2) control flow op is supported by TRT, but not all its subgraphs supported by TRT. Don't assign this node to TRT.
     * (3) control flow op is not supported by TRT, but its subgraphs all supported by TRT. Don't assign this node to TRT.
     * (4) control flow op is not supported by TRT, and not all its subgraphs supported by TRT. Don't assign this node to TRT.
     *
     * For cases 2, 3, 4, even though the control flow op is not assigned to TRT, any portion of its subgraphs that can run in TRT will be still fused and assigned to TRT EP.
     */
    if (control_flow_op_set_.find(node->OpType()) != control_flow_op_set_.end()) {
      auto supported_control_flow_op = [&](const Node* node) {
        auto sub_graphs = node->GetSubgraphs();
        if (sub_graphs.size() != 0) {
          for (auto sub_graph : sub_graphs) {
            // TRT EP should consider the empty subgraph is fully supported by TRT.
            if (sub_graph->CreateGraphViewer()->NumberOfNodes() == 0) {
              continue;
            }
            if (!AllNodesAssignedToSpecificEP(*(sub_graph->CreateGraphViewer()), kTensorrtExecutionProvider)) {
              // if not all its subgraphs are supported, we need to exclude this control flow op
              return false;
            }
          }
        }
        return true;
      };
      supported_node = supported_control_flow_op(node);
    }

    // Exclude any ops, if applicable
    if (exclude_ops_set.find(node->OpType()) != exclude_ops_set.end()) {
      supported_node = false;
    }

    if (supported_node) {
      if (new_subgraph) {
        parser_nodes_vector.emplace_back();
        // Mark all new graphs as "UnKnown" which will later be parsed by TRT parser
        parser_nodes_vector.back().second = false;
        new_subgraph = false;
      }
      parser_nodes_vector.back().first.emplace_back(index);
    } else {
      new_subgraph = true;
    }
  }

  bool early_termination = false;
  supported_nodes_vector = GetSupportedList(parser_nodes_vector, 0, max_partition_iterations_, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

  for (auto it = supported_nodes_vector.begin(); it != supported_nodes_vector.end();) {
    const size_t subgraph_size = it->first.size();
    if (subgraph_size < min_subgraph_size_) {
      it = supported_nodes_vector.erase(it);
    } else {
      ++it;
    }
  }

  // Detect and remove cycles from supported node list
  DetectTensorRTGraphCycles(supported_nodes_vector, graph, model_hash);

  // Consolidate supported node list
  if (supported_nodes_vector.size() > 1) {
    nodes_vector.clear();
    for (const auto& group : supported_nodes_vector) {
      if (!group.first.empty()) {
        nodes_vector.insert(nodes_vector.end(), group.first.begin(), group.first.end());
      }
    }
    SubGraphCollection_t consolidated_supported_nodes_vector = {{nodes_vector, true}};
    if (DetectTensorRTGraphCycles(consolidated_supported_nodes_vector, graph, model_hash, false)) {
      LOGS_DEFAULT(INFO) << "[TensorRT EP] TensorRT nodes are not consolidated because graph will have cycles after consolidation";
    } else {
      LOGS_DEFAULT(INFO) << "[TensorRT EP] TensorRT nodes are consolidated into one subgraph";
      supported_nodes_vector = consolidated_supported_nodes_vector;
    }
  }

  // Handle the case where the graph is subgraph of control flow op.
  // The purpose is to make control flow op as well as its subgraphs run on TRT.
  // Here we need to check whether subgraph is fully supported by TRT and don't fuse the nodes of the subgraph until control flow op level.
  if (IsSubGraphOfControlFlowOp(graph) && IsSubGraphFullySupported(supported_nodes_vector, number_of_ort_nodes)) {
    const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1 /*priority-based topological sort*/);
    bool all_subgraphs_are_supported = true;

    // "If" control flow op has two subgraph bodies, "then" body and "else" body respectively.
    // Check its parent node's another subgraph to see whether that subgraph is also fully supported by TRT.
    if (graph.ParentNode()->OpType() == "If") {
      all_subgraphs_are_supported = false;
      SubGraphCollection_t subgraph_supported_nodes_vector;
      auto sub_graphs = graph.ParentNode()->GetSubgraphs();
      for (auto sub_graph : sub_graphs) {
        if (sub_graph.get() != &graph.GetGraph()) {
          auto sub_graph_viewer = sub_graph->CreateGraphViewer();
          const int number_of_ort_subgraph_nodes = sub_graph_viewer->NumberOfNodes();
          std::vector<size_t> subgraph_nodes_vector(number_of_ort_subgraph_nodes);
          std::iota(std::begin(subgraph_nodes_vector), std::end(subgraph_nodes_vector), 0);
          SubGraphCollection_t parser_subgraph_nodes_vector = {{subgraph_nodes_vector, false}};
          bool subgraph_early_termination = false;

          // Another subgraph of "If" control flow op has no nodes.
          // In this case, TRT EP should consider this empty subgraph is fully supported by TRT.
          if (sub_graph_viewer->NumberOfNodes() == 0) {
            all_subgraphs_are_supported = true;
            break;
          }
          // Another subgraph of "If" control flow op has been parsed by GetCapability before and all subgraph's nodes assigned to TRT EP.
          else if (AllNodesAssignedToSpecificEP(*sub_graph_viewer, kTensorrtExecutionProvider)) {
            all_subgraphs_are_supported = true;
            break;
          }
          // Another subgraph of "If" control flow has been parsed by GetCapability and not all subgraph's nodes assigned to TRT EP.
          // (Note: GetExecutionProviderType() returns "" meaning node has not yet been assigned to any EPs)
          else if (!AllNodesAssignedToSpecificEP(*sub_graph_viewer, "")) {
            all_subgraphs_are_supported = false;
            break;
          }

          // Another subgraph of "If" control flow has not yet been parsed by GetCapability.
          subgraph_supported_nodes_vector = GetSupportedList(parser_subgraph_nodes_vector, 0, max_partition_iterations_, *sub_graph_viewer, &subgraph_early_termination);
          all_subgraphs_are_supported = IsSubGraphFullySupported(subgraph_supported_nodes_vector, number_of_ort_subgraph_nodes);
          break;
        }
      }
    }

    if (all_subgraphs_are_supported) {
      // We want the subgraph nodes to be assigned to TRT EP but don't want them to be fused until later at the control flow op level.
      // Simply request the subgraph nodes with a single ComputeCapability for each with no MetaDef (i.e. what the default implementation for IExecutionProvider::GetCapability does).
      for (const auto& group : supported_nodes_vector) {
        if (!group.first.empty()) {
          for (const auto& index : group.first) {
            std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
            sub_graph->Nodes().push_back(node_index[index]);
            result.push_back(ComputeCapability::Create(std::move(sub_graph)));
          }
        }
      }
      LOGS_DEFAULT(INFO) << "[TensorRT EP] Whole graph will run on TensorRT execution provider";

      // The context map is only used during EP compile time, release it to save memory space.
      subgraph_context_map_.clear();
      return result;
    }
  }

  /**
   * Enable EP related L2+ graph optimizations:
   *
   * 1. Calls provider bridge API to lookup pre-defined optimizer by name and get selection function.
   *    - Example: g_host->GetOptimizerByName(optimizer_name, graph_optimizer_registry, selection_func)
   * 2. Executes the selection function to obtain the selection ComputeCapability.
   *    - ComputeCapability.optimize_func would be set by the optimizer to the function that does the optimization.
   * 3. Uses the selection ComputeCapability to create the optimization ComputeCapability.
   * 4. Returns the final ComputeCapability, with nodes_to_optimize set to the optimization ComputeCapability.
   *
   * Current available optimizations:
   *   - (ConstantFoldingDQ) constant folding on DQ nodes, i.e. dequantize INT32, UINT16, INT16 constant to FP32.
   */

  SelectionFunc selection_func;
  std::vector<std::unique_ptr<ComputeCapability>> selection_cc;

  // Prepare for ConstantFoldingDQ optimizer
  // Note: The NodeIndex here is the node index in the graph, not the index in node vector in supported_nodes_vector.
  std::unordered_set<NodeIndex> trt_selection_node_set;     // The qualified dq nodes selected by TRT EP
  std::unordered_map<NodeIndex, NodeIndex> consumer_to_dq;  // consumer node -> dq node

  if (dla_enable_) {
    std::string optimizer_name = "ConstantFoldingDQ";
    const std::unordered_map<std::string, std::string> key_value_config;
    auto status = g_host->GetOptimizerByName(optimizer_name, graph_optimizer_registry, selection_func);
    if (status == Status::OK()) {
      if (selection_func) {
        selection_cc = selection_func(graph, key_value_config, graph_optimizer_registry);
        SelectQualifiedDQNode(graph, trt_selection_node_set, consumer_to_dq);
      }
    } else {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Can't get optimizer " << optimizer_name;
    }
  }

  // Create ComputeCapability
  int number_of_trt_nodes = 0, subgraph_index = 0;
  for (auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      if (!selection_cc.empty()) {
        // Include DQ nodes that are filtered out by TRT parser
        UpdateSupportedNodeVectorForDQ(graph, group, supported_nodes_vector, consumer_to_dq);
      }

      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, graph, model_hash, subgraph_index);
      auto compute_capability = ComputeCapability::Create(std::move(sub_graph));

      // add optimization ComputeCapability to node_to_optimize
      for (auto& cc : selection_cc) {
        std::unique_ptr<ComputeCapability> optimization_cc = CreateOptimizationComputeCapability(cc.get(), trt_selection_node_set, compute_capability.get());
        compute_capability->add_nodes_to_optimize(std::move(optimization_cc));
      }

      result.push_back(std::move(compute_capability));
      number_of_trt_nodes += static_cast<int>(group.first.size());
      subgraph_index++;
    }
  }

  const size_t number_of_subgraphs = supported_nodes_vector.size();
  if (number_of_trt_nodes == 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] No graph will run on TensorRT execution provider";
  } else if (number_of_trt_nodes == number_of_ort_nodes) {
    LOGS_DEFAULT(INFO) << "[TensorRT EP] Whole graph will run on TensorRT execution provider";
  } else {
    LOGS_DEFAULT(INFO) << "[TensorRT EP] Graph is partitioned and number of subgraphs running on TensorRT execution provider is " << number_of_subgraphs;
  }

  // The context map is only used during EP compile time, release it to save memory space.
  subgraph_context_map_.clear();
  return result;
}

/**
 * Refit the weight-stripped engine
 */
common::Status TensorrtExecutionProvider::RefitEngine(std::string onnx_model_filename,
                                                      std::string& onnx_model_folder_path,
                                                      std::string& weight_stripped_engine_cath_path,
                                                      bool path_check,
                                                      const void* onnx_model_bytestream,
                                                      size_t onnx_model_bytestream_size,
                                                      const void* onnx_external_data_bytestream,
                                                      size_t onnx_external_data_bytestream_size,
                                                      nvinfer1::ICudaEngine* trt_engine,
                                                      bool serialize_refitted_engine,
                                                      bool detailed_build_log) {
#if NV_TENSORRT_MAJOR >= 10
  bool refit_from_file = onnx_model_bytestream == nullptr && onnx_model_bytestream_size == 0;
  bool refit_with_external_data = onnx_external_data_bytestream != nullptr && onnx_external_data_bytestream_size != 0;
  bool refit_complete = false;
  std::filesystem::path onnx_model_path{onnx_model_folder_path};
  if (refit_from_file) {
    if (!onnx_model_filename.empty()) {
      onnx_model_path.append(onnx_model_filename);
    }
    if (onnx_model_path.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "The ONNX model was not provided as path. "
                             "Please use provide an ONNX bytestream to enable refitting the weightless engine."
                             "When providing a bytestream during session initialization, it should also be set as trt_onnx_bytes_stream");
    } else {
      // check if file path to ONNX is legal
      if (path_check && IsAbsolutePath(onnx_model_path.string())) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "For security purpose, the ONNX model path should be set with "
                               "a relative path, but it is an absolute path: " +
                                   onnx_model_path.string());
      }
      if (path_check && IsRelativePathToParentPath(onnx_model_path.string())) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "The ONNX model path has '..'. For security purpose, it's not "
                               "allowed to point outside the directory.");
      }

      if (!(std::filesystem::exists(onnx_model_path) && std::filesystem::is_regular_file(onnx_model_path))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "The ONNX model " + onnx_model_path.string() +
                                   " does not exist.");
      }
    }
  }

  // weight-stripped engine refit logic
  TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log);
  auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*trt_engine, trt_logger));
  auto parser_refitter = std::unique_ptr<nvonnxparser::IParserRefitter>(
      nvonnxparser::createParserRefitter(*refitter, trt_logger));

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
  // New refit APIs
  if (refit_with_external_data) {
    // A valid model bytestream must be passed.
    if (refit_from_file) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP's refit with external data must be called with a valid ONNX model bytestream");
    }

    if (!parser_refitter->loadModelProto(onnx_model_bytestream, onnx_model_bytestream_size, nullptr)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP's IParserRefitter could not load model from provided onnx_model_bytestream");
    }

    // Extract weight information from the Refitter.
    int required_weights = refitter->getAllWeights(0, nullptr);
    std::vector<char const*> refit_names(required_weights);
    refitter->getAllWeights(required_weights, refit_names.data());
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Refitter requires " << required_weights << " weights";

    // Vectors to keep track of data pointers.
    std::vector<std::string> names;
    names.reserve(required_weights);
    std::vector<const char*> bytes;
    bytes.reserve(required_weights);
    std::vector<int64_t> sizes;
    sizes.reserve(required_weights);

    auto onnx_model = ModelProto::Create();
    TensorProtos* allInitializers_byte_stream;

    // Reconstruct onnx model view.
    const auto onnx_model_view = std::string((const char*)onnx_model_bytestream,
                                             onnx_model_bytestream_size);
    if (!onnx_model->ParseFromString(onnx_model_view)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "The provided ONNX bytestream to refit could not be parsed.");
    }

    // Extract graph and initializer information.
    auto const& graph = onnx_model->mutable_graph();
    allInitializers_byte_stream = graph->mutable_initializer();
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Initializers that were found " << allInitializers_byte_stream->size();

    // Loop through all initializers
    int missing_initializer_data = 0;
    for (int initializer_idx = 0; initializer_idx < allInitializers_byte_stream->size(); ++initializer_idx) {
      auto& proto = allInitializers_byte_stream->at(initializer_idx);
      auto& proto_name = proto.name();
      bool weight_is_refittable = std::find(refit_names.begin(), refit_names.end(), proto_name) != refit_names.end();
      if (weight_is_refittable) {
        if (proto.has_data_location()) {
          if (proto.data_location() == TensorProto_DataLocation_EXTERNAL) {
            // Default values for reading into external_data blob.
            int64_t offset = 0;
            size_t length = 0;
            auto external_data = proto.mutable_external_data();
            const std::string kOffset = "offset", kLength = "length";
            for (int entry_idx = 0; entry_idx < external_data->size(); ++entry_idx) {
              auto current_key = external_data->at(entry_idx).mutable_key();
              auto current_value = external_data->at(entry_idx).mutable_value();
              if (*current_key == kOffset && !current_value->empty()) {
                offset = std::stoll(*current_value);
              } else if (*current_key == kLength && !current_value->empty()) {
                length = std::stoul(*current_value);
              }
            }
            names.push_back(proto.name());
            bytes.push_back(static_cast<const char*>(onnx_external_data_bytestream) + offset);
            sizes.push_back(length);
          } else {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "[TensorRT EP] Proto: " + proto_name + " expected to have external datalocation, but default datalocation was provided instead.");
          }
        } else if (proto.has_raw_data()) {
          auto& raw_data = proto.raw_data();
          names.push_back(proto.name());
          bytes.push_back(raw_data.c_str());
          sizes.push_back(raw_data.size());
        } else {
          LOGS_DEFAULT(WARNING) << "[TensorRT EP] Proto: " + proto_name + " has no raw nor external data.";
          ++missing_initializer_data;
        }
      } else {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Initializer with name: " << proto_name << " was not marked as refittable";
      }
    }
    if (missing_initializer_data) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "[TensorRT EP] RefitEngine is missing " + std::to_string(missing_initializer_data) + " initializers.");
    }

    // Load extracted initializers into the parser
    if (!names.empty()) {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Number of initializers submitted to refitter " << names.size();
      for (size_t i = 0; i < names.size(); i++) {
        bool refloadInit = parser_refitter->loadInitializer(names[i].c_str(), bytes[i], sizes[i]);
        if (!refloadInit) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in the provided bytestream");
        }
      }
    }
    // Perform refit.
    if (!parser_refitter->refitModelProto()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP's IParserRefitter refitModelProto() failed with the provided external data bytestream.");
    }
    refit_complete = true;
  }
#else
  // Refitting with external data is not supported prior to TensorRT 10.13. Log a warning in this case for the user.
  if (refit_with_external_data) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Refitting with an onnx_external_data_bytestream is only supported on TensorRT versions >= 10.13! This parameter will be ignored for refitting, and the resulting refitted engine may be incorrect.";
  }
#endif  // (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
  // If new refit flow was not completed, then fallback to refit_from_file.
  if (!refit_complete) {
    if (refit_from_file) {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Refitting from file on disk: " << onnx_model_path.string();
      if (!parser_refitter->refitFromFile(onnx_model_path.string().c_str())) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in: " + onnx_model_path.string());
      }
    } else {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Refitting from byte array";
      if (!parser_refitter->refitFromBytes(onnx_model_bytestream, onnx_model_bytestream_size)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in the provided bytestream");
      }
    }
  }
  if (refitter->refitCudaEngine()) {
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Successfully refitted the weight-stripped engine.";
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                           "TensorRT EP's IRefitter could not refit deserialized weight-stripped engine with weights contained in: " + onnx_model_path.string());
  }

  // serialize the refitted engine to disk
  if (serialize_refitted_engine) {
    std::string refitted_engine_cache = GetWeightRefittedEnginePath(weight_stripped_engine_cath_path);
    nvinfer1::IHostMemory* serialized_engine = trt_engine->serialize();
    std::ofstream engine_file(refitted_engine_cache, std::ios::binary | std::ios::out);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialize the refitted engine to " << refitted_engine_cache;
  }
  return Status::OK();
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP's IParserRefitter can only be used on TRT 10.0 onwards.");
#endif
}

common::Status TensorrtExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;
    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, size_t> input_map;
    const auto& input_defs = fused_node.InputDefs();
    input_map.reserve(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, size_t> output_map;
    const auto& output_defs = fused_node.OutputDefs();
    output_map.reserve(output_defs.size());
    for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    Status status;
    if (GraphHasCtxNode(graph_body_viewer)) {
      status = CreateNodeComputeInfoFromPrecompiledEngine(graph_body_viewer,
                                                          fused_node,
                                                          input_map,
                                                          output_map,
                                                          node_compute_funcs);
    } else {
      status = CreateNodeComputeInfoFromGraph(graph_body_viewer, fused_node, input_map, output_map, node_compute_funcs);
    }
    if (status != Status::OK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, status.ErrorMessage());
    }
  }
  return Status::OK();
}

Status TensorrtExecutionProvider::CreateNodeComputeInfoFromGraph(const GraphViewer& graph_body_viewer,
                                                                 const Node& fused_node,
                                                                 std::unordered_map<std::string, size_t>& input_map,
                                                                 std::unordered_map<std::string, size_t>& output_map,
                                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Reconstruct graph proto from fused node's function body
  auto model = graph_body_viewer.CreateModel(*GetLogger());
  auto model_proto = model->ToProto();

  auto userWeights = std::make_unique<std::vector<TensorrtUserWeights>>();

  if (load_user_initializer_) {
    auto allInitializers = graph_body_viewer.GetAllInitializedTensors();

    for (auto& entry : allInitializers) {
      auto name = entry.first;
      auto* tp = entry.second;
      if (tp->has_raw_data()) {
        userWeights->emplace_back(
            TensorrtUserWeights(tp->name(), tp->raw_data()));
      } else if (utils::HasExternalDataInMemory(*tp)) {
        std::unique_ptr<ONNX_NAMESPACE::TensorProto> full_init;
        ORT_THROW_IF_ERROR(utils::GetTensorProtoWithDataIfInMemory(*tp, full_init));
        userWeights->emplace_back(
            TensorrtUserWeights(full_init->name(), full_init->raw_data()));
      }
    }
  }

  // ORT's default topological sort is using reversed DFS.
  // When creating model proto from graph viewer, let ORT use priority-based topological sort based on node index.
  // The reason is, in some cases, for example ResNet50, using default topological sort will end up with generating
  // the model proto that has different node ordering compared to original onnx model.
  graph_body_viewer.ToProto(*model_proto->mutable_graph(), true, true, 1 /*priority-based topological sort*/, !load_user_initializer_ /*include_initializer_data*/);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string string_buf;
  model_proto->SerializeToString(string_buf);

  if (dump_subgraphs_) {
    // Dump TensorRT subgraphs
    std::fstream dump(fused_node.Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }

  TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log_);
  auto trt_builder = GetBuilder(trt_logger);
  auto network_flags = 0;
#if NV_TENSORRT_MAJOR > 8
  network_flags |= (fp16_enable_ || int8_enable_ || bf16_enable_) ? 0 : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
  network_flags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  auto trt_network = std::unique_ptr<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
  auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
  auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
  if (load_user_initializer_) {
    trt_parser->loadModelProto(string_buf.data(), string_buf.size(), model_path_);
    for (auto const& userWeight : *userWeights) {
      trt_parser->loadInitializer(userWeight.Name(), userWeight.Data(), userWeight.Size());
    }
    trt_parser->parseModelProto();
  } else {
    trt_parser->parse(string_buf.data(), string_buf.size(), model_path_);
  }
#else
  trt_parser->parse(string_buf.data(), string_buf.size(), model_path_);
#endif
  if (max_workspace_size_ > 0) {
    trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
  }

  // Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  if ((fp16_enable_ || bf16_enable_) && layer_norm_fp32_fallback_) {
    for (auto idx = 1; idx < trt_network->getNbLayers() - 1; ++idx) {
      auto layer = trt_network->getLayer(idx);
      auto next_layer = trt_network->getLayer(idx + 1);
      if (layer->getType() == nvinfer1::LayerType::kELEMENTWISE && next_layer->getType() == nvinfer1::LayerType::kREDUCE && (static_cast<nvinfer1::IElementWiseLayer*>(layer))->getOperation() == nvinfer1::ElementWiseOperation::kPOW) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow";
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        next_layer->setPrecision(nvinfer1::DataType::kFLOAT);
        layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        next_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
      }
    }
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

  int num_inputs = trt_network->getNbInputs();
  int num_outputs = trt_network->getNbOutputs();
  std::unordered_map<std::string, size_t> input_indexes(num_inputs);
  std::unordered_map<std::string, size_t> output_indexes(num_outputs);
  std::unordered_map<std::string, size_t> output_types(num_outputs);

  /*
   * Initialize shape range for each dynamic shape input tensor:
   *   1) If user explicitly specifies optimization profiles via provider options, TRT EP will create those profiles during EP compile time.
   *      It won't make adjustment for profile values during EP compute time.
   *
   *   2) If no explicit optimization profiles provided by user, TRT EP will firstly set min/max/opt shape to [INT_MAX, INT_MIN, INT_MIN].
   *      Later in EP compute time, the shape will be adjusted to [min_input_value, max_input_value, max_input_value] based on input tensor value.
   *
   *
   * Once the TRT profiles are created:
   *   1) If all the dynamic shape input tensors have associated profiles explicitly provided by user, those profiles will be applied to TRT builder config
   *      and the engine will be built at EP compile time.
   *
   *   2) As long as one of the dynamic shape input tensors has no explicitly associated profile, TRT EP will create default shape as described above,
   *      and all the profiles won't be applied and engine won't be built until EP compute time.
   */
  bool has_dynamic_shape = false;  // True if input tensor has dynamic shape and no explicit profile is specified, otherwise false.
  bool has_explicit_profile = false;
  bool apply_explicit_profile = false;
  int num_profiles = 0;
  std::vector<nvinfer1::IOptimizationProfile*> trt_profiles;

  // Following c++ map data structure is used to help serialize/deserialize profiles where it saves dynamic shape dimension(s) and min/max/opt values for dynamic shape input tensor.
  //
  // (1) Single profile case:
  // For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
  // has one dynamic shape dimension: dim_1. The data will be:
  // {
  //   tensor_a: {
  //              dim_0: [[min_shape, max_shape, opt_shape]],
  //              dim_2: [[min_shape, max_shape, opt_shape]]
  //   },
  //   tensor_b: {
  //              dim_1: [[min_shape, max_shape, opt_shape]]
  //   }
  // }
  //
  // (2) Multiple profiles case:
  // For example, assume tensor_a has one dynamic shap dimension: dim 0, and tensor_b has one dynamic shape dimension: dim_1,
  // and both of the tensors have two profiles. The data will be:
  // {
  //   tensor_a: {
  //     dim_0: [[min_shape_0, max_shape_0, opt_shape_0], [min_shape_1, max_shape_1, opt_shape_1]]
  //   },
  //   tensor_b: {
  //     dim_1: [[min_shape_2, max_shape_2, opt_shape_2], [min_shape_3, max_shape_3, opt_shape_3]]
  //   }
  // }
  ShapeRangesMap input_explicit_shape_ranges;
  ShapeRangesMap input_implicit_shape_ranges;

  if ((!profile_min_shapes_.empty()) && (!profile_max_shapes_.empty()) && (!profile_opt_shapes_.empty())) {
    has_explicit_profile = true;
    num_profiles = GetNumProfiles(profile_min_shapes_);
    for (int i = 0; i < num_profiles; i++) {
      trt_profiles.push_back(trt_builder->createOptimizationProfile());
    }
  }

  // Iterate all input tensors to check dynamic shape
  for (unsigned int i = 0, end = num_inputs; i < end; ++i) {
    auto input = trt_network->getInput(i);
    const std::string& input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    // Apply explicit optimization profiles provided by user
    if (has_explicit_profile) {
      apply_explicit_profile = ApplyProfileShapesFromProviderOptions(trt_profiles, input, profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_, input_explicit_shape_ranges);
    }

    // If no explicit optimization profile is being applied, TRT EP will later set min/max/opt shape values based on input tensor values at EP compute time
    if (!apply_explicit_profile) {
      if (input->isShapeTensor()) {
        // Shape tensor
        std::vector<std::vector<int64_t>> profile_vector;
        std::vector<int64_t> shape_vector{INT_MAX, INT_MIN, INT_MIN};
        profile_vector.push_back(shape_vector);  // only one profile needed
        input_implicit_shape_ranges[input_name][0] = profile_vector;
        has_dynamic_shape = true;
      } else {
        // Execution tensor
        for (int j = 0, end = nb_dims; j < end; ++j) {
          if (dims.d[j] == -1) {
            std::vector<std::vector<int64_t>> profile_vector;
            std::vector<int64_t> shape_vector{INT_MAX, INT_MIN, INT_MIN};
            profile_vector.push_back(shape_vector);  // only one profile needed
            input_implicit_shape_ranges[input_name][j] = profile_vector;
            has_dynamic_shape = true;
          }
        }
      }
      apply_explicit_profile = false;
    }
  }

  // Set explicit profiles in TRT config if all dynamic shape inputs have associated profiles provided by user
  if (has_explicit_profile) {
    // TRT EP has a constraint here.
    // Users need to provide all the dynamic shape inputs with associated profiles if they want to explicitly specify profiles through provider options.
    if (has_dynamic_shape) {
      std::ostringstream msg;
      msg << "User needs to provide all the dynamic shape inputs with associated profiles if they want to explicitly set profiles through provider options.\n";
      msg << "Please note that main graph could be partitioned into TRT/CUDA/CPU subgraphs, in this case, user also needs to provide shape profiles for the TRT subgraph's input if it's dynamic shape input.\n";
      msg << "Following input(s) has no associated shape profiles provided: ";
      auto begin = input_implicit_shape_ranges.begin();
      auto end = input_implicit_shape_ranges.end();
      auto it = begin;
      if (it != end) {
        msg << it->first;
        ++it;
      }
      for (; it != end; ++it) {
        msg << "," << it->first;
      }
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, msg.str());
    } else {
      for (auto trt_profile : trt_profiles) {
        trt_config->addOptimizationProfile(trt_profile);
      }
    }
  }
  // If no explicit profile is applied and the input has dynamic shape, TRT EP simply creates one profile by default.
  // It will later set proper min/max/opt shape values duing EP compute time.
  else if (!has_explicit_profile && has_dynamic_shape) {
    trt_profiles.push_back(trt_builder->createOptimizationProfile());
  }

  // Check platform availability for low precision
  if (fp16_enable_ || bf16_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (!trt_builder->platformHasFastFp16()) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      fp16_enable_ = false;
      bf16_enable_ = false;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_FP16_ENABLE or ORT_TENSORRT_BF16_ENABLE is set, but platform doesn't support fast native fp16/bf16";
    }
  }

  if (int8_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (!trt_builder->platformHasFastInt8()) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      int8_enable_ = false;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_INT8_ENABLE is set, but platform doesn't support fast native int8";
    }
  }

  // Load INT8 calibration table
  std::unordered_map<std::string, float> dynamic_range_map;
  if (int8_enable_ && int8_calibration_cache_available_) {
    const std::string calibration_cache_path = GetCachePath(cache_path_, int8_calibration_cache_name_);
    if (!ReadDynamicRange(calibration_cache_path, int8_use_native_tensorrt_calibration_table_, dynamic_range_map)) {
      throw std::runtime_error("Failed to read INT8 calibration table " + calibration_cache_path);
    }
  }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  // Set precision flags
  std::string trt_node_name_with_precision = fused_node.Name();
  if (fp16_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    trt_node_name_with_precision += "_fp16";
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] FP16 mode is enabled";
  }
  if (bf16_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kBF16);
    trt_node_name_with_precision += "_bf16";
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] BF16 mode is enabled";
  }
  if (int8_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    trt_node_name_with_precision += "_int8";
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] INT8 mode is enabled";
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  // Set DLA
  if (fp16_enable_ || int8_enable_) {
    if (dla_enable_ && dla_core_ >= 0) {  // DLA can only run with FP16 and INT8
      int number_of_dla_core = trt_builder->getNbDLACores();
      if (number_of_dla_core == 0) {
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] Try to use DLA core, but platform doesn't have any DLA core";
        dla_enable_ = false;
      } else {
        if (dla_core_ >= number_of_dla_core) {
          LOGS_DEFAULT(WARNING) << "[TensorRT EP] Try to use DLA core #" << dla_core_ << ", but it exceeds platform's maximum DLA core number " << number_of_dla_core << ". Use DLA core 0 instead.";
          dla_core_ = 0;
        }
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] use DLA core " << dla_core_;
        trt_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        trt_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        trt_config->setDLACore(dla_core_);
        trt_node_name_with_precision += "_dlacore" + std::to_string(dla_core_);
      }
    }
  }

  // enable sparse weights
  if (sparsity_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Sparse weights are allowed";
  }
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR == 5
  if (build_heuristics_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kENABLE_TACTIC_HEURISTIC);
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Builder heuristics are enabled."
                          << " For TRT > 8.5, trt_build_heuristics_enable is deprecated, please set builder optimization level as 2 to enable builder heuristics.";
  }
#elif NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // for TRT 8.6 onwards, heuristic-based tactic option is automatically enabled by setting builder optimization level 2
  if (build_heuristics_enable_) {
    if (builder_optimization_level_ == 2) {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] Builder heuristics are automatically enabled by builder optimization level 2. trt_build_heuristics_enable is deprecated on TRT 8.6 onwards.";
    } else {
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] trt_build_heuristics_enable is deprecated on TRT 8.6 onwards. Please set builder optimization level as 2 to enable builder heuristics.";
    }
  }
#endif

#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // switch optimizaion level
  if (builder_optimization_level_ != 3) {
    trt_config->setBuilderOptimizationLevel(builder_optimization_level_);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Builder optimization level is set to " << builder_optimization_level_;
  }

  // limit auxiliary streams
  if (auxiliary_streams_ >= 0) {
    trt_config->setMaxAuxStreams(auxiliary_streams_);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Auxiliary streams are se to " << auxiliary_streams_;
  }
#else
  if (builder_optimization_level_ != 3) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Builder optimization level can only be used on TRT 8.6 onwards!";
  }
  if (auxiliary_streams_ >= 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] Auxiliary streams can only be set on TRT 8.6 onwards!";
  }
#endif

  if (weight_stripped_engine_enable_) {
#if NV_TENSORRT_MAJOR >= 10
    trt_config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] STRIP_PLAN is enabled";
    trt_config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] REFIT_IDENTICAL is enabled";
#else
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] weight-stripped engines can only be used on TRT 10.0 onwards!";
#endif
  }

  // limit used tactic sources
  if (!tactic_sources_.empty()) {
    nvinfer1::TacticSources tactics = trt_config->getTacticSources();
    tactics |= GetTacticSourceFromString(tactic_sources_);
    trt_config->setTacticSources(tactics);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Tactic sources are limited using " << tactic_sources_;
  }

  // Set preview feature flags
  for (auto feature : preview_features_) {
    trt_config->setPreviewFeature(feature, true);
  }

  // Build TRT engine (if needed) and load TRT engine if:
  //   (1) Graph has no dynamic shape input
  //   (2) All the dynamic shape inputs have associated explicit profiles specified by user
  //
  // Otherwise engine will be handled at inference time.
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

  std::string cache_path = "";
  std::string cache_suffix = "";
  // Customize cache prefix if assigned
  if (!cache_prefix_.empty()) {
    // Generate cache suffix in case user would like to customize cache prefix
    cache_suffix = "_" + GetCacheSuffix(fused_node.Name(), trt_node_name_with_precision);
    cache_path = GetCachePath(cache_path_, cache_prefix_) + cache_suffix;
  } else {
    cache_path = GetCachePath(cache_path_, trt_node_name_with_precision);
  }

  std::string cache_hw_compat = "_sm" + compute_capability_;
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // Enable hardware compatility mode if assigned
  if (engine_cache_enable_ && engine_hw_compatible_) {
    trt_config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
    cache_hw_compat = "_sm80+";
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Hardware compatibility is enabled when loading and capturing engine cache.";
  }
#endif

  // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
  // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even if they share the same compute capacity
  const std::string cache_path_prefix = cache_path + cache_hw_compat;
  std::string engine_cache_path = cache_path_prefix + ".engine";
  const std::string encrypted_engine_cache_path = engine_cache_path + ".encrypted";
  const std::string profile_cache_path = cache_path_prefix + ".profile";

  // If weight-stripped engine is enabled and refitted engine cache is not present,
  // TRT EP will use the engine cache with ".stripped.engine" appended to the end.
  const std::filesystem::path engine_cache_fs_path = engine_cache_path;
  if (weight_stripped_engine_enable_ && !std::filesystem::exists(engine_cache_fs_path)) {
    engine_cache_path = cache_path_prefix + ".stripped.engine";
    weight_stripped_engine_refit_ = true;
  }

  // Generate file name for dumping ep context model
  if (dump_ep_context_model_ && ctx_model_path_.empty()) {
    ctx_model_path_ = GetCtxModelPath(ep_context_file_path_, model_path_);
  }

  if (!has_dynamic_shape) {
    std::string timing_cache_path = "";
    bool engine_update = false;
    if (timing_cache_enable_) {
      timing_cache_path = GetTimingCachePath(global_cache_path_, compute_capability_);
    }
    {
      // ifstream file check, engine serialization/deserialization and engine build are in critical section. It needs lock protection to prevent race condition when inferencing with multithreading.
      auto lock = GetApiLock();

      // If explicit profile flag is on and engine cache enable flag is on,
      // we need to compare explicit profiles and profiles used to build the engine in order to decide whether to rebuild the engine.
      if (has_explicit_profile && engine_cache_enable_) {
        engine_update = CompareProfiles(profile_cache_path, profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_);
        if (engine_update) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Engine will be built";
        } else {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Engine won't be rebuilt";
        }
      }

      std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
      if (engine_cache_enable_ && !engine_decryption_enable_ && engine_file && !engine_update) {
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);
        trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from cache: " + engine_cache_path);
        }

      } else if (engine_decryption_enable_ && engine_cache_enable_ && std::filesystem::exists(encrypted_engine_cache_path) && !engine_update) {
        // Decrypt engine
        size_t engine_size = 0;
        if (!engine_decryption_(encrypted_engine_cache_path.c_str(), nullptr, &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not get engine buffer size");
        }
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        if (!engine_decryption_(encrypted_engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not call engine decryption function decrypt");
        }
        // Deserialize engine
        trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Decrypted and DeSerialized " + encrypted_engine_cache_path;
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from encrypted cache: " + encrypted_engine_cache_path);
        }
      } else {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        // Set INT8 per tensor dynamic range
        if (int8_enable_ && trt_builder->platformHasFastInt8() && int8_calibration_cache_available_) {
          trt_config->setInt8Calibrator(nullptr);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
          if (!SetDynamicRange(*trt_network, dynamic_range_map)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not set INT8 dynamic range for fused node: " + fused_node.Name());
          }
        }

        // Load timing cache from file. Create a fresh cache if the file doesn't exist
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
        if (timing_cache_enable_) {
          std::vector<char> loaded_timing_cache = loadTimingCacheFile(timing_cache_path);
          timing_cache.reset(trt_config->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()), loaded_timing_cache.size()));
          if (timing_cache == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not create timing cache: " + timing_cache_path);
          }
          trt_config->setTimingCache(*timing_cache, force_timing_cache_match_);
          if (detailed_build_log_) {
            LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Deserialized timing cache from " + timing_cache_path;
          }
        }

        // Build engine
        std::chrono::steady_clock::time_point engine_build_start;
        if (detailed_build_log_) {
          engine_build_start = std::chrono::steady_clock::now();
        }
        std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{trt_builder->buildSerializedNetwork(*trt_network, *trt_config)};
        if (serialized_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP failed to create engine from network for fused node: " + fused_node.Name());
        }
        trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP failed to deserialize engine for fused node: " + fused_node.Name());
        }
        if (detailed_build_log_) {
          auto engine_build_stop = std::chrono::steady_clock::now();
          LOGS_DEFAULT(INFO) << "TensorRT engine build for " << trt_node_name_with_precision << " took: " << std::chrono::duration_cast<std::chrono::milliseconds>(engine_build_stop - engine_build_start).count() << "ms" << std::endl;
        }
        if (engine_cache_enable_) {
          // Serialize engine profile if it has explicit profiles
          if (has_explicit_profile) {
            SerializeProfileV2(profile_cache_path, input_explicit_shape_ranges);
            LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + profile_cache_path;
          }

          if (engine_decryption_enable_) {
            // Encrypt engine. The library is not always deployed with the encrypt function, so check if it is available first.
            if (engine_encryption_ != nullptr) {
              if (!engine_encryption_(encrypted_engine_cache_path.c_str(), reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size())) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                       "TensorRT EP call to engine encryption library failed");
              }
              LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized and encrypted engine " + encrypted_engine_cache_path;
            } else {
              LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine cache encryption function is not found. No cache is written to disk";
            }
          } else {
            std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
            file.write(reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
            LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized engine " + engine_cache_path;
          }
        }
        // serialize and save timing cache
        if (timing_cache_enable_) {
          auto timing_cache = trt_config->getTimingCache();
          std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};
          if (timingCacheHostData == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not serialize timing cache: " + timing_cache_path);
          }
          saveTimingCacheFile(timing_cache_path, timingCacheHostData.get());
          if (detailed_build_log_) {
            LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized timing cache " + timing_cache_path;
          }
        }
        // dump EP context node model
        if (dump_ep_context_model_) {
          // "ep_cache_context" node attribute should be a relative path to context model directory
          if (ep_cache_context_attr_.empty()) {
            auto cache_file_name = std::filesystem::path(engine_cache_path).filename();
            ep_cache_context_attr_ = std::filesystem::path(engine_cache_relative_path_to_context_model_dir).append(cache_file_name.string()).string();
          }
          std::string compute_capability_hw_compat = compute_capability_;
          if (engine_cache_enable_ && engine_hw_compatible_) {
            compute_capability_hw_compat = "80+";
          }
          std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto{CreateCtxModel(graph_body_viewer,
                                                                                 ep_cache_context_attr_,
                                                                                 reinterpret_cast<char*>(serialized_engine->data()),
                                                                                 serialized_engine->size(),
                                                                                 ep_context_embed_mode_,
                                                                                 compute_capability_hw_compat,
                                                                                 model_path_,
                                                                                 GetLogger())};
          DumpCtxModel(model_proto.get(), ctx_model_path_);
        }
      }
    }

    if (weight_stripped_engine_refit_) {
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Refit engine from main ONNX file after engine build";
      auto status = RefitEngine(model_path_,
                                onnx_model_folder_path_,
                                engine_cache_path,
                                false /* path check for security */,
                                onnx_model_bytestream_,
                                onnx_model_bytestream_size_,
                                onnx_external_data_bytestream_,
                                onnx_external_data_bytestream_size_,
                                trt_engine.get(),
                                true /* serialize refitted engine to disk */,
                                detailed_build_log_);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }

    // Build context
    // Note: Creating an execution context from an engine is thread safe per TRT doc
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    if (context_memory_sharing_enable_) {
      // Reset the max_ctx_mem_size_ and context_memory_ since we don't have access to the allocator here.
      max_ctx_mem_size_ = 0;
      context_memory_ = nullptr;
#if NV_TENSORRT_MAJOR < 10
      trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContextWithoutDeviceMemory());
#else
      trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
    } else {
      trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    }
    if (!trt_context) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build execution context for fused node: " + fused_node.Name());
    }
  }

  // Create input to index map
  for (int i = 0; i < num_inputs; ++i) {
    auto input = trt_network->getInput(i);
    const std::string& input_name = input->getName();
    const auto& iter = input_map.find(input_name);
    if (iter != input_map.end()) {
      input_indexes[input_name] = iter->second;
    }
  }

  // Create output to index and type maps
  const auto& graph_output = model_proto->graph().output();
  for (int i = 0; i < num_outputs; ++i) {
    const std::string& output_name = trt_network->getOutput(i)->getName();
    const auto& iter = output_map.find(output_name);
    if (iter != output_map.end()) {
      output_indexes[output_name] = iter->second;
    }
    const auto& tensor_type = graph_output[i].type().tensor_type();
    output_types[output_name] = tensor_type.elem_type();
  }

  // Save TRT engine, other TRT objects and input/output info to map
  parsers_.emplace(fused_node.Name(), std::move(trt_parser));
  engines_.emplace(fused_node.Name(), std::move(trt_engine));
  contexts_.emplace(fused_node.Name(), std::move(trt_context));
  networks_.emplace(fused_node.Name(), std::move(trt_network));
  weights_.emplace(fused_node.Name(), std::move(userWeights));
  input_info_[fused_node.Name()].push_back(input_indexes);
  output_info_[fused_node.Name()].push_back(output_indexes);
  output_info_[fused_node.Name()].push_back(output_types);
  input_shape_ranges_[fused_node.Name()] = input_implicit_shape_ranges;
  profiles_.emplace(fused_node.Name(), std::move(trt_profiles));

  // For dynamic shape input model, firstly TRT EP creates a model proto which includes inputs, outputs and empty engine.
  // TRT EP will serialize the model at inference time due to engine can be updated and the updated engine should be included in the model.
  // However, if the embed_mode is 0 (only includes engine path), TRT EP will serialize it here.
  if (dump_ep_context_model_ && has_dynamic_shape) {
    // "ep_cache_context" node attribute should be a relative path to context model directory
    if (ep_cache_context_attr_.empty()) {
      auto cache_file_name = std::filesystem::path(engine_cache_path).filename();
      ep_cache_context_attr_ = std::filesystem::path(engine_cache_relative_path_to_context_model_dir).append(cache_file_name.string()).string();
    }
    std::string compute_capability_hw_compat = compute_capability_;
    if (engine_cache_enable_ && engine_hw_compatible_) {
      compute_capability_hw_compat = "80+";
    }
    model_proto_.reset(CreateCtxModel(graph_body_viewer,
                                      ep_cache_context_attr_,
                                      nullptr,
                                      0,
                                      ep_context_embed_mode_,
                                      compute_capability_hw_compat,
                                      model_path_,
                                      GetLogger()));
    if (ep_context_embed_mode_ == 0) {
      DumpCtxModel(model_proto_.get(), ctx_model_path_);
    }
  }

  // Create function state
  // TODO: remove default capture
  NodeComputeInfo compute_info;
  compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
    std::unique_ptr<TensorrtFuncState> p = std::make_unique<TensorrtFuncState>();
    // translate tactic sources string to nvinfer1::TacticSources
    nvinfer1::TacticSources tactics = 0;
    if (!tactic_sources_.empty()) {
      tactics = GetTacticSourceFromString(tactic_sources_);
    }
    *p = {context->allocate_func, context->release_func, context->allocator_handle, context->node_name, builder_.get(),
          &parsers_[context->node_name], &engines_[context->node_name], &contexts_[context->node_name],
          &networks_[context->node_name], input_info_[context->node_name], output_info_[context->node_name],
          input_shape_ranges_[context->node_name], &tensorrt_mu_, fp16_enable_, bf16_enable_, int8_enable_, int8_calibration_cache_available_,
          dla_enable_, dla_core_, trt_node_name_with_precision,
          engine_cache_enable_, cache_path_, runtime_.get(), profiles_[context->node_name],
          context_memory_sharing_enable_, &max_ctx_mem_size_, &context_memory_, dynamic_range_map, engine_decryption_enable_,
          engine_decryption_, engine_encryption_, timing_cache_enable_, global_cache_path_, force_timing_cache_match_,
          detailed_build_log_, build_heuristics_enable_, sparsity_enable_, builder_optimization_level_,
          auxiliary_streams_, !tactic_sources_.empty(), tactics, cuda_graph_enable_, cache_prefix_, cache_suffix, engine_hw_compatible_,
          preview_features_, &weights_[context->node_name]};
    *state = p.release();
    return 0;
  };

  // Release function state
  compute_info.release_state_func = [](FunctionState state) {
    delete static_cast<TensorrtFuncState*>(state);
  };

  // Create compute function
  compute_info.compute_func = [this](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    TensorrtFuncState* trt_state = reinterpret_cast<TensorrtFuncState*>(state);

    // The whole compute_function should be considered the critical section where multiple threads may update kernel function state, access one builder, create/serialize/save engine,
    // save profile and serialize/save timing cache. Therefore, those operations should be synchronized across different threads when ORT is using multithreading.
    // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    std::lock_guard<std::mutex> lock(*(trt_state->tensorrt_mu_ptr));
    const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
    const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
    const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
    auto fused_node_name = trt_state->fused_node_name;
    // This map "shape_ranges" contains the shape range info for setting TRT optimization profiles.
    // The info is used for both shape tensor and execution tensor:
    // tensor name->(dimension->[min, max, opt])
    auto& shape_ranges = trt_state->input_shape_ranges;
    std::unordered_map<std::string, std::vector<int32_t>> shape_tensor_values;        // This map holds "shape tensor -> shape values" for the shape tensor input across this inference run
    std::unordered_map<std::string, std::vector<int64_t>> shape_tensor_values_int64;  // same as above but for int64 shape tensor input
    auto& dds_output_allocator_map = this->dds_output_allocator_maps_[fused_node_name];
    auto trt_builder = trt_state->builder;
    auto trt_engine = trt_state->engine->get();
    auto trt_context = trt_state->context->get();
    auto trt_profiles = trt_state->profiles;
    auto context_memory = trt_state->context_memory;
    auto max_context_mem_size_ptr = trt_state->max_context_mem_size_ptr;
    int num_inputs = static_cast<int>(input_indexes.size());
    int num_outputs = static_cast<int>(output_indexes.size());
    bool engine_update = false;
    bool context_update = false;
    std::unordered_set<std::string> input_names;

    OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                     narrow<OrtDevice::DeviceId>(device_id_));
    OrtMemoryInfo mem_info("", OrtAllocatorType::OrtDeviceAllocator, device);
    if (alloc_ == nullptr) {
      Ort::ThrowOnError(api->KernelContext_GetAllocator(context, &mem_info, &alloc_));
    }
    OrtAllocator* alloc = alloc_;

    void* cuda_stream;
    Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &cuda_stream));
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
    // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even if they share the same compute capacity
    // Prepare cache name
    std::string cache_path = "";
    // Customize cache prefix if assigned
    if (!cache_prefix_.empty()) {
      cache_path = GetCachePath(trt_state->engine_cache_path, trt_state->cache_prefix) + trt_state->cache_suffix;
    } else {
      cache_path = GetCachePath(trt_state->engine_cache_path, trt_state->trt_node_name_with_precision);
    }

    // Enable hardware compatility mode if assigned
    std::string cache_hw_compat = "_sm" + compute_capability_;
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
    if (engine_cache_enable_ && engine_hw_compatible_) {
      cache_hw_compat = "_sm80+";
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Hardware compatibility is enabled when loading and capturing engine cache.";
    }
#endif

    // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
    // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even if they share the same compute capacity
    const std::string cache_path_prefix = cache_path + cache_hw_compat;
    std::string engine_cache_path = cache_path_prefix + ".engine";
    const std::string encrypted_engine_cache_path = engine_cache_path + ".encrypted";
    const std::string profile_cache_path = cache_path_prefix + ".profile";
    std::string timing_cache_path = "";
    if (timing_cache_enable_) {
      timing_cache_path = GetTimingCachePath(global_cache_path_, compute_capability_);
    }

    // If weight-stripped engine is enabled and refitted engine cache is not present,
    // TRT EP will use the engine cache with ".stripped.engine" appended to the end.
    const std::filesystem::path engine_cache_fs_path = engine_cache_path;
    if (weight_stripped_engine_enable_ && !std::filesystem::exists(engine_cache_fs_path)) {
      engine_cache_path = cache_path_prefix + ".stripped.engine";
      weight_stripped_engine_refit_ = true;
    }

    // Load serialized engine
    if (trt_state->engine_cache_enable && trt_engine == nullptr) {
      std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
      std::ifstream profile_file(profile_cache_path, std::ios::binary | std::ios::in);
      if (engine_file && !trt_state->engine_decryption_enable && profile_file) {
        // Deserialize profile
        shape_ranges = DeserializeProfileV2(profile_file);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + profile_cache_path;

        // Prepare buffer
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);

        // Deserialize engine
        // Note: Deserializing an engine from a TensorRT runtime is thread safe per TRT doc
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
        trt_state->engine->reset();
        *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(
            trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size));
        if (!(*(trt_state->engine))) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Build Engine.");
        }
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
        trt_engine = trt_state->engine->get();
        context_update = true;

      } else if (trt_state->engine_decryption_enable && std::filesystem::exists(encrypted_engine_cache_path) && profile_file) {
        shape_ranges = DeserializeProfileV2(profile_file);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + profile_cache_path;
        // Decrypt engine
        size_t engine_size = 0;
        if (!trt_state->engine_decryption(encrypted_engine_cache_path.c_str(), nullptr, &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not get engine buffer size");
        }
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        if (!trt_state->engine_decryption(encrypted_engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not call engine decryption function decrypt");
        }
        // Deserialize engine
        // Note: Deserializing an engine from a TensorRT runtime is thread safe per TRT doc
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
        trt_state->engine->reset();
        *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size));
        if (!(*(trt_state->engine))) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from encrypted cache: " + encrypted_engine_cache_path);
        }
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Decrypted and DeSerialized " + encrypted_engine_cache_path;
        trt_engine = trt_state->engine->get();
        context_update = true;
      }
    }

    // Check and update shape ranges for dynamic shape inputs.
    for (int i = 0, end = num_inputs; i < end; ++i) {
      auto input = trt_state->network->get()->getInput(i);
      const std::string& input_name = input->getName();
      input_names.insert(input_name);

      // If there is any input tensor in shape_ranges, it means this input tensor has dynamic shape and its profile shape values have not yet resolved.
      // TRT EP will help determine the min/max/opt profile values based on current input tensor value.
      if (shape_ranges.find(input_name) != shape_ranges.end()) {
        auto status = ApplyProfileShapesFromInputTensorValue(trt_profiles, ctx, input, shape_ranges, input_indexes, shape_tensor_values, shape_tensor_values_int64, stream, &engine_update);
        if (status != Status::OK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to parse input tensor and generate optimization profiles.");
        }
      }
    }

    // Regenerate engine
    if (engine_update) {
      // Destroy the IExecutionContext objects before destroying an engine object, otherwise it will lead to undefined behavior.
      trt_state->context->reset();
      trt_state->engine->reset();
      auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
      if (max_workspace_size_ > 0) {
        trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
      }
      for (auto trt_profile : trt_profiles) {
        trt_config->addOptimizationProfile(trt_profile);
      }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      // Set INT8 Per Tensor Dynamic range
      if (trt_state->int8_enable && trt_builder->platformHasFastInt8() && trt_state->int8_calibration_cache_available) {
        trt_config->setInt8Calibrator(nullptr);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        if (!SetDynamicRange(*trt_state->network->get(), trt_state->dynamic_range_map)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to set INT8 dynamic range.");
        }
      }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      // Set precision
      if (trt_state->int8_enable) {
        trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] INT8 mode is enabled";
      }
      if (trt_state->fp16_enable) {
        trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] FP16 mode is enabled";
      }
      if (trt_state->bf16_enable) {
        trt_config->setFlag(nvinfer1::BuilderFlag::kBF16);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] BF16 mode is enabled";
      }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      // Set DLA (DLA can only run with FP16 or INT8)
      if ((trt_state->fp16_enable || trt_state->int8_enable) && trt_state->dla_enable) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] use DLA core " << trt_state->dla_core;
        trt_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        trt_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        trt_config->setDLACore(trt_state->dla_core);
      }

      // enable sparse weights
      if (trt_state->sparsity_enable) {
        trt_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Sparse weights are allowed";
      }
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR == 5
      // enable builder heuristics
      if (trt_state->build_heuristics_enable) {
        trt_config->setFlag(nvinfer1::BuilderFlag::kENABLE_TACTIC_HEURISTIC);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Builder heuristics are enabled";
      }
#elif NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
      // switch optimizaion level
      if (trt_state->builder_optimization_level != 3) {
        trt_config->setBuilderOptimizationLevel(trt_state->builder_optimization_level);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Builder optimization level is set to " << builder_optimization_level_;
      }

      // limit auxiliary streams
      if (trt_state->auxiliary_streams >= 0) {
        trt_config->setMaxAuxStreams(trt_state->auxiliary_streams);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Auxiliary streams are se to " << trt_state->auxiliary_streams;
      }
#else
      if (trt_state->builder_optimization_level != 3) {
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] Builder optimization level can only be used on TRT 8.6 onwards!";
      }
      if (trt_state->auxiliary_streams >= 0) {
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] Auxiliary streams can only be set on TRT 8.6 onwards!";
      }
#endif
      if (weight_stripped_engine_enable_) {
#if NV_TENSORRT_MAJOR >= 10
        trt_config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] STRIP_PLAN is enabled";
        trt_config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] REFIT_IDENTICAL is enabled";
#else
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] weight-stripped engines can only be used on TRT 10.0 onwards!";
#endif
      }
      // limit used tactic sources
      if (trt_state->filter_tactic_sources) {
        nvinfer1::TacticSources tactics = trt_config->getTacticSources();
        tactics |= trt_state->tactic_sources;
        trt_config->setTacticSources(tactics);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Tactic sources are limited using bitmask " << tactics;
      }

      // Load timing cache from file. Create a fresh cache if the file doesn't exist
      std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
      if (trt_state->timing_cache_enable) {
        std::vector<char> loaded_timing_cache = loadTimingCacheFile(timing_cache_path);
        timing_cache.reset(trt_config->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()), loaded_timing_cache.size()));
        if (timing_cache == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not create timing cache: " + timing_cache_path);
        }
        trt_config->setTimingCache(*timing_cache, force_timing_cache_match_);
        if (detailed_build_log_) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Deserialized timing cache from " + timing_cache_path;
        }
      }

#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
      // Enable hardware compatility mode if assigned
      if (trt_state->engine_hw_compatible) {
        trt_config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
        LOGS_DEFAULT(INFO) << "[TensorRT EP] Re-generate engine with hardware compatibility enabled.";
      }
#endif

      // Set preview feature flags
      for (auto feature : trt_state->preview_features) {
        trt_config->setPreviewFeature(feature, true);
      }

      // Build engine
      std::unique_ptr<nvinfer1::IHostMemory> serialized_engine;
      {
        auto lock = GetApiLock();
        std::chrono::steady_clock::time_point engine_build_start;
        if (detailed_build_log_) {
          engine_build_start = std::chrono::steady_clock::now();
        }
        serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
            trt_builder->buildSerializedNetwork(*trt_state->network->get(), *trt_config));
        if (!serialized_engine) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create engine from network.");
        }
        *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(
            trt_state->runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
        if (!(*(trt_state->engine))) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to deserialize engine.");
        }
        if (detailed_build_log_) {
          auto engine_build_stop = std::chrono::steady_clock::now();
          LOGS_DEFAULT(INFO) << "TensorRT engine build for " << trt_state->trt_node_name_with_precision << " took: " << std::chrono::duration_cast<std::chrono::milliseconds>(engine_build_stop - engine_build_start).count() << "ms" << std::endl;
        }
      }
      if (!(*(trt_state->engine))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Build Engine.");
      }
      trt_engine = trt_state->engine->get();
      if (trt_state->engine_cache_enable) {
        // Serialize engine profile
        SerializeProfileV2(profile_cache_path, shape_ranges);
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + profile_cache_path;

        // Serialize engine
        if (trt_state->engine_decryption_enable) {
          // Encrypt engine. The library is not always deployed with the encrypt function, so check if it is available first.
          if (trt_state->engine_encryption != nullptr) {
            if (!trt_state->engine_encryption(encrypted_engine_cache_path.c_str(), reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size())) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                     "TensorRT EP could not call engine encryption function encrypt");
            }
            LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized and encrypted engine " + encrypted_engine_cache_path;
          } else {
            LOGS_DEFAULT(WARNING) << "[TensorRT EP] Engine cache encryption function is not found. No cache is written to disk";
          }
        } else {
          std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
          file.write(reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + engine_cache_path;
        }
      }

      // serialize and save timing cache
      if (trt_state->timing_cache_enable) {
        auto timing_cache = trt_config->getTimingCache();
        std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};
        if (timingCacheHostData == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not serialize timing cache: " + timing_cache_path);
        }
        saveTimingCacheFile(timing_cache_path, timingCacheHostData.get());
        if (detailed_build_log_) {
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized timing cache " + timing_cache_path;
        }
      }

      // dump ep context model
      if (dump_ep_context_model_ && ep_context_embed_mode_) {
        UpdateCtxNodeModelEngineContext(model_proto_.get(), reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
        DumpCtxModel(model_proto_.get(), ctx_model_path_);
      }
      context_update = true;

      if (weight_stripped_engine_refit_) {
        auto status = RefitEngine(model_path_,
                                  onnx_model_folder_path_,
                                  engine_cache_path,
                                  false /* path check for security */,
                                  onnx_model_bytestream_,
                                  onnx_model_bytestream_size_,
                                  onnx_external_data_bytestream_,
                                  onnx_external_data_bytestream_size_,
                                  trt_engine,
                                  true /* serialize refitted engine to disk */,
                                  detailed_build_log_);
        if (status != Status::OK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
        }
      }
    }

    if (context_update) {
      if (trt_state->context_memory_sharing_enable) {
#if NV_TENSORRT_MAJOR < 10
        *(trt_state->context) = std::unique_ptr<nvinfer1::IExecutionContext>(
            trt_state->engine->get()->createExecutionContextWithoutDeviceMemory());
#else
        *(trt_state->context) = std::unique_ptr<nvinfer1::IExecutionContext>(
            trt_state->engine->get()->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
      } else {
        *(trt_state->context) = std::unique_ptr<nvinfer1::IExecutionContext>(
            trt_state->engine->get()->createExecutionContext());
      }
      if (!(*(trt_state->context))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create context.");
      }
      trt_context = trt_state->context->get();
    }

    // Check before using trt_engine
    if (trt_engine == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "No engine is found.");
    }

    // Get input and output binding names
    int total_bindings = trt_engine->getNbIOTensors();
    std::vector<char const*> input_binding_names, output_binding_names;
    for (int i = 0, end = total_bindings; i < end; ++i) {
      auto const& name = trt_engine->getIOTensorName(i);
      auto const& mode = trt_engine->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        input_binding_names.push_back(name);
      } else {
        output_binding_names.push_back(name);
      }
    }

    /*
     * Set input shapes and bind input buffers
     */
    std::vector<IAllocatorUniquePtr<void>> scratch_buffers;
    for (size_t i = 0, end = input_binding_names.size(); i < end; ++i) {
      char const* input_name = input_binding_names[i];

      size_t input_index = 0;
      const auto iter = input_indexes.find(input_name);
      if (iter != input_indexes.end()) {
        input_index = iter->second;
      }
      auto input_tensor = ctx.GetInput(input_index);
      auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
      const auto tensor_shapes = tensor_info.GetShape();

      auto status = BindContextInput(ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values, shape_tensor_values_int64, scratch_buffers, alloc, stream);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }

    /*
     * Set output shapes and bind output buffers
     */
    std::unordered_map<char const*, void*> buffers;
    buffers.reserve(num_outputs);
    using OutputOrtValue = Ort::UnownedValue;
    std::unordered_map<size_t, OutputOrtValue> output_tensors;
    output_tensors.reserve(num_outputs);
    std::unordered_map<size_t, int> output_dim_sizes;
    output_dim_sizes.reserve(num_outputs);

    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
      char const* output_name = output_binding_names[i];

      size_t output_index = 0;
      const auto& index_iter = output_indexes.find(output_name);
      if (index_iter != output_indexes.end()) {
        output_index = index_iter->second;
      }

      size_t output_type = 0;
      const auto type_iter = output_types.find(output_name);
      if (type_iter != output_types.end()) {
        output_type = type_iter->second;
      }

      Status status = BindContextOutput(ctx, trt_context, output_name, output_index, output_type, i, output_tensors, output_dim_sizes,
                                        dds_output_allocator_map, scratch_buffers, alloc, buffers);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }

    // Set execution context memory
    if (trt_state->context_memory_sharing_enable) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      if (mem_size > *max_context_mem_size_ptr) {
        *max_context_mem_size_ptr = mem_size;
        *context_memory = IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, *max_context_mem_size_ptr, true /*use_reserve*/);
      }
      trt_context->setDeviceMemory((*context_memory).get());
    }

    // Start CUDA graph capture.
    // Note: The reason we don't put graph capture in OnRunStart() like CUDA EP does is because
    // current ORT TRT doesn't get cuda stream until compute time and graph capture requires cuda stream.
    if (cuda_graph_enable_ && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
      LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
      cuda_graph_.SetStream(stream);
      CaptureBegin(0);
    }

    // Run TRT inference
    if (!trt_context->enqueueV3(stream)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP execution context enqueue failed.");
    }

    /*
     * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this function concurrently,
     * TRT EP needs to carefully take care of concurrency here, if not, following concurrent issue might happen:
     *
     * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per stream.
     * In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling InferenceSession::Run() concurrently,
     * the trt execution context instance is shared by all the threads and each thread aquires different stream from ORT.
     * So TRT EP will end up having one trt execution context using multiple streams which is not suggested.
     * But, since the whole compute_func() is protected by the lock and if cudaStreamSynchronize() is enforced here, one trt execution context per stream
     * is guaranteed.
     *
     * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all operations to prevent the concurrent issue mentioned above.
     * However, if cuda graph is enabled, TRT EP won't call cudaStreamSynchronize() since it's not allowed during graph capture.
     */
    if (sync_stream_after_enqueue_) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }

    // Assign TRT output back to ORT output
    // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
    // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
      char const* output_name = output_binding_names[i];

      size_t output_type = 0;
      const auto& iter = output_types.find(output_name);
      if (iter != output_types.end()) {
        output_type = iter->second;
      }

      if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end()) {
        size_t output_index = 0;
        const auto& index_iter = output_indexes.find(output_name);
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        auto status = BindKernelOutput(ctx, &mem_info, dds_output_allocator_map, output_name, output_index, output_type, stream);
        if (status != Status::OK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, status.ErrorMessage());
        }
      } else {
        auto& output_tensor = output_tensors[i];
#if NV_TENSORRT_MAJOR < 10
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          auto output_tensor_ptr = output_tensor.GetTensorMutableData<int64_t>();
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
#endif
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
          auto output_tensor_ptr = output_tensor.GetTensorMutableData<double>();
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<float, double>(stream, reinterpret_cast<float*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
      }
    }

    // End CUDA graph capture.
    // Note: One reason we don't put end of graph capture in OnRunEnd() like CUDA EP does is because of cuda stream mentioned in graph capture
    // above, another reason is because OnRunEnd() is not synchronized with OnRunStart() and ExecuteGraph() per inference_session.cc.
    // It's safe to start/end CUDA graph capture in compute_func() here since cuda graph object is maintained by a per thread basis.
    if (cuda_graph_enable_ && !IsGraphCaptured(0)) {
      if (IsGraphCaptureAllowed()) {
        CaptureEnd(0);
        // CUDA work issued to a capturing stream doesn’t actually run on the GPU,
        // so run the captured graph here to actually execute the work.
        ORT_RETURN_IF_ERROR(ReplayGraph(0));
      } else {
        IncrementRegularRunCountBeforeGraphCapture();
      }
    }

    return Status::OK();
  };

  node_compute_funcs.push_back(compute_info);
  return Status::OK();
}

Status TensorrtExecutionProvider::CreateNodeComputeInfoFromPrecompiledEngine(const GraphViewer& graph_body_viewer,
                                                                             const Node& fused_node,
                                                                             std::unordered_map<std::string, size_t>& input_map,
                                                                             std::unordered_map<std::string, size_t>& output_map,
                                                                             std::vector<NodeComputeInfo>& node_compute_funcs) {
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context;
  std::unordered_map<std::string, size_t> input_indexes;   // TRT engine input name -> ORT kernel context input index
  std::unordered_map<std::string, size_t> output_indexes;  // TRT engine output name -> ORT kernel context output index
  std::unordered_map<std::string, size_t> output_types;    // TRT engine output name -> ORT output tensor type

  // Get engine binary data and deserialize it
  auto trt_cache_model_handler = TensorRTCacheModelHandler(&trt_engine,
                                                           runtime_.get(),
                                                           model_path_,
                                                           compute_capability_,
                                                           weight_stripped_engine_enable_,
                                                           onnx_model_folder_path_,
                                                           onnx_model_bytestream_,
                                                           onnx_model_bytestream_size_,
                                                           onnx_external_data_bytestream_,
                                                           onnx_external_data_bytestream_size_,
                                                           detailed_build_log_);
  auto status = trt_cache_model_handler.GetEpContextFromGraph(graph_body_viewer);
  if (status != Status::OK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
  }

  // Build context
  //
  // Note: Creating an execution context from an engine is thread safe per TRT doc
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  if (context_memory_sharing_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (mem_size > max_ctx_mem_size_) {
      max_ctx_mem_size_ = mem_size;
    }
#if NV_TENSORRT_MAJOR < 10
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContextWithoutDeviceMemory());
#else
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
  } else {
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
  }
  if (!trt_context) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                           "TensorRT EP could not build execution context for fused node: " + fused_node.Name());
  }

  // Create input/output to index maps
  for (int32_t i = 0; i < trt_engine->getNbIOTensors(); ++i) {
    auto const& name = trt_engine->getIOTensorName(i);
    auto const& mode = trt_engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      const auto& iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[name] = iter->second;
      }
    } else {
      const auto& iter = output_map.find(name);
      if (iter != output_map.end()) {
        output_indexes[name] = iter->second;
      }
    }
  }

  // Create output to type map
  for (auto node_arg : graph_body_viewer.GetOutputs()) {
    auto output_name = node_arg->Name();
    auto& type = node_arg->TypeAsProto()->tensor_type();
    output_types[output_name] = type.elem_type();
  }

  // Save TRT engine, TRT context and input/output info to map
  engines_.emplace(fused_node.Name(), std::move(trt_engine));
  contexts_.emplace(fused_node.Name(), std::move(trt_context));
  input_info_[fused_node.Name()].push_back(input_indexes);
  output_info_[fused_node.Name()].push_back(output_indexes);
  output_info_[fused_node.Name()].push_back(output_types);

  // Create function state
  // TODO: remove default capture
  NodeComputeInfo compute_info;
  compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
    std::unique_ptr<TensorrtShortFuncState> p = std::make_unique<TensorrtShortFuncState>();
    *p = {context->allocate_func,
          context->release_func,
          context->allocator_handle,
          context->node_name,
          &engines_[context->node_name],
          &contexts_[context->node_name],
          input_info_[context->node_name],
          output_info_[context->node_name],
          context_memory_sharing_enable_,
          &max_ctx_mem_size_,
          &context_memory_,
          &tensorrt_mu_};
    *state = p.release();
    return 0;
  };

  // Release function state
  compute_info.release_state_func = [](FunctionState state) {
    delete static_cast<TensorrtShortFuncState*>(state);
  };

  // Create compute function
  compute_info.compute_func = [this](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    TensorrtShortFuncState* trt_state = reinterpret_cast<TensorrtShortFuncState*>(state);

    // The whole compute_function should be considered the critical section.
    // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    std::lock_guard<std::mutex> lock(*(trt_state->tensorrt_mu_ptr));

    const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
    const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
    const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
    auto fused_node_name = trt_state->fused_node_name;
    auto& dds_output_allocator_map = this->dds_output_allocator_maps_[fused_node_name];
    auto trt_engine = trt_state->engine->get();
    auto trt_context = trt_state->context->get();
    auto max_context_mem_size_ptr = trt_state->max_context_mem_size_ptr;
    auto context_memory = trt_state->context_memory;
    int num_outputs = static_cast<int>(output_indexes.size());
    std::unordered_map<std::string, std::vector<int32_t>> shape_tensor_values;        // This map holds "shape tensor -> shape values" for the shape tensor input across this inference run
    std::unordered_map<std::string, std::vector<int64_t>> shape_tensor_values_int64;  // same as above but for int64 shape tensor input

    OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA,
                     narrow<OrtDevice::DeviceId>(device_id_));
    OrtMemoryInfo mem_info("", OrtAllocatorType::OrtDeviceAllocator, device);
    if (alloc_ == nullptr) {
      Ort::ThrowOnError(api->KernelContext_GetAllocator(context, &mem_info, &alloc_));
    }
    OrtAllocator* alloc = alloc_;

    void* cuda_stream;
    Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &cuda_stream));
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    // Check before using trt_engine
    if (trt_engine == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "No engine is found.");
    }

    // Get input and output binding names
    int total_bindings = trt_engine->getNbIOTensors();
    std::vector<char const*> input_binding_names, output_binding_names;
    for (int i = 0, end = total_bindings; i < end; ++i) {
      auto const& name = trt_engine->getIOTensorName(i);
      auto const& mode = trt_engine->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        input_binding_names.push_back(name);
      } else {
        output_binding_names.push_back(name);
      }
    }

    /*
     * Set input shapes and bind input buffers
     */
    std::vector<IAllocatorUniquePtr<void>> scratch_buffers;
    for (size_t i = 0, end = input_binding_names.size(); i < end; ++i) {
      char const* input_name = input_binding_names[i];

      size_t input_index = 0;
      const auto iter = input_indexes.find(input_name);
      if (iter != input_indexes.end()) {
        input_index = iter->second;
      }

      Status status = BindContextInput(ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values, shape_tensor_values_int64, scratch_buffers, alloc, stream);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }

    /*
     * Set output shapes and bind output buffers
     */
    std::unordered_map<char const*, void*> buffers;
    buffers.reserve(num_outputs);
    using OutputOrtValue = Ort::UnownedValue;
    std::unordered_map<size_t, OutputOrtValue> output_tensors;
    output_tensors.reserve(num_outputs);
    std::unordered_map<size_t, int> output_dim_sizes;
    output_dim_sizes.reserve(num_outputs);

    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
      char const* output_name = output_binding_names[i];

      size_t output_index = 0;
      const auto& index_iter = output_indexes.find(output_name);
      if (index_iter != output_indexes.end()) {
        output_index = index_iter->second;
      }

      size_t output_type = 0;
      const auto type_iter = output_types.find(output_name);
      if (type_iter != output_types.end()) {
        output_type = type_iter->second;
      }

      Status status = BindContextOutput(ctx, trt_context, output_name, output_index, output_type, i, output_tensors, output_dim_sizes,
                                        dds_output_allocator_map, scratch_buffers, alloc, buffers);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }

    // Set execution context memory
    if (trt_state->context_memory_sharing_enable) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
      size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      if (mem_size > *max_context_mem_size_ptr) {
        *max_context_mem_size_ptr = mem_size;
        *context_memory = IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, *max_context_mem_size_ptr, true /*use_reserve*/);
      }
      trt_context->setDeviceMemory((*context_memory).get());
    }

    // Start CUDA graph capture.
    // Note: The reason we don't put graph capture in OnRunStart() like CUDA EP does is because
    // current ORT TRT doesn't get cuda stream until compute time and graph capture requires cuda stream.
    if (cuda_graph_enable_ && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
      LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
      cuda_graph_.SetStream(stream);
      CaptureBegin(0);
    }

    // Run TRT inference
    if (!trt_context->enqueueV3(stream)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP execution context enqueue failed.");
    }

    /*
     * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this function concurrently,
     * TRT EP needs to carefully take care of concurrency here, if not, following concurrent issue might happen:
     *
     * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per stream.
     * In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling InferenceSession::Run() concurrently,
     * the trt execution context instance is shared by all the threads and each thread aquires different stream from ORT.
     * So TRT EP will end up having one trt execution context using multiple streams which is not suggested.
     * But, since the whole compute_func() is protected by the lock and if cudaStreamSynchronize() is enforced here, one trt execution context per stream
     * is guaranteed.
     *
     * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all operations to prevent the concurrent issue mentioned above.
     * However, if cuda graph is enabled, TRT EP won't call cudaStreamSynchronize() since it's not allowed during graph capture.
     */
    if (sync_stream_after_enqueue_) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }

    // Assign TRT output back to ORT output
    // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
    // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
      char const* output_name = output_binding_names[i];

      size_t output_type = 0;
      const auto& iter = output_types.find(output_name);
      if (iter != output_types.end()) {
        output_type = iter->second;
      }

      if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end()) {
        size_t output_index = 0;
        const auto& index_iter = output_indexes.find(output_name);
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        auto status = BindKernelOutput(ctx, &mem_info, dds_output_allocator_map, output_name, output_index, output_type, stream);
        if (status != Status::OK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, status.ErrorMessage());
        }
      } else {
        auto& output_tensor = output_tensors[i];
#if NV_TENSORRT_MAJOR < 10
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          auto output_tensor_ptr = output_tensor.GetTensorMutableData<int64_t>();
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
#endif
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
          auto output_tensor_ptr = output_tensor.GetTensorMutableData<double>();
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<float, double>(stream, reinterpret_cast<float*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
      }
    }

    // End CUDA graph capture.
    // Note: One reason we don't put end of graph capture in OnRunEnd() like CUDA EP does is because of cuda stream mentioned in graph capture
    // above, another reason is because OnRunEnd() is not synchronized with OnRunStart() and ExecuteGraph() per inference_session.cc.
    // It's safe to start/end CUDA graph capture in compute_func() here since cuda graph object is maintained by a per thread basis.
    if (cuda_graph_enable_ && !IsGraphCaptured(0)) {
      if (IsGraphCaptureAllowed()) {
        CaptureEnd(0);
        // CUDA work issued to a capturing stream doesn’t actually run on the GPU,
        // so run the captured graph here to actually execute the work.
        ORT_RETURN_IF_ERROR(ReplayGraph(0));
      } else {
        IncrementRegularRunCountBeforeGraphCapture();
      }
    }

    return Status::OK();
  };

  node_compute_funcs.push_back(compute_info);
  return Status::OK();
}

void TensorrtExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const {
  auto allocator = allocators[GetOrtDeviceByMemType(OrtMemTypeCPU)];
  RegisterCudaStreamHandles(stream_handle_registry,
                            OrtDevice::GPU,
                            allocator,
                            true /* release_cpu_buffer_on_cuda_stream */,
                            stream_,
                            external_stream_ /* use_existing_stream */,
                            external_cudnn_handle_,
                            external_cublas_handle_,
                            {});
}

OrtDevice TensorrtExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeCPUInput)
    return OrtDevice();
  if (mem_type == OrtMemTypeCPUOutput)
    return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                     default_device_.Id());
  return default_device_;
}

}  // namespace onnxruntime
