// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/util/math.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/providers/webgpu/tensor/pad.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status PadProgram::GenerateShaderCode(ShaderHelper& shader) const {
  if (!dim_value_zero_) {
    shader.AddInput("data", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "tensor/pad.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(dim_value_zero, dim_value_zero_),
                             WGSL_TEMPLATE_PARAMETER(is_float16, is_float16_),
                             WGSL_TEMPLATE_PARAMETER(pad_mode, mode_),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status Pad::ComputeInternal(ComputeContext& context) const {
  const Tensor* input_tensor = context.Input<Tensor>(0);
  auto const& input_shape = input_tensor->Shape();
  size_t dimension_count = input_shape.NumDimensions();

  const PadsVector* p_pads = &pads_;
  const PadsVector* p_slices = &slices_;

  PadsVector pads;
  PadsVector slices;
  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  if (is_dynamic_) {
    size_t data_rank = input_tensor->Shape().NumDimensions();

    const Tensor* pads_tensor = context.Input<Tensor>(1);
    auto pads_tensor_dims = pads_tensor->Shape().GetDims();
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * num_axes] "
                "or a 2D tensor of shape [1, 2 * num_axes]");

    const auto pads_data = pads_tensor->DataAsSpan<int64_t>();

    // Compute Pads by applying axes if specified otherwise copy the supplied pads.
    PadBase::ComputePads(context.KernelContext(), data_rank, pads_data, pads);

    // Separate out any negative pads into the slices array
    PadBase::SeparateNegativeToSlices(pads, slices);

    p_pads = &pads;
    p_slices = &slices;
  }

  auto output_dims(input_shape.AsShapeVector());
  ORT_ENFORCE(dimension_count * 2 == p_pads->size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  std::vector<int32_t> lower_pads(dimension_count);
  for (size_t i = 0; i < dimension_count; i++) {
    int64_t lower_pad = (*p_pads)[i] + (*p_slices)[i];
    int64_t upper_pad = (*p_pads)[i + dimension_count] + (*p_slices)[i + dimension_count];
    lower_pads[i] = static_cast<int32_t>(lower_pad);
    output_dims[i] += lower_pad + upper_pad;
  }
  TensorShape output_shape(output_dims);

  // special case when there is a dim value of 0 in the shape. behavior depends on mode
  bool dim_value_zero = input_shape.Size() == 0;
  if (dim_value_zero) {
    ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode_, input_shape, output_shape));
  }

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = onnxruntime::narrow<uint32_t>(output_shape.Size());
  if (output_size == 0) {
    // Do not need to fill output, return
    return Status::OK();
  }

  // Read constant value and bitcast to uint32.
  uint32_t value_uint32 = 0;
  const auto data_type = input_tensor->GetElementType();
  bool is_float16 = data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  const Tensor* value_tensor = context.Input<Tensor>(2);
  if (!is_dynamic_) {
    if (is_float16) {
      uint16_t value = math::floatToHalf(value_);
      std::memcpy(&value_uint32, &value, sizeof(value));
    } else {
      std::memcpy(&value_uint32, &value_, sizeof(value_uint32));
    }
  } else if (value_tensor) {
    ORT_ENFORCE(value_tensor->DataType() == input_tensor->DataType() && value_tensor->Shape().Size() == 1,
                "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        uint16_t value = value_tensor->Data<MLFloat16>()[0].val;
        std::memcpy(&value_uint32, &value, sizeof(value));
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        std::memcpy(&value_uint32, value_tensor->DataRaw(), sizeof(value_uint32));
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input type: ", static_cast<int>(data_type));
    }
  }

  PadProgram program{mode_, dim_value_zero, is_float16};
  if (!dim_value_zero) {
    program.AddInput({input_tensor, ProgramTensorMetadataDependency::Rank});
  }
  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(mode_)), dim_value_zero)
      .AddUniformVariables({{gsl::span<const int32_t>(lower_pads.data(), lower_pads.size())}, {output_size}, {value_uint32}});

  return context.RunProgram(program);
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    2, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    13, 17,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    18, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pad,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);
ONNX_OPERATOR_KERNEL_EX(
    Pad,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Pad);

}  // namespace webgpu
}  // namespace onnxruntime
