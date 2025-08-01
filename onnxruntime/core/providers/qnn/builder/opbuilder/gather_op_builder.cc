// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles Gather and GatherElements
class GatherOpBuilder : public BaseOpBuilder {
 public:
  GatherOpBuilder() : BaseOpBuilder("GatherOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status GatherOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger) const {
  // On QNN CPU backend, the QNN validator does not properly reject unsupported input shapes.
  // This causes a Qnn graph execution error. So, reject those configs here.
  // We should consider not using QNN CPU backend for onnxruntime unit tests.
  const std::string& op_type = node_unit.OpType();
  if (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU && op_type == "GatherElements") {
    const auto& input0 = node_unit.Inputs()[0];
    std::vector<uint32_t> input0_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input0.node_arg, input0_shape),
                      "Cannot get input[0] shape for ", op_type, " node ", node_unit.Name());

    const size_t input0_rank = input0_shape.size();
    ORT_RETURN_IF_NOT(input0_rank > 1 && input0_rank <= 4,
                      "QNN CPU backend does not support ", op_type, " with input[0] of rank ", input0_rank);
  }

  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

// Makes negative indices positive and converts int64 indices to another integer type (typically int32 or uint32).
// The input and output are both represented as byte arrays.
template <typename SrcType, typename DstType>
static bool FixStaticIndices(const std::vector<uint8_t>& onnx_bytes,
                             int64_t input0_axis_dim,
                             /*out*/ std::vector<uint8_t>& qnn_bytes) {
  const size_t num_elems = onnx_bytes.size() / sizeof(SrcType);
  gsl::span<const SrcType> onnx_indices{reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_elems};

  qnn_bytes.resize(num_elems * sizeof(DstType));
  gsl::span<DstType> qnn_indices{reinterpret_cast<DstType*>(qnn_bytes.data()), num_elems};

  for (size_t i = 0; i < num_elems; i++) {
    SrcType onnx_index = onnx_indices[i];

    // Try to make a negative index positive by adding rank.
    if (onnx_index < 0) {
      onnx_index += static_cast<SrcType>(input0_axis_dim);
    }

    if (onnx_index < 0 || static_cast<int64_t>(onnx_index) >= input0_axis_dim) {
      return false;  // QNN does not support out-of-bounds indices.
    }

    qnn_indices[i] = static_cast<DstType>(onnx_index);
  }

  return true;
}

// Gets the size of input0 on the axis dimension.
static Status GetInput0AxisDimValue(const QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    int64_t default_axis_value,
                                    /*out*/ int64_t& axis_dim_value) {
  const auto& input0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input0_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input0.node_arg, input0_shape),
                    "Cannot get shape for ", node_unit.OpType(), " input[0] ", input0.node_arg.Name());

  int64_t rank = static_cast<int64_t>(input0_shape.size());
  NodeAttrHelper node_helper(node_unit);
  int64_t onnx_axis = node_helper.Get("axis", default_axis_value);
  if (onnx_axis < 0) {
    onnx_axis += rank;
  }
  ORT_RETURN_IF_NOT((onnx_axis >= 0 && onnx_axis < static_cast<int64_t>(input0_shape.size())),
                    "QNN requires axis range [0, rank-1] for ", node_unit.OpType());

  axis_dim_value = static_cast<int64_t>(input0_shape[onnx_axis]);

  return Status::OK();
}

// Processes the indices input to Gather operators.
//
// QNN only supports int32 / uint32 as indices tensor data types.
// When indices tensor is an initializer, statically cast values int64 -> int32.
// When dynamic input, add explicit QNN Cast node for int64 -> int32 conversion.
//
// The HTP backend only supports dynamic int64 indices if they are a graph input.
static Status ProcessIndicesInput(QnnModelWrapper& qnn_model_wrapper,
                                  const NodeUnitIODef& indices_input,
                                  int64_t input0_axis_dim,
                                  const logging::Logger& logger,
                                  std::vector<std::string>& input_names,
                                  bool do_op_validation) {
  const auto& indices_tensor_name = indices_input.node_arg.Name();

  TensorInfo indices_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  std::vector<uint8_t> qnn_indices_bytes;

  // Get raw bytes for static indices.
  // If indices are int64, convert them to int32 and update indices_info.qnn_data_type.
  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*indices_info.initializer_tensor, onnx_indices_bytes));

    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      ORT_RETURN_IF_NOT((FixStaticIndices<int64_t, int32_t>(onnx_indices_bytes, input0_axis_dim, qnn_indices_bytes)),
                        "QNN does not support negative index values for Gather* ops");
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
    } else if (indices_info.qnn_data_type == QNN_DATATYPE_INT_32) {
      ORT_RETURN_IF_NOT((FixStaticIndices<int32_t, int32_t>(onnx_indices_bytes, input0_axis_dim, qnn_indices_bytes)),
                        "QNN does not support negative index values for Gather* ops");
    } else {
      qnn_indices_bytes = std::move(onnx_indices_bytes);
    }
  }

  std::vector<uint32_t> cast_output_shape(indices_info.shape);
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_tensor_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << indices_tensor_name;
  } else {
    QnnTensorWrapper input_tensorwrapper(indices_tensor_name,
                                         qnn_model_wrapper.GetTensorType(indices_tensor_name),
                                         indices_info.qnn_data_type, QnnQuantParamsWrapper(),
                                         std::move(indices_info.shape),
                                         std::move(qnn_indices_bytes));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  // Insert QNN Cast op to convert dynamic indices from int64 to int32.
  std::string indices_casted_name{indices_tensor_name};
  if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
    assert(!indices_info.is_initializer);
    indices_casted_name += "_int32";
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_casted_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << indices_casted_name;
    } else {
      QnnTensorWrapper indices_cast_tensor(indices_casted_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           QNN_DATATYPE_INT_32,
                                           QnnQuantParamsWrapper(),
                                           std::move(cast_output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(indices_cast_tensor)),
                        "Failed to add gather indices cast tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(indices_casted_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {indices_tensor_name},
                                                        {indices_casted_name},
                                                        {},
                                                        do_op_validation),
                        "Failed to add gather indices cast node.");
    }
  }
  input_names.push_back(indices_casted_name);
  return Status::OK();
}

Status GatherOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() != 2, "QNN EP: ", node_unit.OpType(), " operator must have two inputs");
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  int64_t input0_axis_dim = 0;
  ORT_RETURN_IF_ERROR(GetInput0AxisDimValue(qnn_model_wrapper, node_unit, /*default_axis_value=*/0, input0_axis_dim));
  return ProcessIndicesInput(qnn_model_wrapper, inputs[1], input0_axis_dim, logger, input_names, do_op_validation);
}

Status GatherOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  const bool is_gather_elems = node_unit.OpType() == "GatherElements";

  // Create QNN 'axis' parameter.
  std::vector<std::string> param_tensor_names;
  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(),
                             (is_gather_elems ? QNN_OP_GATHER_ELEMENTS_PARAM_AXIS : QNN_OP_GATHER_PARAM_AXIS),
                             axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  if (is_gather_elems) {
    return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                          logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
  }

  // if indicies is scalar shape, then need to add Reshape node
  const auto& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);
  const auto& indices_input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);

  // Calculate the output shape
  std::vector<uint32_t> qnn_output_shape;
  auto input_rank = input_tensor_wrapper.GetTensorRank();
  auto indices_rank = indices_input_tensor_wrapper.GetTensorRank();
  qnn_output_shape.reserve(static_cast<size_t>(input_rank - 1 + indices_rank));

  const auto& gather_indices = node_unit.Inputs()[1];
  std::vector<uint32_t> indices_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_indices.node_arg, indices_shape),
                    "Cannot get shape");

  // replace the dimension for p.axis with the shape from the indices
  std::copy(input_tensor_wrapper.GetTensorDims().begin(), input_tensor_wrapper.GetTensorDims().begin() + axis_value,
            std::back_inserter(qnn_output_shape));

  const auto& indicies_shape = indices_input_tensor_wrapper.GetTensorDims();
  std::copy(indicies_shape.begin(), indicies_shape.end(), std::back_inserter(qnn_output_shape));

  std::copy(input_tensor_wrapper.GetTensorDims().begin() + axis_value + 1, input_tensor_wrapper.GetTensorDims().end(),
            std::back_inserter(qnn_output_shape));

  const auto& gather_output = node_unit.Outputs()[0];
  const auto& output_name = gather_output.node_arg.Name();

  QnnQuantParamsWrapper quantize_param;
  ORT_RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, gather_output));

  const auto* type_proto = gather_output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(quantize_param.IsQuantized(), type_proto, qnn_data_type));

  if (quantize_param.IsPerTensor()) {
    // Make sure the output quantization parameters are equal to the input.
    ORT_RETURN_IF_ERROR(SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                                 0 /*input_index*/, 0 /*output_index*/, qnn_data_type,
                                                                 quantize_param));
  }

  std::vector<uint32_t> target_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_output.node_arg, target_output_shape),
                    "Cannot get shape");

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  bool reshape_required = (qnn_output_shape.size() != target_output_shape.size());

  struct CastNodeInfo {
    std::string node_name;
    std::string input_name;
    std::string output_name;
  };
  std::vector<CastNodeInfo> cast_node_info_vec;

  // Get the output info for the gather output tensor
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(gather_output, output_info));

  // Check if we need to add a cast node for int64
  bool needs_int64_cast = false;
  if (is_graph_output) {
    for (const auto& input_name : input_names) {
      if (input_name.find("_cast_int32") != std::string::npos) {
        needs_int64_cast = true;
        break;
      }
    }
  }

  // If a cast to int64 is needed, add the cast node
  if (needs_int64_cast) {
    std::string cast_node_name = output_name + "_cast_int64";
    std::string cast_input_name = output_name + "_cast_int64_aux";
    std::string cast_output_name = output_name;

    // Create the cast input tensor wrapper
    QnnTensorWrapper cast_input_tensorwrapper(cast_input_name,
                                              QNN_TENSOR_TYPE_NATIVE,
                                              output_info.qnn_data_type,
                                              output_info.quant_param.Copy(),
                                              std::move(qnn_output_shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_input_tensorwrapper)), "Failed to add tensor.");
    cast_node_info_vec.push_back({cast_node_name, cast_input_name, cast_output_name});
    Qnn_TensorType_t cast_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper cast_output(output_name, cast_tensor_type, qnn_data_type, std::move(quantize_param),
                                 std::move(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_output)), "Failed to add tensor.");
  }

  std::string gather_output_name = output_name;
  if (reshape_required) {
    gather_output_name += "_ort_qnn_ep_reshape";
  } else if (needs_int64_cast) {
    gather_output_name += "_cast_int64_aux";
  }

  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper gather_output_wrapper(gather_output_name, tensor_type, qnn_data_type, quantize_param.Copy(),
                                         std::move(qnn_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gather_output_wrapper)), "Failed to add tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {gather_output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");

  if (reshape_required) {
    // Add Reshape Node after Gather.
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, qnn_data_type, std::move(quantize_param),
                                    std::move(target_output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output)), "Failed to add tensor.");
    std::string node_output_name = output_name;

    if (needs_int64_cast) {
      // If needs_int64 is true, the output name should be the input name of the cast node
      node_output_name = output_name + "_cast_int64_aux";
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RESHAPE,
                                                      {gather_output_name},
                                                      {node_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add node.");
  }

  if (needs_int64_cast) {
    for (const auto& cast_node_info : cast_node_info_vec) {
      // Insert cast node.
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_info.node_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {cast_node_info.input_name},
                                                        {cast_node_info.output_name},
                                                        {}),
                        " Failed to add Cast node");
    }
  }

  return Status::OK();
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
