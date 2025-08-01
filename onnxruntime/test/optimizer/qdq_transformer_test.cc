// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include "core/common/inlined_containers_fwd.h"
#include "core/common/span_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/node_unit.h"
#include "core/framework/int4.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/optimizer/double_qdq_pairs_remover.h"
#include "core/optimizer/qdq_transformer/weight_bias_quantization.h"
#include "core/optimizer/qdq_transformer/where_dummy_dq.h"
#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"
#include "core/optimizer/qdq_transformer/qdq_propagation.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"

#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/common/dnnl_op_test_utils.h"

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "qdq_test_utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif  // #if defined(_MSC_VER)

struct QDQOpKeys {
  const char* quantize_linear;
  const char* dequantize_linear;
};

constexpr QDQOpKeys GetQDQOpKeys(bool use_contrib_qdq) {
  if (use_contrib_qdq) {
    return {"com.microsoft.QuantizeLinear", "com.microsoft.DequantizeLinear"};
  }
  return {"QuantizeLinear", "DequantizeLinear"};
}

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
void QDQTransformerConvTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);

      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value &&
                    std::is_same<BiasType, int32_t>::value &&
                    (std::is_same<InputType, uint8_t>::value ||
                     QDQIsInt8Allowed() && std::is_same<WeightType, int8_t>::value)) {
        EXPECT_EQ(op_to_count["QLinearConv"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["Conv"], 1);
        EXPECT_EQ(op_to_count["QLinearConv"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 4);
      }
    };

    TransformerTester(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape,
                                                                                        use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape,
                                                                                        use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape,
                                                                                        use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, Conv_U8X8U8) {
  DNNL_GTEST_SKIP();

  QDQTransformerConvTests<uint8_t, uint8_t, int32_t, uint8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, int32_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_U8X8U8_Bias_Not_i32) {
  // bias not int32_t
  QDQTransformerConvTests<uint8_t, uint8_t, int8_t, uint8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_U8X8S8) {
  // output not uint8_t
  QDQTransformerConvTests<uint8_t, uint8_t, int32_t, int8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, int32_t, int8_t>();
}

TEST(QDQTransformerTests, Conv_S8X8U8) {
  // input not uint8_t
  QDQTransformerConvTests<int8_t, uint8_t, int32_t, uint8_t>();
  QDQTransformerConvTests<int8_t, int8_t, int32_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_S8X8S8) {
  // input not uint8_t and output not uint8_t
  QDQTransformerConvTests<int8_t, uint8_t, int32_t, int8_t>();
  QDQTransformerConvTests<int8_t, int8_t, int32_t, int8_t>();
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_UInt8) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       int opset_version, bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + MaxPool
      auto* dq_maxpool_output = AddQDQNodePair<uint8_t>(builder, conv_output, .0039f, 135, use_contrib_qdq);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<uint8_t>(builder, maxpool_output, .0039f, 135, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], opset_version < 12 ? 2 : 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], opset_version < 12 ? 1 : 0);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      opset_version);
  };

  test_case({1, 12, 37}, {32, 12, 5}, 11);
  test_case({1, 12, 37}, {32, 12, 5}, 12);
  test_case({1, 12, 37}, {32, 12, 5}, 18);
  test_case({1, 12, 37}, {32, 12, 5}, 19);
  test_case({1, 12, 37}, {32, 12, 5}, 11, true);  // Use com.microsoft QDQ ops

  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 11);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 12);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 18);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 19);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 12, true);  // Use com.microsoft QDQ ops

  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 11);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 12);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 18);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 19);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 18, true);  // Use com.microsoft QDQ ops
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 19, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_Int8) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + MaxPool
      auto* dq_maxpool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0039f, 7, use_contrib_qdq);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, maxpool_output, .0039f, 7, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0039f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0039f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const std::vector<std::string> expected_op_types_in_order{
          qdq_keys.quantize_linear,
          "QLinearConv",
          "MaxPool",
          "Reshape"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, true);  // Use com.microsoft QDQ ops
}

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__) || !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, DQ_S8_to_U8) {
  auto test_case = [](bool use_contrib_qdq) {
    const std::vector<int64_t>& input_shape = {19, 37};
    const std::vector<int64_t>& weights_shape = {37, 23};

    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);

      // Use full range weight values to expose avx2 u8s8 overflow problems
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -128, 127);
      auto* output_arg = builder.MakeOutput();

      // add QDQ activation
      typedef std::numeric_limits<uint8_t> Input1Limits;
      auto* dq1_output = AddQDQNodePair<int8_t>(builder, input1_arg, .039f,
                                                (int8_t)((Input1Limits::max() + Input1Limits::min()) / 2 + 1),
                                                use_contrib_qdq);

      // add DQ weight
      auto* dq_w_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);

      builder.AddNode("MatMul", {dq1_output, dq_w_output}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
      EXPECT_EQ(op_to_count["MatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsAvx2PrecisionMode, "1"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);
  };

  test_case(false);  // Use ONNX QDQ ops
  test_case(true);   // Use com.microsoft QDQ ops
}
#endif  // Only for X64 with contrib ops enabled

template <typename InputType, typename OutputType>
void QDQTransformerAveragePoolTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
        EXPECT_EQ(op_to_count["AveragePool"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 0);
        EXPECT_EQ(op_to_count["AveragePool"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(BuildQDQAveragePoolTestCase<InputType, OutputType>(input_shape, 0 /*count_include_pad*/,
                                                                         use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQAveragePoolTestCase<InputType, OutputType>(input_shape, 0 /*count_include_pad*/,
                                                                         use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    // TODO: fix opset 19
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
  test_case({1, 12, 37}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, AveragePool_S8S8) {
  QDQTransformerAveragePoolTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, AveragePool_U8U8) {
  QDQTransformerAveragePoolTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, AveragePool_S8U8) {
  QDQTransformerAveragePoolTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, AveragePool_U8S8) {
  QDQTransformerAveragePoolTests<uint8_t, int8_t>();
}

template <typename InputType, typename OutputType>
void QDQTransformerGlobalAveragePoolTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearGlobalAveragePool"], 1);
        EXPECT_EQ(op_to_count["GlobalAveragePool"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearGlobalAveragePool"], 0);
        EXPECT_EQ(op_to_count["GlobalAveragePool"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(BuildQDQGlobalAveragePoolTestCase<InputType, OutputType>(input_shape, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQGlobalAveragePoolTestCase<InputType, OutputType>(input_shape, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    // TODO: fix opset 19
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
  test_case({1, 12, 37}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, GlobalAveragePool_S8S8) {
  QDQTransformerAveragePoolTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_U8U8) {
  QDQTransformerAveragePoolTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_S8U8) {
  QDQTransformerAveragePoolTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_U8S8) {
  QDQTransformerAveragePoolTests<uint8_t, int8_t>();
}

template <typename Input1Type, typename Input2Type, typename OutputType>
void QDQTransformerBinaryOpTests(const std::string& op_type) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if (std::is_same<Input1Type, Input2Type>::value &&
          std::is_same<Input1Type, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinear" + op_type], 1);
        EXPECT_EQ(op_to_count[op_type], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinear" + op_type], 0);
        EXPECT_EQ(op_to_count[op_type], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 3);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
      }
    };

    TransformerTester(BuildBinaryOpTestCase<Input1Type, Input2Type, OutputType>(input_shape, op_type, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildBinaryOpTestCase<Input1Type, Input2Type, OutputType>(input_shape, op_type, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildBinaryOpTestCase<Input1Type, Input2Type, OutputType>(input_shape, op_type, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
  test_case({1, 12, 37}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, Add) {
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, uint8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Add");
}

TEST(QDQTransformerTests, Add_Have_Different_Types) {
  QDQTransformerBinaryOpTests<uint8_t, int8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<uint8_t, int8_t, uint8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, uint8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, uint8_t>("Add");
}

TEST(QDQTransformerTests, Mul) {
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, uint8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Mul");
}

TEST(QDQTransformerTests, Mul_Have_Different_Types) {
  QDQTransformerBinaryOpTests<uint8_t, int8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<uint8_t, int8_t, uint8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, uint8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, uint8_t>("Mul");
}

template <typename Input1Type, typename Input2Type, typename OutputType>
void QDQTransformerMatMulTests(bool has_output_q) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      // add QDQ 1
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output, use_contrib_qdq);

      // add QDQ 2
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output, use_contrib_qdq);

      if (has_output_q) {
        // add binary operator
        auto* matmul_op_output = builder.MakeIntermediate();
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {matmul_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(matmul_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg, use_contrib_qdq);
      } else {
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {output_arg});
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if (has_output_q) {
        if constexpr (std::is_same<Input1Type, OutputType>::value &&
                      (std::is_same<Input1Type, uint8_t>::value ||
                       QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
        } else {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 3);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
        }
      } else {
        if constexpr (std::is_same<Input1Type, uint8_t>::value ||
                      (QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
        } else {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
        }
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 2, 2}, {1, 2, 4});
  test_case({1, 23, 13, 13}, {13, 13});
  test_case({1, 22, 11, 13, 15}, {1, 22, 11, 15, 15});
  test_case({1, 2, 2}, {1, 2, 4}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, MatMul_U8U8U8) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8S8) {
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8U8S8) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8U8) {
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8S8) {
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8U8) {
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8S8) {
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8U8) {
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(true);
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests(bool has_output_q, bool has_bias, bool beta_not_one = false) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      std::vector<NodeArg*> input_args;

      // add QDQ A
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output, use_contrib_qdq);

      input_args.push_back(dq1_output);

      // add QDQ B
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output, use_contrib_qdq);
      input_args.push_back(dq2_output);

      if (has_bias) {
        auto* dq_bias_output = builder.MakeIntermediate();
        auto* bias = builder.MakeInitializer<BiasType>({input2_shape[1]}, static_cast<BiasType>(0), static_cast<BiasType>(127));
        builder.AddDequantizeLinearNode<BiasType>(bias, 0.00156f,
                                                  0,
                                                  dq_bias_output, use_contrib_qdq);
        input_args.push_back(dq_bias_output);
      }

      Node* gemm_node = nullptr;

      if (has_output_q) {
        auto* gemm_op_output = builder.MakeIntermediate();
        gemm_node = &builder.AddNode("Gemm", input_args, {gemm_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(gemm_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg, use_contrib_qdq);
      } else {
        gemm_node = &builder.AddNode("Gemm", input_args, {output_arg});
      }

      if (beta_not_one) {
        gemm_node->AddAttribute("beta", 2.0f);
      }
    };

    auto check_binary_op_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if ((!has_output_q || std::is_same_v<Input1Type, OutputType>) && (!has_bias || (std::is_same_v<BiasType, int32_t> && !beta_not_one)) &&
          (std::is_same_v<Input1Type, uint8_t> || std::is_same_v<Input2Type, int8_t>)) {
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
        EXPECT_EQ(op_to_count["Gemm"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], has_output_q ? 1 : 0);
      } else {
        int q_count = 2;   // Q for A and B
        int dq_count = 2;  // DQ for A and B
        if (has_bias) {
          dq_count++;
        }
        if (has_output_q) {
          q_count++;
          dq_count++;
        }
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 0);
        EXPECT_EQ(op_to_count["Gemm"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], q_count);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], dq_count);
      }
    };

    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({2, 2}, {2, 4});
  test_case({13, 15}, {15, 15});
  test_case({2, 2}, {2, 4}, true);  // Use com.microsoft QDQ ops
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests() {
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true, true);
}

TEST(QDQTransformerTests, Gemm_U8U8U8) {
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8S8) {
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8U8S8) {
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8U8) {
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8S8) {
  QDQTransformerGemmTests<int8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8U8) {
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8S8) {
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8U8) {
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t, uint8_t>();
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Gather -> Q.
template <typename QuantType>
static void RunGatherDropQDQTestCase(const std::vector<int64_t>& input1_shape,
                                     const std::vector<int64_t>& weights_shape,
                                     bool use_contrib_qdq = false,
                                     int opset = 12) {
  auto build_test_case = [input1_shape, weights_shape, use_contrib_qdq](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>(input1_shape, 0, weights_shape[0] - 1);
    auto* output_arg = builder.MakeOutput();

    // add Gather
    auto* weight = builder.MakeInitializer<QuantType>(weights_shape, std::numeric_limits<QuantType>::min(),
                                                      std::numeric_limits<QuantType>::max());
    auto* dq_w_output = builder.MakeIntermediate();
    auto* gather_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(weight, .003f, 1, dq_w_output, use_contrib_qdq);
    builder.AddNode("Gather", {dq_w_output, input1_arg}, {gather_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(gather_output, .003f, 1, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Gather"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Gather -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, Gather) {
  RunGatherDropQDQTestCase<int8_t>({12, 37}, {24, 12});              // Use ONNX int8 QDQ ops
  RunGatherDropQDQTestCase<int8_t>({12, 37}, {24, 12}, true);        // Use com.microsoft QDQ ops
  RunGatherDropQDQTestCase<int16_t>({12, 37}, {24, 12}, true);       // Use int16 com.microsoft QDQ ops
  RunGatherDropQDQTestCase<int16_t>({12, 37}, {24, 12}, false, 21);  // Use ONNX int16 QDQ ops
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Reshape -> Q.
template <typename QuantType>
static void RunReshapeDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                      const std::vector<int64_t>& new_shape,
                                      bool use_contrib_qdq = false,
                                      int opset = 12) {
  auto build_test_case = [input_shape, new_shape, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    // Add Reshape node
    auto* new_shape_arg = builder.Make1DInitializer(new_shape);
    auto* input_arg_dq = builder.MakeIntermediate();
    auto* reshape_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode("Reshape", {input_arg_dq, new_shape_arg}, {reshape_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(reshape_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Reshape"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Reshape -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, ReshapeDropQDQ) {
  RunReshapeDropQDQTestCase<int8_t>({1, 3, 2, 2}, {1, 12});
  RunReshapeDropQDQTestCase<int8_t>({1, 3, 2, 2}, {1, 12}, true);         // Use com.microsoft QDQ ops
  RunReshapeDropQDQTestCase<int16_t>({1, 3, 2, 2}, {1, 12}, true);        // Use int16 com.microsoft QDQ ops
  RunReshapeDropQDQTestCase<uint16_t>({1, 3, 2, 2}, {1, 12}, true);       // Use int16 com.microsoft QDQ ops
  RunReshapeDropQDQTestCase<int16_t>({1, 3, 2, 2}, {1, 12}, false, 21);   // Use int16 ONNX QDQ ops
  RunReshapeDropQDQTestCase<uint16_t>({1, 3, 2, 2}, {1, 12}, false, 21);  // Use int16 ONNX QDQ ops
}

// Runs a test case that checks if Q/DQ nodes are *not* dropped from DQ -> MaxPool -> Q if the quantization scale is
// negative.
template <typename QuantType>
static void RunMaxPoolNegativeScaleDropQDQTestCase() {
  auto build_test_case = [](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    const std::vector<int64_t> input_shape = {1, 17, 17, 3};
    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();

    constexpr float scale = -0.003f;
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* maxpool_output = builder.MakeIntermediate();

    builder.AddDequantizeLinearNode<QuantType>(input_arg, scale, zero_point, input_arg_dq);

    Node& maxpool_node = builder.AddNode("MaxPool", {input_arg_dq}, {maxpool_output});
    maxpool_node.AddAttribute("auto_pad", "VALID");
    maxpool_node.AddAttribute("kernel_shape", std::vector<int64_t>({2, 2}));

    builder.AddQuantizeLinearNode<QuantType>(maxpool_output, scale, zero_point, output_arg);
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["MaxPool"], 1);
    EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
  };

  constexpr int opset = 21;
  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are *not* dropped from DQ -> MaxPool -> Q for negative scale. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, MaxpoolDontDropQDQForNegativeScale) {
  RunMaxPoolNegativeScaleDropQDQTestCase<int8_t>();
  RunMaxPoolNegativeScaleDropQDQTestCase<uint8_t>();
  RunMaxPoolNegativeScaleDropQDQTestCase<int16_t>();
  RunMaxPoolNegativeScaleDropQDQTestCase<uint16_t>();
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> (Un)Squeeze -> Q.
template <typename QuantType>
static void RunSqueezeUnsqueezeDropQDQTestCase(const std::string& squeeze_type,
                                               const std::vector<int64_t>& input_shape,
                                               const std::vector<int64_t>& axes,
                                               bool use_contrib_qdq = false,
                                               int opset = 13) {
  auto build_test_case = [squeeze_type, input_shape, axes, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    // Add Squeeze node
    auto* axes_arg = builder.Make1DInitializer<int64_t>(axes);
    auto* input_arg_dq = builder.MakeIntermediate();
    auto* xsqueeze_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode(squeeze_type, {input_arg_dq, axes_arg}, {xsqueeze_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(xsqueeze_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [squeeze_type, use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count[squeeze_type], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Squeeze -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, SqueezeDropQDQ) {
  RunSqueezeUnsqueezeDropQDQTestCase<int8_t>("Squeeze", {1, 3, 2, 2}, {0});
  RunSqueezeUnsqueezeDropQDQTestCase<int8_t>("Squeeze", {1, 3, 2, 2}, {0}, true);    // Use MS domain QDQ ops
  RunSqueezeUnsqueezeDropQDQTestCase<int16_t>("Squeeze", {1, 3, 2, 2}, {0}, true);   // Use int16 MS domain QDQ ops
  RunSqueezeUnsqueezeDropQDQTestCase<uint16_t>("Squeeze", {1, 3, 2, 2}, {0}, true);  // Use int16 MS domain QDQ ops

  // With 16-bit ONNX QDQ ops.
  RunSqueezeUnsqueezeDropQDQTestCase<int16_t>("Squeeze", {1, 3, 2, 2}, {0}, false, 21);
  RunSqueezeUnsqueezeDropQDQTestCase<uint16_t>("Squeeze", {1, 3, 2, 2}, {0}, false, 21);
}

// Checks that Q/DQ nodes are dropped from DQ -> Unsqueeze -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, UnsqueezeDropQDQ) {
  RunSqueezeUnsqueezeDropQDQTestCase<int8_t>("Unsqueeze", {1, 3, 2, 2}, {0});
  RunSqueezeUnsqueezeDropQDQTestCase<int8_t>("Unsqueeze", {1, 3, 2, 2}, {0}, true);    // Use MS domain QDQ ops
  RunSqueezeUnsqueezeDropQDQTestCase<int16_t>("Unsqueeze", {1, 3, 2, 2}, {0}, true);   // Use int16 MS domain QDQ ops
  RunSqueezeUnsqueezeDropQDQTestCase<uint16_t>("Unsqueeze", {1, 3, 2, 2}, {0}, true);  // Use int16 MS domain QDQ ops

  // With 16-bit ONNX QDQ ops.
  RunSqueezeUnsqueezeDropQDQTestCase<int16_t>("Unsqueeze", {1, 3, 2, 2}, {0}, false, 21);
  RunSqueezeUnsqueezeDropQDQTestCase<uint16_t>("Unsqueeze", {1, 3, 2, 2}, {0}, false, 21);
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Flatten -> Q.
template <typename QuantType>
static void RunFlattenDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                      int64_t axis = 1,
                                      bool use_contrib_qdq = false,
                                      int opset = 21) {
  auto build_test_case = [input_shape, axis, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* flatten_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    Node& flatten_node = builder.AddNode("Flatten", {input_arg_dq}, {flatten_output});
    flatten_node.AddAttribute("axis", axis);

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(flatten_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Flatten"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Reshape -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, FlattenDropQDQ) {
  for (int64_t axis : {0, 1, 3}) {
    RunFlattenDropQDQTestCase<int8_t>({1, 3, 2, 2}, axis);
    RunFlattenDropQDQTestCase<int8_t>({1, 3, 2, 2}, axis, true, 13);    // Use com.microsoft QDQ ops
    RunFlattenDropQDQTestCase<int16_t>({1, 3, 2, 2}, axis, true, 13);   // Use int16 com.microsoft QDQ ops
    RunFlattenDropQDQTestCase<uint16_t>({1, 3, 2, 2}, axis, true, 13);  // Use int16 com.microsoft QDQ ops
    RunFlattenDropQDQTestCase<int16_t>({1, 3, 2, 2}, axis, false);      // Use int16 ONNX QDQ ops
    RunFlattenDropQDQTestCase<uint16_t>({1, 3, 2, 2}, axis, false);     // Use int16 ONNX QDQ ops
  }
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Expand -> Q.
template <typename QuantType>
static void RunExpandDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& expanded_shape,
                                     bool use_contrib_qdq = false,
                                     int opset = 21) {
  auto build_test_case = [input_shape, expanded_shape, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* expanded_shape_arg = builder.Make1DInitializer(expanded_shape);
    auto* expand_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode("Expand", {input_arg_dq, expanded_shape_arg}, {expand_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(expand_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Expand"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Expand -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, ExpandDropQDQ) {
  RunExpandDropQDQTestCase<int8_t>({1, 3, 1, 1}, {1, 3, 7, 13});
  RunExpandDropQDQTestCase<int8_t>({1, 3, 1, 1}, {1, 3, 7, 13}, true, 13);    // Use com.microsoft QDQ ops
  RunExpandDropQDQTestCase<int16_t>({1, 3, 1, 1}, {1, 3, 7, 13}, true, 13);   // Use int16 com.microsoft QDQ ops
  RunExpandDropQDQTestCase<uint16_t>({1, 3, 1, 1}, {1, 3, 7, 13}, true, 13);  // Use int16 com.microsoft QDQ ops
  RunExpandDropQDQTestCase<int16_t>({1, 3, 1, 1}, {1, 3, 7, 13}, false);      // Use int16 ONNX QDQ ops
  RunExpandDropQDQTestCase<uint16_t>({1, 3, 1, 1}, {1, 3, 7, 13}, false);     // Use int16 ONNX QDQ ops
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Tile -> Q.
template <typename QuantType>
static void RunTileDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& repeats,
                                   bool use_contrib_qdq = false,
                                   int opset = 21) {
  auto build_test_case = [input_shape, repeats, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* repeats_arg = builder.Make1DInitializer(repeats);
    auto* tile_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode("Tile", {input_arg_dq, repeats_arg}, {tile_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(tile_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Tile"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Tile -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, TileDropQDQ) {
  RunTileDropQDQTestCase<int8_t>({1, 3, 2, 2}, {1, 1, 3, 3});
  RunTileDropQDQTestCase<int8_t>({1, 3, 2, 2}, {1, 1, 3, 3}, true, 13);    // Use com.microsoft QDQ ops
  RunTileDropQDQTestCase<int16_t>({1, 3, 2, 2}, {1, 1, 3, 3}, true, 13);   // Use int16 com.microsoft QDQ ops
  RunTileDropQDQTestCase<uint16_t>({1, 3, 2, 2}, {1, 1, 3, 3}, true, 13);  // Use int16 com.microsoft QDQ ops
  RunTileDropQDQTestCase<int16_t>({1, 3, 2, 2}, {1, 1, 3, 3}, false);      // Use int16 ONNX QDQ ops
  RunTileDropQDQTestCase<uint16_t>({1, 3, 2, 2}, {1, 1, 3, 3}, false);     // Use int16 ONNX QDQ ops
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> Slice -> Q.
template <typename QuantType>
static void RunSliceDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    bool use_contrib_qdq = false,
                                    int opset = 21) {
  auto build_test_case = [input_shape, starts, ends, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* starts_arg = builder.Make1DInitializer(starts);
    auto* ends_arg = builder.Make1DInitializer(ends);
    auto* slice_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode("Slice", {input_arg_dq, starts_arg, ends_arg}, {slice_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(slice_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Slice"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> Slice -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, SliceDropQDQ) {
  RunSliceDropQDQTestCase<int8_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4});
  RunSliceDropQDQTestCase<int8_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4}, true, 13);  // Use com.microsoft QDQ ops
  // Use int16 com.microsoft QDQ ops
  RunSliceDropQDQTestCase<int16_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4}, true, 13);
  // Use int16 com.microsoft QDQ ops
  RunSliceDropQDQTestCase<uint16_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4}, true, 13);
  RunSliceDropQDQTestCase<int16_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4}, false);   // Use int16 ONNX QDQ ops
  RunSliceDropQDQTestCase<uint16_t>({1, 3, 5, 5}, {0, 1, 1, 1}, {1, 3, 4, 4}, false);  // Use int16 ONNX QDQ ops
}

// Runs a test case that checks if Q/DQ nodes are dropped from DQ -> GatherElements -> Q.
template <typename QuantType>
static void RunGatherElementsDropQDQTestCase(const std::vector<int64_t>& input_shape,
                                             const std::vector<int64_t>& indices_shape,
                                             const std::vector<int64_t>& indices_data,
                                             bool use_contrib_qdq = false,
                                             int opset = 21) {
  auto build_test_case = [input_shape, indices_shape, indices_data, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* indices_arg = builder.MakeInitializer<int64_t>(indices_shape, indices_data);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* gather_elements_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode("GatherElements", {input_arg_dq, indices_arg}, {gather_elements_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(gather_elements_output, .003f, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["GatherElements"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks that Q/DQ nodes are dropped from DQ -> GatherElements -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, GatherElementsDropQDQ) {
  RunGatherElementsDropQDQTestCase<int8_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0});
  // Use com.microsoft QDQ ops
  RunGatherElementsDropQDQTestCase<int8_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0}, true, 13);
  // Use int16 com.microsoft QDQ ops
  RunGatherElementsDropQDQTestCase<int16_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0}, true, 13);
  // Use int16 com.microsoft QDQ ops
  RunGatherElementsDropQDQTestCase<uint16_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0}, true, 13);
  RunGatherElementsDropQDQTestCase<int16_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0}, false);   // Use int16 ONNX QDQ ops
  RunGatherElementsDropQDQTestCase<uint16_t>({3, 3}, {2, 3}, {1, 2, 0, 2, 0, 0}, false);  // Use int16 ONNX QDQ ops
}

// Runs a test case whether Q/DQ nodes are dropped from DQ -> Reduce(Min|Max) -> Q.
template <typename QuantType>
static void RunReduceExtremumDropQDQTestCase(const std::string& op_type,
                                             const std::vector<int64_t>& input_shape,
                                             float qscale,
                                             bool expect_drop_qdq,
                                             bool use_contrib_qdq = false,
                                             int opset = 21) {
  auto build_test_case = [op_type, input_shape, qscale, use_contrib_qdq](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input_arg = builder.MakeInput<QuantType>(input_shape, qmin, qmax);
    auto* output_arg = builder.MakeOutput();
    QuantType zero_point = 1 + (qmax + qmin) / 2;

    auto* input_arg_dq = builder.MakeIntermediate();
    auto* reduce_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, qscale, zero_point, input_arg_dq, use_contrib_qdq);
    builder.AddNode(op_type, {input_arg_dq}, {reduce_output});

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(reduce_output, qscale, zero_point, output_arg, use_contrib_qdq);
  };

  auto check_graph = [op_type, expect_drop_qdq, use_contrib_qdq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count[op_type], 1);
    if (expect_drop_qdq) {
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    } else {
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    }
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

// Checks whether Q/DQ nodes are dropped from DQ -> Reduce(Min|Max) -> Q. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, ReduceExtremumDropQDQ) {
  // Check that Q/DQ nodes are dropped for positive scale
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMin", {3, 3}, 0.003f, true);
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMin", {3, 3}, 0.003f, true, true, 13);  // Use com.microsoft QDQ ops
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMax", {3, 3}, 0.003f, true);
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMax", {3, 3}, 0.003f, true, true, 13);  // Use com.microsoft QDQ ops

  // Check that Q/DQ nodes are *not* dropped for negative scale
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMin", {3, 3}, -0.003f, false);
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMin", {3, 3}, -0.003f, false, true, 13);  // Use com.microsoft QDQ ops
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMax", {3, 3}, -0.003f, false);
  RunReduceExtremumDropQDQTestCase<int8_t>("ReduceMax", {3, 3}, -0.003f, false, true, 13);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, DoubleQDQ) {
  constexpr uint8_t good_u8_1 = 80;
  constexpr uint8_t good_u8_2 = 40;
  constexpr uint8_t bad_u8 = 13;

  constexpr int8_t good_s8_1 = 99;
  constexpr int8_t good_s8_2 = -112;
  constexpr int8_t bad_s8 = 42;

  constexpr float good_float_point_1 = 4.0f;
  constexpr float good_float_point_2 = 8.0f;
  constexpr float bad_float_point = 1.11f;

  auto expect_succeed = [](bool use_contrib_qdq) -> std::function<void(InferenceSessionWrapper & session)> {
    return [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };
  };

  auto expect_fail = [](bool use_contrib_qdq) -> std::function<void(InferenceSessionWrapper & session)> {
    return [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    };
  };

  auto test_case_all_u8 = [&](bool succeed,
                              uint8_t zp_1, uint8_t zp_2, uint8_t zp_3, uint8_t zp_4,
                              float scale_1, float scale_2, float scale_3, float scale_4,
                              bool use_contrib_qdq = false) {
    TransformerTester(
        BuildDoubleQDQTestCases<uint8_t, uint8_t, uint8_t, uint8_t>(zp_1, zp_2, zp_3, zp_4,
                                                                    scale_1, scale_2, scale_3, scale_4,
                                                                    use_contrib_qdq),
        succeed ? expect_succeed(use_contrib_qdq) : expect_fail(use_contrib_qdq),
        TransformerLevel::Default,
        TransformerLevel::Level1,
        12,
        (scale_1 + scale_3) / 2,
        0.01);
  };

  auto test_case_all_s8 = [&](bool succeed,
                              int8_t zp_1, int8_t zp_2, int8_t zp_3, int8_t zp_4,
                              float scale_1, float scale_2, float scale_3, float scale_4,
                              bool use_contrib_qdq = false) {
    TransformerTester(
        BuildDoubleQDQTestCases<int8_t, int8_t, int8_t, int8_t>(zp_1, zp_2, zp_3, zp_4,
                                                                scale_1, scale_2, scale_3, scale_4,
                                                                use_contrib_qdq),
        succeed ? expect_succeed(use_contrib_qdq) : expect_fail(use_contrib_qdq),
        TransformerLevel::Default,
        TransformerLevel::Level1,
        12,
        (scale_1 + scale_3) / 2,
        0.01);
    TransformerTester(
        BuildDoubleQDQTestCases<int8_t, int8_t, int8_t, int8_t>(zp_1, zp_2, zp_3, zp_4,
                                                                scale_1, scale_2, scale_3, scale_4,
                                                                use_contrib_qdq),
        succeed ? expect_succeed(use_contrib_qdq) : expect_fail(use_contrib_qdq),
        TransformerLevel::Default,
        TransformerLevel::Level1,
        18,
        (scale_1 + scale_3) / 2,
        0.01);
    TransformerTester(
        BuildDoubleQDQTestCases<int8_t, int8_t, int8_t, int8_t>(zp_1, zp_2, zp_3, zp_4,
                                                                scale_1, scale_2, scale_3, scale_4,
                                                                use_contrib_qdq),
        succeed ? expect_succeed(use_contrib_qdq) : expect_fail(use_contrib_qdq),
        TransformerLevel::Default,
        TransformerLevel::Level1,
        19,
        (scale_1 + scale_3) / 2,
        0.01);
  };

  auto test_case_2u8_2s8_failed = [&](uint8_t zp_1, uint8_t zp_2, int8_t zp_3, int8_t zp_4,
                                      float scale_1, float scale_2, float scale_3, float scale_4,
                                      bool use_contrib_qdq = false) {
    TransformerTester(
        BuildDoubleQDQTestCases<uint8_t, uint8_t, int8_t, int8_t>(zp_1, zp_2, zp_3, zp_4,
                                                                  scale_1, scale_2, scale_3, scale_4,
                                                                  use_contrib_qdq),
        expect_fail(use_contrib_qdq),
        TransformerLevel::Default,
        TransformerLevel::Level1);
  };

  // all unsigned type
  test_case_all_u8(true, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_u8(true, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops

  // all signed type
  test_case_all_s8(true, good_s8_1, good_s8_1, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_s8(true, good_s8_1, good_s8_1, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops

  // 2 signed, 2 unsigned
  test_case_2u8_2s8_failed(good_u8_1, good_u8_1, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                           good_float_point_2, good_float_point_2);
  test_case_2u8_2s8_failed(good_u8_1, good_u8_1, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                           good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops

  //  different zero point within a pair
  test_case_all_u8(false, good_u8_1, bad_u8, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_u8(false, good_u8_1, bad_u8, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, bad_u8, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, bad_u8, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops
  test_case_all_s8(false, good_s8_1, bad_s8, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_s8(false, good_s8_1, bad_s8, good_s8_2, good_s8_2, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops
  test_case_all_s8(false, good_s8_1, good_s8_1, good_s8_2, bad_s8, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2);
  test_case_all_s8(false, good_s8_1, good_s8_1, good_s8_2, bad_s8, good_float_point_1, good_float_point_1,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops

  // different scale within a pair
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, bad_float_point,
                   good_float_point_2, good_float_point_2);
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, bad_float_point,
                   good_float_point_2, good_float_point_2, true);  // Use com.microsoft QDQ ops
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   bad_float_point, good_float_point_2);
  test_case_all_u8(false, good_u8_1, good_u8_1, good_u8_2, good_u8_2, good_float_point_1, good_float_point_1,
                   bad_float_point, good_float_point_2, true);  // Use com.microsoft QDQ ops
}

// Verifies fix for bug in the IsQDQPairSupported utility function, which is used by
// various optimizers such as DoubleQDQPairsRemover. The bug causes an exception when
// IsQDQPairIsSupported() is called with a Q -> DQ sequence that uses different scale types.
TEST(QDQTransformerTests, DoubleQDQPairsRemover_Bug_RejectDifferentScaleTypes) {
  // Function that builds a model with a QDQ nodes that use different scale data types:
  // input_fp32 -> Q(scale_fp32) -> DQ(scale_fp16) -> Mul(fp16) -> Q(scale_fp16) -> DQ(scale_fp32) -> output_fp32
  auto build_model_func = [](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({1, 1, 1, 3}, {1.0f, 2.0f, 3.0f});
    NodeArg* q1_output = builder.MakeIntermediate();
    NodeArg* dq1_output = builder.MakeIntermediate();
    NodeArg* mul_output = builder.MakeIntermediate();
    NodeArg* q2_output = builder.MakeIntermediate();
    NodeArg* dq2_output = builder.MakeOutput();
    NodeArg* const_arg = builder.MakeScalarInitializer(MLFloat16(10.0f));

    const float scale_fp32 = 1.0f;
    const MLFloat16 scale_fp16 = MLFloat16(scale_fp32);
    const uint8_t zp = 127;

    builder.AddQuantizeLinearNode<uint8_t, float>(input0_arg, scale_fp32, zp, q1_output);
    builder.AddDequantizeLinearNode<uint8_t, MLFloat16>(q1_output, scale_fp16, zp, dq1_output);
    builder.AddNode("Mul", {dq1_output, const_arg}, {mul_output});
    builder.AddQuantizeLinearNode<uint8_t, MLFloat16>(mul_output, scale_fp16, zp, q2_output);
    builder.AddDequantizeLinearNode<uint8_t, float>(q2_output, scale_fp32, zp, dq2_output);
  };

  // Previously, using different scale data types caused an exception in IsQDQPairSupported.
  // Now, we just reject the sequence. The DoubleQDQPairsRemover optimizer should not change the
  // graph.
  auto graph_checker = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
    EXPECT_EQ(op_to_count["Mul"], 1);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
  };
  TransformerTester(
      build_model_func,
      graph_checker,
      TransformerLevel::Default,
      TransformerLevel::Level1,
      21,
      /*per_sample_tolerance*/ 0.0,
      /*relative_per_sample_tolerance*/ 0.0,
      std::make_unique<DoubleQDQPairsRemover>());
}

template <typename QuantType>
static void RunDoubleQDQWithoutLastNodeBeingOutput(int output_index, int expected_Q_count, int expected_DQ_count,
                                                   bool use_contrib_qdq = false, int opset = 12) {
  auto graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], expected_Q_count);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expected_DQ_count);
  };
  TransformerTester(
      BuildDoubleQDQWithoutLastOutput<QuantType>(output_index, use_contrib_qdq),
      graph,
      TransformerLevel::Default,
      TransformerLevel::Level1,
      opset);
}

TEST(QDQTransformerTests, DoubleQDQ_Without_Last_Node_Being_Output) {
  constexpr bool use_ms_qdq = true;  // For readability.

  // the first node being a graph output doesn't prevent the DQ -> Q in the middle from being removed
  // if they have matching type/scale/zp
  // Q -> DQ -> Q -> DQ
  //  `-> graph output
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(0, 1, 1);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(0, 1, 1, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(0, 1, 1, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<int16_t>(0, 1, 1, use_ms_qdq);

  // EnsureUniqueDQForNodeUnit will duplicate first DQ, but after that the DQ -> Q in the middle can still be removed
  // leaveing one Q and 2 DQ.
  // Q -> DQ -> Q -> DQ
  //       `-> graph output
  // =>
  // Q -> DQ -> Q -> DQ
  //  `-> DQ -> graph output
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(1, 1, 2);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(1, 1, 2, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(1, 1, 2, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<int16_t>(1, 1, 2, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(1, 1, 2, !use_ms_qdq, 21);
  RunDoubleQDQWithoutLastNodeBeingOutput<int16_t>(1, 1, 2, !use_ms_qdq, 21);

  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(2, 2, 2);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(2, 2, 2, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(2, 2, 2, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(2, 2, 2, !use_ms_qdq, 21);

  // last node being a graph output doesn't prevent the DQ -> Q in the middle from being removed
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(3, 1, 1);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint8_t>(3, 1, 1, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(3, 1, 1, use_ms_qdq);
  RunDoubleQDQWithoutLastNodeBeingOutput<uint16_t>(3, 1, 1, !use_ms_qdq, 21);
}

// Utility function that runs a model with a double QDQ sequence (with duplicate end DQs) through
// the DoubleQDQPairsRemover transformer and checks that the resulting graph contains the expected nodes.
// Also checks that the output from the unmodified model matches the output from the modified model.
static void RunDoubleQDQWithDuplicateLastDQs(int expected_Q_count, int expected_DQ_count,
                                             gsl::span<const int64_t> input_shape,
                                             gsl::span<const float> input_data,
                                             gsl::span<const int64_t> zero_points,
                                             gsl::span<const ONNX_NAMESPACE::TensorProto_DataType> zero_point_types,
                                             gsl::span<const float> scales,
                                             size_t graph_output_index,
                                             bool use_contrib_qdq = false,
                                             int opset = 19) {
  auto graph_checker = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], expected_Q_count);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expected_DQ_count);
  };

  auto model_build_fn = BuildDoubleQDQTestCaseWithDuplicateLastDQs(input_shape, input_data, zero_points,
                                                                   zero_point_types, scales, graph_output_index,
                                                                   use_contrib_qdq);
  TransformerTester(model_build_fn,
                    graph_checker,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    opset,
                    /*per_sample_tolerance*/ 0.0,
                    /*relative_per_sample_tolerance*/ 0.0,
                    std::make_unique<DoubleQDQPairsRemover>());
}

// Test QDQDoublePairsRemover when the sequence ends with duplicate DQs.
TEST(QDQTransformerTests, DoubleQDQPairsRemover_DuplicateLastDQs) {
  InlinedVector<int64_t> shape = {1, 2, 2, 2};
  InlinedVector<float> input_data = {-3.0f, -2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

  constexpr auto int8_type = ONNX_NAMESPACE::TensorProto_DataType_INT8;
  constexpr auto uint8_type = ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  constexpr auto int16_type = ONNX_NAMESPACE::TensorProto_DataType_INT16;
  constexpr auto uint16_type = ONNX_NAMESPACE::TensorProto_DataType_UINT16;
  InlinedVector<ONNX_NAMESPACE::TensorProto_DataType> quant_types = {int8_type, uint8_type, int16_type, uint16_type};

  // Input graph:
  // input -> Q1 -> DQ1 -> Q2 --+--> DQ2 -> output0
  //                            |
  //                            ...
  //                            |
  //                            +--> DQ2'' -> outputN
  // Expected graph after DoubleQDQPairsRemover:
  // input -> Q1 --+--> DQ2 -> output0
  //               |
  //               ...
  //               |
  //               +--> DQ2'' -> outputN
  for (auto quant_type : quant_types) {
    for (size_t num_dq2s = 1; num_dq2s <= 3; num_dq2s++) {
      const size_t num_nodes = 3 + num_dq2s;
      InlinedVector<int64_t> zp_vals(num_nodes, 1);
      InlinedVector<ONNX_NAMESPACE::TensorProto_DataType> zp_types(num_nodes, quant_type);
      InlinedVector<float> scale_vals(num_nodes, 0.1f);

      const int expected_q_nodes = 1;
      const int expected_dq_nodes = static_cast<int>(num_dq2s);
      RunDoubleQDQWithDuplicateLastDQs(expected_q_nodes, expected_dq_nodes, shape, input_data, zp_vals, zp_types,
                                       scale_vals, 3, false, 21);
      RunDoubleQDQWithDuplicateLastDQs(expected_q_nodes, expected_dq_nodes, shape, input_data, zp_vals, zp_types,
                                       scale_vals, 3, quant_type == int16_type || quant_type == uint16_type, 19);
    }
  }

  // Should not remove QDQ pair because the middle nodes produce a graph output.
  for (auto quant_type : quant_types) {
    for (size_t output_index = 0; output_index < 3; output_index++) {
      for (size_t num_dq2s = 1; num_dq2s <= 3; num_dq2s++) {
        const size_t num_nodes = 3 + num_dq2s;
        InlinedVector<int64_t> zp_vals(num_nodes, 1);
        InlinedVector<ONNX_NAMESPACE::TensorProto_DataType> zp_types(num_nodes, quant_type);
        InlinedVector<float> scale_vals(num_nodes, 0.1f);

        const int expected_q_nodes = 2;
        int expected_dq_nodes = 1 + static_cast<int>(num_dq2s);
        if (output_index == 1) {
          // EnsureUniqueDQ pass will create a duplicate DQ if it produces a graph output.
          expected_dq_nodes += 1;
        }
        RunDoubleQDQWithDuplicateLastDQs(expected_q_nodes, expected_dq_nodes, shape, input_data, zp_vals, zp_types,
                                         scale_vals, output_index, false, 21);
      }
    }
  }

  // Should not remove any nodes because the Q -> DQ pairs are of different quant types.
  for (size_t num_dq2s = 1; num_dq2s <= 2; num_dq2s++) {
    const size_t num_nodes = 3 + num_dq2s;
    InlinedVector<int64_t> zp_vals(num_nodes, 1);
    InlinedVector<ONNX_NAMESPACE::TensorProto_DataType> zp_types(num_nodes, int8_type);
    for (size_t i = 2; i < num_nodes; i++) {
      // Q2 -> DQ2* have a different type
      zp_types[i] = int16_type;
    }
    InlinedVector<float> scale_vals(num_nodes, 0.1f);

    const int expected_q_nodes = 2;
    const int expected_dq_nodes = 1 + static_cast<int>(num_dq2s);
    RunDoubleQDQWithDuplicateLastDQs(expected_q_nodes, expected_dq_nodes, shape, input_data, zp_vals, zp_types,
                                     scale_vals, 3, false, 21);
  }

  // Should not remove nodes because 1 of the ending DQ2s has a different zero_point.
  const size_t num_dq2s = 2;
  const size_t num_nodes = 3 + num_dq2s;
  InlinedVector<int64_t> zp_vals(num_nodes, 1);
  InlinedVector<ONNX_NAMESPACE::TensorProto_DataType> zp_types(num_nodes, int8_type);
  zp_vals[num_nodes - 1] = 2;  // Last DQ2 has a different zero-point.
  InlinedVector<float> scale_vals(num_nodes, 0.1f);

  const int expected_q_nodes = 2;
  const int expected_dq_nodes = 1 + static_cast<int>(num_dq2s);
  RunDoubleQDQWithDuplicateLastDQs(expected_q_nodes, expected_dq_nodes, shape, input_data, zp_vals, zp_types,
                                   scale_vals, 3, false, 21);
}

// Runs a test that checks if DQ -> Split -> Q (many) is replaced with just Split.
template <typename QuantType>
static void RunDropSplitQDQTestCase(const std::vector<int64_t>& input_shape, int64_t axis,
                                    bool all_same_quant_params, bool use_contrib_qdq = false,
                                    bool should_not_drop = false) {
  auto check_graph = [all_same_quant_params, use_contrib_qdq, should_not_drop](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    int expected_q_ops = all_same_quant_params && !should_not_drop ? 0 : 3;
    int expected_dq_ops = all_same_quant_params && !should_not_drop ? 0 : 1;
    EXPECT_EQ(op_to_count["Split"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], expected_q_ops);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expected_dq_ops);
  };

  std::vector<int> opsets = {12, 13, 18, 19, 21};
  if constexpr (std::is_same_v<QuantType, uint16_t> || std::is_same_v<QuantType, int16_t> ||
                std::is_same_v<QuantType, Int4x2> || std::is_same_v<QuantType, UInt4x2>) {
    opsets = std::vector<int>{21};
  }

  TransformerTester(BuildQDQSplitTestCase<QuantType, QuantType>(input_shape, axis, !all_same_quant_params,
                                                                use_contrib_qdq),
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    opsets);  // Test different ways to specify the split in each opset:
                              // 12 - split into equal parts without explicit 'split' attribute
                              // 13 - use optional 'split' input to split into 3 parts
                              // >= 18 - use 'num_outputs' attribute to split into 3 parts
}

// Test that DQ -> Split -> Q (many) is replaced with just Split for various quantization types.
TEST(QDQTransformerTests, Split) {
  // Test cases that drop Q/DQ ops from DQ -> Split -> Q (many).
  // This happens when all the Q/DQ ops have equal and constant quantization parameters.
  {
    constexpr bool ALL_SAME_QUANT_PARAMS = true;
    constexpr bool USE_CONTRIB_QDQ_OPS = true;
    RunDropSplitQDQTestCase<int8_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS);
    RunDropSplitQDQTestCase<int8_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<int16_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<uint16_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<int16_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<uint16_t>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS);

    // Do not yet support int4 Split, so should not drop
    constexpr bool SHOULD_NOT_DROP = true;
    RunDropSplitQDQTestCase<Int4x2>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS, SHOULD_NOT_DROP);
    RunDropSplitQDQTestCase<UInt4x2>({6, 18, 54}, 0, ALL_SAME_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS, SHOULD_NOT_DROP);
  }

  // Test cases that DO NOT drop Q/DQ ops from DQ -> Split -> Q (many)
  // This happens when the Q/DQ ops do not have equal and constant quantization parameters.
  {
    constexpr bool DIFF_QUANT_PARAMS = false;
    constexpr bool USE_CONTRIB_QDQ_OPS = true;
    RunDropSplitQDQTestCase<int8_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS);
    RunDropSplitQDQTestCase<int8_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<int16_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<uint16_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS, USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<int16_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS);
    RunDropSplitQDQTestCase<uint16_t>({6, 18, 54}, 0, DIFF_QUANT_PARAMS, !USE_CONTRIB_QDQ_OPS);
  }
}

// Because split isn't one the supported ops, this will stay the same
TEST(QDQTransformerTests, Split_without_IdenticalChildrenConsolidation) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const int64_t& axis,
                       bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Split"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
    };
    TransformerTester(BuildConsolidationTestCase<int8_t, int8_t>(input_shape, axis, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2, {12, 18, 19}, {}, {}, nullptr, {},
                      {"IdenticalChildrenConsolidation"});
  };
  test_case({6, 18, 54}, 0);
  test_case({6, 18, 54}, 0, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, Split_with_IdenticalChildrenConsolidation) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const int64_t& axis,
                       bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Split"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
    };
    TransformerTester(BuildConsolidationTestCase<int8_t, int8_t>(input_shape, axis, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      {12, 18, 19});
  };
  test_case({6, 18, 54}, 0);
  test_case({6, 18, 54}, 0, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, Where) {
  auto test_case = [&](const std::vector<int64_t>& cond_shape, const std::vector<int64_t>& x_shape,
                       const std::vector<int64_t>& y_shape, bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearWhere"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };
    TransformerTester(BuildQDQWhereTestCase<int8_t>(cond_shape, x_shape, y_shape, use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };
  test_case({1}, {1}, {1});
  test_case({1}, {1}, {1}, true /*use_contrib_qdq*/);
}

template <typename QuantType>
static void RunDropQDQTransposeTestCase(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms,
                                        bool use_contrib_qdq = false,
                                        int opset = 12) {
  // model has DQ -> Mul -> Q -> DQ -> Transpose -> Q -> output
  // post transform and optimization it should be DQ -> Mul -> Q -> Transpose(uint8) -> output
  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["Transpose"], 1);
    EXPECT_EQ(op_to_count["Mul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  TransformerTester(BuildQDQTransposeTestCase<QuantType, QuantType>(input_shape, perms, use_contrib_qdq),
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    opset);
}

TEST(QDQTransformerTests, Transpose) {
  RunDropQDQTransposeTestCase<uint8_t>({2, 13, 12, 37}, {0, 3, 1, 2});
  RunDropQDQTransposeTestCase<uint8_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunDropQDQTransposeTestCase<uint16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunDropQDQTransposeTestCase<int16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunDropQDQTransposeTestCase<uint16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, false /*use_contrib_qdq*/, 21 /*opset*/);
  RunDropQDQTransposeTestCase<int16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, false /*use_contrib_qdq*/, 21 /*opset*/);
}

template <typename QuantType>
static void RunQDQTransposeNoFusionTestCase(const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& perms,
                                            bool use_contrib_qdq = false, int opset = 12) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<QuantType>(input1_shape, std::numeric_limits<QuantType>::min(),
                                                    std::numeric_limits<QuantType>::max());
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input1_arg, .003f, 1, dq_output, use_contrib_qdq);

    // add Transpose
    auto* transpose_output = builder.MakeOutput();  // transpose output is graph output
    Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
    transpose_node.AddAttribute("perm", perms);

    // add Q
    builder.AddQuantizeLinearNode<QuantType>(transpose_output, .003f, 1, output_arg, use_contrib_qdq);
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);
}

TEST(QDQTransformerTests, Transpose_No_Fusion) {
  RunQDQTransposeNoFusionTestCase<int8_t>({2, 13, 12, 37}, {0, 3, 1, 2});
  RunQDQTransposeNoFusionTestCase<int8_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunQDQTransposeNoFusionTestCase<int16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunQDQTransposeNoFusionTestCase<uint16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
  RunQDQTransposeNoFusionTestCase<int16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, false /*use_contrib_qdq*/, 21 /*opset*/);
  RunQDQTransposeNoFusionTestCase<uint16_t>({2, 13, 12, 37}, {0, 3, 1, 2}, false /*use_contrib_qdq*/, 21 /*opset*/);
}

TEST(QDQTransformerTests, Resize) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape,
                       const std::vector<int64_t>& sizes_shape,
                       bool use_contrib_qdq = false) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    TransformerTester(BuildQDQResizeTestCase(input1_shape,
                                             sizes_shape,
                                             "nearest",             // mode
                                             "half_pixel",          // coordinate_transformation_mode
                                             "round_prefer_floor",  // nearest_mode
                                             false,                 // add_dq_output_float
                                             use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  RandomValueGenerator rand_gen{optional<RandomValueGenerator::RandomSeedType>{2345}};
  test_case({2, 13, 12, 37}, rand_gen.Uniform<int64_t>(std::vector<int64_t>{4}, 1, 16));
  test_case({2, 13, 12, 37}, rand_gen.Uniform<int64_t>(std::vector<int64_t>{4}, 1, 16), true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, Resize_No_Fusion) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& sizes_shape,
                       const std::vector<int64_t>& concat_input2_shape,
                       const int64_t axis,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* roi = builder.MakeInitializer<float>({0}, {});
      auto* scales = builder.MakeInitializer<float>({0}, {});
      auto* sizes = builder.MakeInitializer<int64_t>(sizes_shape, {1, 8, 128, 128});
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .003f, 1, dq_output, use_contrib_qdq);

      // add Resize
      auto* resize_output = builder.MakeIntermediate();
      builder.AddNode("Resize", {dq_output, roi, scales, sizes}, {resize_output});

      // add Concat
      std::vector<NodeArg*> concat_input_args;
      concat_input_args.push_back(resize_output);
      concat_input_args.push_back(builder.MakeInput<float>(concat_input2_shape,
                                                           std::numeric_limits<float>::min(),
                                                           std::numeric_limits<float>::max()));
      auto* concat_output = builder.MakeIntermediate();
      Node& concat_node = builder.AddNode("Concat", concat_input_args, {concat_output});
      concat_node.AddAttribute("axis", axis);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(resize_output, .003f, 1, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count["Concat"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({1, 8, 64, 64}, {4}, {1, 4, 128, 128}, 1);
  test_case({1, 8, 64, 64}, {4}, {1, 4, 128, 128}, 1, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ResizeReshapeSqueezeUnsqueeze) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& sizes_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape,
                                                 std::numeric_limits<float>::min(),
                                                 std::numeric_limits<float>::max());
      auto* roi = builder.MakeInitializer<float>({0}, {});
      auto* scales = builder.MakeInitializer<float>({0}, {});
      auto* sizes = builder.MakeInitializer<int64_t>(sizes_shape, {1, 2, 52, 82});

      // add QDQ + Resize
      auto* qdq_input = AddQDQNodePair<uint8_t>(builder, input_arg, .003f, 1, use_contrib_qdq);
      auto* resize_output = builder.MakeIntermediate();
      builder.AddNode("Resize", {qdq_input, roi, scales, sizes}, {resize_output});

      // add QDQ + Reshape
      auto* qdq_resize_output = AddQDQNodePair<uint8_t>(builder, resize_output, .003f, 1, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({1, 2, 52, 82});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {qdq_resize_output, reshape_shape}, {reshape_output});

      // add QDQ + Squeeze
      auto* qdq_squeeze_output = AddQDQNodePair<uint8_t>(builder, reshape_output, .003f, 1, use_contrib_qdq);
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {qdq_squeeze_output, squeeze_axes}, {squeeze_output});

      // add QDQ + Unsqueeze
      auto* qdq_unsqueeze_output = AddQDQNodePair<uint8_t>(builder, squeeze_output, .003f, 1, use_contrib_qdq);
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {qdq_unsqueeze_output, unsqueeze_axes}, {unsqueeze_output});

      // add QDQ
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, unsqueeze_output, .003f, 1, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/);

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/);
  };

  test_case({1, 2, 26, 42}, {4});
  test_case({1, 2, 26, 42}, {4}, true /*use_contrib_qdq*/);
}

// Runs a test case that checks if the DQ node is dropped from DQ -> Op (e.g., ArgMax).
template <typename QuantType>
static void RunArgMaxDropDQTestCase(const std::vector<int64_t>& input_shape,
                                    int axis,
                                    int keepdims,
                                    int select_last_index,
                                    bool use_contrib_qdq,
                                    bool expect_drop_dq = true) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<QuantType>(input_shape,
                                                   std::numeric_limits<QuantType>::min(),
                                                   std::numeric_limits<QuantType>::max());
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_arg, .003f, 1, dq_output, use_contrib_qdq);

    // add ArgMax
    Node& argmax_node = builder.AddNode("ArgMax", {dq_output}, {output_arg});
    argmax_node.AddAttribute("axis", static_cast<int64_t>(axis));
    argmax_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));
    argmax_node.AddAttribute("select_last_index", static_cast<int64_t>(select_last_index));
  };

  auto check_graph = [use_contrib_qdq, expect_drop_dq](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["ArgMax"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expect_drop_dq ? 0 : 1);
  };

  std::vector<int> opsets = {13, 19, 21};

  if constexpr (std::is_same_v<QuantType, uint16_t> || std::is_same_v<QuantType, int16_t>) {
    if (!use_contrib_qdq) {
      opsets = std::vector<int>{21};  // Only ONNX opset 21 supports 16-bit ints.
    }
  }

  TransformerTester(build_test_case, check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    opsets);
}

// Checks that the DQ node is dropped from DQ -> ArgMax. Uses 8-bit and 16-bit Q/DQ ops.
TEST(QDQTransformerTests, ArgMax) {
  RunArgMaxDropDQTestCase<uint8_t>({2, 13, 12, 37}, 1, 0, 0, false);
  RunArgMaxDropDQTestCase<uint8_t>({2, 13, 12, 37}, 1, 0, 0, true /*use_contrib_qdq*/);

  // Should *not* drop DQ for 16-bit DQ -> ArgMax (because ORT does not support 16-bit input types for ArgMax).
  RunArgMaxDropDQTestCase<uint16_t>({2, 13, 12, 37}, 1, 0, 0, true /*use_contrib_qdq*/, false /*expect_drop_dq*/);
  RunArgMaxDropDQTestCase<int16_t>({2, 13, 12, 37}, 1, 0, 0, true /*use_contrib_qdq*/, false /*expect_drop_dq*/);
  RunArgMaxDropDQTestCase<uint16_t>({2, 13, 12, 37}, 1, 0, 0, false /*use_contrib_qdq*/, false /*expect_drop_dq*/);
  RunArgMaxDropDQTestCase<int16_t>({2, 13, 12, 37}, 1, 0, 0, false /*use_contrib_qdq*/, false /*expect_drop_dq*/);

  RunArgMaxDropDQTestCase<uint8_t>({2, 13, 12, 37}, 0, 1, 0, false);
  RunArgMaxDropDQTestCase<uint8_t>({2, 13, 12, 37}, 0, 0, 1, false);
}

TEST(QDQTransformerTests, QLinearMatMul) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129, use_contrib_qdq);
      auto* dq_matmul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129, use_contrib_qdq);
      builder.AddNode("MatMul", {dq_matmul_output1, dq_matmul_output2}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, MatMul_No_Fusion) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129, use_contrib_qdq);
      builder.AddNode("MatMul", {dq_matmul_output1, input2_arg}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, MatMul_1st_Input_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int8_t>(input1_shape, -128, 127);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add DQ with type int8
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input1_arg, .004f, 1, dq_output_1, use_contrib_qdq);

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129, use_contrib_qdq);
      builder.AddNode("MatMul", {dq_output_1, dq_matmul_output2}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, MatMulIntegerToFloat) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<uint8_t>(input1_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* input2_arg = builder.MakeInput<uint8_t>(input2_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input1_arg, .0035f, 135, dq_output_1, use_contrib_qdq);

      auto* dq_output_2 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input2_arg, .0035f, 135, dq_output_2, use_contrib_qdq);

      builder.AddNode("MatMul", {dq_output_1, dq_output_2}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      1e-5 /*per_sample_tolerance*/,
                      1e-5 /*relative_per_sample_tolerance*/);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      1e-5 /*per_sample_tolerance*/,
                      1e-5 /*relative_per_sample_tolerance*/);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ConvRelu) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool is_zp_zero, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add Relu
      auto* relu_output = builder.MakeIntermediate();
      builder.AddNode("Relu", {conv_output}, {relu_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(relu_output, .0039f, is_zp_zero ? 0 : 1, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if (is_zp_zero) {
        EXPECT_EQ(op_to_count["QLinearConv"], 1);
        EXPECT_EQ(op_to_count["Conv"], 0);
        EXPECT_EQ(op_to_count["Relu"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      } else {
        EXPECT_EQ(op_to_count["QLinearConv"], 0);
        EXPECT_EQ(op_to_count["Conv"], 0);
        EXPECT_EQ(op_to_count["Relu"], 0);
        EXPECT_EQ(op_to_count["com.microsoft.FusedConv"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({1, 12, 37}, {32, 12, 5}, true, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, true, true /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, false, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, false, true /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, true, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, true, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_UInt8) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_averagepool_output = AddQDQNodePair<uint8_t>(builder, conv_output, .0035f, 135, use_contrib_qdq);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<uint8_t>(builder, averagepool_output, .0035f, 135, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
    // TODO: fix opset 19
  };

  test_case({1, 12, 37}, {32, 12, 5}, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_Int8) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_averagepool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7, use_contrib_qdq);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, averagepool_output, .0035f, 7, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
    // TODO: fix opset 19
  };

  test_case({1, 12, 37}, {32, 12, 5}, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_Int8_Fail) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int8_t>(input_shape, -128, 127);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add DQ + Conv
      auto* dq_output = builder.MakeIntermediate();
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input_arg, .004f, 1, dq_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_averagepool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7, use_contrib_qdq);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, averagepool_output, .0035f, 7, use_contrib_qdq);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["Conv"], 1);
      EXPECT_EQ(op_to_count["QLinearConv"], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
    };

    // TODO: fix opset 19
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      {12, 18} /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  test_case({1, 12, 37}, {32, 12, 5}, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, {32, 12, 5}, true /*use_contrib_qdq*/);
}

template <typename InputType, typename OutputType>
void QDQTransformerLeakyReluTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      // add QDQ + LeakyRelu
      auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .0035f, 7, use_contrib_qdq);
      auto* leakyrelu_output = builder.MakeIntermediate();
      Node& leakyrelu_node = builder.AddNode("LeakyRelu", {dq_output}, {leakyrelu_output});
      leakyrelu_node.AddAttribute("alpha", 0.2f);

      // add QDQ output
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<OutputType>(leakyrelu_output,
                                                .0038f,
                                                std::numeric_limits<OutputType>::max() / 2,
                                                q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                  .0039f,
                                                  std::numeric_limits<OutputType>::max() / 2,
                                                  output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearLeakyRelu"], 1);
        EXPECT_EQ(op_to_count["LeakyRelu"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearLeakyRelu"], 0);
        EXPECT_EQ(op_to_count["LeakyRelu"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37}, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, true /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, LeakyRelu_S8S8) {
  QDQTransformerLeakyReluTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_U8U8) {
  QDQTransformerLeakyReluTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_S8U8) {
  QDQTransformerLeakyReluTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_U8S8) {
  QDQTransformerLeakyReluTests<uint8_t, int8_t>();
}

template <typename InputType, typename OutputType>
void QDQTransformerSigmoidTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      // add QDQ + Sigmoid
      auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .0035f, 7, use_contrib_qdq);
      auto* sigmoid_output = builder.MakeIntermediate();
      builder.AddNode("Sigmoid", {dq_output}, {sigmoid_output});

      // add QDQ output
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<OutputType>(sigmoid_output,
                                                .0038f,
                                                std::numeric_limits<OutputType>::max() / 2,
                                                q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                  .0039f,
                                                  std::numeric_limits<OutputType>::max() / 2,
                                                  output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSigmoid"], 1);
        EXPECT_EQ(op_to_count["Sigmoid"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSigmoid"], 0);
        EXPECT_EQ(op_to_count["Sigmoid"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37}, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, true /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, false /*use_contrib_qdq*/);
  test_case({1, 22, 11, 13, 15}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, Sigmoid_S8S8) {
  QDQTransformerSigmoidTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, Sigmoid_U8U8) {
  QDQTransformerSigmoidTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Sigmoid_S8U8) {
  QDQTransformerSigmoidTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Sigmoid_U8S8) {
  QDQTransformerSigmoidTests<uint8_t, int8_t>();
}

TEST(QDQTransformerTests, ConvTranspose_QBackward) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& perms, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {conv_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(transpose_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, {0, 3, 1, 2}, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, QBackward_MutilpleSteps) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {conv_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {reshape_output}, {transpose_output});
      transpose_node.AddAttribute("perm", std::vector<int64_t>({1, 0}));

      // add Unsqueeze
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {transpose_output, unsqueeze_axes}, {unsqueeze_output});

      // add Squeeze
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {unsqueeze_output, squeeze_axes}, {squeeze_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(squeeze_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(squeeze_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/);
    // TODO: fix opset 19
  };

  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, ConvTranspose_DQForward) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& perms, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ
      auto* dq_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(transpose_output, dq_w_output, conv_output);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(conv_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(conv_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12, 0.0, 0.0, nullptr, {},  // defaults that we're not overriding
                      {"TransposeOptimizer"});    // disable TransposeOptimizer for simplicity
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18, 0.0, 0.0, nullptr, {},  // defaults that we're not overriding
                      {"TransposeOptimizer"});    // disable TransposeOptimizer for simplicity
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19, 0.0, 0.0, nullptr, {},  // defaults that we're not overriding
                      {"TransposeOptimizer"});    // disable TransposeOptimizer for simplicity
  };

  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2}, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, DQForward_MutilpleSteps) {
  DNNL_GTEST_SKIP();

  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& perms, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add Transpose
      auto* qdq_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1, use_contrib_qdq);
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {qdq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add Unsqueeze
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {maxpool_output, unsqueeze_axes}, {unsqueeze_output});

      // add Squeeze
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {unsqueeze_output, squeeze_axes}, {squeeze_output});

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);
      builder.AddConvNode(squeeze_output, dq_w_output, conv_output);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {conv_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/,
                      0.0, 0.0, nullptr, {},    // defaults that we're not overriding
                      {"TransposeOptimizer"});  // disable TransposeOptimizer for simplicity
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.0, 0.0, nullptr, {},    // defaults that we're not overriding
                      {"TransposeOptimizer"});  // disable TransposeOptimizer for simplicity
    // TODO: fix opset 19
  };

  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2}, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, Clip) {
  constexpr float epsilon = std::numeric_limits<float>::epsilon();

  auto test_case = [&](float scale, auto zero_point, int clip_count, int opset_version,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int8_t>({1, 32, 112, 112},
                                                  std::numeric_limits<int8_t>::min(),
                                                  std::numeric_limits<int8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input_arg, .0035f, 7, dq_output, use_contrib_qdq);

      // add Clip
      auto* clip_output = builder.MakeIntermediate();
      constexpr float min = .0f;
      constexpr float max = 6.0f;
      auto opset = builder.DomainToVersionMap().find(kOnnxDomain)->second;
      EXPECT_EQ(opset_version, opset);
      if (opset >= 11) {
        auto* min_initializer = builder.MakeScalarInitializer<float>(min);
        auto* max_initializer = builder.MakeScalarInitializer<float>(max);
        builder.AddNode("Clip", {dq_output, min_initializer, max_initializer}, {clip_output});
      } else {
        Node& argmax_node = builder.AddNode("Clip", {dq_output}, {clip_output});
        argmax_node.AddAttribute("min", min);
        argmax_node.AddAttribute("max", max);
      }

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode(clip_output, scale, zero_point, q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode(q_output, scale, zero_point, output_arg, use_contrib_qdq);
    };

    auto check_clip_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count["Clip"], clip_count);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    };

    TransformerTester(build_test_case, check_clip_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level2,
                      opset_version,
                      epsilon,
                      epsilon);
  };

  constexpr int16_t int16_min = std::numeric_limits<int16_t>::min();
  constexpr uint16_t uint16_min = std::numeric_limits<uint16_t>::min();

  std::vector<int> opsets{12, 18, 19};
  for (auto opset : opsets) {
    test_case(.0235294122248888f, static_cast<int8_t>(-128), 0, opset);        // [0, 6]
    test_case(.0235294122248888f, static_cast<int8_t>(-128), 0, opset, true);  // [0, 6] contrib qdq
    test_case(9.15541313801785e-5f, int16_min, 0, opset, true);                // [0, 6] contrib 16-bit qdq
    test_case(0.0009f, int16_min, 1, opset, true);                             // [0, 58.98] contrib 16-bit qdq
    test_case(.02f, static_cast<int8_t>(-128), 0, opset);                      // [0, 5.1]
    test_case(.02f, static_cast<int8_t>(-128), 0, opset, true);                // [0, 5.1] contrib qdq
    test_case(.03f, static_cast<int8_t>(-128), 1, opset);                      // [0, 7.65]
    test_case(.03f, static_cast<int8_t>(-128), 1, opset, true);                // [0, 7.65] contrib qdq
    test_case(.02f, static_cast<int8_t>(127), 1, opset);                       // [-5.1 , 0]
    test_case(.02f, static_cast<int8_t>(127), 1, opset, true);                 // [-5.1 , 0] contrib qdq
    test_case(.02f, static_cast<int8_t>(0), 1, opset);                         // [-2.56, 2.54]
    test_case(.02f, static_cast<int8_t>(0), 1, opset, true);                   // [-2.56, 2.54] contrib qdq
    test_case(.04f, static_cast<int8_t>(-97), 1, opset);                       // [-1.24, 8.96]
    test_case(.04f, static_cast<int8_t>(-97), 1, opset, true);                 // [-1.24, 8.96] contrib qdq
    test_case(.02352941176f, static_cast<uint8_t>(0), 0, opset);               // [0, 6]
    test_case(.02352941176f, static_cast<uint8_t>(0), 0, opset, true);         // [0, 6] contrib qdq
    test_case(9.15541313801785e-5f, uint16_min, 0, opset, true);               // [0, 6] contrib 16-bit qdq
    test_case(0.0009f, uint16_min, 1, opset, true);                            // [0, 58.98] contrib 16-bit qdq
    test_case(.02f, static_cast<uint8_t>(0), 0, opset);                        // [0, 5.1]
    test_case(.02f, static_cast<uint8_t>(0), 0, opset, true);                  // [0, 5.1] contrib qdq
    test_case(.03f, static_cast<uint8_t>(0), 1, opset);                        // [0, 7.65]
    test_case(.03f, static_cast<uint8_t>(0), 1, opset, true);                  // [0, 7.65] contrib qdq
    test_case(.02f, static_cast<uint8_t>(255), 1, opset);                      // [-5.1, 0]
    test_case(.02f, static_cast<uint8_t>(255), 1, opset, true);                // [-5.1, 0] contrib qdq
    test_case(.02f, static_cast<uint8_t>(128), 1, opset);                      // [-2.56, 2.54]
    test_case(.02f, static_cast<uint8_t>(128), 1, opset, true);                // [-2.56, 2.54] contrib qdq
    test_case(.04f, static_cast<uint8_t>(31), 1, opset);                       // [-1.24, 8.96]
    test_case(.04f, static_cast<uint8_t>(31), 1, opset, true);                 // [-1.24, 8.96] contrib qdq
  }

  // opset_version = 10
  test_case(.02f, static_cast<int8_t>(-128), 0, 10);        // [0, 5.1]
  test_case(.02f, static_cast<int8_t>(-128), 0, 10, true);  // [0, 5.1] contrib qdq
  test_case(.03f, static_cast<int8_t>(-128), 1, 10);        // [0, 7.65]
  test_case(.03f, static_cast<int8_t>(-128), 1, 10, true);  // [0, 7.65] contrib qdq
  test_case(.02f, static_cast<uint8_t>(0), 0, 10);          // [0, 5.1]
  test_case(.02f, static_cast<uint8_t>(0), 0, 10, true);    // [0, 5.1] contrib qdq
  test_case(.03f, static_cast<uint8_t>(0), 1, 10);          // [0, 7.65]
  test_case(.03f, static_cast<uint8_t>(0), 1, 10, true);    // [0, 7.65] contrib qdq

  // difference between lower/upper and min/max are within epsilon
  for (auto opset : opsets) {
    test_case(epsilon, static_cast<int8_t>(-127), 0, opset);                    // [-epsilon, x] (x <= 6 + epsilon)
    test_case(epsilon, static_cast<int8_t>(-127), 0, opset, true);              // [-epsilon, x] (x <= 6 + epsilon)
    test_case((6 + epsilon) / 255, static_cast<int8_t>(-128), 0, opset);        // [0, 6 + epsilon]
    test_case((6 + epsilon) / 255, static_cast<int8_t>(-128), 0, opset, true);  // [0, 6 + epsilon]
    test_case(epsilon, static_cast<uint8_t>(1), 0, opset);                      // [-epsilon, x] (x <= 6 + epsilon)
    test_case(epsilon, static_cast<uint8_t>(1), 0, opset, true);                // [-epsilon, x] (x <= 6 + epsilon)
    test_case((6 + epsilon) / 255, static_cast<uint8_t>(0), 0, opset);          // [0, 6 + epsilon]
    test_case((6 + epsilon) / 255, static_cast<uint8_t>(0), 0, opset, true);    // [0, 6 + epsilon]
  }
}

// Test that the ReluQuantFusion transformer only runs for optimization level >= 2.
TEST(QDQTransformerTests, ReluQuantFusion_Level2Only) {
  auto test_case = [&](TransformerLevel opt_level, int8_t zp) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int8_t>({1, 2, 2, 2},
                                                  {-4, -3, -2, 0, 1, 2, 3, 4});
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input_arg, 1.0f, zp, dq_output);

      // add Relu
      auto* relu_output = builder.MakeIntermediate();
      builder.AddNode("Relu", {dq_output}, {relu_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<int8_t>(relu_output, 1.0f, zp, q_output);
      builder.AddDequantizeLinearNode<int8_t>(q_output, 1.0f, zp, output_arg);
    };

    auto check_relu_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
      // Only fuse relu into Q if level >= 2 and zero_point == -128 for int8.
      // Level1 graph:   input -> DQ -> Relu -> Q -> DQ -> output
      // Level2+ graph: input -> DQ -> output (QuantReluFusion + QDQFinalCleanupTransformer transformers)
      const bool fuse_relu = (zp == -128) &&
                             (opt_level == TransformerLevel::Level2 || opt_level == TransformerLevel::Level3);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], fuse_relu ? 0 : 1);
      EXPECT_EQ(op_to_count["Relu"], fuse_relu ? 0 : 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], fuse_relu ? 1 : 2);
    };

    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    TransformerTester(build_test_case, check_relu_graph,
                      TransformerLevel::Default,
                      opt_level,
                      18,
                      epsilon,
                      epsilon);
  };

  test_case(TransformerLevel::Level1, -128);  // Will not fuse Relu into QuantizeLinear due to level1 opt.
  test_case(TransformerLevel::Level2, -128);  // Will fuse Relu into QuantizeLinear.
  test_case(TransformerLevel::Level3, -128);  // Will fuse Relu into QuantizeLinear.
  test_case(TransformerLevel::Level3, 0);     // Will not fuse Relu into QuantizeLinear due to zero-point != -128
}

template <typename ScaleType, typename ZpType>
void TestWhereWithDqInput(bool is_dq_1,
                          bool is_dq_2,
                          int expected_num_where,
                          int expected_num_dq,
                          int expected_num_q,
                          bool expected_modified) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("WhereDummyDqTester", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  NodeArg* where_in1 = nullptr;
  NodeArg* where_in2 = nullptr;
  if (is_dq_1) {
    // DQ
    auto* dq_Input = builder.MakeInput<ZpType>({4, 3, 32}, 0.0, 1.0);
    auto* dq_scale = builder.MakeInitializer<ScaleType>({}, 0.0, 1.0);
    auto* dq_zp = builder.MakeInitializer<ZpType>({}, 0.0, 1.0);
    where_in1 = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {dq_Input, dq_scale, dq_zp}, {where_in1});
  } else {
    where_in1 = builder.MakeInitializer<float>({}, 0.0, 1.0);
  }
  if (is_dq_2) {
    // DQ
    auto* dq_Input = builder.MakeInput<ZpType>({4, 3, 32}, 0.0, 1.0);
    auto* dq_scale = builder.MakeInitializer<ScaleType>({}, 0.0, 1.0);
    auto* dq_zp = builder.MakeInitializer<ZpType>({}, 0.0, 1.0);
    where_in2 = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {dq_Input, dq_scale, dq_zp}, {where_in2});
  } else {
    where_in2 = builder.MakeInitializer<float>({}, 0.0, 1.0);
  }

  // Where
  auto* where_cond = builder.MakeInputBool({4, 3, 32});
  auto* where_out = builder.MakeIntermediate();
  builder.AddNode("Where", {where_cond, where_in1, where_in2}, {where_out});

  // Q
  auto* q_scale = builder.MakeInitializer<float>({}, 0.0, 1.0);
  auto* q_zp = builder.MakeInitializer<uint16_t>({}, 0.0, 1.0);
  auto* q_out = builder.MakeOutput();
  builder.AddNode("QuantizeLinear", {where_out, q_scale, q_zp}, {q_out});

  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  auto where_optimizer = std::make_unique<WhereDummyDq>();
  bool modified = false;
  ASSERT_STATUS_OK(where_optimizer->Apply(graph, modified, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Where"], expected_num_where);
  ASSERT_EQ(op_to_count["DequantizeLinear"], expected_num_dq);
  ASSERT_EQ(op_to_count["QuantizeLinear"], expected_num_q);
  ASSERT_EQ(modified, expected_modified);

  return;
};

TEST(QDQTransformerTests, WhereDummyDqTest) {
  TestWhereWithDqInput<float, uint8_t>(true, true, 1, 2, 1, false);
  TestWhereWithDqInput<float, uint8_t>(true, false, 1, 2, 1, true);
  TestWhereWithDqInput<float, uint8_t>(false, true, 1, 2, 1, true);
  TestWhereWithDqInput<float, uint8_t>(false, false, 1, 0, 1, false);
  TestWhereWithDqInput<float, uint16_t>(true, true, 1, 2, 1, false);
  TestWhereWithDqInput<float, uint16_t>(true, false, 1, 2, 1, true);
  TestWhereWithDqInput<float, uint16_t>(false, true, 1, 2, 1, true);
  TestWhereWithDqInput<float, uint16_t>(false, false, 1, 0, 1, false);
}

TEST(QDQTransformerTests, Concat) {
  auto test_case = [&](const std::vector<std::vector<int64_t>>& input_shapes,
                       int64_t axis,
                       bool has_input_float,
                       bool has_input_int8,
                       bool has_output_int8,
                       bool use_contrib_qdq) {
    auto check_graph = [&input_shapes, has_input_float, has_input_int8, has_output_int8,
                        use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if (has_input_float || has_input_int8 || has_output_int8) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearConcat"], 0);
      } else {
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], static_cast<int>(input_shapes.size()));
        EXPECT_EQ(op_to_count["com.microsoft.QLinearConcat"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      }
    };

    TransformerTester(BuildQDQConcatTestCase(input_shapes,
                                             axis,
                                             has_input_float,
                                             has_input_int8,
                                             has_output_int8,
                                             use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQConcatTestCase(input_shapes,
                                             axis,
                                             has_input_float,
                                             has_input_int8,
                                             has_output_int8,
                                             use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(BuildQDQConcatTestCase(input_shapes,
                                             axis,
                                             has_input_float,
                                             has_input_int8,
                                             has_output_int8,
                                             use_contrib_qdq),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({{1, 6, 36}, {1, 3, 36}}, 1,
            false,   // has_input_float
            false,   // has_input_int8
            false,   // has_output_int8
            false);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 3, 36}}, 1,
            false,  // has_input_float
            false,  // has_input_int8
            false,  // has_output_int8
            true);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            false,   // has_input_float
            false,   // has_input_int8
            false,   // has_output_int8
            false);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            true,    // has_input_float
            false,   // has_input_int8
            false,   // has_output_int8
            false);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            true,   // has_input_float
            false,  // has_input_int8
            false,  // has_output_int8
            true);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            false,   // has_input_float
            true,    // has_input_int8
            false,   // has_output_int8
            false);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            false,  // has_input_float
            true,   // has_input_int8
            false,  // has_output_int8
            true);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            false,   // has_input_float
            false,   // has_input_int8
            true,    // has_output_int8
            false);  // use_contrib_qdq
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2,
            false,  // has_input_float
            false,  // has_input_int8
            true,   // has_output_int8
            true);  // use_contrib_qdq
}

template <typename InputType, typename OutputType>
void QDQTransformerSoftmaxTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, int64_t axis, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -5.f, 5.f);
      auto* output_arg = builder.MakeOutput();
      // add QDQ + Softmax
      auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .105f,
                                                  (std::numeric_limits<OutputType>::max() / 255 * 255) / 2,
                                                  use_contrib_qdq);
      auto* softmax_output = builder.MakeIntermediate();
      auto& softmax_node = builder.AddNode("Softmax", {dq_output}, {softmax_output});
      softmax_node.AddAttribute("axis", axis);
      // add QDQ output
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<OutputType>(softmax_output,
                                                1.0f / (std::numeric_limits<OutputType>::max() + 1),
                                                0,
                                                q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                  1.0f / (std::numeric_limits<OutputType>::max() + 1),
                                                  0,
                                                  output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSoftmax"], 1);
        EXPECT_EQ(op_to_count["Softmax"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSoftmax"], 0);
        EXPECT_EQ(op_to_count["Softmax"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37}, -1, false /*use_contrib_qdq*/);
  test_case({1, 12, 37}, -1, true /*use_contrib_qdq*/);
  test_case({1, 23, 13, 13}, -2, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, Softmax_S8S8) {
  QDQTransformerSoftmaxTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, Softmax_U8U8) {
  QDQTransformerSoftmaxTests<uint8_t, uint8_t>();
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, QDQPropagation_QBackward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       size_t maxpool_dim,
                       const std::vector<int64_t>& perms,
                       bool add_op_boundary,
                       bool include_zp,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      auto* transpose_input = add_op_boundary ? builder.MakeIntermediate() : input_arg;
      if (add_op_boundary) {
        // add Sign as boundary for QDQ propagation
        builder.AddNode("Sign", {input_arg}, {transpose_input});
      }

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {transpose_input}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((maxpool_dim - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(maxpool_dim - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_output = builder.MakeIntermediate();
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      // add Q
      constexpr float qdq_scale = 0.004f;
      if (include_zp) {
        constexpr uint8_t qdq_zero_point = 129;
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, qdq_scale, qdq_zero_point, output_arg, use_contrib_qdq);
      } else {
        builder.AddQuantizeLinearNode(reshape_output, qdq_scale, output_arg, use_contrib_qdq);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      std::vector<std::string> expected_op_types_in_order{};
      if (add_op_boundary) {
        expected_op_types_in_order.push_back("Sign");
      }
      expected_op_types_in_order.insert(
          expected_op_types_in_order.end(),
          {qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
           "Transpose",
           qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
           "MaxPool",
           qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
           "Reshape",
           qdq_keys.quantize_linear});

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, true, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, false, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, true, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false, true /*use_contrib_qdq*/);
#endif
}

// Test backwards propagation of a QuantizeLinear node that uses the "output_dtype" attribute
// to set the quantization type (i.e., does not have an explicit zero-point input). This tests
// the copying of attributes for QDQ propagation.
TEST(QDQTransformerTests, QDQPropagation_QBackward_NoZP_OutputDtypeAttribute) {
  auto test_case = [&](ONNX_NAMESPACE::TensorProto_DataType q_output_type) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({1, 2, 2}, {-2.0f, 0.0f, 1.0f, 2.0f});
      auto* output_arg = builder.MakeOutput();

      // add Add
      auto* const_1_input = builder.MakeScalarInitializer<float>(1.0f);
      auto* add_output = builder.MakeIntermediate();
      builder.AddNode("Add", {input_arg, const_1_input}, {add_output});

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      builder.AddNode("Transpose", {add_output}, {transpose_output});

      // add Q with a "output_dtype" attribute. Omit the zero-point input (defaults to 0).
      constexpr float qdq_scale = 1.0f;
      Node& q_node = builder.AddQuantizeLinearNode(transpose_output, qdq_scale, output_arg);
      q_node.AddAttribute("output_dtype", static_cast<int64_t>(q_output_type));
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
      std::vector<std::string> expected_op_types_in_order = {
          "Add",
          qdq_keys.quantize_linear,
          qdq_keys.dequantize_linear,
          "Transpose",
          qdq_keys.quantize_linear,
      };

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      21);  // Opset >= 21 supports the "output_dtype" attribute
  };

  test_case(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  test_case(ONNX_NAMESPACE::TensorProto_DataType_INT8);
  test_case(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
  test_case(ONNX_NAMESPACE::TensorProto_DataType_INT16);
}

TEST(QDQTransformerTests, QDQPropagation_DQForward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       size_t maxpool_dim,
                       const std::vector<int64_t>& perms,
                       bool add_op_boundary,
                       bool include_zp,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      constexpr float qdq_scale = 0.004f;
      auto* dq_output = builder.MakeIntermediate();
      if (include_zp) {
        constexpr uint8_t qdq_zero_point = 129;
        builder.AddDequantizeLinearNode<uint8_t>(input_arg, qdq_scale, qdq_zero_point, dq_output, use_contrib_qdq);
      } else {
        builder.AddDequantizeLinearNode(input_arg, qdq_scale, dq_output, use_contrib_qdq);
      }

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((maxpool_dim - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(maxpool_dim - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = add_op_boundary ? builder.MakeIntermediate() : output_arg;
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      if (add_op_boundary) {
        // add Sign as boundary for QDQ propagation
        builder.AddNode("Sign", {reshape_output}, {output_arg});
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      std::vector<std::string> expected_op_types_in_order{
          qdq_keys.dequantize_linear,
          "Transpose",
          qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
          "MaxPool",
          qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
          "Reshape",
          qdq_keys.quantize_linear, qdq_keys.dequantize_linear};
      if (add_op_boundary) {
        expected_op_types_in_order.push_back("Sign");
      }

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      12, 0.0, 0.0, nullptr, {},  // defaults that we're not overriding
                      {"TransposeOptimizer"});    // disable TransposeOptimizer for simplicity
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      18, 0.0, 0.0, nullptr, {},  // defaults that we're not overriding
                      {"TransposeOptimizer"});    // disable TransposeOptimizer for simplicity
    // TODO: fix opset 19
  };

  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, true, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, false, false /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, true, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false, true /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, true, true /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, false, true /*use_contrib_qdq*/);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, true, true /*use_contrib_qdq*/);
#endif
}

TEST(QDQTransformerTests, QDQPropagation_StopAtOtherQDQ) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool same_scale, bool same_zp,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ
      auto* qdq_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129, use_contrib_qdq);

      // Reshape
      auto* reshape_output = builder.MakeIntermediate();
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      builder.AddNode("Reshape", {qdq_output, reshape_shape}, {reshape_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, same_scale ? .004f : .0039f, same_zp ? 129 : 128,
                                             output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const std::vector<std::string> expected_op_types_in_order{
          qdq_keys.quantize_linear,
          qdq_keys.dequantize_linear,
          "Reshape",
          qdq_keys.quantize_linear};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, false, false, false);
  test_case({1, 13, 13, 23}, false, true, false);
  test_case({1, 13, 13, 23}, true, false, false);
  test_case({1, 13, 13, 23}, true, true, false);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case({1, 13, 13, 23}, false, false, true);
  test_case({1, 13, 13, 23}, false, true, true);
  test_case({1, 13, 13, 23}, true, false, true);
  test_case({1, 13, 13, 23}, true, true, true);
#endif
}

TEST(QDQTransformerTests, QDQPropagation_Q_No_Parent) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {input_arg}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const std::vector<std::string> expected_op_types_in_order{
          qdq_keys.quantize_linear,
          qdq_keys.dequantize_linear,
          "Transpose",
          qdq_keys.quantize_linear};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, true /*use_contrib_qdq*/);
#endif
}

TEST(QDQTransformerTests, QDQPropagation_DQ_No_Children) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .0035f, 135, dq_output, use_contrib_qdq);

      // add transpose
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {output_arg});
      transpose_node.AddAttribute("perm", perms);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const std::vector<std::string> expected_op_types_in_order{
          qdq_keys.dequantize_linear,
          "Transpose",
          qdq_keys.quantize_linear, qdq_keys.dequantize_linear};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, true /*use_contrib_qdq*/);
#endif
}

TEST(QDQTransformerTests, QDQPropagation_Per_Layer_No_Propagation) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ with per layer scale and zp values
      auto* dq_output = builder.MakeIntermediate();
      auto* dq_scale = builder.Make1DInitializer(std::vector<float>(input_shape[1], 0.0035f));
      auto* dq_zp = builder.Make1DInitializer(std::vector<uint8_t>(input_shape[1], 135));
      builder.AddNode("DequantizeLinear", {input_arg, dq_scale, dq_zp}, {dq_output},
                      use_contrib_qdq ? kMSDomain : "");

      // add transpose
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {output_arg});
      transpose_node.AddAttribute("perm", perms);
    };

    bool use_transpose_optimizer = false;

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);

      // if the transpose optimizer isn't used the DQ doesn't propagate past the Transpose
      // TODO: Should it? It makes it easier for an EP to do a quantized Tranpose if it's in a QDQ node unit as it
      // doesn't have to special-case looking for a solo Transpose.
      std::vector<std::string> expected_op_types_in_order{qdq_keys.dequantize_linear,
                                                          "Transpose"};
      if (use_transpose_optimizer) {
        // fixup of QDQ node units would have put the Transpose in a QDQ node unit for consistency IFF
        // the scale and zero point inputs are constant (which they are here)
        expected_op_types_in_order.push_back(qdq_keys.quantize_linear);
        expected_op_types_in_order.push_back(qdq_keys.dequantize_linear);
      }

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);

      if (use_transpose_optimizer) {
        // the trailing Q/DQ should have updated axis based on the transpose. default axis of 1 moves to 3 with
        // transpose of {0,2,3,1} (NCHW -> NHWC)
        GraphViewer graph_viewer{session.GetGraph()};
        const auto& ordered_nodes = graph_viewer.GetNodesInTopologicalOrder();
        const auto& q_node = *graph_viewer.GetNode(ordered_nodes.back() - 1);
        const auto& dq_node = *graph_viewer.GetNode(ordered_nodes.back());

        EXPECT_EQ(graph_utils::GetNodeAttribute(q_node, std::string("axis"))->i(), 3);
        EXPECT_EQ(graph_utils::GetNodeAttribute(dq_node, std::string("axis"))->i(), 3);
      }
    };

    auto run_test = [&](int opset) {
      use_transpose_optimizer = true;
      TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset);

      use_transpose_optimizer = false;
      TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset,
                        // defaults that we're not overriding
                        0.0, 0.0, nullptr, {},
                        // disable generic L1 and CPU EP specific L2 TransposeOptimizer
                        {"TransposeOptimizer", std::string("TransposeOptimizer_") + kCpuExecutionProvider});
    };

    run_test(12);
    run_test(18);
    run_test(19);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, {0, 2, 3, 1}, true /*use_contrib_qdq*/);
#endif
}

TEST(QDQTransformerTests, QDQPropagation_DQ_Q) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .0035f, 135, dq_output, use_contrib_qdq);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(dq_output, .0035f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const std::vector<std::string> expected_op_types_in_order{
          qdq_keys.dequantize_linear,
          qdq_keys.quantize_linear};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, false /*use_contrib_qdq*/);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case({1, 13, 13, 23}, true /*use_contrib_qdq*/);
#endif
}

// Test propagating a DQ forward through a chain of Slice and Transpose operators that have multiple consumers.
// original model:
//   in0 -> DQ -> Slice --+--> slice_out
//                        |
//                        +--> Add -> out0
//                        |
//                        +--> Transpose --+--> Pow -> out1
//                        |                |
//                        |                +--> Pow -> out2
//                        |
//                        +--> Transpose --+--> Pow -> out3
//                                         |
//                                         +--> Pow -> out4
// expected model:
//   in0 -> DQ -> Slice -> Q --+--> DQ -> slice_out
//                             |
//                             +--> DQ -> Add -> out0
//                             |
//                             +--> DQ -> TP -> Q --+--> DQ -> Pow -> out1
//                             |                    |
//                             |                    +--> DQ -> Pow -> out2
//                             |
//                             +--> DQ -> TP -> Q --+--> DQ -> Pow -> out3
//                                                  |
//                                                  +--> DQ -> Pow -> out4
TEST(QDQTransformerTests, QDQPropagation_DQForward_SliceMultipleConsumers) {
  auto run_test_case = [&](bool slice_has_graph_output) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<int64_t> input0_shape = {1, 2, 2, 2};
      std::vector<int64_t> input1_shape = {1, 1, 1, 1};
      auto* input0_arg = builder.MakeInput<uint8_t>(input0_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* input1_arg = builder.MakeInput<float>(input1_shape, {0.0f});
      auto* output0_arg = builder.MakeOutput();
      auto* output1_arg = builder.MakeOutput();
      auto* output2_arg = builder.MakeOutput();
      auto* output3_arg = builder.MakeOutput();
      auto* output4_arg = builder.MakeOutput();

      // DQ
      constexpr float qdq_scale = 1.0f;
      constexpr uint8_t qdq_zero_point = 128;
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input0_arg, qdq_scale, qdq_zero_point, dq_output);

      // Slice
      auto* slice_output = slice_has_graph_output ? builder.MakeOutput() : builder.MakeIntermediate();
      auto* slice_starts = builder.Make1DInitializer(std::vector<int64_t>{0, 0, 0, 0});
      auto* slice_ends = builder.Make1DInitializer(std::vector<int64_t>{1, 1, 1, 1});
      builder.AddNode("Slice", {dq_output, slice_starts, slice_ends}, {slice_output});

      // Add
      builder.AddNode("Add", {slice_output, input1_arg}, {output0_arg});

      // Transpose
      auto* transpose0_output = builder.MakeIntermediate();
      builder.AddNode("Transpose", {slice_output}, {transpose0_output});

      // Transpose
      auto* transpose1_output = builder.MakeIntermediate();
      builder.AddNode("Transpose", {slice_output}, {transpose1_output});

      // Pows
      auto* pow_exp = builder.MakeScalarInitializer(2.0f);
      builder.AddNode("Pow", {transpose0_output, pow_exp}, {output1_arg});
      builder.AddNode("Pow", {transpose0_output, pow_exp}, {output2_arg});
      builder.AddNode("Pow", {transpose1_output, pow_exp}, {output3_arg});
      builder.AddNode("Pow", {transpose1_output, pow_exp}, {output4_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
      std::vector<std::string> expected_op_types_in_order;
      expected_op_types_in_order.reserve(20);
      expected_op_types_in_order.insert(expected_op_types_in_order.end(),
                                        {qdq_keys.dequantize_linear,
                                         "Slice",
                                         qdq_keys.quantize_linear});

      if (slice_has_graph_output) {
        // Should have a DQ before the graph output generated by the Slice.
        expected_op_types_in_order.push_back(qdq_keys.dequantize_linear);
      }

      expected_op_types_in_order.insert(expected_op_types_in_order.end(),
                                        {qdq_keys.dequantize_linear,
                                         "Add",
                                         qdq_keys.dequantize_linear,
                                         "Transpose",
                                         qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
                                         "Pow",
                                         qdq_keys.dequantize_linear,
                                         "Pow",
                                         qdq_keys.dequantize_linear,
                                         "Transpose",
                                         qdq_keys.quantize_linear, qdq_keys.dequantize_linear,
                                         "Pow",
                                         qdq_keys.dequantize_linear,
                                         "Pow"});

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      18, 0.0, 0.0, std::make_unique<QDQPropagationTransformer>());
  };

  run_test_case(/*slice_has_graph_output*/ false);
  run_test_case(/*slice_has_graph_output*/ true);
}

#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
static void VerifyIODef(const NodeUnitIODef& io_def, const Node& node) {
  const auto& op_type = node.OpType();
  const bool is_dq = op_type == "DequantizeLinear";
  const bool is_q = op_type == "QuantizeLinear";
  ASSERT_TRUE(is_dq || is_q);
  const auto input_defs = node.InputDefs();
  if (is_dq) {
    ASSERT_EQ(&io_def.node_arg, input_defs[0]);
  } else {  // is_q
    ASSERT_EQ(&io_def.node_arg, node.OutputDefs()[0]);
  }

  ASSERT_EQ(&io_def.quant_param->scale, input_defs[1]);

  // [optional] zero point should be consistent between NodeUnitIODef and Input/OutputDefs
  ASSERT_EQ(input_defs.size() == 3, io_def.quant_param->zero_point != nullptr);
  if (input_defs.size() == 3) {
    // we have zero point
    ASSERT_EQ(io_def.quant_param->zero_point, input_defs[2]);
  }
}
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)

TEST(QDQTransformerTests, QDQ_Selector_Test) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/transform/qdq_conv.onnx");
  const auto& logger = DefaultLoggingManager().DefaultLogger();

  SessionOptions so;
  // We want to keep the graph un-optimized to prevent QDQ transformer to kick in
  so.graph_optimization_level = TransformerLevel::Default;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  const Graph& graph = session_object.GetGraph();
  const auto* conv_node = graph.GetNode(3);

  // Make sure node 3 is the conv node
  ASSERT_TRUE(nullptr != conv_node);
  ASSERT_EQ("Conv", conv_node->OpType());

  onnxruntime::QDQ::ConvNodeGroupSelector conv_selector;

  // Initialize SelectorManager
  QDQ::SelectorManager selector_mgr;

  // Create a GraphViewer covers the whole graph
  const GraphViewer whole_graph_viewer(graph);

  // Make sure the conv QDQ group is selected for the full graph
  {
    const auto result = conv_selector.GetQDQSelection(whole_graph_viewer, *conv_node);
    ASSERT_TRUE(result.has_value());
    const auto& qdq_group = *result;
    ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
    ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
    ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
  }

  // Check if SelectorManager get a conv qdq group selection as expected
  {
    const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer, logger);
    ASSERT_FALSE(result.empty());
    const auto& qdq_group = result.at(0);
    ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
    ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
    ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
  }

// The function GetAllNodeUnits is used by NNAPI, XNNPACK and QNN
#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
  {
    // Get all the NodeUnits in the graph_viewer
    std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
    std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

    std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(whole_graph_viewer, logger);

    // We should get a single QDQ Node unit in the result
    ASSERT_EQ(1, node_unit_holder.size());
    ASSERT_EQ(5, node_unit_map.size());
    const auto& qdq_node_unit = *node_unit_holder[0];
    ASSERT_EQ(NodeUnit::Type::QDQGroup, qdq_node_unit.UnitType());

    ASSERT_EQ(3, qdq_node_unit.Inputs().size());
    ASSERT_EQ(1, qdq_node_unit.Outputs().size());
    ASSERT_EQ(conv_node, &qdq_node_unit.GetNode());

    // We know the graph has 5 nodes, DQ_input, DQ_weight, DQ_bias, Conv, Q_output (index 0-4)
    VerifyIODef(qdq_node_unit.Inputs()[0], *whole_graph_viewer.GetNode(0));   // DQ_input
    VerifyIODef(qdq_node_unit.Inputs()[1], *whole_graph_viewer.GetNode(1));   // DQ_weight
    VerifyIODef(qdq_node_unit.Inputs()[2], *whole_graph_viewer.GetNode(2));   // DQ_bias
    VerifyIODef(qdq_node_unit.Outputs()[0], *whole_graph_viewer.GetNode(4));  // Q_output
  }
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)

  // Create a graph viewer covers part of the graph
  // Make sure the qdq conv selector will fail for the partial graph
  {
    // Get 3 nodes out of 5 nodes in the graph
    std::vector<const Node*> nodes{
        graph.GetNode(0),
        graph.GetNode(3),
        graph.GetNode(4),
    };

    // Generate the indexed subgraph
    const auto compute_capability = utils::MakeComputeCapability(
        whole_graph_viewer, nodes,
        []() { return "sub_graph"; },
        "Test Provider",
        /*drop_constant_initializers*/ false);

    const GraphViewer partial_graph_viewer(graph, *compute_capability->sub_graph);
    ASSERT_EQ(3, partial_graph_viewer.NumberOfNodes());

    // Check there is no qdq selection for the given nodes
    {
      const auto result = conv_selector.GetQDQSelection(partial_graph_viewer, *conv_node);
      ASSERT_FALSE(result.has_value());
    }

    // Check SelectorManager will get empty result
    {
      const auto result = selector_mgr.GetQDQSelections(partial_graph_viewer, logger);
      ASSERT_TRUE(result.empty());
    }
  }
}

TEST(QDQTransformerTests, QDQ_Selector_Test_Conv_Relu) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Relu is redundant.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 2, 4, 4}, std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* weight_arg = builder.MakeInput<uint8_t>({2, 1, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* bias_arg =
          builder.MakeInput<int32_t>({2}, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
      // DQ
      auto* dq_input = builder.MakeIntermediate();
      auto* dq_weight = builder.MakeIntermediate();
      auto* dq_bias = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_arg, 0.02348f, uint8_t(0), dq_input, false);
      builder.AddDequantizeLinearNode(weight_arg, 0.307f, uint8_t(0), dq_weight, false);
      builder.AddDequantizeLinearNode(bias_arg, 0.007f, int32_t(0), dq_bias, false);

      // Conv
      auto* conv_output = builder.MakeIntermediate();
      Node& conv_node = builder.AddNode("Conv", {dq_input, dq_weight, dq_bias}, {conv_output});
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("group", int64_t(2));
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

      // Relu
      auto* relu_output = builder.MakeIntermediate();
      builder.AddNode("Relu", {conv_output}, {relu_output});

      // Q
      auto* q_output = builder.MakeOutput();
      builder.AddQuantizeLinearNode(relu_output, 0.02348f, uint8_t(0), q_output, false);
    };

    // Build the model for this test.
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 18;
    domain_to_version[kMSDomain] = 1;
    Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, logger);
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    build_test_case(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    const GraphViewer whole_graph_viewer(graph);

    // Make sure node 3 is the conv node
    const auto* conv_node = graph.GetNode(3);
    ASSERT_TRUE(nullptr != conv_node);
    ASSERT_EQ("Conv", conv_node->OpType());

    // Make sure the conv QDQ group is selected for the full graph
    {
      onnxruntime::QDQ::ConvNodeGroupSelector conv_selector;
      const auto result = conv_selector.GetQDQSelection(whole_graph_viewer, *conv_node);
      ASSERT_TRUE(result.has_value());
      const auto& qdq_group = *result;
      ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
      ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
      ASSERT_EQ(NodeIndex(4), qdq_group.redundant_clip_node);
      ASSERT_EQ(std::vector<NodeIndex>({5}), qdq_group.q_nodes);
    }

    // Check if SelectorManager get a conv qdq group selection as expected
    {
      QDQ::SelectorManager selector_mgr;
      const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer, logger);
      ASSERT_FALSE(result.empty());
      const auto& qdq_group = result.at(0);
      ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
      ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
      ASSERT_EQ(NodeIndex(4), qdq_group.redundant_clip_node);
      ASSERT_EQ(std::vector<NodeIndex>({5}), qdq_group.q_nodes);
    }

// The function GetAllNodeUnits is used by NNAPI, XNNPACK and QNN
#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
    {
      // Get all the NodeUnits in the graph_viewer
      std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
      std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

      std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(whole_graph_viewer, logger);

      // We should get a single QDQ Node unit in the result
      ASSERT_EQ(1, node_unit_holder.size());
      ASSERT_EQ(6, node_unit_map.size());
      const auto& qdq_node_unit = *node_unit_holder[0];
      ASSERT_EQ(NodeUnit::Type::QDQGroup, qdq_node_unit.UnitType());

      ASSERT_EQ(3, qdq_node_unit.Inputs().size());
      ASSERT_EQ(1, qdq_node_unit.Outputs().size());
      ASSERT_EQ(conv_node, &qdq_node_unit.GetNode());
      ASSERT_EQ(whole_graph_viewer.GetNode(4), qdq_node_unit.GetRedundantClipNode());

      // We know the graph has 5 nodes, DQ_input, DQ_weight, DQ_bias, Conv, Relu, Q_output (index 0-5)
      VerifyIODef(qdq_node_unit.Inputs()[0], *whole_graph_viewer.GetNode(0));   // DQ_input
      VerifyIODef(qdq_node_unit.Inputs()[1], *whole_graph_viewer.GetNode(1));   // DQ_weight
      VerifyIODef(qdq_node_unit.Inputs()[2], *whole_graph_viewer.GetNode(2));   // DQ_bias
      VerifyIODef(qdq_node_unit.Outputs()[0], *whole_graph_viewer.GetNode(5));  // Q_output
    }
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
  }

  // Relu is NOT redundant.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 2, 4, 4}, std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* weight_arg = builder.MakeInput<uint8_t>({2, 1, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* bias_arg =
          builder.MakeInput<int32_t>({2}, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
      // DQ
      auto* dq_input = builder.MakeIntermediate();
      auto* dq_weight = builder.MakeIntermediate();
      auto* dq_bias = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_arg, 0.02348f, uint8_t(0), dq_input, false);
      builder.AddDequantizeLinearNode(weight_arg, 0.307f, uint8_t(0), dq_weight, false);
      builder.AddDequantizeLinearNode(bias_arg, 0.007f, int32_t(0), dq_bias, false);

      // Conv
      auto* conv_output = builder.MakeIntermediate();
      Node& conv_node = builder.AddNode("Conv", {dq_input, dq_weight, dq_bias}, {conv_output});
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("group", int64_t(2));
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});

      // Relu
      auto* relu_output = builder.MakeIntermediate();
      builder.AddNode("Relu", {conv_output}, {relu_output});

      // Q
      auto* q_output = builder.MakeOutput();
      builder.AddQuantizeLinearNode(relu_output, 0.02348f, uint8_t(127), q_output, false);
    };

    // Build the model for this test.
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 18;
    domain_to_version[kMSDomain] = 1;
    Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, logger);
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    build_test_case(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    const GraphViewer whole_graph_viewer(graph);

    // Make sure node 3 is the conv node
    const auto* conv_node = graph.GetNode(3);
    ASSERT_TRUE(nullptr != conv_node);
    ASSERT_EQ("Conv", conv_node->OpType());

    // Make sure the conv QDQ group is selected for the full graph
    {
      onnxruntime::QDQ::ConvNodeGroupSelector conv_selector;
      const auto result = conv_selector.GetQDQSelection(whole_graph_viewer, *conv_node);
      ASSERT_FALSE(result.has_value());
    }

    // Check if SelectorManager get a conv qdq group selection as expected
    {
      QDQ::SelectorManager selector_mgr;
      const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer, logger);
      ASSERT_TRUE(result.empty());
    }

// The function GetAllNodeUnits is used by NNAPI, XNNPACK and QNN
#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
    {
      std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
      std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
      std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(whole_graph_viewer, logger);
      ASSERT_EQ(6, node_unit_holder.size());
      ASSERT_EQ(6, node_unit_map.size());
    }
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
  }
}

TEST(QDQTransformerTests, QDQ_Selector_Test_Add_Clip) {
  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Clip is redundant.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_0_arg = builder.MakeInput<uint8_t>({2, 3, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                     std::numeric_limits<uint8_t>::max());
      auto* input_1_arg = builder.MakeInput<uint8_t>({2, 1, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                     std::numeric_limits<uint8_t>::max());

      // DQ
      auto* dq_input_0 = builder.MakeIntermediate();
      auto* dq_input_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_0_arg, 0.02348f, uint8_t(0), dq_input_0, false);
      builder.AddDequantizeLinearNode(input_1_arg, 0.307f, uint8_t(0), dq_input_1, false);

      // Add
      auto* add_output = builder.MakeIntermediate();
      builder.AddNode("Add", {dq_input_0, dq_input_1}, {add_output});

      // Clip
      auto* clip_min_arg = builder.MakeInitializer<float>({}, {0.0f});
      auto* clip_max_arg = builder.MakeInitializer<float>({}, {6.0f});
      auto* clip_output = builder.MakeIntermediate();
      builder.AddNode("Clip", {add_output, clip_min_arg, clip_max_arg}, {clip_output});

      // Q
      auto* q_output = builder.MakeOutput();
      builder.AddQuantizeLinearNode(clip_output, 0.02348f, uint8_t(0), q_output, false);
    };

    // Build the model for this test.
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 18;
    domain_to_version[kMSDomain] = 1;
    Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, logger);
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    build_test_case(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    const GraphViewer whole_graph_viewer(graph);

    const auto* add_node = graph.GetNode(2);
    ASSERT_TRUE(nullptr != add_node);
    ASSERT_EQ("Add", add_node->OpType());

    // Make sure the add QDQ group is selected for the full graph
    {
      onnxruntime::QDQ::BinaryNodeGroupSelector add_selector;
      const auto result = add_selector.GetQDQSelection(whole_graph_viewer, *add_node);
      ASSERT_TRUE(result.has_value());
      const auto& qdq_group = *result;
      ASSERT_EQ(std::vector<NodeIndex>({0, 1}), qdq_group.dq_nodes);
      ASSERT_EQ(NodeIndex(2), qdq_group.target_node);
      ASSERT_EQ(NodeIndex(3), qdq_group.redundant_clip_node);
      ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
    }

    // Check if SelectorManager get a add qdq group selection as expected
    {
      QDQ::SelectorManager selector_mgr;
      const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer, logger);
      ASSERT_FALSE(result.empty());
      const auto& qdq_group = result.at(0);
      ASSERT_EQ(std::vector<NodeIndex>({0, 1}), qdq_group.dq_nodes);
      ASSERT_EQ(NodeIndex(2), qdq_group.target_node);
      ASSERT_EQ(NodeIndex(3), qdq_group.redundant_clip_node);
      ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
    }

// The function GetAllNodeUnits is used by NNAPI, XNNPACK and QNN
#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
    {
      // Get all the NodeUnits in the graph_viewer
      std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
      std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

      std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(whole_graph_viewer, logger);

      // We should get a single QDQ Node unit in the result
      ASSERT_EQ(1, node_unit_holder.size());
      ASSERT_EQ(5, node_unit_map.size());
      const auto& qdq_node_unit = *node_unit_holder[0];
      ASSERT_EQ(NodeUnit::Type::QDQGroup, qdq_node_unit.UnitType());

      ASSERT_EQ(2, qdq_node_unit.Inputs().size());
      ASSERT_EQ(1, qdq_node_unit.Outputs().size());
      ASSERT_EQ(add_node, &qdq_node_unit.GetNode());

      // We know the graph has 5 nodes, DQ_input_0, DQ_input_1, Add, Clip, Q_output (index 0-4)
      VerifyIODef(qdq_node_unit.Inputs()[0], *whole_graph_viewer.GetNode(0));   // DQ_input_0
      VerifyIODef(qdq_node_unit.Inputs()[1], *whole_graph_viewer.GetNode(1));   // DQ_input_1
      VerifyIODef(qdq_node_unit.Outputs()[0], *whole_graph_viewer.GetNode(4));  // Q_output
    }
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
  }

  // Clip is NOT redundant.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_0_arg = builder.MakeInput<uint8_t>({2, 3, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                     std::numeric_limits<uint8_t>::max());
      auto* input_1_arg = builder.MakeInput<uint8_t>({2, 1, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                     std::numeric_limits<uint8_t>::max());

      // DQ
      auto* dq_input_0 = builder.MakeIntermediate();
      auto* dq_input_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_0_arg, 0.02348f, uint8_t(0), dq_input_0, false);
      builder.AddDequantizeLinearNode(input_1_arg, 0.307f, uint8_t(0), dq_input_1, false);

      // Add
      auto* add_output = builder.MakeIntermediate();
      builder.AddNode("Add", {dq_input_0, dq_input_1}, {add_output});

      // Clip
      auto* clip_min_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* clip_max_arg = builder.MakeInitializer<float>({}, {5.0f});
      auto* clip_output = builder.MakeIntermediate();
      builder.AddNode("Clip", {add_output, clip_min_arg, clip_max_arg}, {clip_output});

      // Q
      auto* q_output = builder.MakeOutput();
      builder.AddQuantizeLinearNode(clip_output, 0.02348f, uint8_t(0), q_output, false);
    };

    // Build the model for this test.
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 18;
    domain_to_version[kMSDomain] = 1;
    Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, logger);
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    build_test_case(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    const GraphViewer whole_graph_viewer(graph);

    const auto* add_node = graph.GetNode(2);
    ASSERT_TRUE(nullptr != add_node);
    ASSERT_EQ("Add", add_node->OpType());

    {
      onnxruntime::QDQ::BinaryNodeGroupSelector add_selector;
      const auto result = add_selector.GetQDQSelection(whole_graph_viewer, *add_node);
      ASSERT_FALSE(result.has_value());
    }

    {
      QDQ::SelectorManager selector_mgr;
      const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer, logger);
      ASSERT_TRUE(result.empty());
    }

// The function GetAllNodeUnits is used by NNAPI, XNNPACK and QNN
#if defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
    {
      // Get all the NodeUnits in the graph_viewer
      std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
      std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
      std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(whole_graph_viewer, logger);
      ASSERT_EQ(5, node_unit_holder.size());
      ASSERT_EQ(5, node_unit_map.size());
    }
#endif  // defined(USE_NNAPI) || defined(USE_QNN) || defined(USE_XNNPACK)
  }
}

// regression test to validate TransposeOptimizer and QDQ Propagation don't loop
// see https://github.com/microsoft/onnxruntime/issues/11605
TEST(QDQTransformerTests, QDQPropagation_GH11605_Opset12_19) {
  auto test_case = [&](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 4, 4},
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_arg, 0.123f, uint8_t(0), dq_output, use_contrib_qdq);

      // add Transpose 0, 2, 1
      const std::vector<int64_t>& perms{0, 2, 1};
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Softmax with axis=2 (to block the Transpose moving past it due to the transpose perms)
      auto* softmax_output = builder.MakeIntermediate();
      Node& softmax_node = builder.AddNode("Softmax", {transpose_output}, {softmax_output});
      softmax_node.AddAttribute("axis", int64_t(2));

      // add second Transpose. this is so the check in TransposeOptimizer::ProcessTranspose for outputs leading to
      // a Transpose is satisfied, allowing the first Transpose to move past the Q/DQ inserted by QDQ Propagation
      Node& transpose_node2 = builder.AddNode("Transpose", {softmax_output}, {builder.MakeOutput()});
      transpose_node2.AddAttribute("perm", perms);
    };

    // check that an edge case where transpose optimization gets blocked is handled gracefully.
    // Original: DQ -> Tr -> SoftM -> Tr
    // QDQ Prop inserts a Q/DQ pair to create a QDQ node group for the Transpose: DQ -> Tr -> Q -> DQ -> SoftM -> Tr
    // Transpose opt phase 1 moves the Tr down until it blocks on the SoftMax: DQ -> Q -> DQ -> Tr -> SoftM -> Tr
    // Transpose opt phase 2 repairs the QDQ node units: DQ -> Q -> DQ -> Tr -> Q -> DQ -> SoftM -> Tr
    // and removes the unnecessary DQ/Q pair at the start: DQ -> Tr -> Q -> DQ -> SoftM -> Tr
    // The L2 CPU EP QDQ handling converts the DQ -> Tr -> Q to a Transpose with 8-bit data: Tr -> DQ -> SoftM -> Tr
    //   Note: This L2 CPU EP QDQ handling is currently only enabled when contrib ops are enabled.
    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
#if !defined(DISABLE_CONTRIB_OPS)
      std::vector<std::string> expected_op_types_in_order{
          "Transpose",
          qdq_keys.dequantize_linear,
          "Softmax",
          "Transpose"};
#else
      std::vector<std::string> expected_op_types_in_order{
          qdq_keys.dequantize_linear,
          "Transpose",
          qdq_keys.quantize_linear,
          qdq_keys.dequantize_linear,
          "Softmax",
          "Transpose"};
#endif

      const auto& graph = session.GetGraph();
      GraphViewer graph_viewer(graph);
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(graph, true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);

      auto first_node = graph_viewer.GetNode(graph_viewer.GetNodesInTopologicalOrder().front());
      EXPECT_EQ(*first_node->InputDefs()[0]->Type(), "tensor(uint8)");
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level2,
                      12);

    // TODO: fix opset 18, 19
  };

  test_case(false);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(true);  // Use contrib QDQ ops
#endif
}

TEST(QDQTransformerTests, QDQPropagation_GH11605_Opset13) {
  auto test_case = [&](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 4, 4},
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode(input_arg, 0.123f, uint8_t(0), dq_output, use_contrib_qdq);

      // add Transpose 0, 2, 1
      const std::vector<int64_t>& perms{0, 2, 1};
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Softmax with axis=2 (to block the Transpose moving past it due to the transpose perms)
      auto* softmax_output = builder.MakeIntermediate();
      Node& softmax_node = builder.AddNode("Softmax", {transpose_output}, {softmax_output});
      softmax_node.AddAttribute("axis", int64_t(2));

      // add second Transpose. this is so the check in TransposeOptimizer::ProcessTranspose for outputs leading to
      // a Transpose is satisfied, allowing the first Transpose to move past the Q/DQ inserted by QDQ Propagation
      Node& transpose_node2 = builder.AddNode("Transpose", {softmax_output}, {builder.MakeOutput()});
      transpose_node2.AddAttribute("perm", perms);
    };

    // check that an edge case where transpose optimization gets blocked is handled gracefully.
    // Original: DQ -> Tr -> SoftM -> Tr
    // QDQ Prop inserts a Q/DQ pair to create a QDQ node group for the Transpose: DQ -> Tr -> Q -> DQ -> SoftM -> Tr
    // Transpose opt phase 1 moves the Tr down until it blocks on the SoftMax: DQ -> Q -> DQ -> Tr -> SoftM -> Tr
    // Transpose opt phase 2 flips the Tr to prior to the DQ as it's not part of a QDQ node group at that point, as
    // running the transpose on 8-bit data should be cheaper: DQ -> Q -> Tr -> DQ -> SoftM -> Tr
    // QDQ cleanup in Level2 removes the unnecessary DQ/Q pair at the start: Tr -> DQ -> SoftM -> Tr
    // this is the optimal result as the Transpose is using 8-bit data and we have no surplus Q/DQ pairs
    auto check_graph = [&](InferenceSessionWrapper& session) {
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      std::vector<std::string> expected_op_types_in_order{
          qdq_keys.dequantize_linear,
          "Softmax"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level2,
                      13);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level2,
                      19);
  };

  test_case(false);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(true);  // Use contrib QDQ ops
#endif
}

// test removal of Q->DQ pairs by QDQFinalCleanupTransformer
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_BasicQDQCleanup) {
  auto test_case = [&](const std::vector<std::vector<int64_t>>& input_shapes,
                       bool block_removal_of_last_dq,
                       bool block_removal_of_first_dq,
                       bool use_contrib_qdq = false) {
    // create model with float input to multiple -> Q -> DQ -> Concat -> Q -> DQ -> output
    // If we enable cleanup and don't run the QDQ transformer we should drop all the Q->DQ pairs
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto input_count = input_shapes.size();
      std::vector<NodeArg*> input_args;
      std::vector<NodeArg*> q_input_args;
      for (size_t i = 0; i < input_count; i++) {
        input_args.push_back(builder.MakeInput<float>(input_shapes[i], -1.f, 1.f));
        q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.05f, 128, use_contrib_qdq));

        if (i == 0 && block_removal_of_first_dq) {
          // add another edge to the DQ node
          auto* output = builder.MakeOutput();
          builder.AddNode("Identity", {q_input_args.back()}, {output});
        }
      }
      auto* concat_output = builder.MakeIntermediate();
      Node& concat_node = builder.AddNode("Concat", q_input_args, {concat_output});
      concat_node.AddAttribute("axis", int64_t(1));

      auto* q_concat_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(concat_output, 0.05f, 128, q_concat_output, use_contrib_qdq);

      auto* output_arg = builder.MakeOutput();
      Node& dq_node = builder.AddDequantizeLinearNode<uint8_t>(q_concat_output, 0.05f, 128, output_arg,
                                                               use_contrib_qdq);

      if (block_removal_of_last_dq) {
        // add another edge to the DQ node
        auto* output = builder.MakeOutput();
        builder.AddNode("Identity", {dq_node.MutableOutputDefs()[0]}, {output});
      }
    };

    // if we block removal of the DQ node the Q node in the pair will not be removed either
    // TODO(yilyu): block_removal_of_first_dq is not functional, need to fix it
    // TODO(yilyu): block_removal_of_last_dq is not functional, need to fix it
    const int expected_qdq_count = 0;  // + (block_removal_of_first_dq ? 1 : 0) + (block_removal_of_last_dq ? 1 : 0);
    // blocking removal of DQ by adding an additional edge will cause EnsureUniqueDQForNodeUnit to duplicate the DQ,
    // so we expect twice as many DQ's as original QDQ pairs
    const int expected_dq_count = expected_qdq_count * 2;

    auto check_graph = [expected_qdq_count, expected_dq_count,
                        use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], expected_qdq_count);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expected_dq_count);
      EXPECT_EQ(op_to_count["Concat"], 1);
    };

    auto add_session_options = [](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
    };

    // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
    // essentially rounding the input values.
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
  };

  test_case({{1, 2, 4}, {1, 3, 4}}, false, false);  // Do not block removal
  test_case({{1, 2, 4}, {1, 3, 4}}, true, false);   // Block removal of last dq
  test_case({{1, 2, 4}, {1, 3, 4}}, false, true);   // Block removal of first dq
  test_case({{1, 2, 4}, {1, 3, 4}}, true, true);    // Block removal of first and last dq

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case({{1, 2, 4}, {1, 3, 4}}, false, false, true);  // Do not block removal
  test_case({{1, 2, 4}, {1, 3, 4}}, true, false, true);   // Block removal of last dq
  test_case({{1, 2, 4}, {1, 3, 4}}, false, true, true);   // Block removal of first dq
  test_case({{1, 2, 4}, {1, 3, 4}}, true, true, true);    // Block removal of first and last dq
#endif
}

// test removal of Q->DQs pairs by QDQFinalCleanupTransformer
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_BasicQDQsCleanup) {
  auto test_case = [&](bool is_input_q,
                       bool is_dq1_output,
                       bool is_dq2_output,
                       bool use_contrib_qdq) {
    // create model with float Input -> (Transpose1 ->) Q -> DQ1 -> (Transpose2 ->) Output1
    //                                                    -> DQ2 -> (Transpose3 ->) Output2
    // If we enable cleanup and don't run the QDQ transformer we should drop all the Q->DQ pairs
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_args = builder.MakeInput<float>({1, 2, 4}, -1.f, 1.f);

      // Add Input -> (Transpose1 ->)
      NodeArg* q_input = nullptr;
      if (is_input_q) {
        q_input = input_args;
      } else {
        auto* transpose_output = builder.MakeIntermediate();
        builder.AddNode("Transpose", {input_args}, {transpose_output});
        q_input = transpose_output;
      }

      // Add Q ->
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(q_input, 0.05f, 128, q_output, use_contrib_qdq);

      // Add DQ1 -> (Transpose2 ->) Output1
      if (is_dq1_output) {
        auto* dq1_output = builder.MakeOutput();
        builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.05f, 128, dq1_output, use_contrib_qdq);
      } else {
        auto* dq1_output = builder.MakeIntermediate();
        builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.05f, 128, dq1_output, use_contrib_qdq);
        auto* output1 = builder.MakeOutput();
        builder.AddNode("Transpose", {dq1_output}, {output1});
      }

      // Add DQ2 -> (Transpose3 ->) Output2
      if (is_dq2_output) {
        auto* dq2_output = builder.MakeOutput();
        builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.05f, 128, dq2_output, use_contrib_qdq);
      } else {
        auto* dq2_output = builder.MakeIntermediate();
        builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.05f, 128, dq2_output, use_contrib_qdq);
        auto* output2 = builder.MakeOutput();
        builder.AddNode("Transpose", {dq2_output}, {output2});
      }
    };

    const int expected_transpose_count = (is_input_q ? 0 : 1) + (is_dq1_output ? 0 : 1) + (is_dq2_output ? 0 : 1);

    auto check_graph = [expected_transpose_count,
                        use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["Transpose"], expected_transpose_count);
    };

    auto add_session_options = [](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
    };

    // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
    // essentially rounding the input values.
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
  };

  test_case(true, true, true, false);    // is_input_q, is_dq1_output, is_dq2_output
  test_case(false, true, true, false);   // is_dq1_output, is_dq2_output
  test_case(true, false, true, false);   // is_input_q, is_dq2_output
  test_case(false, false, true, false);  // is_dq2_output
  test_case(true, true, false, false);   // is_input_q, is_dq1_output
  test_case(false, true, false, false);  // is_dq1_output
  test_case(true, false, false, false);  // is_input_q
  test_case(false, false, false, false);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case(true, true, true, true);    // is_input_q, is_dq1_output, is_dq2_output
  test_case(false, true, true, true);   // is_dq1_output, is_dq2_output
  test_case(true, false, true, true);   // is_input_q, is_dq2_output
  test_case(false, false, true, true);  // is_dq2_output
  test_case(true, true, false, true);   // is_input_q, is_dq1_output
  test_case(false, true, false, true);  // is_dq1_output
  test_case(true, false, false, true);  // is_input_q
  test_case(false, false, false, true);
#endif
}

TEST(QDQTransformerTests, QDQFinalCleanupTransformer_BasicDQQCleanUp) {
  auto test_case = [](bool use_matching_qdq_params, bool use_contrib_qdq) {
    // input -> Q -> DQ -> Q -> DQ -> output
    auto build_test_case = [&](ModelTestBuilder& builder) {
      constexpr float scale_1 = 0.05f;
      constexpr uint8_t zp_1 = 128;
      auto* const input = builder.MakeInput<float>({1, 2, 4}, -1.0f, 1.0f);
      auto* const dq_1_out = AddQDQNodePair<uint8_t>(builder, input, scale_1, zp_1, use_contrib_qdq);

      const float scale_2 = use_matching_qdq_params ? scale_1 : scale_1 + 0.01f;
      const uint8_t zp_2 = use_matching_qdq_params ? zp_1 : zp_1 + 1;
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, dq_1_out, scale_2, zp_2, use_contrib_qdq);
    };

    auto check_graph = [&](const InferenceSessionWrapper& session) {
      const auto ops_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph(), true);
      const auto expected_ops_in_order = [&]() -> std::vector<std::string> {
        const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
        // In either case both DQ and Q will be removed and fused due to DoubleQDQPairsRemover
        return {qdq_keys.quantize_linear, qdq_keys.dequantize_linear};
      }();

      EXPECT_EQ(ops_in_order, expected_ops_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.0f /*per_sample_tolerance*/,
                      0.0f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(false /*enable_q_dq_cleanup*/));

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.0f /*per_sample_tolerance*/,
                      0.0f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(false /*enable_q_dq_cleanup*/));

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.0f /*per_sample_tolerance*/,
                      0.0f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(false /*enable_q_dq_cleanup*/));
  };

  test_case(true, false);   // Matching QDQ params
  test_case(false, false);  // Non-matching QDQ params

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case(true, true);   // Matching QDQ params
  test_case(false, true);  // Non-matching QDQ params
#endif
}

// test removal of DQ->Qs pairs by QDQFinalCleanupTransformer
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_BasicDQQsCleanup) {
  auto test_case = [&](bool is_input_dq,
                       bool is_q1_output,
                       bool is_q2_output,
                       bool use_contrib_qdq) {
    // create model with float Input -> (Transpose1 ->) DQ -> Q1 -> (Transpose2 ->) Output1
    //                                                     -> Q2 -> (Transpose3 ->) Output2
    // If we enable cleanup and don't run the QDQ transformer we should drop all the DQ->Q pairs
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_args = builder.MakeInput<uint8_t>({1, 2, 4},
                                                       std::numeric_limits<uint8_t>::min(),
                                                       std::numeric_limits<uint8_t>::max());

      // Add Input -> (Transpose1 ->)
      NodeArg* dq_input = nullptr;
      if (is_input_dq) {
        dq_input = input_args;
      } else {
        auto* transpose_output = builder.MakeIntermediate();
        builder.AddNode("Transpose", {input_args}, {transpose_output});
        dq_input = transpose_output;
      }

      // Add DQ ->
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(dq_input, 0.05f, 128, dq_output, use_contrib_qdq);

      // Add Q1 -> (Transpose2 ->) Output1
      if (is_q1_output) {
        auto* q1_output = builder.MakeOutput();
        builder.AddQuantizeLinearNode<uint8_t>(dq_output, 0.05f, 128, q1_output, use_contrib_qdq);
      } else {
        auto* q1_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<uint8_t>(dq_output, 0.05f, 128, q1_output, use_contrib_qdq);
        auto* output1 = builder.MakeOutput();
        builder.AddNode("Transpose", {q1_output}, {output1});
      }

      // Add Q2 -> (Transpose3 ->) Output2
      if (is_q2_output) {
        auto* q2_output = builder.MakeOutput();
        builder.AddQuantizeLinearNode<uint8_t>(dq_output, 0.05f, 128, q2_output, use_contrib_qdq);
      } else {
        auto* q2_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<uint8_t>(dq_output, 0.05f, 128, q2_output, use_contrib_qdq);
        auto* output2 = builder.MakeOutput();
        builder.AddNode("Transpose", {q2_output}, {output2});
      }
    };

    const int expected_transpose_count = (is_input_dq ? 0 : 1) + (is_q1_output ? 0 : 1) + (is_q2_output ? 0 : 1);

    auto check_graph = [expected_transpose_count,
                        use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["Transpose"], expected_transpose_count);
    };

    auto add_session_options = [](SessionOptions& so) {
      // The function EnsureUniqueDQForEachExplicitOutputEdge does not account for this particular case. Disable it to
      // prevent test failures.
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));

      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
    };

    // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
    // essentially rounding the input values.
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(true /*enable_q_dq_cleanup*/),
                      add_session_options);
  };

  test_case(true, true, true, false);    // is_input_dq, is_q1_output, is_q2_output
  test_case(false, true, true, false);   // is_q1_output, is_q2_output
  test_case(true, false, true, false);   // is_input_dq, is_q2_output
  test_case(false, false, true, false);  // is_q2_output
  test_case(true, true, false, false);   // is_input_dq, is_q1_output
  test_case(false, true, false, false);  // is_q1_output
  test_case(true, false, false, false);  // is_input_dq
  test_case(false, false, false, false);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case(true, true, true, true);    // is_input_dq, is_q1_output, is_q2_output
  test_case(false, true, true, true);   // is_q1_output, is_q2_output
  test_case(true, false, true, true);   // is_input_dq, is_q2_output
  test_case(false, false, true, true);  // is_q2_output
  test_case(true, true, false, true);   // is_input_dq, is_q1_output
  test_case(false, true, false, true);  // is_q1_output
  test_case(true, false, false, true);  // is_input_dq
  test_case(false, false, false, true);
#endif
}

// test removal when we have graph input -> Q/DQ pair -> graph output
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_GraphInputToOutput) {
  auto test_case = [](bool is_q_dq, bool use_contrib_qdq) {
    // create model with input -> Q/DQ pair -> output
    auto build_test_case = [&](ModelTestBuilder& builder) {
      constexpr float scale = 0.05f;
      constexpr uint8_t zp = 128;
      NodeArg* input = is_q_dq ? builder.MakeInput<float>({1, 2, 4}, -1.f, 1.f)
                               : builder.MakeInput<uint8_t>({1, 2, 4},
                                                            std::numeric_limits<uint8_t>::min(),
                                                            std::numeric_limits<uint8_t>::max());

      NodeArg* first_node_output = builder.MakeIntermediate();

      is_q_dq ? builder.AddQuantizeLinearNode<uint8_t>(input, scale, zp, first_node_output, use_contrib_qdq)
              : builder.AddDequantizeLinearNode<uint8_t>(input, scale, zp, first_node_output, use_contrib_qdq);

      auto* second_node_output = builder.MakeOutput();

      is_q_dq ? builder.AddDequantizeLinearNode<uint8_t>(first_node_output, scale, zp, second_node_output,
                                                         use_contrib_qdq)
              : builder.AddQuantizeLinearNode<uint8_t>(first_node_output, scale, zp, second_node_output,
                                                       use_contrib_qdq);
    };

    // with the Q/DQ pair being dropped we should have inserted an Identity node
    // to connect the graph input to the graph output
    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["Identity"], 1);
    };

    auto add_session_options = [&](SessionOptions& so) {
      if (is_q_dq) {
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
      }
    };

    const auto [per_sample_tolerance, relative_per_sample_tolerance] =
        is_q_dq
            // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
            // essentially rounding the input values.
            ? std::pair{0.025f, 0.01f}
            : std::pair{0.0f, 0.0f};

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      per_sample_tolerance,
                      relative_per_sample_tolerance,
                      std::make_unique<QDQFinalCleanupTransformer>(is_q_dq /*enable_q_dq_cleanup*/),
                      add_session_options);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      per_sample_tolerance,
                      relative_per_sample_tolerance,
                      std::make_unique<QDQFinalCleanupTransformer>(is_q_dq /*enable_q_dq_cleanup*/),
                      add_session_options);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      per_sample_tolerance,
                      relative_per_sample_tolerance,
                      std::make_unique<QDQFinalCleanupTransformer>(is_q_dq /*enable_q_dq_cleanup*/),
                      add_session_options);
  };

  test_case(true, false);   // input -> Q -> DQ -> output
  test_case(false, false);  // input -> DQ -> Q -> output

#if !defined(DISABLE_CONTRIB_OPS)
  // Use contrib QDQ ops
  test_case(true, true);   // input -> Q -> DQ -> output
  test_case(false, true);  // input -> DQ -> Q -> output
#endif
}

#if !defined(DISABLE_CONTRIB_OPS)
TEST(QDQTransformerTests, QDQSoftmaxWithDQProducingGraphOutput) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, int64_t axis,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -5.f, 5.f);
      auto* dq_output_arg = builder.MakeOutput();
      auto* output_arg = builder.MakeOutput();
      // add input QDQ
      auto* input_q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(input_arg,
                                             .105f,
                                             127,
                                             input_q_output,
                                             use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(input_q_output,
                                               .105f,
                                               127,
                                               dq_output_arg,
                                               use_contrib_qdq);

      // add Softmax
      auto* softmax_output = builder.MakeIntermediate();
      auto& softmax_node = builder.AddNode("Softmax", {dq_output_arg}, {softmax_output});
      softmax_node.AddAttribute("axis", axis);

      // add output QDQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(softmax_output,
                                             1.0f / (std::numeric_limits<uint8_t>::max() + 1),
                                             0,
                                             q_output,
                                             use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(q_output,
                                               1.0f / (std::numeric_limits<uint8_t>::max() + 1),
                                               0,
                                               output_arg,
                                               use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);

      // expect fusion because DQ duplication ensures that the node unit has unique DQ nodes
      EXPECT_EQ(op_to_count["com.microsoft.QLinearSoftmax"], 1);
      EXPECT_EQ(op_to_count["Softmax"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);  // duplicate of first DQ and original second DQ
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/);
  };

  test_case({1, 12, 37}, -1, false);
  test_case({1, 12, 37}, -1, true);  // Use contrib QDQ ops
}

// DQ produces graph output - special case for DropDQ path where there is only a DQ -> Node with no trailing Q
TEST(QDQTransformerTests, DropDQSelectorWithDQProducingGraphOutput) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, int64_t axis, bool dq_produces_graph_output,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -5.f, 5.f);
      auto* output_arg = builder.MakeOutput();

      // add input QDQ
      auto* input_q_output = builder.MakeIntermediate();
      auto* dq_output_arg = dq_produces_graph_output ? builder.MakeOutput() : builder.MakeIntermediate();

      builder.AddQuantizeLinearNode<uint8_t>(input_arg, .105f, 127, input_q_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(input_q_output, .105f, 127, dq_output_arg, use_contrib_qdq);

      // add ArgMax
      auto* argmax_output = builder.MakeIntermediate();
      auto& argmax_node = builder.AddNode("ArgMax", {dq_output_arg}, {argmax_output});
      argmax_node.AddAttribute("axis", axis);

      // add output Identity
      builder.AddNode("Identity", {argmax_output}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const Graph& graph = session.GetGraph();

      auto op_to_count = CountOpsInGraph(graph);
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      const auto expected_dq_count =
          dq_produces_graph_output
              ? 1   // EnsureUniqueDQForNodeUnit duplicates one DQ and DropDQ drops one DQ
              : 0;  // DropDQ drops one DQ
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], expected_dq_count);

      const auto& nodes = graph.Nodes();
      const auto argmax_node_it = std::find_if(nodes.cbegin(),
                                               nodes.cend(),
                                               [](const Node& node) { return node.OpType() == "ArgMax"; });
      ASSERT_NE(argmax_node_it, nodes.cend());

      // the DQ from Q -> DQ -> ArgMax should have been dropped, look for the Q -> ArgMax edge
      ASSERT_EQ(argmax_node_it->GetInputEdgesCount(), static_cast<size_t>(1));
      EXPECT_EQ(argmax_node_it->InputEdgesBegin()->GetNode().OpType(), "QuantizeLinear");
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/);

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/);
  };

  // test with and without the DQ producing a graph output to validate the test hits DropDQ

  // DQ does not produce graph output
  test_case({1, 4, 8}, -1, false, false);
  test_case({1, 4, 8}, -1, false, true);  // Use contrib QDQ ops

  // DQ produces graph output
  test_case({1, 4, 8}, -1, true, false);
  test_case({1, 4, 8}, -1, true, true);  // Use contrib QDQ ops
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, WeightBiasQuantization_Conv_Bias) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg = builder.MakeInput<uint8_t>({1, 24, 128, 128}, std::numeric_limits<uint8_t>::min(),
                                                      std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<uint8_t>({24, 1, 3, 3}, std::numeric_limits<uint8_t>::min(),
                                                             std::numeric_limits<uint8_t>::max());
      NodeArg* bias_arg = builder.MakeInitializer<float>({24}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* weight_dq_arg = builder.MakeIntermediate();
      NodeArg* conv_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.07f, static_cast<uint8_t>(0), input_dq_arg,
                                               use_contrib_qdq);
      auto& weight_dq_node = builder.AddDequantizeLinearNode<uint8_t>(weight_arg, std::vector<float>(24, 0.05f),
                                                                      std::vector<uint8_t>(24, static_cast<uint8_t>(0)),
                                                                      weight_dq_arg, nullptr, use_contrib_qdq);
      weight_dq_node.AddAttribute("axis", static_cast<int64_t>(0));
      auto& conv_node = builder.AddNode("Conv", {input_dq_arg, weight_dq_arg, bias_arg}, {conv_dq_arg});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
      conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("group", static_cast<int64_t>(24));
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      builder.AddQuantizeLinearNode<uint8_t>(conv_dq_arg, 0.14f, static_cast<uint8_t>(127), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(true);
#endif
}

TEST(QDQTransformerTests, WeightBiasQuantization_Conv_Weight_Bias) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg = builder.MakeInput<uint8_t>({1, 24, 67, 67}, std::numeric_limits<uint8_t>::min(),
                                                      std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<float>({24, 1, 5, 5}, -0.1f, 0.1f);
      NodeArg* bias_arg = builder.MakeInitializer<float>({24}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* conv_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.014f, static_cast<uint8_t>(127), input_dq_arg,
                                               use_contrib_qdq);
      auto& conv_node = builder.AddNode("Conv", {input_dq_arg, weight_arg, bias_arg}, {conv_dq_arg});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{5, 5});
      conv_node.AddAttribute("strides", std::vector<int64_t>{2, 2});
      conv_node.AddAttribute("group", static_cast<int64_t>(24));
      conv_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
      builder.AddQuantizeLinearNode<uint8_t>(conv_dq_arg, 0.014f, static_cast<uint8_t>(127), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(true);
#endif
}

// Tests that the WeightBiasQuantization optimizer does not process nodes that do not
// already have an output that is consumed by a single QuantizeLinear node.
TEST(QDQTransformerTests, WeightBiasQuantization_SkipIfOutputNotQuantized) {
  auto test_case = [](bool add_final_reshape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg = builder.MakeInput<uint8_t>({1, 24, 67, 67}, std::numeric_limits<uint8_t>::min(),
                                                      std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<float>({24, 1, 5, 5}, -0.1f, 0.1f);
      NodeArg* bias_arg = builder.MakeInitializer<float>({24}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* conv_output_arg = add_final_reshape ? builder.MakeIntermediate() : builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.014f, static_cast<uint8_t>(127), input_dq_arg);
      auto& conv_node = builder.AddNode("Conv", {input_dq_arg, weight_arg, bias_arg}, {conv_output_arg});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{5, 5});
      conv_node.AddAttribute("strides", std::vector<int64_t>{2, 2});
      conv_node.AddAttribute("group", static_cast<int64_t>(24));
      conv_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

      // Make adding a final Reshape node configurable to test two cases:
      //  - Conv produces a graph output
      //  - Conv output is consumed by some node that is NOT a QuantizeLinear
      // In either case, the WeightBiasQuantization optimizer should skip this node.
      if (add_final_reshape) {
        NodeArg* reshape_output_arg = builder.MakeOutput();
        NodeArg* new_shape_arg = builder.Make1DInitializer<int64_t>({1, -1});
        builder.AddNode("Reshape", {conv_output_arg, new_shape_arg}, {reshape_output_arg});
      }
    };

    auto check_graph = [add_final_reshape](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(false);

      // Should retain the same nodes in the original graph.
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
      EXPECT_EQ(op_to_count["Conv"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count["Reshape"], static_cast<int>(add_final_reshape));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      21,
                      /*per_sample_tolerance*/ 0.0,
                      /*relative_per_sample_tolerance*/ 0.0,
                      std::make_unique<WeightBiasQuantization>());
  };

  test_case(false);  // Conv produces a graph output directly
  test_case(true);   // Conv -> Reshape -> graph_output
}

TEST(QDQTransformerTests, WeightBiasQuantization_ConvTranspose_Weight) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg = builder.MakeInput<uint8_t>({1, 3, 4, 4}, std::numeric_limits<uint8_t>::min(),
                                                      std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<float>({3, 3, 3, 3}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* conv_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.014f, static_cast<uint8_t>(127), input_dq_arg,
                                               use_contrib_qdq);
      auto& conv_node = builder.AddNode("ConvTranspose", {input_dq_arg, weight_arg}, {conv_dq_arg});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
      conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
      conv_node.AddAttribute("group", static_cast<int64_t>(1));
      conv_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
      builder.AddQuantizeLinearNode<uint8_t>(conv_dq_arg, 0.014f, static_cast<uint8_t>(127), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      // No QLinearConvTranspose CPU kernel. Check the graph only.
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"] + op_to_count["com.microsoft.DequantizeLinear"], 2);
      EXPECT_EQ(op_to_count["ConvTranspose"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
#if !defined(DISABLE_CONTRIB_OPS)
  test_case(true);
#endif
}

#if !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, WeightBiasQuantization_Gemm_Bias) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg =
          builder.MakeInput<uint8_t>({1, 32}, std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<uint8_t>({16, 32}, std::numeric_limits<uint8_t>::min(),
                                                             std::numeric_limits<uint8_t>::max());
      NodeArg* bias_arg = builder.MakeInitializer<float>({16}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* weight_dq_arg = builder.MakeIntermediate();
      NodeArg* gemm_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.001f, static_cast<uint8_t>(0), input_dq_arg,
                                               use_contrib_qdq);
      builder.AddDequantizeLinearNode<uint8_t>(weight_arg, 0.26f, static_cast<uint8_t>(0), weight_dq_arg,
                                               use_contrib_qdq);
      auto& gemm_node = builder.AddNode("Gemm", {input_dq_arg, weight_dq_arg, bias_arg}, {gemm_dq_arg});
      gemm_node.AddAttribute("transB", static_cast<int64_t>(1));
      builder.AddQuantizeLinearNode<uint8_t>(gemm_dq_arg, 0.144f, static_cast<uint8_t>(69), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
  test_case(true);
}

TEST(QDQTransformerTests, WeightBiasQuantization_Gemm_Weight) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg =
          builder.MakeInput<uint8_t>({1, 32}, std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<float>({32, 16}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* gemm_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.001f, static_cast<uint8_t>(127), input_dq_arg,
                                               use_contrib_qdq);
      builder.AddNode("Gemm", {input_dq_arg, weight_arg}, {gemm_dq_arg});
      builder.AddQuantizeLinearNode<uint8_t>(gemm_dq_arg, 0.144f, static_cast<uint8_t>(127), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
  test_case(true);
}

TEST(QDQTransformerTests, WeightBiasQuantization_Gemm_Weight_Bias) {
  auto test_case = [](bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      NodeArg* input_arg =
          builder.MakeInput<uint8_t>({1, 32}, std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
      NodeArg* weight_arg = builder.MakeInitializer<float>({16, 32}, -0.1f, 0.1f);
      NodeArg* bias_arg = builder.MakeInitializer<float>({16}, -0.1f, 0.1f);
      NodeArg* input_dq_arg = builder.MakeIntermediate();
      NodeArg* gemm_dq_arg = builder.MakeIntermediate();
      NodeArg* output_arg = builder.MakeOutput();

      builder.AddDequantizeLinearNode<uint8_t>(input_arg, 0.001f, static_cast<uint8_t>(127), input_dq_arg,
                                               use_contrib_qdq);
      auto& gemm_node = builder.AddNode("Gemm", {input_dq_arg, weight_arg, bias_arg}, {gemm_dq_arg});
      gemm_node.AddAttribute("transB", static_cast<int64_t>(1));
      builder.AddQuantizeLinearNode<uint8_t>(gemm_dq_arg, 0.144f, static_cast<uint8_t>(127), output_arg,
                                             use_contrib_qdq);
    };

    auto check_graph = [use_contrib_qdq](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 18);

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2, 19);
  };

  test_case(false);
  test_case(true);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
