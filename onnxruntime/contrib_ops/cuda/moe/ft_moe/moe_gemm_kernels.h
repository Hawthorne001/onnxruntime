/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "contrib_ops/cuda/moe/cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ort_fastertransformer {

struct MoEGemmConfigMap {
  using MoEGemmConfigMapT = std::unordered_map<int64_t, CutlassGemmConfig>;

  MoEGemmConfigMapT map;
  std::mutex mutex;

  void Insert(int64_t key, CutlassGemmConfig config) {
    std::lock_guard<std::mutex> lock(mutex);
    map[key] = config;
  }

  bool Contains(int64_t key) {
    std::lock_guard<std::mutex> lock(mutex);
    return map.find(key) != map.end();
  }

  CutlassGemmConfig Get(int64_t key) {
    std::lock_guard<std::mutex> lock(mutex);
    return map[key];
  }
};

enum class ActivationType { Gelu,
                            Relu,
                            Silu,
                            GeGLU,
                            ReGLU,
                            SiGLU,
                            SwiGLU,
                            Identity,
                            InvalidType };

template <typename T, /*The type used for activations/scales/compute*/
          typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner {
 public:
  MoeGemmRunner();

  void initialize(int sm);

  void moe_gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                         int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                         int num_experts, ActivationType activation_type, cudaStream_t stream);

  void moe_gemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                int num_experts, cudaStream_t stream);

  static MoEGemmConfigMap& GetGemmConfigMap() {
    static MoEGemmConfigMap gFactory;
    return gFactory;
  }

 private:
  template <typename EpilogueTag>
  void dispatch_to_arch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                        int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                        int num_experts, CutlassGemmConfig gemm_config, cudaStream_t stream,
                        int* occupancy = nullptr);

  template <typename EpilogueTag>
  void profile_gemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                    int num_experts, cudaStream_t stream, int64_t key);

  template <typename EpilogueTag>
  void run_gemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                int num_experts, cudaStream_t stream);

 private:
  int sm_;
  int multi_processor_count_;
};

}  // namespace ort_fastertransformer
