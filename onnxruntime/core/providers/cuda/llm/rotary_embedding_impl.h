// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status LaunchRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const int64_t* position_ids,
    const T* cos_cache,
    const T* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int rotary_embedding_dim,
    const int max_sequence_length,
    const int position_ids_format,
    const bool interleaved,
    const int max_threads_per_block,
    const bool is_input_bnsh_format);

template <typename T>
Status LaunchRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const int64_t* position_ids,
    const T* cos_cache,
    const T* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int rotary_embedding_dim,
    const int max_sequence_length,
    const int position_ids_format,
    const bool interleaved,
    const int max_threads_per_block,
    int4 in_strides,
    int4 out_strides);

}  // namespace cuda
}  // namespace onnxruntime
