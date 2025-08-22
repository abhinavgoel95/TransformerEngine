/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_TRANSPOSE_HADAMARD_TRANSFORM_H_
#define TRANSFORMER_ENGINE_COMMON_TRANSPOSE_HADAMARD_TRANSFORM_H_

#include "../common.h"

namespace transformer_engine::detail {

void hadamard_transform_amax_sm100_plus(
    const SimpleTensor& input, SimpleTensor& output_pre_rht_amax,
    // TODO(Frank): It doesn't matter if the type is SimpleTensor or float*.
    SimpleTensor& output_identity_amax, SimpleTensor& output_transpose_amax,
    // TODO(Frank): Are we having random_sign_mask and random_sign_mask_t or RHT 16x16 matrix?
    uint16_t random_sign_mask, uint16_t random_sign_mask_t, int32_t hadamard_dimension,
    bool return_pre_rht_amax, bool return_identity_amax, bool return_transposed_amax,
    cudaStream_t stream);

// Placeholder.
// Perform the hadamard transform on the input tensor.
// Returns:
// - output: The output tensor with 16x16 hadamard transform applied in identity layout.
//          Semantically: output = (input.reshape(-1, 16) @ H_16x16).reshape(input.shape)
// - output_t: The output tensor with 16x16 hadamard transform applied in identity layout.
//          Semantically: output_t = (input.transpose(0, 1).reshape(-1, 16) @ HT_16x16).transpose(0, 1).reshape(input.shape)
void hadamard_transform_sm90_plus(const SimpleTensor& input, SimpleTensor& output,
                                  SimpleTensor& output_t, uint16_t random_sign_mask,
                                  uint16_t random_sign_mask_t, int32_t hadamard_dimension,
                                  bool return_identity, bool return_transpose, cudaStream_t stream);

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_COMMON_TRANSPOSE_CAST_TRANSPOSE_H_
