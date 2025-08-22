/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file hadamard_transform_kernels.cuh
 */

#ifndef TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_KERNELS_CUH_
#define TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../common.h"
#include "../transpose/hadamard_transform.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

namespace detail {

namespace {
constexpr uint16_t kHadamardDimension = 16;

}
void hadamard_transform_helper(const NVTETensor input, NVTETensor output,
                               const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  const Tensor *input_tensor = convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);

  // NOTE: This is non-intuitive, we are writing the result of transposed RHT to the
  // output of rowwise.
  auto &output_transpose = output_tensor->data;

  bool return_transpose = output_transpose.dptr != nullptr;

  // TODO(Frank): Add random sign mask.
  uint16_t random_sign_mask_t = 0;

  auto empty = SimpleTensor();
  hadamard_transform_sm90_plus(
      /*input=*/input_tensor->data,
      /*output_identity=*/empty,
      /*output_transpose=*/output_transpose,
      /*random_sign_mask=*/0,
      /*random_sign_mask_t=*/random_sign_mask_t,
      /*hadamard_dimension=*/kHadamardDimension,
      /*return_identity=*/false,
      /*return_transpose=*/return_transpose,
      /*stream=*/stream);
}

void hadamard_transform_amax_helper(const NVTETensor input, NVTETensor output,
                                    const NVTEQuantizationConfig quant_config,
                                    cudaStream_t stream) {
  const Tensor *input_tensor = convertNVTETensorCheck(input);
  Tensor *output_tensor = convertNVTETensorCheck(output);

  // TODO(Frank): We have a either/or relationship between identity_amax and pre_rht_amax.
  auto &pre_rht_amax = output_tensor->amax;
  auto &transpose_amax = output_tensor->columnwise_amax;

  bool return_pre_rht_amax = pre_rht_amax.dptr != nullptr;
  bool return_transposed_amax = transpose_amax.dptr != nullptr;

  // TODO(Frank): Add random sign mask.
  uint16_t random_sign_mask_t = 0;

  auto empty = SimpleTensor();
  hadamard_transform_amax_sm100_plus(
      /*input=*/input_tensor->data,
      /*output_pre_rht_amax=*/pre_rht_amax,
      /*output_identity_amax=*/empty,
      /*output_transpose_amax=*/transpose_amax,
      /*random_sign_mask=*/0,  // Unused
      /*random_sign_mask_t=*/random_sign_mask_t,
      /*hadamard_dimension=*/kHadamardDimension,
      /*return_pre_rht_amax=*/return_pre_rht_amax,
      /*return_identity_amax=*/false,  // Unused
      /*return_transposed_amax=*/return_transposed_amax,
      /*stream=*/stream);
}

}  // namespace detail
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_KERNELS_CUH_
