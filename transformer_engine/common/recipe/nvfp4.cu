/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cassert>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {
namespace nvfp4_recipe {

constexpr float factor = 6.0 * 6.0 * 448.0 * 448.0;

// single thread kernel that takes in two amax pointers, calculate the following
// alpha = amax_A * amax_B / factor
// Kernel to compute alpha = amax_A * amax_B / factor
__global__ void compute_nvfp4_per_tensor_scale_kernel(const float *amax_A, const float *amax_B,
                                                      float *alpha_ptr) {
  // factor is defined in the enclosing namespace
  *alpha_ptr = (*amax_A) * (*amax_B) / factor;
}

}  // namespace nvfp4_recipe
}  // namespace transformer_engine

void nvte_nvfp4_compute_per_tensor_scale(const NVTETensor inpA, const bool inpA_is_columnwise,
                                         const NVTETensor inpB, const bool inpB_is_columnwise,
                                         NVTETensor out, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_compute_per_tensor_scale);
  using namespace transformer_engine;

  auto *tA = convertNVTETensor(inpA);
  auto *tB = convertNVTETensor(inpB);
  auto *tOut = convertNVTETensor(out);

  void *amax_A_ptr = inpA_is_columnwise ? tA->columnwise_amax.dptr : tA->amax.dptr;
  void *amax_B_ptr = inpB_is_columnwise ? tB->columnwise_amax.dptr : tB->amax.dptr;
  void *alpha_ptr = tOut->scale.dptr;

  // check for not null pointers
  NVTE_CHECK(amax_A_ptr != nullptr, "amax_A_ptr is null");
  NVTE_CHECK(amax_B_ptr != nullptr, "amax_B_ptr is null");
  NVTE_CHECK(alpha_ptr != nullptr, "alpha_ptr is null");

  nvfp4_recipe::compute_nvfp4_per_tensor_scale_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const float *>(amax_A_ptr), reinterpret_cast<const float *>(amax_B_ptr),
      reinterpret_cast<float *>(alpha_ptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}
