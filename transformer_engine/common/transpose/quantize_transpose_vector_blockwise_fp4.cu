/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cuda/barrier>
#include <utility>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/utils.cuh"
#include "nvfp4_utils.cuh"

namespace quantize_transpose_nvfp4 {
namespace {

template <typename T>
constexpr T RoundUpDivide(const T x, const T y) {
  static_assert(std::is_integral<T>::value, "RoundUpDivide requires integral types.");
  static_assert(std::is_unsigned<T>::value, "RoundUpDivide requires unsigned types.");
  return (x + (y - 1)) / y;
}

constexpr int kThreadsPerWarp = 32;

#if CUDA_VERSION >= 12080
// for fp4, we use uint8_t to store 2 fp4 numbers
constexpr int kNFP4PerContainer = 2;

// Hyperparameters for performance tuning
constexpr int kTileDim = 128;
// constexpr int kScaleDim = 32;
constexpr int kNVecIn = 8;             // The number of elements each LDG touches
constexpr int kNVecOut = 16;           // The number of elements each STG touches
constexpr int kNVecSMem = 2;           // The number of elements each LDS/STS touches
constexpr int kThreadsPerBlock = 256;  // Thread block size, 8 warps in total

constexpr int kNVecContainer = kNVecOut / kNFP4PerContainer;

// Auto-calculated constants, do not modify directly)
static_assert(kNVecIn % kNVecSMem == 0, "kNVecIn must be divisible by kNVecSMem");
static_assert(kNVecOut % kNVecSMem == 0, "kNVecOut must be divisible by kNVecSMem");
constexpr int kSMemRow = kTileDim;
constexpr int kSMemCol = (kTileDim / kNVecSMem) + 1;
constexpr int kSMemSize = kSMemRow * kSMemCol * kNVecSMem;
constexpr int kNumThreadsLoad = kTileDim / kNVecIn;
constexpr int kNumThreadsStore = kTileDim / kNVecOut;
// constexpr int kNumThreadsReduce = kScaleDim / kNVecOut;
static_assert(kNumThreadsLoad <= kThreadsPerWarp, "kNumThreadsLoad must be <= kThreadsPerWarp");
static_assert(kNumThreadsStore <= kThreadsPerWarp, "kNumThreadsStore must be <= kThreadsPerWarp");

template <class ScaleType>
__device__ __forceinline__ size_t scale_factor_swizzled_offset(size_t row_idx, size_t col_idx,
                                                               uint32_t col_length) {
  // This function takes in indices from the scale factor matrix and returns an offset in the
  // swizzled format. row_idx, col_idx are original indices from the scale factor matrix (unswizzled
  // index). col_length is the column length of the scale factor matrix. tile_scales_inv is the
  // pointer to the scale factor matrix.

  // https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
  // For any scale factor matrix, it's 512B base block. Each base block consists of 128 rows and 4
  // columns. Base block is divided into 4 column blocks, each column block has 32 rows and 4
  // columns.

  // NOTE: There are not a lot of good illustrations about the swizzled scale factor matrix.
  // To think in high level, the swizzled scale factor matrix could be composed as:
  // unswizzled_scale_factor_matrix = torch.empty((M, N // 16), dtype=torch.uint8)
  // cbg_cnt = N // 16 // 4  # Assuming N is divisible by 64
  // rb_cnt = M // 128  # Assuming M is divisible by 128
  // tmp = unswizzled_scale_factor_matrix.reshape(rb_cnt, 4, 32, cbg_cnt, 4)
  // tmp = torch.permute(tmp, (0, 3, 2, 1, 4))
  // swizzled_scale_factor_matrix = tmp.reshape((-1, 128, 4))

  constexpr uint32_t kTotalRowsPerBaseBlock = 128;
  constexpr uint32_t kRowsPerBaseBlockCol = 32;
  constexpr uint32_t kColsPerBaseBlockCol = 4;

  const size_t rb = row_idx / kTotalRowsPerBaseBlock;
  const size_t rem = row_idx % kTotalRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;

  const size_t cbg_cnt = RoundUpDivide(col_length, kColsPerBaseBlockCol);
  // row-major offset in the logical shape
  // (rb_cnt , cbg_cnt , 32 , 4 , 4)
  // Magic number 16 below comes from the fact we have kColsPerBaseBlockCol = 4, and d4 ([0-128] /
  // 32 = [0-4])
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

template <bool kReturnIdentity, bool kReturnTranspose, bool kIsE8Scaling, bool kAligned,
          typename CType, typename IType, typename OType, typename ScaleType, bool kSwizzledScale>
__global__ void __launch_bounds__(kThreadsPerBlock) block_scaled_1d_cast_transpose_kernel(
    const IType* const input, const float* global_amax, OType* const output_c,
    OType* const output_t, ScaleType* const tile_scales_inv_c, ScaleType* const tile_scales_inv_t,
    const size_t row_length, const size_t num_rows, const size_t scale_stride_x,
    const size_t scale_stride_y, const size_t scale_t_stride_x, const size_t scale_t_stride_y,
    const size_t kScaleBlockDim, const float epsilon) {
  using SMemVec = Vec<IType, kNVecSMem>;
  using OVec = Vec<OType, kNVecContainer>;
  union IVec {
    Vec<IType, kNVecIn> input_type;
    Vec<SMemVec, kNVecIn / kNVecSMem> smem_type;
  };

  extern __shared__ char smem_base[];
  SMemVec* smem = reinterpret_cast<SMemVec*>(&smem_base[0]);

  // Step 1: Load input to shared memory
  {
    constexpr int r_stride = kThreadsPerBlock / kNumThreadsLoad;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsLoad) * (kNVecIn / kNVecSMem);         // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsLoad;                             // Row in shared memory
    const size_t c_g = (size_t)blockIdx.x * kTileDim + c_s * kNVecSMem;  // Column in global memory
    size_t r_g = (size_t)blockIdx.y * kTileDim + r_s;                    // Row in global memory
    const size_t stride_g = (size_t)r_stride * row_length;               // Stride in global memory
    const size_t num_ele =
        c_g < row_length ? min((size_t)kNVecIn, row_length - c_g) : 0;  // For not aligned case
    const IType* input_g = &input[r_g * row_length + c_g];  // Input address in global memory
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      IVec input_vec;
      // Step 1.1: Load from global memory (input) to registers
      if constexpr (kAligned) {
        input_vec.input_type.VecLoadFrom(input_g);
      } else {
        if (r_g < num_rows) {
          input_vec.input_type.EleLoadFromIfNeeded(input_g, 0, num_ele);
        } else {
          input_vec.input_type.clear();
        }
      }
      // Step 1.2: Write to shared memory
#pragma unroll
      for (int i = 0; i < kNVecIn / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem[r * kSMemCol + c] = input_vec.smem_type.data.ele[i];
      }
      // Step 1.3: Update input address, row index of shared memory, (and row index of global memory
      // for not aligned case)
      input_g += stride_g;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  __syncthreads();

  const int kNumThreadsReduce = kScaleBlockDim / kNVecOut;
  const float global_encode_scale =
      kIsE8Scaling ? 1.0f : ComputeGlobalEncodeScaleFP4(global_amax[0]);
  // Step 2: Cast and store to output_c
  if constexpr (kReturnIdentity) {
    constexpr int r_stride =
        kThreadsPerBlock / kNumThreadsStore;  // stride in rows of shared memory
    constexpr int num_iterations = kTileDim / r_stride;
    const int c_s =
        (threadIdx.x % kNumThreadsStore) * (kNVecOut / kNVecSMem);       // Column in shared memory
    int r_s = threadIdx.x / kNumThreadsStore;                            // Row in shared memory
    const size_t c_g = (size_t)blockIdx.x * kTileDim + c_s * kNVecSMem;  // Column in global memory
    size_t r_g = (size_t)blockIdx.y * kTileDim + r_s;                    // Row in global memory
    const size_t stride_g = (size_t)r_stride * row_length;               // Stride in global memory
    const size_t num_ele = c_g < row_length ? min((size_t)kNVecOut / kNFP4PerContainer,
                                                  (row_length - c_g) / kNFP4PerContainer)
                                            : 0;  // For not aligned case
    OType* output_g =
        &output_c[(r_g * row_length + c_g) / kNFP4PerContainer];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane =
        (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut / kNVecSMem];
      // Step 2.1: Load from shared memory to registers
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
        int c = c_s + i;
        int r = r_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
      // Step 2.2: Compute local amax
      CType amax = 0;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; ++j) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, fabsf(smem_vec[i].data.ele[j]));
        }
      }
      // Step 2.3: Reduce amax
      if constexpr (kIsE8Scaling) {
#pragma unroll
        for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
          const float other_amax = __shfl_down_sync(mask, amax, delta);
          __builtin_assume(amax >= 0);
          __builtin_assume(other_amax >= 0);
          amax = fmaxf(amax, other_amax);
        }
        amax = __shfl_sync(mask, amax, src_lane);
      }
      // Step 2.4: Compute scale
      ScaleType scale_inv =
          ComputeDecodeScaleFP4<OType, ScaleType, kIsE8Scaling>(amax, global_encode_scale);
      float encode_scale =
          ComputeEncodeScaleFP4<ScaleType, kIsE8Scaling>(scale_inv, global_encode_scale);
      // Step 2.5: Write scale_inv
      bool write_scale_inv = is_src_lane;
      if constexpr (!kAligned) {
        write_scale_inv &= (r_g < num_rows);
        write_scale_inv &= (c_g < row_length);
      }
      if (write_scale_inv) {
        size_t row_idx = (size_t)blockIdx.y * kTileDim + r_s;
        size_t col_idx = (size_t)blockIdx.x * (kNumThreadsStore / kNumThreadsReduce) +
                         ((size_t)threadIdx.x % kNumThreadsStore) / kNumThreadsReduce;
        if constexpr (kSwizzledScale) {
          size_t offset = scale_factor_swizzled_offset<ScaleType>(
              row_idx, col_idx, RoundUpDivide(row_length, kScaleBlockDim));
          tile_scales_inv_c[offset] = scale_inv;
        } else {
          tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
        }
      }
      // Step 2.6: Quantize
      OVec output_vec;
#pragma unroll
      for (int i = 0; i < kNVecOut / kNVecSMem; ++i) {
#pragma unroll
        for (int j = 0; j < kNVecSMem; j += kNFP4PerContainer) {
          // Pack two elements into __nv_bfloat162
          float2 f2;
          f2.x = ComputeOutputFP4<IType, ScaleType, kIsE8Scaling>(smem_vec[i].data.ele[j],
                                                                  encode_scale);
          f2.y = ComputeOutputFP4<IType, ScaleType, kIsE8Scaling>(smem_vec[i].data.ele[j + 1],
                                                                  encode_scale);
          // Convert to __nv_fp4x2_e2m1
          output_vec.data.ele[i] = __nv_cvt_float2_to_fp4x2(f2, __NV_E2M1, cudaRoundNearest);
        }
      }
      // Step 2.7: Store output_c
      if constexpr (kAligned) {
        output_vec.VecStoreTo(output_g);
      } else {
        if (r_g < num_rows) {
          output_vec.EleStoreToIfNeeded(output_g, 0, num_ele);
        }
      }
      // Step 2.8: Update output address, row index of shared memory (and row index of global memory
      // for not aligned case)
      output_g += stride_g / kNFP4PerContainer;
      r_s += r_stride;
      if constexpr (!kAligned) {
        r_g += r_stride;
      }
    }
  }

  // Step 3: Transpose, cast and store to output_t
  if constexpr (kReturnTranspose) {
    constexpr int c_stride =
        kThreadsPerBlock / kNumThreadsStore;  // Stride in columns of shared memory
    constexpr int num_iterations = kTileDim / (c_stride * kNVecSMem);
    const int r_s = (threadIdx.x % kNumThreadsStore) * kNVecOut;      // Row in shared memory
    int c_s = threadIdx.x / kNumThreadsStore;                         // Column in shared memory
    size_t r_g = (size_t)blockIdx.x * kTileDim + c_s * kNVecSMem;     // Row in global memory
    const size_t c_g = (size_t)blockIdx.y * kTileDim + r_s;           // Column in global memory
    const size_t stride_g = (size_t)c_stride * kNVecSMem * num_rows;  // Stride in global memory
    const size_t num_ele = c_g < num_rows ? min((size_t)kNVecOut / kNFP4PerContainer,
                                                (num_rows - c_g) / kNFP4PerContainer)
                                          : 0;  // For not aligned case
    OType* output_g =
        &output_t[(r_g * num_rows + c_g) / kNFP4PerContainer];  // Output address in global memory
    // Each kNumThreadsStore threads form a warp process one row, we need to find the lane id of
    // the first thread to do the reduction.
    const unsigned src_lane =
        (threadIdx.x % kThreadsPerWarp) / kNumThreadsReduce * kNumThreadsReduce;
    // This mask represents which threads should do the reduction together.
    const unsigned mask = ((1 << kNumThreadsReduce) - 1) << src_lane;
    const bool is_src_lane = (threadIdx.x % kNumThreadsReduce) == 0;
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      SMemVec smem_vec[kNVecOut];
      // Step 3.1: Load from shared memory to registers
#pragma unroll
      for (int i = 0; i < kNVecOut; ++i) {
        int r = r_s + i;
        int c = c_s;
        smem_vec[i] = smem[r * kSMemCol + c];
      }
#pragma unroll
      for (int smem_idx = 0; smem_idx < kNVecSMem; ++smem_idx) {
        // Step 3.2: Compute local amax
        CType amax = 0;
#pragma unroll
        for (int i = 0; i < kNVecOut; ++i) {
          amax = fmaxf(amax, fabsf(smem_vec[i].data.ele[smem_idx]));
        }
        // Step 3.3: Reduce amax
        if constexpr (kIsE8Scaling) {
#pragma unroll
          for (int delta = kNumThreadsReduce / 2; delta > 0; delta /= 2) {
            const float other_amax = __shfl_down_sync(mask, amax, delta);
            __builtin_assume(amax >= 0);
            __builtin_assume(other_amax >= 0);
            amax = fmaxf(amax, other_amax);
          }
          amax = __shfl_sync(mask, amax, src_lane);
        }
        // Step 3.4: Compute scale
        ScaleType scale_inv =
            ComputeDecodeScaleFP4<OType, ScaleType, kIsE8Scaling>(amax, global_encode_scale);
        float encode_scale =
            ComputeEncodeScaleFP4<ScaleType, kIsE8Scaling>(scale_inv, global_encode_scale);
        // Step 3.5: Write scale_inv_t
        bool write_scale_inv = is_src_lane;
        if constexpr (!kAligned) {
          write_scale_inv &= (r_g + smem_idx < row_length);
          write_scale_inv &= (c_g < num_rows);
        }
        if (write_scale_inv) {
          size_t row_idx = (size_t)blockIdx.x * kTileDim + c_s * kNVecSMem + smem_idx;
          size_t col_idx = (size_t)blockIdx.y * (kNumThreadsStore / kNumThreadsReduce) +
                           ((size_t)threadIdx.x % kNumThreadsStore) / kNumThreadsReduce;
          if constexpr (kSwizzledScale) {
            size_t offset = scale_factor_swizzled_offset<ScaleType>(
                row_idx, col_idx, RoundUpDivide(num_rows, kScaleBlockDim));
            tile_scales_inv_t[offset] = scale_inv;
          } else {
            tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
          }
        }
        // Step 3.6: Quantize
        OVec output_vec;
#pragma unroll
        for (int i = 0; i < kNVecOut / kNFP4PerContainer; i += 1) {
          // Pack two elements into __nv_bfloat162
          float2 f2;
          f2.x = ComputeOutputFP4<IType, ScaleType, kIsE8Scaling>(
              smem_vec[2 * i].data.ele[smem_idx], encode_scale);
          f2.y = ComputeOutputFP4<IType, ScaleType, kIsE8Scaling>(
              smem_vec[2 * i + 1].data.ele[smem_idx], encode_scale);
          // Convert to __nv_fp4x2_e2m1
          output_vec.data.ele[i] = __nv_cvt_float2_to_fp4x2(f2, __NV_E2M1, cudaRoundNearest);
        }
        // Step 3.7: Store output_t
        if constexpr (kAligned) {
          output_vec.VecStoreTo(output_g + smem_idx * num_rows / kNFP4PerContainer);
        } else {
          if (r_g + smem_idx < row_length) {
            output_vec.EleStoreToIfNeeded(output_g + smem_idx * num_rows / kNFP4PerContainer, 0,
                                          num_ele);
          }
        }
      }
      // Step 3.8: Update output address, column index of shared memory (and row index of global
      // memory for not aligned case)
      output_g += stride_g / kNFP4PerContainer;
      c_s += c_stride;
      if constexpr (!kAligned) {
        r_g += c_stride * kNVecSMem;
      }
    }
  }
}

#endif  // if CUDA_VERSION >= 12080

}  // namespace
}  // namespace quantize_transpose_nvfp4

namespace transformer_engine::detail {

void quantize_transpose_vector_blockwise_fp4(const SimpleTensor& input,
                                             const SimpleTensor& global_amax,
                                             SimpleTensor& scale_inv, SimpleTensor& scale_inv_t,
                                             SimpleTensor& output, SimpleTensor& output_t,
                                             const float epsilon, const bool return_identity,
                                             const bool return_transpose, const bool pow2_scale,
                                             const bool swizzled_scale, cudaStream_t stream) {
  // #if CUDA_VERSION >= 12080

  NVTE_API_CALL(quantize_transpose_vector_blockwise_fp4);

  // pow 2 scale is for MXFP4 since it's using E8M0 scaling
  // raise error if pow2_scale is true
  NVTE_CHECK(!pow2_scale, "No support for pow2_scale for MXFP4 for now");

  if (!return_identity && !return_transpose) {
    return;
  }

  const size_t row_length = input.shape[1];
  const size_t num_rows = input.shape[0];

  size_t scale_stride_x = 0;
  size_t scale_stride_y = 0;

  if (return_identity) {
    scale_stride_x = 1;
    scale_stride_y = scale_inv.shape[1];
  }

  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  if (return_transpose) {
    scale_t_stride_x = 1;
    scale_t_stride_y = scale_inv_t.shape[1];
  }

  using namespace quantize_transpose_nvfp4;

  const size_t num_blocks_x = RoundUpDivide(row_length, (size_t)kTileDim);
  const size_t num_blocks_y = RoundUpDivide(num_rows, (size_t)kTileDim);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP4x2_ONLY(
          output.dtype, 2, OutputType,

          dim3 grid(num_blocks_x, num_blocks_y, 1);

          using ScaleType = fp8e4m3; constexpr int kScaleBlockDim = 16;
          constexpr bool kPow2Scale = false;

          const bool full_tile = row_length % kTileDim == 0 && num_rows % kTileDim == 0;

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_identity, kReturnIdentity,

              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  return_transpose, kReturnTranspose,

                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      full_tile, kAligned,

                      TRANSFORMER_ENGINE_SWITCH_CONDITION(
                          swizzled_scale, kSwizzledScale,

                          size_t smem_bytes = kSMemSize * sizeof(InputType);
                          if (smem_bytes >= 48 * 1024) {
                            cudaError_t err = cudaFuncSetAttribute(
                                &block_scaled_1d_cast_transpose_kernel<
                                    kReturnIdentity, kReturnTranspose, kPow2Scale, kAligned, float,
                                    InputType, OutputType, ScaleType, kSwizzledScale>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                            NVTE_CHECK(err == cudaSuccess,
                                       "Failed to set dynamic shared memory size.");
                          } block_scaled_1d_cast_transpose_kernel<kReturnIdentity, kReturnTranspose,
                                                                  kPow2Scale, kAligned, float,
                                                                  InputType, OutputType, ScaleType,
                                                                  kSwizzledScale>
                          <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                              reinterpret_cast<const InputType*>(input.dptr),
                              reinterpret_cast<const float*>(global_amax.dptr),
                              reinterpret_cast<OutputType*>(output.dptr),
                              reinterpret_cast<OutputType*>(output_t.dptr),
                              reinterpret_cast<ScaleType*>(scale_inv.dptr),
                              reinterpret_cast<ScaleType*>(scale_inv_t.dptr), row_length, num_rows,
                              scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y,
                              kScaleBlockDim,
                              epsilon);)  // kSwizzledScale
                      )                   // kAligned
                  )                       // kReturnTranspose
              )                           // kReturnIdentity
          )                               // OutputType
      )                                   // InputType

  NVTE_CHECK_CUDA(cudaGetLastError());

  // #else
  //   NVTE_CHECK(false, "Quantize vector blockwise fp4 is not supported for CUDA version < 12.8");
  // #endif // if CUDA_VERSION >= 12080
}

}  // namespace transformer_engine::detail
