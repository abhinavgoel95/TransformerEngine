/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ActivationType {
    Identity,
    GeLU,
    SiLU,
    ReLU,
    QGeLU,
    SReLU
};

double2 cvt_fp4x2_to_double2(fp4e2m1x2 fp4_pair) {
    const __half2_raw raw_truncated_to_fp4e2m1_pair =
        __nv_cvt_fp4x2_to_halfraw2(*reinterpret_cast<__nv_fp4x2_storage_t*>(&fp4_pair), __NV_E2M1);

    const __half2 truncated_to_fp4e2m1_pair(raw_truncated_to_fp4e2m1_pair);
    const double truncated_to_fp4e2m1_x = static_cast<double>(truncated_to_fp4e2m1_pair.x);
    const double truncated_to_fp4e2m1_y = static_cast<double>(truncated_to_fp4e2m1_pair.y);
    return {truncated_to_fp4e2m1_x, truncated_to_fp4e2m1_y};
}

template <typename InputType>
std::vector<InputType> create_transpose(const InputType* const input, const size_t rows, size_t cols) {
    std::vector<InputType> input_t(cols * rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            const size_t idx_t = j * rows + i;
            input_t[idx_t] = input[idx];
        }
    }
    return input_t;
}

// Compute the global encode scale factor for a given global amax
float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  constexpr float fp8_max = 448.0f;     // 448.0f;
  constexpr float fp4_max = 6.0f;       // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, Numeric_Traits<float>::maxNorm);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

template <typename InputType>
void quantize_nvfp4(float (*OP)(const float),
                    const InputType* const input,
                    fp4e2m1x2* const output,
                    fp8e4m3* const scales,
                    const size_t rows,
                    const size_t cols,
                    const size_t scales_stride,
                    const float global_amax) {

    // Compute a global encoding/decoding scaling factor for all S_dec_b
    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax);

    constexpr size_t block_size_X = 16;
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    std::array<float, block_size_X> cache_buffer;
    for (size_t i = 0; i < block_size_X; ++i) {
        cache_buffer[i] = 0.0f;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = j_min + block_size_X;

            // Find block amax
            float block_amax = 0.0f;
            for (size_t j = j_min; j < j_max; ++j) {
                const size_t idx = i * cols + j;
                const size_t cache_idx = j - j_min;

                const float input_elt = static_cast<float>(input[idx]);
                const float act_elt = OP(input_elt);

                // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                cache_buffer[cache_idx] = elt;
                block_amax = std::max(block_amax, std::abs(elt));
            }

            // 2. Compute E4M3 scaling factor
            // Compute per-block encoding/decoding scaling factor
            const float S_dec_b = block_amax / 6.0f;

            // Scale & Store per-block decoding scaling factor
            const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(S_dec_b * S_enc);

            // Compute "correct" per-block encoding scaling factor
            const float S_enc_b_fp8 = S_enc / static_cast<float>(S_dec_b_fp8);

            const size_t scale_idx = i * scales_stride + block_X;
            scales[scale_idx] = S_dec_b_fp8;
            const float scale_reciprocal = S_enc_b_fp8;

            for (size_t j = j_min; j < j_max; j += 2) {
                const int idx_pair = (i * cols + j) / 2;
                const int cache_idx_x = j - j_min;
                const int cache_idx_y = cache_idx_x + 1;
                const float cached_x = cache_buffer[cache_idx_x];
                const float cached_y = cache_buffer[cache_idx_y];
                const float scaled_elt_x = cached_x * scale_reciprocal;
                const float scaled_elt_y = cached_y * scale_reciprocal;
                const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                output[idx_pair] = casted_to_e2m1_pair;

                // const double2 truncated_pair = cvt_fp4x2_to_double2(casted_to_e2m1_pair);
            }
        }
    }
}

template <typename InputType>
void compute_ref(float (*OP)(const float),
                 const InputType* input,
                 fp4e2m1x2* output,
                 fp4e2m1x2* output_t,
                 fp8e4m3* scales,
                 fp8e4m3* scales_t,
                 const float global_amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const size_t scales_stride_t)
{
    std::vector<InputType> input_t = create_transpose(input, rows, cols);
    quantize_nvfp4(OP, input, output, scales, rows, cols, scales_stride, global_amax);
    quantize_nvfp4(OP, input_t.data(), output_t, scales_t, cols, rows, scales_stride_t, global_amax);
}

void compare_nvfp4_tensors(const std::string& name,
                           const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                           const int rows, const int cols,
                           double atol = 1e-5, double rtol = 1e-8) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 2) {
            const int idx = i * cols + j;
            double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
            double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));

            for (int k = 0; k < 2; ++k) {
                const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);

                bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
                /* For Float32 the floating point comparison is enough to error out */
                bool assertion = false;
                if (mismatch && !assertion) {
                    /* Check if it is just a failure of round to nearest choosing different
                        side of the real value */
                    const double mean = (t + r) / 2;
                    const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
                    const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
                    const double cast_mean_p = static_cast<double>(static_cast<fp4e2m1>(mean_p));
                    const double cast_mean_m = static_cast<double>(static_cast<fp4e2m1>(mean_m));
                    assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
                }
                if (assertion) {
                    ASSERT_FALSE(assertion) << "Error in tensor " << name << " in " << std::endl
                                            << "Mismatch at place "
                                            << " (" << std::to_string(idx + k) << "): "
                                            << t << " vs " << r;
                }
            }
        }
    }
}

void compareResults_nvfp4(const Tensor &test,
                          const void *ref, const void *ref_t, const int rows, const int cols,
                          double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true) {
    if (if_on_gpus) test.to_cpu();

    const fp4e2m1 *test_data = test.rowwise_cpu_dptr<fp4e2m1>();
    const fp4e2m1 *test_data_t = test.columnwise_cpu_dptr<fp4e2m1>();
    const fp4e2m1 *ref_data = reinterpret_cast<const fp4e2m1*>(ref);
    const fp4e2m1 *ref_data_t = reinterpret_cast<const fp4e2m1*>(ref_t);

    compare_nvfp4_tensors("output", test_data, ref_data, rows, cols);
    compare_nvfp4_tensors("output_t", test_data_t, ref_data_t, cols, rows);
}

template <typename InputType>
void performTest(float (*OP)(const float),
                 const std::vector<size_t>& shape) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);
    const std::array<size_t,4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t blocks_Y_t = scale_dims_t[2];
    const size_t blocks_X_t = scale_dims_t[3];
    const size_t scales_stride_t = blocks_X_t;

    Tensor input("input", shape, itype);
    Tensor output("output", shape, otype, true, true, NVTE_NVFP4_1D_SCALING);

    std::unique_ptr<fp4e2m1x2[]> ref_output   = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
    std::unique_ptr<fp4e2m1x2[]> ref_output_t = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
    std::unique_ptr<fp8e4m3[]> ref_scales     = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);
    std::unique_ptr<fp8e4m3[]> ref_scales_t   = std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);

    fillCase<fp32>(&input, InputsFillCase::uniform);

    // Find global amax
    float amax = 0.0f;
    const InputType* input_dptr = input.rowwise_cpu_dptr<InputType>();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            amax = fmaxf(amax, static_cast<float>(input_dptr[idx]));
        }
    }
    // Set 2nd stage NVFP4 scaling factor
    output.set_scale(amax);

    compute_ref<InputType>(OP,
                           input.rowwise_cpu_dptr<InputType>(),
                           ref_output.get(),
                           ref_output_t.get(),
                           ref_scales.get(),
                           ref_scales_t.get(),
                           output.scale(),
                           rows,
                           cols,
                           scales_stride,
                           scales_stride_t);

    auto nvte_quantize_operation = &nvte_quantize_v2;

    QuantizationConfigWrapper quant_config;
    const size_t RNG_seed = 123;
    const size_t RNG_sequence = 321;
    quant_config.set_rng_seed(RNG_seed);
    quant_config.set_rng_sequence(RNG_sequence);

    // if (OP == &gelu)       { nvte_quantize_operation = &nvte_gelu; }
    // else if (OP == &silu)  { nvte_quantize_operation = &nvte_silu; }
    // else if (OP == &relu)  { nvte_quantize_operation = &nvte_relu; }
    // else if (OP == &qgelu) { nvte_quantize_operation = &nvte_qgelu; }
    // else if (OP == &srelu) { nvte_quantize_operation = &nvte_srelu; }

    nvte_quantize_operation(input.data(), output.data(), quant_config, 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    const double atol = 0.05;
    const double rtol = 0.1;
    compareResults_nvfp4(output, ref_output.get(), ref_output_t.get(), rows, cols, atol, rtol);

    size_t scale_mismatches_num = 0;
    compare_scaling_factors<fp8e4m3>("scales", output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                      ref_scales.get(),
                                      unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                                      scale_mismatches_num);

    compare_scaling_factors<fp8e4m3>("scales_t", output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                      ref_scales_t.get(),
                                      unpadded_blocks_Y_t, unpadded_blocks_X_t, scales_stride_t,
                                      scale_mismatches_num);
}

std::vector<std::vector<size_t>> tensor_dims = {
    // {32, 32},
    // {32, 64},
    // {64, 32},
    // {64, 96},
    // {128, 128},
    // {256, 256},
    // {512, 512},
    // {1024, 1024},
    // {2048, 2048},
    // {128, 256},
    // {8192, 128},
    // {2048, 160},
    // {8, 32, 1024},
    // {16, 8, 4, 512},
    {1024, 16384},
    {4096, 13312},
};

// Only GeLU activation tests are supported
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity,
    // ActivationType::GeLU,
    // ActivationType::SiLU,
    // ActivationType::ReLU,
    // ActivationType::QGeLU,
    // ActivationType::SReLU,
};

}  // namespace

class FusedCastTransposeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<ActivationType,
                std::vector<size_t>,
                transformer_engine::DType>> {};

TEST_P(FusedCastTransposeNVFP4TestSuite, TestFusedCastTransposeNVFP4) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const ActivationType Act_type = std::get<0>(GetParam());
    const auto tensor_dims = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());

    // Skip tests if the input tensor is 1D
    if (tensor_dims.size() < 2) {
        GTEST_SKIP();
    }

    // Forward activations
    auto OP = &identity;
    switch (Act_type) {
        case ActivationType::GeLU: OP = &gelu; break;
        case ActivationType::SiLU: OP = &silu; break;
        case ActivationType::ReLU: OP = &relu; break;
        case ActivationType::QGeLU: OP = &qgelu; break;
        case ActivationType::SReLU: OP = &srelu; break;
    }

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        performTest<InputType>(OP, tensor_dims);
    );
}

std::string to_string(const ActivationType Act_type) {
    switch (Act_type) {
        case ActivationType::Identity:  return "CAST_ONLY";
        case ActivationType::GeLU:      return "GeLU";
        case ActivationType::SiLU:      return "SiLU";
        case ActivationType::ReLU:      return "ReLU";
        case ActivationType::QGeLU:     return "QGeLU";
        case ActivationType::SReLU:     return "SReLU";
        default: return "";
    }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),
        ::testing::ValuesIn(tensor_dims),
        ::testing::Values(DType::kBFloat16)),
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        std::string name = to_string(std::get<0>(info.param));
      const auto& shape = std::get<1>(info.param);
      for ( const auto& s: shape) {
        name += "X" + std::to_string(s);
      }
      name += "X" + test::typeName(std::get<2>(info.param));
        return name;
    });
