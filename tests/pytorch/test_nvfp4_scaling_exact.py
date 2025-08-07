# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple
import math
import os
import pathlib
import pytest
import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    NVFP4Quantizer,
    NVFP4Tensor,
)
from transformer_engine.pytorch.fp8 import fp8_autocast, get_fp4_te_dtype
from test_float8_current_scaling_exact import (
    TestFP8RecipeLinearBase,
)


TENSOR_DUMP_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "tensor_dumps"
tensor_dump_dir_env = os.getenv("NVTE_TEST_NVFP4_SCALING_EXACT_TENSOR_DUMP_DIR")
if tensor_dump_dir_env is not None:
    TENSOR_DUMP_DIR = pathlib.Path(tensor_dump_dir_env)
recipe_available, reason_for_no_recipe = FP8GlobalStateManager.is_nvfp4_available()


class GetRecipes:

    @staticmethod
    def none():
        return None

    @staticmethod
    def nvfp4():
        # return default configs
        return NVFP4BlockScaling()


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
) -> None:
    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # baseline quantizer
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    if use_cpp_allocator:
        x_nvfp4_sut = nvfp4_quantizer(x)
    else:
        x_nvfp4_sut = nvfp4_quantizer.make_empty(
            (M, N), dtype=x_dtype, device=device, requires_grad=False
        )
        x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    # x_nvfp4_sut = nvfp4_quantizer(x)

    ref_amax = torch.max(torch.abs(x.float())).reshape(1)

    assert x_nvfp4_sut._rowwise_data is not None
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_t_amax = x_nvfp4_sut._amax_columnwise

    # print them all
    print(f"qx: {qx}")
    print(f"sx: {sx}")
    print(f"qx_t: {qx_t}")
    print(f"sx_t: {sx_t}")
    print(f"qx_amax: {qx_amax}")
    print(f"qx_t_amax: {qx_t_amax}")

    # load tensor from TE root dir / tensor dumps / quantizer_test
    # load_dir = TENSOR_DUMP_DIR / "quantizer_test"
    # qx_ref = torch.load(load_dir / "qx.pt")
    # sx_ref = torch.load(load_dir / "sx.pt")
    # qx_t_ref = torch.load(load_dir / "qx_t.pt")
    # sx_t_ref = torch.load(load_dir / "sx_t.pt")

    # print them all
    # print(f"qx_ref: {qx_ref}")
    # print(f"sx_ref: {sx_ref}")
    # print(f"qx_t_ref: {qx_t_ref}")
    # print(f"sx_t_ref: {sx_t_ref}")

    # torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)
    # torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)
    # torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
    # torch.testing.assert_close(sx_t, sx_t_ref, atol=0.0, rtol=0.0)

    # Check
    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)
    # torch.testing.assert_close(qx.float(), qx_ref.float(), atol=0.0, rtol=0.0)
    # # Zero out values that are don't care values
    # # Scale format has padding.
    # scale_mask = torch.ones(
    #     (math.ceil(M / tile_size[0]), math.ceil(N / tile_size[1])), device=sx.device
    # )
    # scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
    #     QuantizeResult(qx, scale_mask, None, None), tile_size
    # ).scale
    # sx = sx * scale_mask
    # torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)

    # if return_transpose:
    #     assert qx_t is not None
    #     qx_t = qx_t.view(dtype=quant_dtype)
    #     assert qx_t_ref is not None
    #     assert sx_t is not None
    #     assert sx_t_ref is not None
    #     scale_mask = torch.ones(
    #         (math.ceil(N / tile_size[0]), math.ceil(M / tile_size[1])),
    #         device=sx_t.device,
    #     )
    #     scale_mask = ref_quantizer.scale_munger.munge_scale_shapes_for_backend(
    #         QuantizeResult(qx_t, scale_mask, None, None), tile_size
    #     ).scale
    #     sx_t = sx_t * scale_mask
    #     torch.testing.assert_close(qx_t.float(), qx_t_ref.float(), atol=0.0, rtol=0.0)
    #     torch.testing.assert_close(sx_t, sx_t_ref, atol=0.0, rtol=0.0)
    # else:
    #     # should be None
    #     assert qx_t is None and qx_t_ref is None
    #     assert sx_t is None and sx_t_ref is None


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
# @pytest.mark.parametrize(
#     "M, N",
#     [
#         # full tile cases
#         (128, 128),
#         (256, 256),
#         (256, 1024),
#         (1024, 256),
#         # Padding required cases
#         (256, 272),
#         (303, 300),
#         (305, 256),
#         # Some larger tiles.
#         (2000, 2000),
#         (2048, 2000),
#         (2000, 1024),
#         (2048, 1024),
#     ],
# )
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        # (128, 128),
        (1024, 4096),
    ],
)
# @pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
# @pytest.mark.parametrize(
#     "return_transpose", [True, False], ids=["quantize_transpose", "quantize_only"]
# )
@pytest.mark.parametrize("return_transpose", [True], ids=["quantize_transpose"])
# @pytest.mark.parametrize(
#     "swizzled_scale", [True, False], ids=["swizzled_scale", "linear_scale"]
# )
@pytest.mark.parametrize("swizzled_scale", [False], ids=["linear_scale"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype, M, N, return_transpose, swizzled_scale, use_cpp_allocator
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
class TestNVFP4BlockScalingRecipeLinear(TestFP8RecipeLinearBase):

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def compare_recipe(
        self,
        recipe1,
        recipe2,
        batch_size,
        hidden_size,
        out_size,
        use_bias,
        seed,
        dtype,
        y_error=0.0,
        dgrad_error=0.0,
        wgrad_error=0.0,
        bgrad_error=0.0,
        recipe1_golden_tensors=None,
        recipe2_golden_tensors=None,
    ):

        x, w, bias, gradient = self._prepare_data(
            batch_size, hidden_size, out_size, use_bias, seed=seed, dtype=dtype
        )

        # print shape of x and w
        print(f"x shape: {x.shape}")
        print(f"w shape: {w.shape}")
        print(f"gradient shape: {gradient.shape}")

        use_nvfp4_recipe1 = recipe1() != GetRecipes.none()
        if use_nvfp4_recipe1:
            with fp8_autocast(enabled=True, fp8_recipe=recipe1()):
                y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(x, w, bias, gradient)
        else:
            y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(x, w, bias, gradient)

        # recipe2
        use_nvfp4_recipe2 = recipe2() != GetRecipes.none()
        if use_nvfp4_recipe2:
            with fp8_autocast(enabled=True, fp8_recipe=recipe2()):
                y_q, dgrad, wgrad, bgrad = self.run_linear(x, w, bias, gradient)
        else:
            y_q, dgrad, wgrad, bgrad = self.run_linear(x, w, bias, gradient)

        # print out y_q, dgrad, wgrad, bgrad and their abs sum
        print(f"y_q: {y_q}\ny_q abs sum: {y_q.abs().sum()}\n")
        print(f"dgrad: {dgrad}\ndgrad abs sum: {dgrad.abs().sum()}\n")
        print(f"wgrad: {wgrad}\nwgrad abs sum: {wgrad.abs().sum()}\n")
        if use_bias:
            print(f"bgrad: {bgrad}\nbgrad abs sum: {bgrad.abs().sum()}\n")

        # also print the gradient and its abs sum
        print(f"gradient: {gradient}\ngradient abs sum: {gradient.abs().sum()}\n")

        # # Compare results (mean abs relative error)
        # assert (
        #     self._get_mean_abs_relative_error(y_q, y_q_ref).item() < y_error
        # ), "y and y_ref has too large mean abs relative error"
        # assert (
        #     self._get_mean_abs_relative_error(dgrad, dgrad_ref) < dgrad_error
        # ), "dgrad and dgrad_ref has too large mean abs relative error"
        # assert (
        #     self._get_mean_abs_relative_error(wgrad, wgrad_ref).item() < wgrad_error
        # ), "wgrad and wgrad_ref has too large mean abs relative error"
        # if use_bias:
        #     assert (
        #         self._get_mean_abs_relative_error(bgrad, bgrad_ref).item() < bgrad_error
        #     ), "bgrad and bgrad_ref has too large mean abs relative error"

        # # enforce zero tolerance check when we can find golden tensor value dump
        # if recipe2_golden_tensors is not None:
        #     torch.testing.assert_close(
        #         y_q.float(), recipe2_golden_tensors["y"].float(), atol=0, rtol=0.0
        #     )
        #     torch.testing.assert_close(dgrad, recipe2_golden_tensors["dgrad"], atol=0.0, rtol=0.0)
        #     torch.testing.assert_close(wgrad, recipe2_golden_tensors["wgrad"], atol=0.0, rtol=0.0)
        #     if use_bias:
        #         torch.testing.assert_close(
        #             bgrad, recipe2_golden_tensors["bgrad"], atol=0.0, rtol=0.0
        #         )

    @pytest.mark.parametrize(
        "batch_size, hidden_size, out_size",
        [
            (32, 256, 128),
            # (128, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
    @pytest.mark.parametrize(
        "recipe1, recipe2",
        [
            (GetRecipes.none, GetRecipes.nvfp4),
        ],
    )
    def test_nvfp4_scaling_with_linear_module(
        self,
        recipe1,
        recipe2,
        batch_size,
        hidden_size,
        out_size,
        dtype,
        use_bias=True,
    ):

        # TODO: add tensor dump for nvfp4

        # fp8_zero_tolerance_tensor_dumps_recipe2 = None
        # # check tensor dumps dir, if the dir exists, then read files to get y, dgrad, wgrad, bgrad
        # # if we cannot get all four tensors, then still set the tensor dump to None
        # tensor_map = self._check_golden_tensor_dumps(
        #     TENSOR_DUMP_DIR, recipe2, (batch_size, hidden_size, out_size), dtype, use_bias
        # )
        # if tensor_map is not None:
        #     fp8_zero_tolerance_tensor_dumps_recipe2 = tensor_map

        self.compare_recipe(
            recipe1,
            recipe2,
            batch_size,
            hidden_size,
            out_size,
            use_bias,
            seed=torch.initial_seed(),
            dtype=dtype,
            y_error=0.5,
            dgrad_error=1,
            wgrad_error=1,
            bgrad_error=0.5,
            recipe1_golden_tensors=None,
            recipe2_golden_tensors=None,
        )
