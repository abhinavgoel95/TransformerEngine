# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import dataclasses
import enum
import os
from typing import Optional

import torch

from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.experimental import quantization
from transformer_engine.pytorch.experimental import quantization_microblock_ref
from transformer_engine.pytorch.experimental.quantization import MMParams


@dataclasses.dataclass()
class QLinearParams:
    """Quantization parameters of linear layer.

    Contains ready-to-use quantizers for input (x), weight (w), and gradient (g) tensors.
    """

    x_quantizer: Optional[quantization.ExperimentalQuantizer] = None
    w_quantizer: Optional[quantization.ExperimentalQuantizer] = None
    g_quantizer: Optional[quantization.ExperimentalQuantizer] = None

    mm_fprop: Optional[MMParams] = None
    mm_dgrad: Optional[MMParams] = None
    mm_wgrad: Optional[MMParams] = None


@enum.unique
class QuantizeRecipe(enum.Enum):
    """Pre-defined quantization recipes for linear layers."""

    NON_QUANTIZE = "non_quantize"
    NVFP4_REF = "nvfp4_ref"
    NVFP4_REF_RHT_ONLY = "nvfp4_ref_rht_only"


def get_qlinear_params_from_predefined(
    recipe: QuantizeRecipe,
) -> Optional[QLinearParams]:
    """Get quantization parameters for linear layer based on recipe."""
    if recipe == QuantizeRecipe.NON_QUANTIZE:
        return None
    elif recipe == QuantizeRecipe.NVFP4_REF:
        return QLinearParams(
            x_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
            ),
            w_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
            ),
            g_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
            ),
        )
    elif recipe == QuantizeRecipe.NVFP4_REF_RHT_ONLY:
        return QLinearParams(
            x_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=True,
            ),
            w_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=False,
            ),
            g_quantizer=quantization_microblock_ref.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported quantize recipe: {recipe}")


def get_qlinear_params_from_qat_params(qat_params_idx: int) -> Optional[QLinearParams]:
    assert qat_params_idx > 0, "QAT_PARAMS is not set."

    if qat_params_idx == 6010:
        return get_qlinear_params_from_predefined(QuantizeRecipe.NVFP4_REF)
    elif qat_params_idx == 960109:
        return get_qlinear_params_from_predefined(QuantizeRecipe.NVFP4_REF_RHT_ONLY)
    raise ValueError(f"Unsupported QAT params index: {qat_params_idx}")


def set_qlinear_params(
    qlinear_params: Optional[QLinearParams] = None,
    layer_number: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> Optional[QLinearParams]:
    """Set quantization parameters based on configuration.

    Args:
        qlinear_params: Quantization parameters. If None, loaded from environment.
        layer_number: The numerical index of this layer in the model structure.
        layer_name: The name for this layer.

    Returns:
        QLinearParams: The finalized quantization parameters for this layer.
    """
    if qlinear_params is None:
        qat_params_idx = int(os.getenv("QAT_PARAMS", "0"))
        if qat_params_idx == 0:
            return None
        return get_qlinear_params_from_qat_params(qat_params_idx)

    # Apply layer-specific overrides
    if layer_number is not None:
        raise NotImplementedError("Layer-specific overrides are not supported yet.")
    if layer_name is not None:
        raise NotImplementedError("Layer-specific overrides are not supported yet.")

    return qlinear_params


def get_experimental_quantizers(fp8: bool, qlinear_params: QLinearParams):
    """Replacement of _get_quantizers() in TE modules."""
    if not fp8:
        raise ValueError("FP8 is required to be enabled for experimental quantization.")
    input_quantizer = qlinear_params.x_quantizer
    weight_quantizer = qlinear_params.w_quantizer
    output_quantizer = None
    grad_input_quantizer = None
    grad_weight_quantizer = None
    grad_output_quantizer = qlinear_params.g_quantizer

    return (
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
    )
