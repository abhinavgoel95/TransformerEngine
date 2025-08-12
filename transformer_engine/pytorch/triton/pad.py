# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

import triton
import triton.language as tl


@triton.jit
def zero_pad_kernel(
    inp_dim_0, inp_dim_1, inp_ptr, out_ptr, inp_stride_0, inp_stride_1, out_stride_0, out_stride_1
):
    """Kernel for 0.0 padding a tensor."""

    # Calculate the thread's row and column in the output tensor
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Handle out-of-bounds rows and columns (padding with zero)
    if row < inp_dim_0 and col < inp_dim_1:
        # Load data from the input tensor A
        value = tl.load(inp_ptr + row * inp_stride_0 + col * inp_stride_1)
    else:
        # Zero padding in the out-of-bounds area
        value = tl.cast(0, tl.uint8)

    # Write the value to the output tensor
    tl.store(out_ptr + row * out_stride_0 + col * out_stride_1, value)


def pad_columnwise_scale_inv(
    tensor: torch.Tensor, dtype: torch.dtype = torch.uint8
) -> torch.Tensor:
    """Pads a tensor assuming it's a columnwise scaling inverse."""

    assert tensor.ndim == 2, "Unsupported ndim."
    dim0, dim1 = tensor.shape

    # Compute padding sizes.
    pad_x = (128 - dim0 % 128) % 128
    pad_y = (4 - dim1 % 4) % 4

    # No padding needed.
    if pad_x == 0 and pad_y == 0:
        return tensor

    # Create the output tensor.
    out = torch.empty((dim0 + pad_x, dim1 + pad_y), device=tensor.device, dtype=dtype)

    # Launch Triton kernel to perform the padding.
    zero_pad_kernel[(dim0 + pad_x, dim1 + pad_y)](
        dim0, dim1, tensor, out, tensor.stride(0), tensor.stride(1), out.stride(0), out.stride(1)
    )

    return out
