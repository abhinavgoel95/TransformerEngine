# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with NVFP4 data"""
from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex

from transformer_engine.common.recipe import NVFP4BlockScaling, Recipe
from ..constants import NVFP4_BLOCK_SCALING_SIZE
from ..utils import devices_match, round_up_to_nearest_multiple

from ._internal.nvfp4_tensor_base import NVFP4TensorBase, _FromNVFP4Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten


class NVFP4Quantizer(Quantizer):
    """Builder class for NVFP4 tensors with NV block scaling"""

    dtype: TE_DType
    """Random Hadamard Transform"""
    with_rht: bool
    with_post_rht_amax: bool
    """amax reduction options"""
    with_amax_reduction: bool
    amax_reduction_group: Optional[dist_group_type]

    def __init__(
        self,
        fp4_dtype: TE_DType = tex.DType.kFloat4E2M1,
        rowwise: bool = True,
        columnwise: bool = True,
        with_amax_reduction: bool = False,
        amax_reduction_group: Optional[dist_group_type] = None,
        with_rht: bool = False,
        with_post_rht_amax: bool = False,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp4_dtype
        self.with_rht = with_rht
        self.with_post_rht_amax = with_post_rht_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, NVFP4Tensor), f"Cannot store quantized NVFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        return dst

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for NVFP4 1D blockwise quantization.

        This method determines the shape of the scaling tensor needed for blockwise quantization,
        taking into account the input tensor shape and whether columnwise scaling is used.

        Parameters
        ----------
        shape : Iterable[int]
            Shape of the input tensor to be quantized
        columnwise : bool
            Whether to use columnwise scaling (True) or rowwise scaling (False)

        Returns
        -------
        Tuple[int, int]
            Shape of the scaling tensor as (outer_dim, inner_dim)
            For NVFP4 1D blockwise quantization, blocksize is 16
            - If columnwise: (round_to_multiple(K, 128), round_to_multiple(roundup(M / 16), 4))
            - If rowwise: (round_to_multiple(M, 128), round_to_multiple(roundup(K / 16), 4))
        Swizzle kernel will be performed before GEMM to suit the need of CuBLAS.
        CuBLAS doc: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
        """
        M, K = 1, 1
        M = math.prod(shape[:-1])
        K = shape[-1]

        if columnwise:
            outer = round_up_to_nearest_multiple(K, 128)
            inner = round_up_to_nearest_multiple(math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4)
            return (outer, inner)
        else:
            # rowwise
            outer = round_up_to_nearest_multiple(M, 128)
            inner = round_up_to_nearest_multiple(math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4)
            return (outer, inner)

    def get_columnwise_shape(self, shape: Iterable[int]) -> Tuple[int, ...]:
        """Calculate the shape of a tensor after columnwise quantization.

        For NVFP4 columnwise quantization, it's performing 16x1 quantization block scaling.

        Parameters
        ----------
        shape : Iterable[int]
            Original shape of the tensor

        Returns
        -------
        Tuple[int, ...]
            New shape with dimensions rearranged for columnwise layout.
            For a shape (d1, d2, ..., dn), returns (dn, d1, d2, ..., dn-1).
            Returns empty tuple for empty input shape.
        """
        if len(shape) == 0:
            return tuple()
        # and then after AG, a reorganize kernel will be called to restore the shape
        colwise_shape = [shape[-1]]
        for i in range(len(shape) - 1):
            colwise_shape.append(shape[i])
        return tuple(colwise_shape)

    def convert_shape_for_fp4(self, shape: Iterable[int]) -> Tuple[int, ...]:
        """Convert shape for FP4 data by dividing the last dimension by 2"""
        shape = list(shape)
        shape[-1] = shape[-1] // 2
        return tuple(shape)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> NVFP4Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert shape[-1] % NVFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for NVFP4. Tensor dims must be divisible by"
            f" {NVFP4_BLOCK_SCALING_SIZE}"
        )

        flat_first_dim = math.prod(shape[:-1])
        assert flat_first_dim % NVFP4_BLOCK_SCALING_SIZE == 0, (
            f"Incorrect shape {shape} for NVFP4. Tensor dims must be divisible by"
            f" {NVFP4_BLOCK_SCALING_SIZE}"
        )

        # Allocate FP4 data
        data = None
        scale_inv = None
        amax_rowwise = None
        if self.rowwise_usage:
            data = torch.empty(self.convert_shape_for_fp4(shape), dtype=torch.uint8, device=device)
            scale_shape = self.get_scale_shape(shape, columnwise=False)
            scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)
            # Allocate per tensor scale inverse. FP32 format.
            amax_rowwise = torch.zeros(1, dtype=torch.float32, device=device)

        # Allocate FP8 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        amax_columnwise = None
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                self.convert_shape_for_fp4(self.get_columnwise_shape(shape)),
                dtype=torch.uint8,
                device=device,
            )
            columnwise_scale_shape = self.get_scale_shape(shape, columnwise=True)
            columnwise_scale_inv = torch.empty(
                columnwise_scale_shape, dtype=torch.uint8, device=device
            )
            amax_columnwise = torch.zeros(1, dtype=torch.float32, device=device)

        # Construct FP8 tensor
        return NVFP4Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            amax_rowwise=amax_rowwise,
            amax_columnwise=amax_columnwise,
            fp4_dtype=self.dtype,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # TODO(ksivamani): No calibration?
        pass

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return NVFP4BlockScaling


class NVFP4Tensor(NVFP4TensorBase, QuantizedTensor):
    """Experimental tensor class with Hybrid FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw FP8 data in a uint4 tensor
    fp8_scale_inv: torch.Tensor
                   Reciprocal of the scaling factor applied when
                   casting to FP8, i.e. the scaling factor that must
                   be applied when casting from FP8 to higher
                   precision.
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.
    """

    # NOTE: We reorder the *args so that we can instantiate a NVFP4TensorBase with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        fp4_dtype: TE_DType,
        quantizer: Quantizer,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            amax_rowwise,
            amax_columnwise,
            fp4_dtype,
            quantizer,
            *args,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        return f"NVFP4Tensor, data={self.dequantize(dtype=self.dtype)})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from NVFP4Tensor

        By default the resulting tensor's dtype is the
        NVFP4Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromNVFP4Func.apply(self, dtype)
        return _FromNVFP4Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return NVFP4Quantizer()

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> NVFP4Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def detach(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        # TODO(ksivamani): Fix the detach bug
        return NVFP4Tensor.make_like(self)

    def clone(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
        rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> NVFP4Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("NVFP4Tensor does not support different memory formats!")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            return NVFP4Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                amax_rowwise=tensor._amax_rowwise,
                amax_columnwise=tensor._amax_columnwise,
                quantizer=tensor._quantizer,
                requires_grad=False,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor,
        fp4_dtype: TE_DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
    ) -> NVFP4Tensor:
        """Build NVFP4Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return NVFP4Tensor(
            shape=shape,
            dtype=dtype,
            fp4_dtype=fp4_dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            amax_rowwise=amax_rowwise,
            amax_columnwise=amax_columnwise,
            quantizer=quantizer,
            requires_grad=False,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            NVFP4Tensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._amax_rowwise,
                self._amax_columnwise,
                self._fp4_dtype,
                self.dtype,
                self._quantizer,
            ),
        )

    def _get_data(self) -> NVFP4Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a NVFP4Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is NVFP4Tensor
        if isinstance(tensor, NVFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    NVFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(NVFP4Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            self._amax_rowwise = tensor._amax_rowwise
            self._amax_columnwise = tensor._amax_columnwise
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.update_quantized(tensor, self)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting NVFP4Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)
        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(ctx.shape[-1], -1)
            else:
                new_columnwise_data = None
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
