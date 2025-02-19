# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
This file contains tests for saving and loading TransformerEngine torch checkpoints.

The purpose of this test is to validate the TransformerEngine hooks for saving FP8 metadata
in torch checkpoints, which are called as part of torch.save() and torch.load().
The test verifies the values of FP8 metadata object after saving and loading a checkpoint
are identical to the original values.
"""

import io
import tempfile
from typing import Iterable, Union

import pytest
import torch
import transformer_engine.common
import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import fp8_gemm, cast_to_fp8
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


def init_meta(size: int = 1):
    meta = tex.FP8TensorMeta()
    meta.scale = torch.ones(size, dtype=torch.float32, device="cuda")
    meta.scale_inv = torch.ones(size, dtype=torch.float32, device="cuda")
    meta.amax_history = torch.zeros(1, size, dtype=torch.float32, device="cuda")
    return meta


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("scale_fwd", [224, 112, 66])
@pytest.mark.parametrize("scale_bwd", [448, 33])
@pytest.mark.parametrize("history_fwd", [1.23, 4.56])
@pytest.mark.parametrize("history_bwd", [2.34, 5.67])
def test_export_loaded_checkpoint(scale_fwd, scale_bwd, history_fwd, history_bwd):

    tmp_filename = tempfile.NamedTemporaryFile().name

    precision = torch.float32

    class Test_TE_Export(TransformerEngineBaseModule):
        def __init__(self, precision, use_bias):
            super().__init__()
            self.use_bias = use_bias
            self.precision = precision

            self.fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT
            self.fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
            nb_inp_scales = nb_weight_scales = 1
            self.meta_inp = init_meta(nb_inp_scales)
            self.meta_weight = init_meta(nb_weight_scales)

            bias_size = nb_weight_scales
            self.bias = torch.randn(bias_size, dtype=precision, device="cuda")

            self.inp_type = tex.DType.kFloat8E4M3
            self.weights_type = tex.DType.kFloat8E4M3
            self.outp_type = precision

        def get_fp8_weights_scratchpad(self, is_first_microbatch):
            raise RuntimeError(
                "Method get_fp8_weights_scratchpad is dummy and should not be invoked."
            )

        def forward(self, inp, weight):
            inp_fp8 = cast_to_fp8(inp, self.meta_inp, self.fp8_tensor_inp, self.inp_type)

            weight_fp8 = cast_to_fp8(
                weight, self.meta_weight, self.fp8_tensor_weight, self.weights_type
            )

            ret = fp8_gemm(
                weight_fp8,
                self.meta_weight.scale_inv,
                self.fp8_tensor_weight,
                self.inp_type,
                inp_fp8,
                self.meta_inp.scale_inv,
                self.fp8_tensor_inp,
                self.weights_type,
                self.outp_type,
                get_workspace(),
                bias=self.bias,
                use_bias=self.use_bias,
                use_split_accumulator=False,
            )
            return ret

    model_in = Test_TE_Export(precision, True)
    with te.fp8_autocast(enabled=True):
        model_in.init_fp8_metadata()
        # scaling fwd
        model_in.fp8_meta["scaling_fwd"].scale = (
            torch.ones(3, dtype=torch.float32, device="cuda") * scale_fwd
        )
        model_in.fp8_meta["scaling_fwd"].scale_inv = (
            torch.ones(3, dtype=torch.float32, device="cuda") / scale_fwd
        )
        model_in.fp8_meta["scaling_fwd"].amax_history = (
            torch.ones(3, dtype=torch.float32, device="cuda") * history_fwd
        )
        # scaling bwd
        model_in.fp8_meta["scaling_bwd"].scale = (
            torch.ones(2, dtype=torch.float32, device="cuda") * scale_bwd
        )
        model_in.fp8_meta["scaling_bwd"].scale_inv = (
            torch.ones(2, dtype=torch.float32, device="cuda") / scale_bwd
        )
        model_in.fp8_meta["scaling_bwd"].amax_history = (
            torch.ones(2, dtype=torch.float32, device="cuda") * history_bwd
        )

    torch.save(model_in.state_dict(), tmp_filename)

    model_out = Test_TE_Export(precision, True)
    model_out.load_state_dict(torch.load(tmp_filename))
    model_out.eval()

    # scaling fwd
    assert torch.allclose(
        model_in.fp8_meta["scaling_fwd"].scale, model_out.fp8_meta["scaling_fwd"].scale
    )
    assert torch.allclose(
        model_in.fp8_meta["scaling_fwd"].scale_inv, model_out.fp8_meta["scaling_fwd"].scale_inv
    )
    assert torch.allclose(
        model_in.fp8_meta["scaling_fwd"].amax_history,
        model_out.fp8_meta["scaling_fwd"].amax_history,
    )
    # scaling bwd
    assert torch.allclose(
        model_in.fp8_meta["scaling_bwd"].scale, model_out.fp8_meta["scaling_bwd"].scale
    )
    assert torch.allclose(
        model_in.fp8_meta["scaling_bwd"].scale_inv, model_out.fp8_meta["scaling_bwd"].scale_inv
    )
    assert torch.allclose(
        model_in.fp8_meta["scaling_bwd"].amax_history,
        model_out.fp8_meta["scaling_bwd"].amax_history,
    )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("save_fp8_model", [True, False])
@pytest.mark.parametrize("load_fp8_model", [True, False])
def test_fp8_model_checkpoint(
    save_fp8_model: bool,
    load_fp8_model: bool,
    dims: Iterable[int] = [32, 32],
    dtype: torch.dtype = torch.float32,
    device: Union[torch.device, str] = "cuda",
):

    # Construct model
    dims = list(dims)
    hidden_dim = dims[-1]
    with te.fp8_model_init(enabled=save_fp8_model):
        model = te.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
            params_dtype=dtype,
            device=device,
        )
    # Keep track of model output
    x = torch.randn(dims, dtype=dtype, device=device)
    with te.fp8_autocast():
        y_ref = model(x.detach().clone()).detach().clone()

    fp8_meta_ref = {"scaling_fwd": {}, "scaling_bwd": {}}
    with te.fp8_autocast(), torch.no_grad():
        fp8_meta_fwd = model.fp8_meta["scaling_fwd"]
        fp8_meta_bwd = model.fp8_meta["scaling_bwd"]
        fp8_meta_fwd_ref = fp8_meta_ref["scaling_fwd"]
        fp8_meta_bwd_ref = fp8_meta_ref["scaling_bwd"]
        fp8_meta_fwd_ref["scale"] = torch.rand_like(fp8_meta_fwd.scale) + 0.5
        fp8_meta_fwd_ref["scale_inv"] = fp8_meta_fwd_ref["scale"].reciprocal()
        fp8_meta_bwd_ref["scale"] = torch.rand_like(fp8_meta_bwd.scale) + 0.5
        fp8_meta_bwd_ref["scale_inv"] = fp8_meta_bwd_ref["scale"].reciprocal()
        fp8_meta_fwd.scale.copy_(fp8_meta_fwd_ref["scale"])
        fp8_meta_fwd.scale_inv.copy_(fp8_meta_fwd_ref["scale_inv"])
        fp8_meta_bwd.scale.copy_(fp8_meta_bwd_ref["scale"])
        fp8_meta_bwd.scale_inv.copy_(fp8_meta_bwd_ref["scale_inv"])
        del fp8_meta_fwd, fp8_meta_bwd

    # [ This is part of logic that tests save_fp8_model=False and load_fp8_model=True ]
    # This line copies the fp8 scale_inv from the model metadata to the weight fp8 tensor.
    # The sole purpose of the following lines is to set the scale_inv of the weight tensor, which is the simplest method.
    # It is essential for these values to be equal, so setting scale_inv only in the model metadata is insufficient.
    model.weight.data.copy_(model.weight.float().cuda())
    # After copying, the tensor computes the meta scale_inv based on the amax history; we then reset these values.
    model.fp8_meta["scaling_fwd"].scale = fp8_meta_fwd_ref["scale"]
    model.fp8_meta["scaling_fwd"].scale_inv = fp8_meta_fwd_ref["scale_inv"]

    # Keep track of weights and FP8 scaling factors
    weight_ref = model.weight.float().detach().clone()

    # Save checkpoint
    byte_stream = io.BytesIO()
    torch.save(model.state_dict(), byte_stream)
    model_bytes = byte_stream.getvalue()
    del byte_stream

    # Disturb and destroy model
    with torch.no_grad():
        model.weight.zero_()
    model.fp8_meta = {"This": "is", "filled": "with", "nonsense": 1234}
    del model

    # Construct new model
    with te.fp8_model_init(enabled=load_fp8_model):
        model = te.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
            params_dtype=dtype,
            device=device,
        )

    # Make sure new model does not match saved model
    tols = dict(rtol=0.125, atol=0.0675)  # fp8e4me3 epsilon = 0.0625
    with pytest.raises(AssertionError):
        torch.testing.assert_close(model.weight, weight_ref, **tols)
    with te.fp8_autocast():
        model.init_fp8_metadata()
        fp8_meta_fwd = model.fp8_meta["scaling_fwd"]
        fp8_meta_bwd = model.fp8_meta["scaling_bwd"]
        fp8_meta_fwd_ref = fp8_meta_ref["scaling_fwd"]
        fp8_meta_bwd_ref = fp8_meta_ref["scaling_bwd"]
        with pytest.raises(AssertionError):
            torch.testing.assert_close(fp8_meta_fwd.scale, fp8_meta_fwd_ref["scale"])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(fp8_meta_fwd.scale_inv, fp8_meta_fwd_ref["scale_inv"])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(fp8_meta_bwd.scale, fp8_meta_bwd_ref["scale"])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(fp8_meta_bwd.scale_inv, fp8_meta_bwd_ref["scale_inv"])
    with te.fp8_autocast():
        y = model(x.detach().clone())
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y, y_ref, **tols)

    # [ This is part of logic that tests save_fp8_model=False and load_fp8_model=True ]
    # When save_fp8_model=True, we load a model with weights in high precision,
    # which does not include _scale_inv,
    # but has the fp8 scaling factor in the meta data. This scenario can occur
    # when using te.fp8_autocast(enabled=False, calibrating=True).
    #
    # In such cases, the default behavior of load_state_dict is incorrect - it loads tensors first,
    # followed by the fp8 metadata. This results in an incorrect _scale_inv for the tensor. This behavior
    # is corrected by overriding the _load_state_dict method from PyTorch in TransformerEngineBaseModule,
    # to load the fp8 metadata before loading tensors.
    #
    # Load checkpoint
    model.load_state_dict(torch.load(io.BytesIO(model_bytes)))
    del model_bytes

    # Check that loaded model matches saved model
    torch.testing.assert_close(model.weight, weight_ref, **tols)
    with te.fp8_autocast():
        fp8_meta_fwd = model.fp8_meta["scaling_fwd"]
        fp8_meta_bwd = model.fp8_meta["scaling_bwd"]
        fp8_meta_fwd_ref = fp8_meta_ref["scaling_fwd"]
        fp8_meta_bwd_ref = fp8_meta_ref["scaling_bwd"]
        torch.testing.assert_close(fp8_meta_fwd.scale, fp8_meta_fwd_ref["scale"])
        torch.testing.assert_close(fp8_meta_fwd.scale_inv, fp8_meta_fwd_ref["scale_inv"])
        torch.testing.assert_close(fp8_meta_bwd.scale, fp8_meta_bwd_ref["scale"])
        torch.testing.assert_close(fp8_meta_bwd.scale_inv, fp8_meta_bwd_ref["scale_inv"])
    with te.fp8_autocast():
        y = model(x.detach().clone())
        torch.testing.assert_close(y, y_ref, **tols)

    if load_fp8_model:
        # [ This is part of logic that tests save_fp8_model=False and load_fp8_model=True ]
        # We need to ensure that the tensor's scale_inv parameter matches its meta data.
        # This is crucial to avoid confusion about which value is correct.
        meta_index = model.weight._fp8_meta_index
        torch.testing.assert_close(
            model.weight._scale_inv.item(), fp8_meta_fwd_ref["scale_inv"][meta_index].item()
        )


@pytest.mark.parametrize("fp8", (False, True))
@pytest.mark.parametrize("save_fp8_model", (False, True))
@pytest.mark.parametrize("load_fp8_model", (False, True))
def test_sequential_model(
    *,
    in_shape: Iterable[int] = (16, 16),
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    save_steps: int = 2,
    load_steps: int = 2,
    fp8: bool,
    save_fp8_model: bool,
    load_fp8_model: bool,
) -> None:

    # Skip invalid configurations
    if fp8 or save_fp8_model or load_fp8_model:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if torch.device(device).type != "cuda":
            pytest.skip("FP8 is only supported on CUDA devices")

    # FP8 recipe
    margin = 2
    fp8_format = transformer_engine.common.recipe.Format.E4M3
    recipe = transformer_engine.common.recipe.DelayedScaling(
        margin=margin,
        fp8_format=fp8_format,
        amax_history_len=8,
        amax_compute_algo="max",
    )

    # Construct model to save to checkpoint
    with te.fp8_model_init(enabled=save_fp8_model):
        model = te_ops.Sequential(
            te_ops.Linear(in_shape[-1], in_shape[-1], device=device, dtype=dtype),
        )
    with torch.no_grad():
        torch.rand(model[0].weight.size(), out=model[0].weight)
        torch.rand(model[0].bias.size(), out=model[0].bias)

    # Synthetic data
    xs_ref = [
        torch.rand(in_shape, dtype=dtype, device=device) for _ in range(save_steps + load_steps)
    ]
    dys_ref = [
        torch.rand(in_shape, dtype=dtype, device=device) for _ in range(save_steps + load_steps)
    ]

    def train_step(
        model: te_ops.Sequential,
        x: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function to perform training step"""
        x = x.detach().clone().requires_grad_()
        dy = dy.detach().clone()
        with te.fp8_autocast(enabled=fp8, fp8_recipe=recipe):
            y = model(x)
        y.backward(dy)
        with torch.no_grad():
            for param in model.parameters():
                param += 0.125
        return (
            y.detach().clone(),
            x.grad.detach().clone(),
            model[0].weight.detach().float().clone(),
        )

    # Initial training steps with saved model
    ys_ref = []
    dxs_ref = []
    ws_ref = []
    for step in range(save_steps):
        y, dx, w = train_step(model, xs_ref[step], dys_ref[step])
        ys_ref.append(y)
        dxs_ref.append(dx)
        ws_ref.append(w)

    # Keep track of FP8 metadata if needed
    fp8_meta_ref = dict(input={}, param={}, grad_output={})
    if fp8:
        for fp8_meta_type, fp8_meta_key in (
            ("input", "scaling_fwd"),
            ("param", "scaling_fwd"),
            ("grad_output", "scaling_bwd"),
        ):
            m_model = model[0].basic_ops[0].get_fp8_meta(fp8_meta_type)[fp8_meta_key]
            m_ref = fp8_meta_ref[fp8_meta_type]
            m_ref["amax"] = m_model.amax_history.detach().clone()
            m_ref["scale"] = m_model.scale.detach().clone()
            m_ref["scale_inv"] = m_model.scale_inv.detach().clone()
            del m_model, m_ref

    # Save checkpoint
    byte_stream = io.BytesIO()
    torch.save(model.state_dict(), byte_stream)
    model_bytes = byte_stream.getvalue()
    del byte_stream

    # More training steps with saved model
    for step in range(save_steps, save_steps + load_steps):
        y, dx, w = train_step(model, xs_ref[step], dys_ref[step])
        ys_ref.append(y)
        dxs_ref.append(dx)
        ws_ref.append(w)

    # Disturb and destroy model
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    model[0].basic_ops[0]._fp8_metas = None
    del model

    # Construct new model to load from checkpoint
    with te.fp8_model_init(enabled=load_fp8_model):
        model = te_ops.Sequential(
            te_ops.Linear(in_shape[-1], in_shape[-1], device=device, dtype=dtype),
        )

    # Tolerances for numerical checks
    tols = {}
    if fp8 or save_fp8_model or load_fp8_model:
        tols = dict(rtol=0.125, atol=0.0675)  # fp8e4me3 epsilon = 0.0625
    exact_tols = dict(rtol=0, atol=0)

    # Training steps with dummy data
    for step in range(save_steps):
        y, dx, w = train_step(
            model,
            torch.zeros_like(xs_ref[step]),
            torch.zeros_like(dys_ref[step]),
        )

        # Make sure results don't match saved model
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y, ys_ref[step], **tols)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(dx, dxs_ref[step], **tols)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(w, ws_ref[step], **tols)

    # Make sure new model's FP8 metadata doesn't match saved model
    if fp8:
        for fp8_meta_type, fp8_meta_key in (
            ("input", "scaling_fwd"),
            ("param", "scaling_fwd"),
            ("grad_output", "scaling_bwd"),
        ):
            m_model = model[0].basic_ops[0].get_fp8_meta(fp8_meta_type)[fp8_meta_key]
            m_ref = fp8_meta_ref[fp8_meta_type]
            with pytest.raises(AssertionError):
                torch.testing.assert_close(m_model.amax_history, m_ref["amax"], **exact_tols)
            with pytest.raises(AssertionError):
                torch.testing.assert_close(m_model.scale, m_ref["scale"], **exact_tols)
            with pytest.raises(AssertionError):
                torch.testing.assert_close(m_model.scale_inv, m_ref["scale_inv"], **exact_tols)

    # Load checkpoint
    model.load_state_dict(torch.load(io.BytesIO(model_bytes)))
    del model_bytes

    # Check that new model's FP8 metadata matches saved model
    if fp8:
        for fp8_meta_type, fp8_meta_key in (
            ("input", "scaling_fwd"),
            ("param", "scaling_fwd"),
            ("grad_output", "scaling_bwd"),
        ):
            m_model = model[0].basic_ops[0].get_fp8_meta(fp8_meta_type)[fp8_meta_key]
            m_ref = fp8_meta_ref[fp8_meta_type]
            torch.testing.assert_close(m_model.amax_history, m_ref["amax"], **exact_tols)
            torch.testing.assert_close(m_model.scale, m_ref["scale"], **exact_tols)
            torch.testing.assert_close(m_model.scale_inv, m_ref["scale_inv"], **exact_tols)

    # More training steps with loaded model
    for step in range(save_steps, save_steps + load_steps):
        y, dx, w = train_step(model, xs_ref[step], dys_ref[step])
        torch.testing.assert_close(y, ys_ref[step], **tols)
        torch.testing.assert_close(dx, dxs_ref[step], **tols)
        torch.testing.assert_close(w, ws_ref[step], **tols)
