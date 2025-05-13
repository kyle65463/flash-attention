import torch
from torch.autograd import Function

# --- Naïve kernel wrapper -------------------------------------------------

try:
    import flash._naive as _naive_cuda
except ImportError as e:  # pragma: no cover
    _naive_cuda = None
    print("warning: naive CUDA extension not built", e)


def naive_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Reference CUDA implementation (O(n^2) memory) – outputs same shape as SDPA."""
    assert _naive_cuda is not None, "naive kernel not compiled"
    return _naive_cuda.forward(q, k, v)


# --- Flash‑2 kernel autograd Function -------------------------------------

try:
    import flash._flash2 as _flash2_cuda
except ImportError as e:  # pragma: no cover
    _flash2_cuda = None
    print("warning: flash2 CUDA extension not built", e)


class _Flash2Func(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        assert _flash2_cuda is not None, "flash2 kernel not compiled"
        out = _flash2_cuda.forward(q, k, v)
        # save for backward once implemented
        ctx.save_for_backward(q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _flash2_cuda.backward(q, k, v, out, grad_out)
        return dq, dk, dv


def flash2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return _Flash2Func.apply(q, k, v)


# --- Flash‑1 kernel autograd Function -------------------------------------

try:
    import flash._flash1 as _flash1_cuda
except ImportError as e:  # pragma: no cover
    _flash1_cuda = None
    print("warning: flash1 CUDA extension not built", e)


class _Flash1Func(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        assert _flash1_cuda is not None, "flash1 kernel not compiled"
        return _flash1_cuda.forward(q, k, v)


def flash1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return _Flash1Func.apply(q, k, v)
