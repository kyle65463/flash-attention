import time
import json
from collections import OrderedDict

import numpy as np
import torch

from baseline import (
    pytorch_attn_efficient,
    pytorch_attn_flash,
    pytorch_attn_math,
    pytorch_manual_attn,
    flashattn_lib_attn,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

BYTES_PER_EL = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8}

def attn_ops_and_bytes(b: int, h: int, s: int, d: int, dtype):
    """Rough FLOP / byte model for standard scaled‑dot attention."""
    flops = 4 * b * h * s * s * d         # QKᵀ + softmax + AV (≈4 matmuls)
    bytes_mv = 4 * b * h * s * d * BYTES_PER_EL[dtype]  # read Q,K,V, write O
    return flops, bytes_mv


def run_timed(func, *args, runs: int = 5):
    """Return output, mean latency (ms) and mean peak memory (MB) over *runs*."""
    times, mems, output = [], [], None

    func(*args)  # warm‑up
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = func(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        mems.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return output, float(np.mean(times)), float(np.mean(mems))

# -----------------------------------------------------------------------------
# Benchmark core
# -----------------------------------------------------------------------------

def bench(
    *,
    seq: int,
    d: int = 64,
    heads: int = 32,
    batch: int = 4,
    device: str = "cuda",
    dtype=torch.float16,
    configs: dict,
    reference_impl: str | None = None,
):
    qkv = [torch.randn(batch, heads, seq, d, device=device, dtype=dtype) for _ in range(3)]
    flops_model, bytes_model = attn_ops_and_bytes(batch, heads, seq, d, dtype)

    ref_out = None
    row = OrderedDict({"N_CTX": seq})

    for name, cfg in configs.items():
        if not cfg["enabled"]:
            for suf in ("time_ms", "mem_mb", "gflops", "ai", "tok_s"):
                row[f"{name}_{suf}"] = "-"
            continue
        try:
            out, t_ms, m_mb = run_timed(cfg["func"], *qkv, runs=cfg.get("runs", 5))
            if name == reference_impl:
                ref_out = out

            if ref_out is not None and name != reference_impl and cfg.get("check_correctness"):
                if not torch.allclose(out, ref_out, atol=1e-2):
                    t_ms = f"{t_ms:.3f}*"  # flag mismatch

            secs = float(t_ms) / 1e3 if isinstance(t_ms, (int, float)) else np.nan
            gflops = flops_model / secs / 1e9 if secs else np.nan
            ai = flops_model / bytes_model
            tok_s = batch * seq / secs if secs else np.nan

            row[f"{name}_time_ms"] = round(t_ms, 3) if isinstance(t_ms, (int, float)) else t_ms
            row[f"{name}_mem_mb"] = round(m_mb, 2)
            row[f"{name}_gflops"] = round(gflops, 2)
            row[f"{name}_ai"] = round(ai, 2)
            row[f"{name}_tok_s"] = round(tok_s, 1)
        except Exception as e:
            print(f"{name} failed: {e}")
            for suf in ("time_ms", "mem_mb", "gflops", "ai", "tok_s"):
                row[f"{name}_{suf}"] = "ERROR"
    return row

# -----------------------------------------------------------------------------
# Config discovery
# -----------------------------------------------------------------------------

def get_configs():
    cfgs = OrderedDict({
        "PyTorch-Math": {
            "enabled": torch.backends.cuda.is_built(),
            "func": pytorch_attn_math,
            "runs": 5,
        },
        "PyTorch-Flash": {
            "enabled": getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)(),
            "func": pytorch_attn_flash,
        },
        "PyTorch-Efficient": {
            "enabled": getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)(),
            "func": pytorch_attn_efficient,
        },
        "PyTorch-Manual": {
            "enabled": True,
            "func": pytorch_manual_attn,
            "check_correctness": True,
        },
    })

    try:
        import flash_attn  # noqa: F401
        cfgs["FlashAttn-Lib"] = {
            "enabled": True,
            "func": flashattn_lib_attn,
            "check_correctness": True,
        }
    except ImportError:
        print("flash-attn library not available")

    try:
        import flash  # noqa: F401
        cfgs["CUDA-Naive"] = {
            "enabled": True,
            "func": flash.naive_attn,
            "check_correctness": True,
        }
        if hasattr(flash, "flash2"):
            cfgs["Flash2"] = {
                "enabled": True,
                "func": flash.flash2,
                "check_correctness": True,
            }
    except ImportError:
        print("Flash module not available")

    return cfgs

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

def main():
    seqs = [32, 128, 512, 1024, 2048, 4096]
    cfgs = get_configs()

    results = [bench(seq=s, configs=cfgs, reference_impl="PyTorch-Math") for s in seqs]

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": str(torch.float16),
        "gpu_name": torch.cuda.get_device_name(0),
        "cc": torch.cuda.get_device_capability(0),
        "results": results,
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("✔ benchmark_results.json saved")


if __name__ == "__main__":
    main()
