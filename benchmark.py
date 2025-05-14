import json
import time
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
import flash

# ----- simple FLOP / byte model ------------------------------------------------
BYTES_PER_EL = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8}


def attn_ops_and_bytes(b, h, s, d, dtype):
    flops = 4 * b * h * s * s * d  # QKᵀ + softmax + AV  (≈4 matmuls)
    bytes_mv = 4 * b * h * s * d * BYTES_PER_EL[dtype]  # Q,K,V + O
    return flops, bytes_mv


# ----- timing helper -----------------------------------------------------------
def _run(func, *args, runs=5):
    times, mems = [], []
    out = func(*args)  # warm-up

    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = func(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        mems.append(torch.cuda.max_memory_allocated() / 1024**2)
    return out, float(np.mean(times)), float(np.mean(mems))


# ----- benchmark core ----------------------------------------------------------
def bench(
    *,
    seq,
    d=64,
    heads=32,
    batch=4,
    device="cuda",
    dtype=torch.float16,
    configs,
    reference_impl=None,
):
    qkv = [
        torch.randn(batch, heads, seq, d, device=device, dtype=dtype) for _ in range(3)
    ]
    flops_model, bytes_model = attn_ops_and_bytes(batch, heads, seq, d, dtype)
    reference = None

    result = {"seq": seq, "data": {}}

    for name, cfg in configs.items():
        entry = {}
        if not cfg["enabled"]:
            result["data"][name] = None  # disabled on this machine
            continue

        try:
            print(f"Running {name} with {seq} sequence length")
            out, t_ms, m_mb = _run(cfg["func"], *qkv, runs=cfg.get("runs", 5))
            if name == reference_impl:
                reference = out

            if (
                reference is not None
                and name != reference_impl
                and cfg.get("check_correctness")
            ):
                entry["mismatch"] = not torch.allclose(out, reference, atol=1e-2)

            secs = t_ms / 1e3
            entry.update(
                time_ms=round(t_ms, 3),
                mem_mb=round(m_mb, 2),
                gflops=round(flops_model / secs / 1e9, 2),
                ai=round(flops_model / bytes_model, 2),
                tok_s=round(batch * seq / secs, 1),
            )
        except Exception as e:
            entry["error"] = str(e)

        result["data"][name] = entry
    return result


# ----- config discovery --------------------------------------------------------
def get_configs():
    cfgs = OrderedDict(
        {
            "PyTorch-Math": {
                "enabled": torch.backends.cuda.is_built(),
                "func": pytorch_attn_math,
            },
            "PyTorch-Flash": {
                "enabled": getattr(
                    torch.backends.cuda, "flash_sdp_enabled", lambda: False
                )(),
                "func": pytorch_attn_flash,
            },
            "PyTorch-Efficient": {
                "enabled": getattr(
                    torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False
                )(),
                "func": pytorch_attn_efficient,
            },
            "PyTorch-Manual": {
                "enabled": True,
                "func": pytorch_manual_attn,
                "check_correctness": True,
            },
            "FlashAttn-Lib": {
                "enabled": True,
                "func": flashattn_lib_attn,
                "check_correctness": True,
            },
            "Naive": {
                "enabled": True,
                "func": flash.naive_attn,
                "check_correctness": True,
            },
            "Flash1": {
                "enabled": True,
                "func": flash.flash1,
                "check_correctness": True,
            },
            "Flash2": {
                "enabled": True,
                "func": flash.flash2,
                "check_correctness": True,
            },
        }
    )

    return cfgs


# ----- main --------------------------------------------------------------------
def main():
    seqs = [32, 128, 512, 1024, 2048]
    cfgs = get_configs()

    results = [bench(seq=s, configs=cfgs, reference_impl="PyTorch-Math") for s in seqs]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dtype": str(torch.float16),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(0))
            ),
        },
        "configs": list(cfgs.keys()),
        "results": results,
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("✔ benchmark_results.json saved")


if __name__ == "__main__":
    main()
