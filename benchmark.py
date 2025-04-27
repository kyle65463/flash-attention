import time
import torch
import numpy as np
import pandas as pd
from baseline import python_attention, pytorch_attention

try:
    import flash
    HAVE_FLASH = True
    print("Flash module loaded successfully!")
except ImportError as e:
    print(f"Flash module not available: {e}")
    HAVE_FLASH = False

def run_timed(func, *args, runs=5):
    times = []
    output = None
    
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = func(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    
    mean = np.mean(times)
    return output, mean

def bench(seq=32, d=64, heads=32, batch=4, device="cuda", dtype=torch.float16, run_cpu=False):
    print(f"Running benchmark with sequence length: {seq}, dim: {d}, heads: {heads}, batch: {batch}")    
    
    qkv = [
        torch.randn(
            batch,
            heads,
            seq,
            d,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        for _ in range(3)
    ]
    
    print("Running PyTorch SDPA implementation (5 runs)...")
    _ = pytorch_attention(*qkv)  # warmup
    o_pytorch, pytorch_mean = run_timed(pytorch_attention, *qkv)
    
    python_time = None
    if run_cpu:
        print("Running CPU (Python) implementation...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o_python = python_attention(*qkv)
        torch.cuda.synchronize()
        python_time = (time.perf_counter() - t0) * 1000
    
    naive_time_str = "-"
    
    if HAVE_FLASH:
        print("Running naive CUDA implementation (5 runs)...")
        try:
            _ = flash.naive_attn(*qkv)  # warmup
            
            o_naive, naive_mean = run_timed(flash.naive_attn, *qkv)
            naive_time_str = f"{naive_mean:.3f}"
            
            max_diff_naive = torch.max(torch.abs(o_naive - o_pytorch)).item()
            print(f"Max difference between naive CUDA and SDPA: {max_diff_naive:.6f}")
            if not torch.allclose(o_naive, o_pytorch, atol=1e-2):
                print(f"Warning: Naive CUDA and SDPA implementations don't match exactly")
                naive_time_str = f"{naive_mean:.3f}*"
        except Exception as e:
            print(f"Error running naive CUDA implementation: {e}")
            naive_time_str = "ERROR"
    
    result = {
        "N_CTX": seq,
        "Python-CPU": f"{python_time:.3f}" if python_time is not None else "-",
        "PyTorch-SDPA": f"{pytorch_mean:.3f}",
        "CUDA-Naive": naive_time_str,
        "Flash2": "-",
        "Flash2-opt": "-"
    }
    
    return result

if __name__ == "__main__":
    seq_lengths = [32, 128, 512, 2048]
    results = []
    
    for seq in seq_lengths:
        run_cpu = (seq == 32)
        result = bench(seq=seq, run_cpu=run_cpu)
        results.append(result)
        print("\n" + "-"*80 + "\n")
    
    df = pd.DataFrame(results)
    
    formatted_df = pd.DataFrame({
        "| N_CTX |": df["N_CTX"],
        " Python-CPU [ms] |": df["Python-CPU"],
        " PyTorch-SDPA [ms] |": df["PyTorch-SDPA"],
        " CUDA-Naive [ms] |": df["CUDA-Naive"],
        " Flash2 [ms] |": df["Flash2"],
        " Flash2-opt [ms] |": df["Flash2-opt"]
    })
    
    print("\nBenchmark Results Summary:")
    print(formatted_df.to_string(index=False, justify="center"))