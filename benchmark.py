import time
import torch
import numpy as np
import pandas as pd
from baseline import (
    pytorch_attn_efficient,
    pytorch_attn_flash,
    pytorch_attn_math,
    pytorch_manual_attn,
    flashattn_lib_attn,
)
from collections import OrderedDict


def run_timed(func, *args, runs=5):
    """Run a function multiple times and return the output and mean execution time."""
    times = []
    output = None

    # Warmup
    _ = func(*args)

    # Run the function multiple times
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = func(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean = np.mean(times)

    return output, mean


def bench(
    seq=32,
    d=64,
    heads=32,
    batch=4,
    device="cuda",
    dtype=torch.float16,
    configs=None,
    reference_impl=None,
):
    """Benchmark different attention implementations."""
    print(
        f"Running benchmark with sequence length: {seq}, dim: {d}, heads: {heads}, batch: {batch}"
    )

    # Generate random query, key, value tensors
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

    results = OrderedDict({"N_CTX": seq})
    reference_output = None

    # Run each enabled implementation
    for name, config in configs.items():
        if not config["enabled"]:
            results[name] = "-"
            continue

        impl_func = config["func"]
        runs = config.get("runs", 5)

        print(
            f"Running {name} implementation{f' ({runs} runs)' if runs > 1 else ''}..."
        )

        try:
            # Time the implementation
            output, mean_time = run_timed(impl_func, *qkv, runs=runs)
            time_str = f"{mean_time:.3f}"

            # Set reference output if this is the reference implementation
            if name == reference_impl:
                reference_output = output

            # Compare with reference if available
            if reference_output is not None and name != reference_impl:
                if config.get("check_correctness", False) and not torch.allclose(
                    output, reference_output, atol=1e-2
                ):
                    print(
                        f"Warning: {name} and {reference_impl} implementations don't match exactly"
                    )
                    time_str = f"{time_str}*"

            results[name] = time_str

        except Exception as e:
            print(f"Error running {name} implementation: {e}")
            results[name] = "ERROR"

    return results


def get_configs():
    """Return default configurations for all implementations."""
    configs = OrderedDict(
        {
            "PyTorch-Math": {
                "enabled": torch.backends.cuda.is_built(),
                "func": pytorch_attn_math,
                "runs": 5,
            },
            "PyTorch-Flash": {
                "enabled": hasattr(torch.backends.cuda, "flash_sdp_enabled")
                and torch.backends.cuda.flash_sdp_enabled(),
                "func": pytorch_attn_flash,
                "runs": 5,
            },
            "PyTorch-Efficient": {
                "enabled": hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
                and torch.backends.cuda.mem_efficient_sdp_enabled(),
                "func": pytorch_attn_efficient,
                "runs": 5,
            },
            "PyTorch-Manual": {
                "enabled": True,
                "func": pytorch_manual_attn,
                "runs": 5,
                "check_correctness": True,
            },
        }
    )

    # Add flash-attn library implementation
    try:
        import flash_attn

        configs["FlashAttn-Lib"] = {
            "enabled": True,
            "func": flashattn_lib_attn,
            "runs": 5,
            "check_correctness": True,
        }
    except ImportError:
        print("flash-attn library not available")

    # Add Flash implementations if available
    try:
        import flash

        configs["CUDA-Naive"] = {
            "enabled": True,
            "func": flash.naive_attn,
            "runs": 5,
            "check_correctness": True,
        }

        # Uncomment these when implemented
        # configs["Flash2"] = {
        #     "enabled": hasattr(flash, "flash2"),
        #     "func": flash.flash2,
        #     "runs": 5,
        #     "check_correctness": True,
        # }
    except ImportError:
        print("Flash module not available")

    return configs


def format_results_table(results_list):
    """Format benchmark results into a nice table."""
    df = pd.DataFrame(results_list)
    columns = {
        f"| {col} |" if col == "N_CTX" else f" {col} [ms] |": df[col]
        for col in df.columns
    }
    return pd.DataFrame(columns)


if __name__ == "__main__":
    # Sequence lengths to test
    seq_lengths = [32, 128, 512, 1024]

    # Get default configurations
    configs = get_configs()

    results = []
    for seq in seq_lengths:
        result = bench(seq=seq, configs=configs, reference_impl="PyTorch-Math")
        results.append(result)
        print("\n" + "-" * 80 + "\n")

    # Format and display results
    formatted_df = format_results_table(results)
    print("\nBenchmark Results Summary:")
    print(formatted_df.to_string(index=False, justify="center"))
