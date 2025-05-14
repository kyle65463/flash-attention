import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ─── Tweakables ────────────────────────────────────────────────────────────────
DISABLED = {"PyTorch-Flash", "PyTorch-Efficient", "PyTorch-Math"}  # kernels to hide
REF = "PyTorch-Manual"  # baseline for Plot B
FIXED_LEN = 2048  # seq length for Plot B
OUTDIR = Path("plots")  # output folder

# GPU peak numbers for Roofline (edit to match your HW)
PEAK_GFLOPS = 45_000  # e.g. 45 TFLOP/s FP16  ⇒ 45 000 GFLOP/s
PEAK_BW_GB_S = 900  # e.g. 900 GB/s HBMe3   ⇒ mem-roof slope
# ───────────────────────────────────────────────────────────────────────────────


# ---------- helpers -----------------------------------------------------------
def load_results(jpath):
    with open(jpath) as f:
        js = json.load(f)
    return js["results"], js["configs"]


def collect(results, cfgs, field):
    """Return {kernel: ([seqs],[vals])}, skipping DISABLED + errors."""
    out = {c: ([], []) for c in cfgs if c not in DISABLED}
    for row in results:
        seq = row["seq"]
        for k in out:
            entry = row["data"].get(k, {})
            val = entry.get(field)
            if isinstance(val, (int, float)):
                out[k][0].append(seq)
                out[k][1].append(val)
    return out


def get_row(results, seq):
    return next((r["data"] for r in results if r["seq"] == seq), None)


# ---------- Plot A ------------------------------------------------------------
def plot_latency(results, cfgs, outdir):
    data = collect(results, cfgs, "time_ms")
    plt.figure(figsize=(6, 4))
    for k, (x, y) in data.items():
        if x:
            plt.plot(x, y, marker="o", label=k)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Latency [ms]")
    plt.title("Latency vs sequence length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "A.latency.png", dpi=200)
    plt.close()


# ---------- Plot B ------------------------------------------------------------
def plot_memory(results, cfgs, outdir):
    data = collect(results, cfgs, "mem_mb")
    plt.figure(figsize=(6, 4))
    for k, (x, y) in data.items():
        if x:
            plt.plot(x, np.array(y) / 1024, marker="o", label=k)  # MB→GB
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Peak memory [GB]")
    plt.title("Memory vs sequence length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "B.memory.png", dpi=200)
    plt.close()


# ---------- Plot C ------------------------------------------------------------
def plot_roofline(results, cfgs, outdir):
    # ---- gather per-kernel series ------------------------------------------
    series = defaultdict(lambda: ([], []))  # {kernel: ([AI], [GFLOPS])}
    for row in results:
        ai = row["data"][REF]["ai"]
        for k in cfgs:
            if k in DISABLED:
                continue
            d = row["data"].get(k, {})
            if isinstance(d.get("gflops"), (int, float)):
                series[k][0].append(ai)
                series[k][1].append(d["gflops"])

    if not series:
        print("roofline skipped (no data)")
        return

    # ---- plot points + tiny line so the colour is obvious -------------------
    plt.figure(figsize=(8, 4))
    for k, (xs, ys) in series.items():
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        plt.plot(xs, ys, marker="o", label=k, linewidth=0.8)  # thin line

    # ---- draw the roofs -----------------------------------------------------
    ai_min, ai_max = (
        0.5 * min(min(xs) for xs, _ in series.values()),
        2 * max(max(xs) for xs, _ in series.values()),
    )
    ai_line = np.logspace(np.log10(ai_min), np.log10(ai_max), 256)
    plt.plot(ai_line, PEAK_BW_GB_S * ai_line, "k--", label="Mem roof")
    plt.axhline(PEAK_GFLOPS, color="k", linestyle=":", label="Compute roof")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic intensity [FLOP/byte]")
    plt.ylabel("Achieved GFLOP/s")
    plt.title("Roofline")
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(outdir / "C.roofline.png", dpi=200)
    plt.close()


# ---------- Plot D ------------------------------------------------------------
def plot_throughput(results, cfgs, outdir):
    data = collect(results, cfgs, "tok_s")
    plt.figure(figsize=(6, 4))
    for k, (x, y) in data.items():
        if x:
            plt.plot(x, y, marker="o", label=k)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / s")
    plt.title("Throughput vs sequence length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "D.throughput.png", dpi=200)
    plt.close()


# ---------- main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="benchmark_results.json")
    ap.add_argument(
        "--outdir", type=Path, default=OUTDIR, help="directory to write PNGs"
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results, cfgs = load_results(args.json_path)

    plot_latency(results, cfgs, args.outdir)  # A
    plot_memory(results, cfgs, args.outdir)  # B
    plot_roofline(results, cfgs, args.outdir)  # C
    plot_throughput(results, cfgs, args.outdir)  # D

    print("✓ plots saved to", args.outdir.resolve())


if __name__ == "__main__":
    main()
