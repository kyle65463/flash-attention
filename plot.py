#!/usr/bin/env python3
# plot_results.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -------- tweakables ---------------------------------------------------------
DISABLED  = {"PyTorch-Flash", "PyTorch-Efficient"}  # hide these kernels
REF       = "PyTorch-Manual"                        # baseline for speed-up
FIXED_LEN = 2048                                    # sequence length for B/D
OUTDIR    = Path("./plots")                               # write PNGs here
# -----------------------------------------------------------------------------


def load(path):
    with open(path) as f:
        js = json.load(f)
    return js["results"], js["configs"]


# ---------- helpers to collect series ----------------------------------------
def collect(results, cfgs, field):
    out = {c: ([], []) for c in cfgs if c not in DISABLED}
    for row in results:
        seq = row["seq"]
        for c in out:
            v = row["data"].get(c, {})
            val = v.get(field)
            if isinstance(val, (int, float)):
                out[c][0].append(seq)
                out[c][1].append(val)
    return out


# ---------- Plot A  -----------------------------------------------------------
def plot_latency(results, cfgs):
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
    plt.savefig(OUTDIR / "latency.png")
    plt.close()


# ---------- Plot B  -----------------------------------------------------------
def plot_speedup(results, cfgs):
    target = next((r["data"] for r in results if r["seq"] == FIXED_LEN), None)
    if not target or not target.get(REF, {}).get("time_ms"):
        print("speed-up plot skipped (missing baseline)")
        return

    lat_ref = target[REF]["time_ms"]
    names, su = [], []
    for c in cfgs:
        if c in DISABLED or c == REF:
            continue
        lat = target.get(c, {}).get("time_ms")
        if lat:
            names.append(c)
            su.append(lat_ref / lat)

    plt.figure(figsize=(6, 3))
    plt.bar(names, su)
    plt.ylabel(f"Speed-up over {REF} @ {FIXED_LEN}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "speedup.png")
    plt.close()


# ---------- Plot C  -----------------------------------------------------------
def plot_memory(results, cfgs):
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
    plt.savefig(OUTDIR / "memory.png")
    plt.close()


# ---------- Plot D  -----------------------------------------------------------
def plot_roofline(results, cfgs):
    row = next((r["data"] for r in results if r["seq"] == FIXED_LEN), None)
    if not row:
        print("roofline plot skipped (seq missing)")
        return
    xs, ys, labels = [], [], []
    for c in cfgs:
        if c in DISABLED:
            continue
        d = row.get(c, {})
        if isinstance(d.get("ai"), (int, float)) and isinstance(
            d.get("gflops"), (int, float)
        ):
            xs.append(d["ai"])
            ys.append(d["gflops"])
            labels.append(c)

    if not xs:
        print("roofline plot skipped (no data)")
        return

    plt.figure(figsize=(5, 4))
    plt.scatter(xs, ys)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic intensity [FLOP / byte]")
    plt.ylabel("Achieved GFLOP/s")
    plt.title(f"Roofline @ seq {FIXED_LEN}")
    plt.tight_layout()
    plt.savefig(OUTDIR / "roofline.png")
    plt.close()


# ---------- Plot E  -----------------------------------------------------------
def plot_throughput(results, cfgs):
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
    plt.savefig(OUTDIR / "throughput.png")
    plt.close()


# ---------- main  -------------------------------------------------------------
def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="benchmark_results.json")
    ap.add_argument("--out", default=OUTDIR, type=Path, help="output directory")
    args = ap.parse_args()

    OUTDIR = args.out
    OUTDIR.mkdir(parents=True, exist_ok=True)

    results, cfgs = load(args.json_path)

    plot_latency(results, cfgs)     # A
    plot_speedup(results, cfgs)     # B
    plot_memory(results, cfgs)      # C
    plot_roofline(results, cfgs)    # D
    plot_throughput(results, cfgs)  # E

    print("✓ plots saved to", OUTDIR.resolve())


if __name__ == "__main__":
    main()
