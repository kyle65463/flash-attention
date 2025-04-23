import argparse, time, torch, flash


def bench(seq, d=128, heads=8, batch=1, device="cuda"):
    qkv = [
        torch.randn(
            batch,
            heads,
            seq,
            d,
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )
        for _ in range(3)
    ]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = flash.naive_attn(*qkv)
    torch.cuda.synchronize()
    print(f"naive  : {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    o2 = torch.nn.functional.scaled_dot_product_attention(*qkv)
    torch.cuda.synchronize()
    print(f"torch  : {(time.perf_counter() - t0) * 1000:.1f} ms")

    # TODO flash2 once implemented
    assert torch.allclose(o, o2, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=512)
    args = parser.parse_args()
    bench(args.seq)
