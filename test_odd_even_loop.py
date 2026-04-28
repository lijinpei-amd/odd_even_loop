import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odd_even_loop as ext_module


def check_variant(name, fn, a, b, expected, rows, cols) -> None:
    out = fn(a, b)
    torch.cuda.synchronize()
    actual = out[rows[:, None], cols[None, :]]
    ref = expected[rows[:, None], cols[None, :]]
    torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)
    print(f"{name} correctness: passed")
    print(f"{name} out shape/stride:", tuple(out.shape), tuple(out.stride()))


def bench_variant(name, fn, a, b, iters=100) -> float:
    M, K = a.shape
    N = b.shape[1]
    ms = triton.testing.do_bench(lambda: fn(a, b), warmup=25, rep=iters)
    tflops = (2.0 * M * N * K) / ms * 1e-9
    print(f"{name:20s} {ms:.6f} ms {tflops:.0f} TFLOP/s")
    return tflops


def bench_torch_matmul(a, b, iters=100) -> float:
    M, K = a.shape
    N = b.shape[1]
    ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=iters)
    tflops = (2.0 * M * N * K) / ms * 1e-9
    print(f"{'torch.matmul':20s} {ms:.6f} ms {tflops:.0f} TFLOP/s")
    return tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=16384)
    args = parser.parse_args()

    M = args.m
    N = args.n
    K = args.k
    device = "cuda"

    ext = ext_module.load_ext(verbose=False)
    print("extension:", ext.__file__)
    print(f"problem: M={M} N={N} K={K}")

    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    b_storage = torch.randn((N, K), device=device, dtype=torch.bfloat16)
    b = b_storage.T

    expected = torch.matmul(a, b)
    rows = torch.tensor([0, 1, 257, M - 1], device=device)
    cols = torch.tensor([0, 1, 17, 1024, N - 1], device=device)

    check_variant(
        "original",
        ext_module.matmul_original,
        a,
        b,
        expected,
        rows,
        cols,
    )
    check_variant(
        "same_loop",
        ext_module.matmul_same_loop,
        a,
        b,
        expected,
        rows,
        cols,
    )

    bench_variant("original", ext_module.matmul_original, a, b)
    bench_variant("same_loop", ext_module.matmul_same_loop, a, b)
    bench_torch_matmul(a, b)


if __name__ == "__main__":
    main()
