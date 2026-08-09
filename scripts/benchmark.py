#!/usr/bin/env python
"""Measure MSLRT throughput and project the wall clock for a full screen.

Run this before committing GPU hours: the cost is linear in
``n_targets * n_genes * nr * newton_iterations``, and the constant depends enough on the
device and the CV regime to be worth measuring rather than assuming.

    python scripts/benchmark.py                     # synthetic, perturb-seq-like
    python scripts/benchmark.py --nr 1000 --shards 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import cvequality as cvq

# Sizing of a typical genome-wide Perturb-seq screen, used only to project wall clock.
# Override with --targets/--genes/--n-ref/--n-median for your own design.
SCREEN_TARGETS = 2000
SCREEN_GENES = 18000
SCREEN_N_REF = 200_000
SCREEN_N_MEDIAN = 700


def synth(T: int, k: int, seed: int = 0):
    """A batch with single-cell-like group sizes and CVs (the slow regime for the MLE)."""
    rng = np.random.default_rng(seed)
    n = np.empty((T, k))
    n[:, 0] = SCREEN_N_REF
    n[:, 1:] = SCREEN_N_MEDIAN
    x = rng.uniform(0.05, 40.0, (T, k))
    cv = rng.uniform(0.5, 3.0, (T, k))
    return n, x, cv * x


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, default=100_000, help="tests per timing batch")
    ap.add_argument("--nr", type=int, default=1000)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--shards", type=int, default=3, help="GPUs the real run will use")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    dev = cvq._backend.resolve_device(a.device)
    print(f"device={dev} dtype={a.dtype} nr={a.nr} k={a.k}")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        print(f"  {torch.cuda.get_device_name(dev)}  {free/2**30:.1f}/{total/2**30:.1f} GiB free")

    n, x, s = synth(a.tests, a.k)

    # Warm up: first CUDA call pays context creation and kernel autotuning.
    cvq.mslr_test2_batch(n=n[:64], x=x[:64], s=s[:64], nr=64, device=dev, dtype=dtype)

    _, t_asym = timed(lambda: cvq.asymptotic_test2_batch(n=n, s=s, x=x, device=dev, dtype=dtype))
    print(f"\nasymptotic: {a.tests:,} tests in {t_asym:.3f}s = {a.tests/t_asym:,.0f} tests/s")
    full_asym = SCREEN_TARGETS * SCREEN_GENES / (a.tests / t_asym)
    print(f"  full screen ({SCREEN_TARGETS*SCREEN_GENES:,} tests): {full_asym:.1f}s")

    res, t_ms = timed(
        lambda: cvq.mslr_test2_batch(
            n=n, x=x, s=s, nr=a.nr, seed=0, device=dev, dtype=dtype, chunk=a.chunk
        )
    )
    rate = a.tests / t_ms
    print(f"\nmslrt (nr={a.nr}): {a.tests:,} tests in {t_ms:.2f}s = {rate:,.0f} tests/s")
    ok = int((res.status == int(cvq.Status.OK)).sum())
    print(f"  status OK: {ok:,}/{a.tests:,}   median bootstrap draws used: "
          f"{int(res.n_valid.median())}/{a.nr}")

    full = SCREEN_TARGETS * SCREEN_GENES / rate
    print(f"\nprojected screen, {SCREEN_TARGETS:,} targets x {SCREEN_GENES:,} genes:")
    print(f"  1 GPU : {full/3600:6.2f} h")
    print(f"  {a.shards} GPUs: {full/a.shards/3600:6.2f} h  ({full/a.shards/60:.0f} min per shard)")
    for frac in (0.5, 0.25):
        print(f"  with min_frac_expressed dropping {100*(1-frac):.0f}% of genes: "
              f"{full*frac/a.shards/3600:6.2f} h on {a.shards} GPUs")

    print("\nchunk-size sweep (throughput vs memory):")
    for ch in (2048, 8192, 32768, None):
        try:
            _, t = timed(
                lambda ch=ch: cvq.mslr_test2_batch(
                    n=n[:20_000], x=x[:20_000], s=s[:20_000], nr=a.nr, seed=0,
                    device=dev, dtype=dtype, chunk=ch,
                )
            )
            label = "auto" if ch is None else f"{ch:,}"
            print(f"  chunk={label:>7s}: {20_000/t:>10,.0f} tests/s")
        except torch.cuda.OutOfMemoryError:
            print(f"  chunk={ch}: OOM")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
