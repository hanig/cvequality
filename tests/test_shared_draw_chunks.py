"""Shared bootstrap draws are call-wide and unaffected by chunk boundaries."""

import numpy as np
import torch

import cvequality as cvq
from cvequality._backend import Status


def _shared_inputs():
    n = np.tile(np.array([100.0, 100.0]), (6, 1))
    x = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.4, 0.9],
            [0.8, 1.6],
            [2.0, 1.2],
            [1.1, 1.8],
        ]
    )
    s = np.array(
        [
            [0.3, 0.5],
            [0.3, 0.5],
            [0.6, 0.2],
            [0.4, 0.7],
            [1.1, 0.8],
            [0.5, 1.0],
        ]
    )
    return n, x, s


def _assert_bootstrap_equal(left, right, left_index=slice(None), right_index=slice(None)):
    for field in ("MSLRT", "p_value", "null_mean", "null_sd"):
        torch.testing.assert_close(
            getattr(left, field)[left_index], getattr(right, field)[right_index], rtol=0, atol=0
        )
    torch.testing.assert_close(
        left.n_valid[left_index], right.n_valid[right_index], rtol=0, atol=0
    )


def test_repeated_rows_share_draws_across_multiple_chunks():
    n, x, s = _shared_inputs()
    result = cvq.mslr_test2_batch(
        n=n, x=x, s=s, nr=100, seed=123, device="cpu", chunk=2, share_draws=True
    )

    # These identical rows occupy the first chunk; append copies in later chunks to prove
    # every chunk uses the same normal and chi-square bases.
    n = np.concatenate((n, n[:1], n[:1]))
    x = np.concatenate((x, x[:1], x[:1]))
    s = np.concatenate((s, s[:1], s[:1]))
    across = cvq.mslr_test2_batch(
        n=n, x=x, s=s, nr=100, seed=123, device="cpu", chunk=2, share_draws=True
    )

    for field in ("MSLRT", "p_value", "null_mean", "null_sd", "n_valid"):
        values = getattr(across, field)[[0, 1, 6, 7]]
        torch.testing.assert_close(values, values[:1].expand_as(values), rtol=0, atol=0)
    _assert_bootstrap_equal(result, across, right_index=slice(0, 6))


def test_fixed_seed_shared_results_are_independent_of_chunk_size():
    n, x, s = _shared_inputs()
    kwargs = dict(n=n, x=x, s=s, nr=128, seed=17, device="cpu", share_draws=True)
    unchunked = cvq.mslr_test2_batch(**kwargs, chunk=10_000)

    for chunk in (1, 2, 4):
        _assert_bootstrap_equal(unchunked, cvq.mslr_test2_batch(**kwargs, chunk=chunk))


def test_seed_none_uses_global_rng_without_chunk_dependence():
    n, x, s = _shared_inputs()
    kwargs = dict(n=n, x=x, s=s, nr=64, seed=None, device="cpu", share_draws=True)
    torch.manual_seed(2026)
    one_at_a_time = cvq.mslr_test2_batch(**kwargs, chunk=1)
    torch.manual_seed(2026)
    unchunked = cvq.mslr_test2_batch(**kwargs, chunk=10_000)

    _assert_bootstrap_equal(one_at_a_time, unchunked)


def test_shared_chunks_compact_degenerate_rows_and_handle_empty_batches():
    n, x, s = _shared_inputs()
    clean = cvq.mslr_test2_batch(
        n=n, x=x, s=s, nr=64, seed=9, device="cpu", chunk=1, share_draws=True
    )
    x_mixed = np.insert(x, [1, 4], [[0.0, 1.0], [1.0, 1.0]], axis=0)
    s_mixed = np.insert(s, [1, 4], [[0.2, 0.2], [0.0, 0.2]], axis=0)
    n_mixed = np.insert(n, [1, 4], [[100.0, 100.0], [100.0, 100.0]], axis=0)
    mixed = cvq.mslr_test2_batch(
        n=n_mixed, x=x_mixed, s=s_mixed, nr=64, seed=9, device="cpu", chunk=3,
        share_draws=True,
    )

    ok = mixed.status == int(Status.OK)
    _assert_bootstrap_equal(clean, mixed, right_index=ok)
    assert bool((mixed.n_valid[~ok] == 0).all())

    all_bad = cvq.mslr_test2_batch(
        n=n[:2], x=np.zeros_like(x[:2]), s=s[:2], nr=8, seed=9, device="cpu", chunk=1,
        share_draws=True,
    )
    assert bool((all_bad.status == int(Status.NONPOSITIVE_MEAN)).all())
    assert bool((all_bad.n_valid == 0).all())

    empty = np.empty((0, 2))
    result = cvq.mslr_test2_batch(
        n=empty, x=empty, s=empty, nr=8, seed=None, device="cpu", chunk=1,
        share_draws=True,
    )
    assert result.MSLRT.shape == (0,)
