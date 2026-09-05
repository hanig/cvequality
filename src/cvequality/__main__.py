"""Command-line entry point, so a sharded SLURM job needs no bespoke Python.

    python -m cvequality stats        --source screen.h5ad --out tp10k.pt --transform tp10k
    python -m cvequality vs-reference --source screen.h5ad --out shard0.parquet \\
        --test all --mean-stats tp10k.pt --shard 0/3
    python -m cvequality omnibus      --source screen.h5ad --out omnibus.parquet --test sd_ratio
    python -m cvequality null         --source screen.h5ad --out null.parquet \\
        --test sd_ratio --match-size 722

``stats`` writes the sufficient statistics so later runs (other transforms aside) skip the
full matrix pass entirely.
"""

from __future__ import annotations

import argparse
import sys


def _shard(text: str):
    try:
        rank, world = text.split("/")
        return int(rank), int(world)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("shard must look like RANK/WORLD, e.g. 0/3") from exc


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="path to a CSR .h5ad")
    p.add_argument("--out", required=True, help="output path (.parquet, or .pt for `stats`)")
    p.add_argument("--group-key", default="target_gene_name", help="obs column with group labels")
    p.add_argument("--transform", default="log1p", choices=["counts", "tp10k", "log1p"])
    p.add_argument("--target-sum", type=float, default=1e4,
                   help="depth-normalization constant; changes the CV itself for log1p")
    p.add_argument("--layer", default=None, help="layer name (default: X)")
    p.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    p.add_argument("--stats", default=None, help="reuse sufficient statistics from this .pt")
    p.add_argument("--max-cells", type=int, default=None, help="smoke-test cap on cells streamed")
    p.add_argument("--quiet", action="store_true")


def _test_args(p: argparse.ArgumentParser, default_test: str) -> None:
    p.add_argument(
        "--test", default=default_test,
        choices=["both", "asymptotic", "mslrt", "sd_ratio", "all"],
    )
    p.add_argument("--nr", type=int, default=1000, help="bootstrap replicates for MSLRT")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", default="newton", choices=["newton", "fixedpoint"])
    p.add_argument("--chunk", type=int, default=None, help="tests per bootstrap step")
    p.add_argument("--min-cells", type=int, default=30)
    p.add_argument("--min-frac-expressed", type=float, default=0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cvequality", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("vs-reference", help="per-group vs a shared reference (k=2)")
    _common(p)
    _test_args(p, "both")
    p.add_argument("--reference", default="non-targeting")
    p.add_argument("--shard", type=_shard, default=(0, 1), help="RANK/WORLD")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--share-draws", action="store_true",
                   help="reuse bootstrap draws across genes within a group (common random numbers)")
    p.add_argument("--mean-stats", default=None,
                   help="ratio-scale sufficient statistics for log2_mean_ratio")

    p = sub.add_parser("omnibus", help="one test per gene across all groups")
    _common(p)
    _test_args(p, "asymptotic")
    p.add_argument("--exclude", nargs="*", default=[], help="group labels to leave out")

    p = sub.add_parser("null", help="reference-vs-itself calibration diagnostic")
    _common(p)
    _test_args(p, "both")
    p.add_argument("--reference", default="non-targeting")
    p.add_argument("--n-splits", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--match-size", type=int, default=None,
                   help="cells per half, to match a real group size")

    p = sub.add_parser("stats", help="compute and save sufficient statistics only")
    _common(p)

    a = ap.parse_args(argv)
    import cvequality as cvq
    from cvequality.sufficient import GroupStats, group_sufficient_stats

    progress = not a.quiet
    stats = GroupStats.load(a.stats, device=a.device) if a.stats else None

    if a.cmd == "stats":
        st = group_sufficient_stats(
            a.source, group_key=a.group_key, layer=a.layer, transform=a.transform,
            target_sum=a.target_sum, device=a.device, max_cells=a.max_cells, progress=progress,
        )
        st.save(a.out)
        print(f"wrote {a.out}: {len(st.group_names)} groups x {len(st.var_names)} genes "
              f"(transform={st.transform})")
        return 0

    shared = dict(
        stats=stats, group_key=a.group_key, transform=a.transform,
        target_sum=a.target_sum, layer=a.layer,
        test=a.test, nr=a.nr, seed=a.seed, solver=a.solver, min_cells=a.min_cells,
        min_frac_expressed=a.min_frac_expressed, device=a.device, progress=progress,
    )
    source = None if stats is not None else a.source
    if stats is None and a.max_cells is not None:
        # max_cells belongs to the streaming pass, which only happens without precomputed stats
        shared["stats"] = group_sufficient_stats(
            a.source, group_key=a.group_key, layer=a.layer, transform=a.transform,
            target_sum=a.target_sum, device=a.device, max_cells=a.max_cells, progress=progress,
        )
        source = None

    if a.cmd == "vs-reference":
        mean_stats = GroupStats.load(a.mean_stats, device=a.device) if a.mean_stats else None
        df = cvq.vs_reference(
            source, reference=a.reference, shard=a.shard, max_targets=a.max_targets,
            chunk=a.chunk, share_draws=a.share_draws, mean_stats=mean_stats, out=a.out, **shared,
        )
    elif a.cmd == "omnibus":
        df = cvq.omnibus(source, exclude=tuple(a.exclude), chunk=a.chunk, out=a.out, **shared)
    else:
        # The null diagnostic must recompute statistics: it groups cells by a random
        # half-split of the reference, which no precomputed table can describe.
        shared.pop("min_cells")
        if shared.pop("stats") is not None:
            print("note: --stats is ignored by `null` (it needs statistics for its own "
                  "half-split grouping)", file=sys.stderr)
        df = cvq.null_ntc_split(
            a.source, reference=a.reference, n_splits=a.n_splits, split_seed=a.split_seed,
            match_size=a.match_size, **shared,
        )
        df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
