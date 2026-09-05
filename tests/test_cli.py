"""The CLI, since `scripts/run_sharded.sbatch` depends on it and sbatch failures are slow."""

import numpy as np
import pytest

from cvequality.__main__ import main


@pytest.fixture(scope="module")
def h5ad(tmp_path_factory):
    anndata = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    n, g = 600, 40
    groups = np.repeat(["non-targeting", "T1", "T2"], 200).astype(object)
    X = rng.poisson(np.outer(rng.gamma(20, 1 / 20, n), rng.gamma(1.5, 3.0, g))).astype(np.float32)
    ad = anndata.AnnData(X=sp.csr_matrix(X), obs={"target_gene_name": groups})
    ad.var_names = [f"g{i}" for i in range(g)]
    p = tmp_path_factory.mktemp("cli") / "toy.h5ad"
    ad.write_h5ad(p)
    return str(p)


def test_stats_then_vs_reference_shards(h5ad, tmp_path):
    import pandas as pd

    stats = str(tmp_path / "stats.pt")
    assert main(["stats", "--source", h5ad, "--out", stats, "--device", "cpu", "--quiet"]) == 0

    parts = []
    for r in range(2):
        out = str(tmp_path / f"shard{r}.parquet")
        assert main([
            "vs-reference", "--source", h5ad, "--stats", stats, "--out", out,
            "--shard", f"{r}/2", "--test", "both", "--nr", "64", "--device", "cpu", "--quiet",
        ]) == 0
        parts.append(pd.read_parquet(out))

    df = pd.concat(parts, ignore_index=True)
    assert set(df["perturbation"]) == {"T1", "T2"}
    assert len(df) == 2 * 40
    assert not set(parts[0]["perturbation"]) & set(parts[1]["perturbation"])


def test_vs_reference_all_tests(h5ad, tmp_path):
    import pandas as pd

    out = str(tmp_path / "all.parquet")
    assert main([
        "vs-reference", "--source", h5ad, "--out", out, "--test", "all", "--nr", "64",
        "--device", "cpu", "--quiet",
    ]) == 0
    df = pd.read_parquet(out)
    assert {
        "pval_asymptotic", "pval_mslrt", "pval_sd_ratio", "fdr_sd_ratio",
    } <= set(df.columns)


def test_vs_reference_sd_ratio_with_mean_stats(h5ad, tmp_path):
    import pandas as pd

    mean_stats = str(tmp_path / "tp10k.pt")
    assert main([
        "stats", "--source", h5ad, "--out", mean_stats, "--transform", "tp10k",
        "--device", "cpu", "--quiet",
    ]) == 0

    out = str(tmp_path / "sd-ratio.parquet")
    assert main([
        "vs-reference", "--source", h5ad, "--out", out, "--test", "sd_ratio",
        "--mean-stats", mean_stats, "--device", "cpu", "--quiet",
    ]) == 0
    assert set(pd.read_parquet(out)["mean_ratio_transform"]) == {"tp10k"}


def test_sd_ratio_omnibus_and_null(h5ad, tmp_path):
    import pandas as pd

    out = str(tmp_path / "sd-ratio-omni.parquet")
    assert main([
        "omnibus", "--source", h5ad, "--out", out, "--test", "sd_ratio",
        "--device", "cpu", "--quiet",
    ]) == 0
    assert "pval_sd_ratio" in pd.read_parquet(out)

    out = str(tmp_path / "sd-ratio-null.parquet")
    assert main([
        "null", "--source", h5ad, "--out", out, "--test", "sd_ratio",
        "--match-size", "100", "--device", "cpu", "--quiet",
    ]) == 0
    assert "pval_sd_ratio" in pd.read_parquet(out)


def test_omnibus_and_null(h5ad, tmp_path):
    import pandas as pd

    out = str(tmp_path / "omni.parquet")
    assert main(["omnibus", "--source", h5ad, "--out", out, "--test", "asymptotic",
                 "--device", "cpu", "--quiet"]) == 0
    assert len(pd.read_parquet(out)) == 40

    out = str(tmp_path / "null.parquet")
    assert main(["null", "--source", h5ad, "--out", out, "--test", "asymptotic",
                 "--match-size", "100", "--device", "cpu", "--quiet"]) == 0
    assert len(pd.read_parquet(out)) == 40


def test_null_ignores_stats_with_a_warning(h5ad, tmp_path, capsys):
    stats = str(tmp_path / "s.pt")
    main(["stats", "--source", h5ad, "--out", stats, "--device", "cpu", "--quiet"])
    out = str(tmp_path / "n.parquet")
    assert main(["null", "--source", h5ad, "--stats", stats, "--out", out,
                 "--test", "asymptotic", "--match-size", "100",
                 "--device", "cpu", "--quiet"]) == 0
    assert "ignored by `null`" in capsys.readouterr().err


def test_max_cells_truncates(h5ad, tmp_path):
    stats = str(tmp_path / "part.pt")
    assert main(["stats", "--source", h5ad, "--out", stats, "--max-cells", "300",
                 "--device", "cpu", "--quiet"]) == 0
    from cvequality.sufficient import GroupStats

    st = GroupStats.load(stats, device="cpu")
    assert float(st.n.sum()) == 300
    # only the leading cells are counted, so later groups are empty
    assert float(st.n[st.group_index("non-targeting")]) == 200


def test_bad_shard_is_rejected():
    with pytest.raises(SystemExit):
        main(["vs-reference", "--source", "x.h5ad", "--out", "y.parquet", "--shard", "oops"])
