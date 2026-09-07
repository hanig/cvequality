from collections import Counter

import cvequality as cvq
from cvequality.sufficient import GroupStats, group_sufficient_stats


def test_vs_reference_materializes_derived_tables_once(monkeypatch, toy_adata):
    """Derived (groups, genes) tables must not be recomputed per target."""
    stats = group_sufficient_stats(
        toy_adata, group_key="target_gene_name", transform="log1p", moments=True,
        device="cpu", progress=False,
    )
    mean_stats = group_sufficient_stats(
        toy_adata, group_key="target_gene_name", transform="tp10k", moments=False,
        device="cpu", progress=False,
    )
    counts = Counter()

    for name in ("mean", "sd", "frac_expressed", "kurtosis"):
        original = GroupStats.__dict__[name].fget

        def counted(self, _name=name, _original=original):
            counts[(_name, id(self))] += 1
            return _original(self)

        monkeypatch.setattr(GroupStats, name, property(counted))

    result = cvq.vs_reference(
        stats=stats, mean_stats=mean_stats, reference="ntc", transform="log1p",
        test="sd_ratio", min_cells=30, min_frac_expressed=0.1,
        device="cpu", progress=False,
    )

    assert result["perturbation"].nunique() == 3
    by_property = Counter(name for name, _object_id in counts)
    assert by_property == Counter({
        "mean": 2,  # CV stats plus the separate ratio-scale mean_stats
        "sd": 1,
        "frac_expressed": 1,
        "kurtosis": 1,
    })
