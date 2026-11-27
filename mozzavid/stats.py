"""Group comparison, post-hoc tests, ICC and the text report (Requirements 15-17).

Pure Python using scipy for every statistical primitive (Levene, ANOVA, Kruskal-Wallis,
Tukey HSD, ranking, normal distribution). The Sample is the statistical unit for `compare`
and `posthoc`; `icc1` works on per-volume values grouped by sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """One omnibus comparison of a metric across cheese types (Requirement 15.5)."""

    test_name: str            # "one-way ANOVA" | "Kruskal-Wallis"
    statistic: float
    p_value: float
    effect_size: float        # eta^2 (ANOVA) | epsilon^2 (Kruskal-Wallis)
    effect_size_name: str
    levene_p: float
    n_groups: int
    n_per_group: dict


def _groups(per_sample: pd.DataFrame, metric: str, group_col: str,
            drop_undersized: bool = False) -> dict:
    """Return {group: values} with NaN dropped; raise or drop groups with < 2 values (Req 15.1, 15.7).

    `compare` raises for any undersized group (an under-powered test would be misleading).
    `icc1` passes `drop_undersized=True`, since single-volume samples carry no within-sample
    information and are simply excluded.
    """
    groups, undersized = {}, []
    for name, chunk in per_sample.groupby(group_col, sort=True):
        values = chunk[metric].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size < 2:
            undersized.append(str(name))
            continue
        groups[str(name)] = values
    if not drop_undersized and undersized:
        raise ValueError(f"metric {metric!r}: groups with < 2 non-NaN values: {undersized}")
    if len(groups) < 2:
        raise ValueError(f"metric {metric!r}: need >= 2 groups with >= 2 values, got {len(groups)}")
    return groups


def compare(per_sample: pd.DataFrame, metric: str = "foreground_fraction",
            group_col: str = "cheese_type", alpha: float = 0.05) -> TestResult:
    """One-way ANOVA when Levene's test passes, else Kruskal-Wallis (Requirement 15).

    Levene (median-centred) checks equal variance. `levene_p >= alpha` selects ANOVA with
    η² = SS_between/SS_total; otherwise Kruskal-Wallis with ε² = H·(n+1)/(n²−1).
    """
    named = _groups(per_sample, metric, group_col)
    values = list(named.values())
    levene_p = float(scipy.stats.levene(*values, center="median").pvalue)
    sizes = {name: int(group.size) for name, group in named.items()}
    if levene_p >= alpha:
        res = scipy.stats.f_oneway(*values)
        return _result("one-way ANOVA", res, _eta_squared(values), "eta_squared", levene_p, sizes)
    res = scipy.stats.kruskal(*values)
    effect = _epsilon_squared(float(res.statistic), values)
    return _result("Kruskal-Wallis", res, effect, "epsilon_squared", levene_p, sizes)


def _result(name: str, res, effect_size: float, effect_name: str, levene_p: float,
            sizes: dict) -> TestResult:
    """Assemble a TestResult from a scipy omnibus result and the computed effect size."""
    return TestResult(test_name=name, statistic=float(res.statistic), p_value=float(res.pvalue),
                      effect_size=effect_size, effect_size_name=effect_name, levene_p=levene_p,
                      n_groups=len(sizes), n_per_group=sizes)


def _eta_squared(groups: list) -> float:
    """Return η² = SS_between / SS_total for a one-way layout."""
    pooled = np.concatenate(groups)
    grand_mean = pooled.mean()
    ss_total = float(((pooled - grand_mean) ** 2).sum())
    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups))
    return ss_between / ss_total if ss_total > 0 else 0.0


def _epsilon_squared(h_statistic: float, groups: list) -> float:
    """Return ε² = H·(n+1)/(n²−1), clipped to 0 with a log message if negative (Req 15.4, 15.6)."""
    n = sum(g.size for g in groups)
    epsilon = h_statistic * (n + 1) / (n**2 - 1)
    if epsilon < 0:
        log.info("epsilon_squared %.6g < 0; clipping to 0", epsilon)
        return 0.0
    return float(epsilon)


def posthoc(per_sample: pd.DataFrame, metric: str, result: TestResult,
            group_col: str = "cheese_type") -> pd.DataFrame:
    """Tukey HSD after ANOVA, else Dunn's test with Holm adjustment (Requirement 16).

    Returns columns `group_a`, `group_b`, `statistic`, `p_raw`, `p_adj`. The post-hoc test is
    matched to the omnibus test so it never relies on an assumption the omnibus test rejected.
    """
    named = _groups(per_sample, metric, group_col)
    if result.test_name == "one-way ANOVA":
        return _tukey_frame(scipy.stats.tukey_hsd(*named.values()), list(named))
    return _holm(_dunn(named))


def _tukey_frame(tukey, names: list) -> pd.DataFrame:
    """Flatten a scipy Tukey HSD result into the post-hoc table; p_adj equals p_raw for Tukey."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = float(tukey.pvalue[i, j])
            rows.append({"group_a": names[i], "group_b": names[j],
                         "statistic": float(tukey.statistic[i, j]), "p_raw": p, "p_adj": p})
    return pd.DataFrame(rows)


def _dunn(named: dict) -> pd.DataFrame:
    """Dunn's test on pooled ranks with tie correction; two-sided normal p-values (Req 16.2)."""
    names = list(named)
    pooled = np.concatenate([named[n] for n in names])
    ranks = scipy.stats.rankdata(pooled)
    sizes = np.array([named[n].size for n in names])
    mean_ranks = _group_mean_ranks(ranks, sizes)
    total = int(sizes.sum())
    tie_term = _tie_term(ranks, total)
    rows = [_dunn_pair(names, mean_ranks, sizes, total, tie_term, i, j)
            for i in range(len(names)) for j in range(i + 1, len(names))]
    return pd.DataFrame(rows)


def _group_mean_ranks(ranks: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return the mean rank of each group from the pooled rank vector, in group order."""
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return np.array([ranks[bounds[i]:bounds[i + 1]].mean() for i in range(sizes.size)])


def _tie_term(ranks: np.ndarray, total: int) -> float:
    """Return the tie correction T = Σ(t³−t)/(12(N−1)) for the pooled ranks (Requirement 16.2)."""
    _, counts = np.unique(ranks, return_counts=True)
    return float((counts**3 - counts).sum()) / (12.0 * (total - 1)) if total > 1 else 0.0


def _dunn_pair(names: list, mean_ranks: np.ndarray, sizes: np.ndarray, total: int,
               tie_term: float, i: int, j: int) -> dict:
    """Return one Dunn pair row: z-statistic and two-sided normal p-value."""
    sigma = np.sqrt((total * (total + 1) / 12.0 - tie_term) * (1.0 / sizes[i] + 1.0 / sizes[j]))
    z = (mean_ranks[i] - mean_ranks[j]) / sigma
    p = 2.0 * float(scipy.stats.norm.sf(abs(z)))
    return {"group_a": names[i], "group_b": names[j], "statistic": float(z),
            "p_raw": min(p, 1.0), "p_adj": np.nan}


def _holm(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Holm step-down adjusted p-values, monotone and capped at 1 (Requirement 16.4)."""
    frame = frame.sort_values("p_raw", kind="stable").reset_index(drop=True)
    m = len(frame)
    raw = frame["p_raw"].to_numpy()
    # Holm: multiply the i-th smallest raw p by (m - i), then enforce monotone non-decreasing.
    adjusted = np.maximum.accumulate((m - np.arange(m)) * raw)
    frame["p_adj"] = np.clip(adjusted, 0.0, 1.0)
    return frame


def icc1(per_volume: pd.DataFrame, metric: str, group_col: str = "sample_id") -> float:
    """One-way random-effects ICC(1) of volumes within samples, with the n0 correction (Req 17.4).

    Uses only samples with ≥ 2 non-NaN volume values. Negative values are returned as-is;
    when all values are identical the denominator is 0 and the result is NaN with a warning.
    Raises `ValueError` when fewer than 2 samples qualify.
    """
    groups = list(_groups(per_volume, metric, group_col, drop_undersized=True).values())
    sizes = np.array([g.size for g in groups])
    k, total = len(groups), int(sizes.sum())
    grand_mean = np.concatenate(groups).mean()
    ms_between = sum(n * (g.mean() - grand_mean) ** 2 for n, g in zip(sizes, groups)) / (k - 1)
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (total - k)
    n0 = (total - (sizes**2).sum() / total) / (k - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    if denominator == 0:
        log.warning("icc1 %r: zero variance, returning NaN", metric)
        return float("nan")
    return float((ms_between - ms_within) / denominator)


def format_report(results: dict, posthocs: dict, alpha: float = 0.05) -> str:
    """Return a fixed-width text block per metric for `results/anova_results.txt` (Requirement 17.1)."""
    blocks = [_format_metric(metric, results.get(metric), posthocs.get(metric), alpha)
              for metric in results]
    return "\n".join(blocks)


def _format_metric(metric: str, result, posthoc_frame, alpha: float) -> str:
    """Format one metric block: header, omnibus line, and significant post-hoc pairs."""
    header = f"=== {metric} ===".ljust(60)
    if result is None:
        return f"{header}\ninsufficient data\n"
    lines = [header, _omnibus_line(result), _groups_line(result)]
    lines += _posthoc_lines(posthoc_frame, alpha)
    return "\n".join(lines) + "\n"


def _omnibus_line(result: TestResult) -> str:
    """Return the one-line omnibus summary: test, statistic, p, effect size, Levene p."""
    return (f"{result.test_name}: statistic={result.statistic:.6g} p={result.p_value:.6g} "
            f"{result.effect_size_name}={result.effect_size:.6g} levene_p={result.levene_p:.6g}")


def _groups_line(result: TestResult) -> str:
    """Return the group-count and group-sizes line."""
    return f"groups={result.n_groups} sizes={result.n_per_group}"


def _posthoc_lines(posthoc_frame, alpha: float) -> list:
    """Return the significant post-hoc rows (p_adj < alpha) and the total pairs tested."""
    if posthoc_frame is None or posthoc_frame.empty:
        return ["post-hoc: none"]
    significant = posthoc_frame[posthoc_frame["p_adj"] < alpha]
    rows = [f"  {r.group_a} vs {r.group_b}: statistic={r.statistic:.6g} p_adj={r.p_adj:.6g}"
            for r in significant.itertuples()]
    return [f"post-hoc significant pairs ({len(significant)} of {len(posthoc_frame)} tested):"] + rows
