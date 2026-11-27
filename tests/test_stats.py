"""Tests for mozzavid.stats (Requirements 15-17; design Properties 11, 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from mozzavid import stats


def _frame(groups: dict, metric: str = "foreground_fraction") -> pd.DataFrame:
    """Build a per-sample frame from {cheese_type: values}."""
    rows = [{"cheese_type": name, metric: v} for name, vals in groups.items() for v in vals]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(8))
def test_equal_variance_selects_anova_and_tukey(seed):
    """Property 11: equal-variance normal groups select ANOVA + Tukey (Requirement 15.3, 16.1)."""
    rng = np.random.default_rng(seed)
    groups = {f"cheese_{i}": rng.normal(i, 1.0, size=20) for i in range(4)}
    result = stats.compare(_frame(groups), "foreground_fraction", alpha=0.001)
    assert result.test_name == "one-way ANOVA"
    assert result.effect_size_name == "eta_squared"
    post = stats.posthoc(_frame(groups), "foreground_fraction", result)
    assert set(post.columns) == {"group_a", "group_b", "statistic", "p_raw", "p_adj"}


@pytest.mark.parametrize("seed", range(8))
def test_unequal_variance_selects_kruskal_and_dunn(seed):
    """Property 11: one 10x-variance group selects Kruskal-Wallis + Dunn-Holm (Req 15.4, 16.2)."""
    rng = np.random.default_rng(seed)
    groups = {f"cheese_{i}": rng.normal(0, 1.0, size=10) for i in range(3)}
    groups["cheese_3"] = rng.normal(0, 10.0, size=10)  # inflated variance trips Levene
    result = stats.compare(_frame(groups), "foreground_fraction")
    assert result.test_name == "Kruskal-Wallis"
    assert result.effect_size_name == "epsilon_squared"
    assert result.effect_size >= 0.0
    post = stats.posthoc(_frame(groups), "foreground_fraction", result)
    assert (post["p_adj"] >= post["p_raw"] - 1e-12).all()
    assert (post["p_adj"] <= 1.0 + 1e-12).all()


def test_compare_raises_for_undersized_group():
    """compare raises ValueError naming an undersized group (Requirement 15.7)."""
    groups = {"a": [1.0, 2.0, 3.0], "b": [5.0]}
    with pytest.raises(ValueError, match="foreground_fraction"):
        stats.compare(_frame(groups), "foreground_fraction")


def test_eta_squared_hand_computed():
    """eta^2 matches SS_between / SS_total for a worked example (Requirement 15.3)."""
    groups = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    # grand mean 3.5; SS_between = 3*(2-3.5)^2 + 3*(5-3.5)^2 = 13.5; SS_total = 17.5
    assert stats._eta_squared(groups) == pytest.approx(13.5 / 17.5)


def test_epsilon_squared_hand_computed_and_clip():
    """epsilon^2 matches H*(n+1)/(n^2-1) and clips negatives to 0 (Requirement 15.4, 15.6)."""
    groups = [np.zeros(5), np.ones(5)]
    assert stats._epsilon_squared(4.0, groups) == pytest.approx(4.0 * 11 / 99)
    assert stats._epsilon_squared(-1.0, groups) == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_holm_bounds_and_monotonicity(seed):
    """Property 11: Holm p_adj >= p_raw, <= 1, monotone in sorted order (Requirement 16.4)."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, size=10)
    frame = pd.DataFrame({"group_a": range(10), "group_b": range(10, 20),
                          "statistic": raw, "p_raw": raw, "p_adj": np.nan})
    out = stats._holm(frame)
    assert (out["p_adj"] >= out["p_raw"] - 1e-12).all()
    assert (out["p_adj"] <= 1.0 + 1e-12).all()
    assert (np.diff(out["p_adj"].to_numpy()) >= -1e-12).all()


def test_holm_known_vector():
    """Holm adjustment matches a hand-computed vector (Requirement 16.4, 25.8)."""
    raw = np.array([0.01, 0.02, 0.03, 0.04])
    frame = pd.DataFrame({"group_a": [0, 1, 2, 3], "group_b": [4, 5, 6, 7],
                          "statistic": raw, "p_raw": raw, "p_adj": np.nan})
    out = stats._holm(frame).sort_values("p_raw")
    # multipliers 4,3,2,1 then cumulative max: 0.04, 0.06, 0.06, 0.06
    assert out["p_adj"].tolist() == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_dunn_hand_worked_three_groups_with_ties():
    """Dunn z and p match a hand-worked 3-group example with ties (Requirement 16.2, 25.8)."""
    named = {"a": np.array([1.0, 2.0]), "b": np.array([2.0, 3.0]), "c": np.array([5.0, 6.0])}
    out = stats._dunn(named)
    # pooled values [1,2,2,3,5,6] -> ranks [1,2.5,2.5,4,5,6]; mean ranks a=1.75,b=3.25,c=5.5
    total = 6
    ties = (2**3 - 2) / (12 * (total - 1))  # one tie group of size 2
    sigma = np.sqrt((total * (total + 1) / 12 - ties) * (1 / 2 + 1 / 2))
    expected_ab = (1.75 - 3.25) / sigma
    row_ab = out[(out["group_a"] == "a") & (out["group_b"] == "b")].iloc[0]
    assert row_ab["statistic"] == pytest.approx(expected_ab)
    assert row_ab["p_raw"] == pytest.approx(2 * scipy.stats.norm.sf(abs(expected_ab)))


# --- ICC (Property 15) -------------------------------------------------------------------


def _per_volume(values_by_sample: dict, metric: str = "foreground_fraction") -> pd.DataFrame:
    """Build a per-volume frame from {sample_id: [values]}."""
    rows = [{"sample_id": s, "volume_id": f"{s}/{i}", metric: v}
            for s, vals in values_by_sample.items() for i, v in enumerate(vals)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(6))
def test_icc_row_order_invariant_and_bounded(seed):
    """Property 15: icc1 is row-order invariant and <= 1 (Requirement 17.4)."""
    rng = np.random.default_rng(seed)
    samples = {f"s{i}": rng.normal(i, 0.5, size=rng.integers(2, 5)) for i in range(5)}
    frame = _per_volume({k: v.tolist() for k, v in samples.items()})
    base = stats.icc1(frame, "foreground_fraction")
    shuffled = stats.icc1(frame.sample(frac=1.0, random_state=seed), "foreground_fraction")
    assert base == pytest.approx(shuffled)
    assert base <= 1.0 + 1e-12


def test_icc_unbalanced_hand_computed():
    """icc1 matches a hand-computed unbalanced 3-sample example (Requirement 17.4)."""
    frame = _per_volume({"a": [1.0, 2.0], "b": [4.0, 5.0, 6.0], "c": [9.0, 10.0, 11.0, 12.0]})
    value = stats.icc1(frame, "foreground_fraction")
    expected = _reference_icc1([[1.0, 2.0], [4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]])
    assert value == pytest.approx(expected)


def _reference_icc1(groups: list) -> float:
    """Independent reference ICC(1) implementation for the hand-computed test."""
    arrs = [np.array(g) for g in groups]
    sizes = np.array([a.size for a in arrs])
    k, total = len(arrs), int(sizes.sum())
    grand = np.concatenate(arrs).mean()
    msb = sum(n * (a.mean() - grand) ** 2 for n, a in zip(sizes, arrs)) / (k - 1)
    msw = sum(((a - a.mean()) ** 2).sum() for a in arrs) / (total - k)
    n0 = (total - (sizes**2).sum() / total) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def test_icc_identical_within_sample_near_one():
    """icc1 is approximately 1 when within-sample values are identical (Requirement 17.4)."""
    frame = _per_volume({"a": [1.0, 1.0], "b": [5.0, 5.0], "c": [9.0, 9.0]})
    assert stats.icc1(frame, "foreground_fraction") == pytest.approx(1.0)


def test_icc_all_identical_returns_nan():
    """icc1 returns NaN when all values are identical (Requirement 17.4)."""
    frame = _per_volume({"a": [3.0, 3.0], "b": [3.0, 3.0]})
    assert np.isnan(stats.icc1(frame, "foreground_fraction"))


def test_icc_raises_with_fewer_than_two_qualifying_samples():
    """icc1 raises ValueError with fewer than 2 samples that have >= 2 values (Requirement 17.7)."""
    frame = _per_volume({"a": [1.0, 2.0], "b": [5.0]})
    with pytest.raises(ValueError):
        stats.icc1(frame, "foreground_fraction")


def test_format_report_marks_insufficient_data():
    """format_report writes 'insufficient data' for a None result (Requirement 17.1)."""
    report = stats.format_report({"foreground_fraction": None}, {})
    assert "insufficient data" in report
