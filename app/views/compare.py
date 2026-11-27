"""Compare view: per-cheese-type distribution of each stat metric, with click-to-jump.

Renders one strip plot per metric (25 cheese types side by side, every sample a point),
annotated with the Kruskal-Wallis result parsed from `anova_results.txt`. Clicking a point
selects its sample and cheese type into shared session state so other views can follow.

A strip plot (not a box plot) is used so every drawn point is an individual marker that
carries its row index as `customdata`; Plotly box-plot points do not emit a reliable
per-point identity through the selection event, which made click-to-jump silently fail.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app import data


def render(selected_types: list[str]) -> None:
    """Draw the four per-type strip plots for the currently selected cheese types."""
    per_sample = data.load_per_sample()
    stats = data.load_metric_stats()
    if selected_types:
        per_sample = per_sample[per_sample["cheese_type"].isin(selected_types)]

    if per_sample.empty:
        st.info("No samples match the current cheese-type selection.")
        return

    # One stable positional index per row so a clicked point maps back to its sample.
    plot_df = per_sample.sort_values("cheese_type").reset_index(drop=True)

    st.markdown(
        "**How each cheese type compares on four microstructure metrics.** "
        "Each point is one cheese sample; click a point to select it, and the sidebar "
        "(and later views) will follow. Expand **What this measures** under any chart for "
        "a plain description of the metric."
    )
    with st.expander("How to read the statistics on each chart"):
        st.markdown(data.STATS_NOTE)

    for metric in data.STAT_METRICS:
        _render_metric(plot_df, metric, stats.get(metric))


def _render_metric(plot_df, metric, metric_stats) -> None:
    """Draw one metric's strip plot, its description and Kruskal-Wallis annotation; wire click-to-jump."""
    info = data.METRIC_INFO[metric]
    label = info["label"]
    st.subheader(label)
    st.caption(info["what"])
    with st.expander("What this measures"):
        st.markdown(info["how"])

    if metric_stats is not None:
        st.caption(
            f"Kruskal-Wallis H = {metric_stats.statistic:.1f}, "
            f"p = {metric_stats.p_value:.2e}, "
            f"epsilon² = {metric_stats.epsilon_squared:.3f} "
            f"(across {metric_stats.n_groups} cheese types)"
        )

    fig = px.strip(
        plot_df,
        x="cheese_type",
        y=metric,
        custom_data=["sample_id"],
        hover_data=["sample_id"],
        labels={"cheese_type": "Cheese type", metric: label},
    )
    fig.update_traces(jitter=0.3, marker=dict(size=7, opacity=0.75))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)

    event = st.plotly_chart(
        fig,
        key=f"compare_{metric}",
        on_select="rerun",
        selection_mode="points",
    )
    _apply_selection(event)


def _apply_selection(event) -> None:
    """If the user clicked a point, write its sample/type into shared session state.

    The strip trace carries `sample_id` as the point's first customdata entry, so the
    clicked point identifies its sample directly, with no value-matching. `on_select="rerun"`
    already reruns the script to deliver this event, so no explicit `st.rerun()` is needed
    (a second rerun here would loop).
    """
    selection = (event or {}).get("selection", {})
    points = selection.get("points", [])
    if not points:
        return
    customdata = points[0].get("customdata")
    if not customdata:
        return
    sample_id = customdata[0]
    cheese_type = points[0].get("x")
    if st.session_state.get("selected_sample") == sample_id:
        return
    st.session_state["selected_sample"] = sample_id
    st.session_state["selected_cheese_type"] = cheese_type
    st.toast(f"Selected sample {sample_id}")
