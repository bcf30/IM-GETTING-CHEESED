"""MozzaVID microstructure dashboard: one Streamlit app with linked views over the results.

Launch from the repo root with the project env active:

    streamlit run app/dashboard.py

Step 1 of the build ships the Compare view only; later steps add Distributions, 3D
structure and the Fiji-backed slice inspector. The sidebar holds the shared selection that
links every view.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/dashboard.py` puts the script's own folder (app/) on sys.path, not the
# repo root, so the `app` package is not importable by default. Add the repo root (this
# file's parent's parent) so `from app import ...` resolves regardless of launch directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.views import compare  # noqa: E402

st.set_page_config(page_title="MozzaVID microstructure", layout="wide")


def _sidebar() -> list[str]:
    """Render the shared controls and return the selected cheese types."""
    st.sidebar.title("MozzaVID microstructure")
    st.sidebar.caption("Linked views over the committed segmentation results.")

    all_types = data.cheese_types()
    selected = st.sidebar.multiselect(
        "Cheese types",
        options=all_types,
        default=all_types,
        help="Filters the Compare and (later) Distributions views.",
    )

    sample = st.session_state.get("selected_sample")
    if sample:
        st.sidebar.success(f"Selected sample: {sample}")
        if st.sidebar.button("Clear selection"):
            st.session_state.pop("selected_sample", None)
            st.session_state.pop("selected_cheese_type", None)
            st.rerun()
    else:
        st.sidebar.info("Click a point in Compare to select a sample.")
    return selected


def main() -> None:
    """Route the sidebar selection into the active tab (Compare only in step 1)."""
    selected_types = _sidebar()
    st.title("Cheese microstructure explorer")
    st.markdown(data.OVERVIEW)
    tab_compare, = st.tabs(["Compare"])
    with tab_compare:
        compare.render(selected_types)


main()
