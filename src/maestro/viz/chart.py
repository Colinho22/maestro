"""
MAESTRO viz — matplotlib figure helpers for the dashboard.

Two pieces every view uses:

- ``new_figure`` — create a ``(fig, ax)`` with the house style applied, so a
  view never has to remember to call ``apply_thesis_style`` itself.
- ``render_chart`` — display a figure in Streamlit and offer PNG + SVG
  downloads (publication-quality vector for the thesis, raster for slides).
  It closes the figure afterwards: matplotlib keeps figures alive globally,
  so a long-running dashboard that forgot to close them would leak memory.

Export uses an in-memory buffer (no temp files), so it works regardless of
the host OS or filesystem.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.figure import Figure

from maestro.viz.theme import apply_thesis_style


def new_figure(
    *,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[Figure, plt.Axes]:
    """
    Return a themed ``(fig, ax)``. Applies the house style first (idempotent),
    so callers get consistent typography/spines/grid without extra setup.

    Default size is the design guide's "tall vertical bar" (7 × 4.5, ~1.6:1).
    Views pick a size from the guide's figure-dimensions table to match their
    chart type (e.g. 8 × 5 for Pareto scatter, 10 × 4.5 for wide grouped bars).
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _savefig_bytes(fig: Figure, fmt: str) -> bytes:
    """Render ``fig`` to ``fmt`` (png/svg) bytes via an in-memory buffer."""
    buf = io.BytesIO()
    # bbox_inches='tight' trims surrounding whitespace so exported figures
    # embed cleanly without manual cropping. dpi applies only to raster (PNG):
    # the guide's 200 DPI for slides/previews. SVG is vector, where dpi is a
    # no-op, so it is omitted to avoid implying otherwise.
    kwargs: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        kwargs["dpi"] = 200
    fig.savefig(buf, format=fmt, **kwargs)
    return buf.getvalue()


def render_chart(
    fig: Figure,
    *,
    filename: str,
    key: str,
    caption: str | None = None,
) -> None:
    """
    Display ``fig`` in Streamlit with PNG + SVG download buttons, then close
    it.

    ``filename`` is the download stem (no extension); ``key`` must be unique
    per chart on a page (Streamlit requires distinct widget keys). ``caption``
    is shown under the figure if given.

    ``use_container_width=True`` makes the chart scale to its container rather
    than render at the figure's native inches — so a wide-layout page does
    not let the figure dictate the overall column width. Views can wrap this
    call in an ``st.columns`` block to bound the chart's region further.
    """
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)

    png = _savefig_bytes(fig, "png")
    svg = _savefig_bytes(fig, "svg")

    col_png, col_svg = st.columns(2)
    with col_png:
        st.download_button(
            "Download PNG",
            data=png,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"{key}-png",
        )
    with col_svg:
        st.download_button(
            "Download SVG",
            data=svg,
            file_name=f"{filename}.svg",
            mime="image/svg+xml",
            key=f"{key}-svg",
        )

    # Release the figure — matplotlib holds a global reference otherwise.
    plt.close(fig)
