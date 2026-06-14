"""
MAESTRO viz: render Mermaid source to SVG via the mmdc CLI.

The dashboard renders diagrams with the *same* engine the metric pipeline
uses to compute ``parses_valid`` (mmdc, see analysis/metrics.py), so the
picture a viewer sees is consistent with the validity the data records. This
matters for a thesis artifact: a second, in-browser renderer could disagree
with mmdc and produce a diagram that looks fine but was scored invalid (or
vice versa). mmdc is also deterministic and version-pinnable, so a rendered
figure is reproducible.

When mmdc is not installed the renderer returns ``None`` and the caller falls
back to showing the source: the honest behavior is to not fabricate a render
we cannot produce. Requires: ``npm install -g @mermaid-js/mermaid-cli``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def mmdc_available() -> bool:
    """Whether the mmdc CLI is on PATH."""
    return shutil.which("mmdc") is not None


def render_mermaid_svg(diagram_code: str, *, timeout: int = 15) -> str | None:
    """
    Render ``diagram_code`` to an SVG string via mmdc, or ``None`` on any
    failure (mmdc missing, invalid source, timeout). Callers treat ``None`` as
    "show the source instead".

    Unlike the validate-only path in metrics.py, this reads the produced SVG
    back. Input is written to a temp ``.mmd`` file (not ``/dev/stdin``) so the
    input side is not Unix-only; the output temp file is created in a temp dir
    and read after mmdc exits. Empty/blank source short-circuits to ``None``.
    """
    mmdc = shutil.which("mmdc")
    if mmdc is None or not diagram_code or not diagram_code.strip():
        return None

    # Forward a Puppeteer launch config when one is configured (mirrors
    # metrics.check_mermaid_valid). In a container running as root, Chromium
    # needs --no-sandbox, which mmdc only honours from a -p config file, not an
    # env var. Locally MERMAID_PUPPETEER_CONFIG is unset and mmdc uses its
    # default; without this, in-container renders would silently return None.
    puppeteer_args: list[str] = []
    puppeteer_config = os.environ.get("MERMAID_PUPPETEER_CONFIG")
    if puppeteer_config and Path(puppeteer_config).is_file():
        puppeteer_args = ["-p", puppeteer_config]

    try:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "in.mmd"
            out_path = Path(tmp) / "out.svg"
            in_path.write_text(diagram_code, encoding="utf-8")
            result = subprocess.run(
                [
                    mmdc,
                    *puppeteer_args,
                    "-i",
                    str(in_path),
                    "-o",
                    str(out_path),
                    "-e",
                    "svg",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 or not out_path.exists():
                return None
            return out_path.read_text(encoding="utf-8")
    except (subprocess.TimeoutExpired, OSError):
        return None
