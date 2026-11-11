"""Redraw the section 4 (an amplifier in one arm) report figure in the paper style.

Run from fibre_analysis/:
    uv run python tools/section4_figures.py

Writes ../figures/fig09 (SLD envelope with amplifier 1 in one arm, main run). Figures 10-13 were already made in
this style. The plotting is shared with section 3's SLD envelope (fig07), so it reuses tools/section3_figures.py.
Prints the fitted numbers so they can be checked against REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from section3_figures import _sld_fig
from section6_figures import paper_style


def main() -> None:
    paper_style()
    _sld_fig("amp1_sld_scan_main.csv", "fig09_sld_envelope_with_amplifier.png")


if __name__ == "__main__":
    main()
