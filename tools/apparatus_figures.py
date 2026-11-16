"""Apparatus diagrams for the report (replace the hand drawing and the TikZ macros).

Writes PNGs (600 dpi, used by REPORT.md and the LaTeX report):
  figures/fig01_setup_A.png            Setup A in its three configurations
                                       (a) section 3, (b) section 4, (c) section 5
  figures/fig01b_vibrometer_bench.png  the section 6 vibrometer bench (handout Fig. 5)

Run from fibre_analysis/:  uv run python tools/apparatus_figures.py

To change a configuration, edit CONFIGS below: each arm is a list of component names
drawn left to right after the splitter (see draw_component for the names).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]  # E1/
FIG_DIR = ROOT / "figures"

FIBRE: dict[str, Any] = dict(color="#1f4e9c", lw=1.3, solid_capstyle="round", zorder=1)
BEAM: dict[str, Any] = dict(color="#e8890c", lw=2.6, solid_capstyle="butt", zorder=1)
ELEC: dict[str, Any] = dict(color="0.35", lw=0.9, ls=(0, (3, 2)), zorder=1)
FS = 8  # label font size (pt) at the final 7 in width

# Setup A, per arm, in order from the splitter. Common to every configuration:
# air gap (delay line) -> [phase modulator, arm 2 only]. The one phase modulator is
# driven by the function generator (drawn below it). The single polariser is not in
# an arm: it sits after the combiner (drawn in setup_a_panel).
ARM1 = ["gap", None]
ARM2 = ["gap", "pm"]
CONFIGS = [
    ("(a)", "Section 3: plain interferometer", [None, *ARM1], [None, *ARM2]),
    ("(b)", "Section 4: amplifier in one arm", ["ext", *ARM1], ["amp1", *ARM2]),
    ("(c)", "Section 5: amplifier in each arm", ["amp2", *ARM1], ["amp1", *ARM2]),
]
# x positions of the component slots along each arm
SLOT_X = [5.0, 7.6, 10.0]


def style() -> None:
    plt.rcParams.update({"font.size": FS, "savefig.dpi": 600})


def box(ax, x, y, w, h, text=None, fs=FS, lw=1.0):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=lw, zorder=3))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def lens(ax, x, y, h=0.62):
    ax.add_patch(Ellipse((x, y), 0.18, h, fc="#dbe6f5", ec="black", lw=0.9, zorder=3))


def lens_pair(ax, x, y, half=0.42):
    """Two collimating lenses with a free-space beam between them (centred on x)."""
    ax.plot([x - half, x + half], [y, y], **BEAM)
    lens(ax, x - half, y)
    lens(ax, x + half, y)


def polariser(ax, x, y):
    box(ax, x, y, 0.62, 0.56)
    for dx in (-0.14, 0.0, 0.14):
        ax.plot([x + dx, x + dx], [y - 0.19, y + 0.19], color="black", lw=0.9, zorder=4)


def label(ax, x, y, side, text):
    """Label above (side=+1) or below (side=-1) a component on an arm at height y."""
    ax.text(
        x,
        y + side * 0.48,
        text,
        ha="center",
        va="bottom" if side > 0 else "top",
        fontsize=FS,
        linespacing=1.05,
        zorder=4,
    )


def draw_component(ax, name, x, y, side, gap_name):
    """Draw one in-arm component centred at (x, y). Returns its half-width."""
    if name is None:
        return 0.0
    if name == "gap":
        lens_pair(ax, x, y)
        # micrometer: double arrow under/over the beam
        ax.annotate(
            "",
            (x + 0.42, y - side * 0.42),
            (x - 0.42, y - side * 0.42),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.25", shrinkA=0, shrinkB=0),
        )
        label(ax, x, y, side, f"Air gap\n({gap_name})")
        return 0.52
    if name == "pm":
        box(ax, x, y, 1.05, 0.56)
        # sawtooth symbol
        xs = [x - 0.36, x - 0.12, x - 0.12, x + 0.12, x + 0.12, x + 0.36]
        ys = [y - 0.15, y + 0.15, y - 0.15, y + 0.15, y - 0.15, y + 0.15]
        ax.plot(xs, ys, color="black", lw=0.9, zorder=4)
        # label beside the drive line, and the function generator beyond the arm
        # label on the far side from the generator
        label(ax, x, y, -side, "Phase\nmodulator")
        gy = y + side * 1.4
        box(ax, x, gy, 1.75, 0.78, "Function\ngenerator", fs=FS - 0.5)
        ax.plot([x, x], [y + side * 0.28, gy - side * 0.39], **ELEC)
        return 0.53
    if name in ("amp1", "amp2"):
        box(ax, x, y, 1.15, 0.62)
        ax.add_patch(
            Polygon(
                [[x - 0.26, y - 0.2], [x - 0.26, y + 0.2], [x + 0.3, y]],
                closed=True,
                fc="white",
                ec="black",
                lw=1.0,
                zorder=4,
            )
        )
        label(ax, x, y, side, f"Amplifier {name[-1]}")
        return 0.58
    if name == "ext":
        ax.add_patch(
            Rectangle((x - 0.62, y - 0.36), 1.24, 0.72, fc="none", ec="0.4", lw=0.8, ls=(0, (2, 1.5)), zorder=2)
        )
        lens_pair(ax, x, y)
        label(ax, x, y, side, "Lens-based\nextender")
        return 0.62
    raise ValueError(name)


def setup_a_panel(ax, letter, title, arm1, arm2):
    y1, y2 = 1.3, -1.3
    # source, splitter, combiner, polariser, attenuator, detector
    box(ax, 0.85, 0, 1.5, 0.75, "FP/SLD")
    box(ax, 2.55, 0, 0.85, 0.5, "50:50", fs=FS - 1)
    box(ax, 12.25, 0, 0.85, 0.5, "50:50", fs=FS - 1)
    polariser(ax, 13.45, 0)
    ax.text(13.45, 0.36, "Polariser", ha="center", va="bottom", fontsize=FS)
    box(ax, 14.75, 0, 0.5, 0.32)
    ax.text(14.75, -0.38, "Attenuator", ha="center", va="top", fontsize=FS)
    box(ax, 16.3, 0, 1.5, 0.75, "Detector")
    ax.plot([1.6, 2.12], [0, 0], **FIBRE)
    ax.plot([12.68, 13.14], [0, 0], **FIBRE)
    ax.plot([13.76, 14.5], [0, 0], **FIBRE)
    ax.plot([15.0, 15.55], [0, 0], **FIBRE)

    for y, side, comps, gap in ((y1, 1, arm1, "M1"), (y2, -1, arm2, "M2")):
        sgn = 1 if y > 0 else -1
        # fibre fan-out from the splitter and fan-in to the combiner
        ax.plot([2.97, 3.3, 3.75], [sgn * 0.12, y, y], **FIBRE)
        ax.plot([11.0, 11.45, 11.83], [y, y, sgn * 0.12], **FIBRE)
        # straight fibre along the arm; components are drawn over it
        x_last = 3.75
        for name, x in zip(comps, SLOT_X, strict=True):
            hw = draw_component(ax, name, x, y, side, gap)
            if name is None:
                continue
            ax.plot([x_last, x - hw], [y, y], **FIBRE)
            x_last = x + hw
        ax.plot([x_last, 11.0], [y, y], **FIBRE)
        ax.text(
            3.15,
            y + side * 0.3,
            f"Arm {1 if y > 0 else 2}",
            ha="right",
            va="bottom" if side > 0 else "top",
            fontsize=FS,
            fontweight="bold",
        )

    ax.text(-0.1, 2.55, f"{letter} {title}", ha="left", va="center", fontsize=FS + 1)


def legend(ax, x, y):
    items = [("Fibre", FIBRE), ("Free-space beam", BEAM), ("Electrical", ELEC)]
    for dx, (text, st) in zip((0.0, 2.3, 6.2), items, strict=True):
        x0 = x + dx
        ax.plot([x0, x0 + 0.5], [y, y], **st)
        ax.text(x0 + 0.65, y, text, ha="left", va="center", fontsize=FS)


def setup_a_figure():
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.8))
    for ax, cfg in zip(axes, CONFIGS, strict=True):
        setup_a_panel(ax, *cfg)
        ax.set_xlim(-0.2, 17.15)
        ax.set_ylim(-3.25, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")
    legend(axes[0], 8.0, 2.55)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, hspace=0.05)
    return fig


def vibrometer_figure():
    fig, ax = plt.subplots(figsize=(7, 3.0))
    yb, yt = 0.0, 1.9  # bottom row (reference) and top row (circulator)

    box(ax, 0.85, yb, 1.5, 0.75, "FP laser\n(unpolarised)")
    box(ax, 3.0, yb, 0.85, 0.5, "90:10", fs=FS - 1)
    box(ax, 6.4, yb, 0.85, 0.5, "90:10", fs=FS - 1)
    box(ax, 8.9, yb, 1.3, 0.75, "Detector")
    ax.plot([1.6, 2.57], [yb, yb], **FIBRE)

    # 90 % path: up to the circulator
    ax.plot([3.0, 3.0, 4.95], [yb + 0.25, yt, yt], **FIBRE)
    ax.text(2.9, 0.75, "90%", ha="right", va="center", fontsize=FS)
    # 10 % path: straight to the second splitter
    ax.plot([3.43, 5.97], [yb, yb], **FIBRE)
    ax.text(4.0, yb - 0.12, "10%", ha="center", va="top", fontsize=FS)

    # circulator
    cx, r = 5.4, 0.42
    ax.add_patch(Circle((cx, yt), r, fc="white", ec="black", lw=1.0, zorder=3))
    # clockwise arrow: 1 -> 2 -> 3
    ra = 0.24
    ax.add_patch(Arc((cx, yt), 2 * ra, 2 * ra, theta1=-50, theta2=200, lw=0.9, zorder=4))
    tip, tail = np.radians(-72), np.radians(-50)
    ax.annotate(
        "",
        (cx + ra * np.cos(tip), yt + ra * np.sin(tip)),
        (cx + ra * np.cos(tail), yt + ra * np.sin(tail)),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", mutation_scale=7, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.text(cx, yt + r + 0.08, "Circulator", ha="center", va="bottom", fontsize=FS)
    ax.text(cx - r - 0.06, yt + 0.06, "1", ha="right", va="bottom", fontsize=FS - 1)
    ax.text(cx + r + 0.06, yt + 0.06, "2", ha="left", va="bottom", fontsize=FS - 1)
    ax.text(cx + 0.06, yt - r - 0.06, "3", ha="left", va="top", fontsize=FS - 1)

    # port 3 down to the second splitter
    ax.plot([cx, cx, 6.4, 6.4], [yt - r, 1.0, 1.0, yb + 0.25], **FIBRE)
    ax.text(6.5, 0.65, "90%", ha="left", va="center", fontsize=FS)
    ax.text(5.95, yb - 0.12, "10%", ha="right", va="top", fontsize=FS)
    ax.plot([6.83, 8.25], [yb, yb], **FIBRE)

    # port 2: fibre to the collimating lens, free space to the mirror and back
    xl, xm = 8.2, 10.6
    ax.plot([cx + r, xl - 0.09], [yt, yt], **FIBRE)
    ax.plot([xl, xm], [yt, yt], **BEAM)
    ax.annotate(
        "",
        (xl + 0.6, yt + 0.16),
        (xm - 0.4, yt + 0.16),
        arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="0.25", mutation_scale=7),
    )
    lens(ax, xl, yt)
    ax.text(xl, yt + 0.4, "Lens", ha="center", va="bottom", fontsize=FS)
    # mirror on a loudspeaker
    ax.add_patch(Rectangle((xm, yt - 0.42), 0.09, 0.84, fc="0.55", ec="black", lw=0.8, zorder=3))
    ax.add_patch(
        Polygon(
            [[xm + 0.09, yt - 0.42], [xm + 0.09, yt + 0.42], [xm + 0.6, yt + 0.2], [xm + 0.6, yt - 0.2]],
            closed=True,
            fc="white",
            ec="black",
            lw=0.9,
            zorder=3,
        )
    )
    ax.add_patch(Rectangle((xm + 0.6, yt - 0.2), 0.35, 0.4, fc="0.8", ec="black", lw=0.9, zorder=3))
    ax.text(xm + 0.45, yt + 0.55, "Mirror on\nloudspeaker", ha="center", va="bottom", fontsize=FS, linespacing=1.05)
    ax.annotate(
        "",
        (xm + 0.35, yt - 0.6),
        (xm - 0.25, yt - 0.6),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="black", shrinkA=0, shrinkB=0),
    )
    ax.text(xm + 0.05, yt - 0.7, r"$\Delta x$", ha="center", va="top", fontsize=FS)
    ax.text(9.4, yt - 0.22, "double pass", ha="center", va="top", fontsize=FS - 1, style="italic")

    # speaker drive
    box(ax, 11.4, yb, 1.6, 0.75, "Function\ngenerator")
    ax.plot([11.4, 11.4, xm + 0.95], [yb + 0.375, yt, yt], **ELEC)

    ax.set_xlim(-0.2, 12.4)
    ax.set_ylim(-0.95, 3.15)
    ax.set_aspect("equal")
    ax.axis("off")
    items = [("Fibre", FIBRE), ("Free-space beam", BEAM), ("Electrical", ELEC)]
    for i, (text, st) in enumerate(items):
        x0 = i * 3.0
        ax.plot([x0, x0 + 0.7], [-0.75, -0.75], **st)
        ax.text(x0 + 0.85, -0.75, text, ha="left", va="center", fontsize=FS)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def save(fig, stem):
    for ext in ("png",):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        print("wrote", out)
    plt.close(fig)


def main():
    style()
    save(setup_a_figure(), "fig01_setup_A")
    save(vibrometer_figure(), "fig01b_vibrometer_bench")


if __name__ == "__main__":
    main()
