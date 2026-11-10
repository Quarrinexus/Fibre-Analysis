"""Trace the manufacturer spectra off the scanned Setup A data sheets.

    uv run python tools/digitise_datasheets.py ../Setup_A_manuals_v2.pdf

Needs `pdftoppm` (poppler) to render the PDF pages at 300 dpi.

For each plot:
1. find the evenly spaced gridlines, and map pixels to data with a straight-line
   fit through them (their known values are below);
2. keep only the curve's blue pixels, which drops the gridlines, text and the pen
   marks someone drew on the SLD plot;
3. take the top edge of the line in each pixel column and undo the line's
   thickness by a disk erosion (keeps narrow peaks sharp), interpolating across
   short gaps where pen marks cross the curve.

Writes src/fibre_analysis/datasheets/{fp,sld}_spectrum.csv and, for checking by
eye, output/datasheets/datasheet_trace_{fp,sld}.png with the trace drawn over the scan.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import grey_erosion
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "fibre_analysis" / "datasheets"


@dataclass
class Plot:
    name: str
    page: int
    box: tuple[int, int, int, int]  # crop at 300 dpi: left, top, right, bottom
    x_values: np.ndarray  # data values of the vertical gridlines, left to right
    y_values: np.ndarray  # data values of the horizontal gridlines, top to bottom
    x_label: str
    y_label: str
    source: str


PLOTS = [
    Plot(
        "fp",
        6,
        (560, 480, 1780, 1420),
        np.arange(1530, 1571, 10.0),
        np.arange(1.0, -0.01, -0.2),
        "wavelength_nm",
        "intensity_norm",
        "LPS-PM1550-FC-1 S/N 230906-39 data sheet, '230906-39 Spectrum' "
        "(Tcase=25 C, Iop=24.1 mA, Yokogawa AQ6370), PDF page 6",
    ),
    Plot(
        "sld",
        2,
        (300, 1760, 1720, 2680),
        np.arange(1425, 1666, 20.0),
        np.arange(0.0, -46.0, -5.0),
        "wavelength_nm",
        "power_db",
        "SLD1005 S/N SLD-16063 data sheet, Figure 2 'SLED Spectrum' (I=650 mA), PDF page 2",
    ),
]


def render(pdf: Path, page: int, folder: Path) -> np.ndarray:
    stem = folder / f"p{page}"
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-f", str(page), "-l", str(page), str(pdf), str(stem)], check=True)
    png = next(folder.glob(f"p{page}*.png"))
    return np.asarray(Image.open(png).convert("RGB")).astype(int)


def masks(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > r + 40) & (b > g + 15)
    grid = (np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (r < 235) & ~blue
    return blue, grid


def grid_lines(profile: np.ndarray, n: int, tol: float = 6) -> tuple[np.ndarray, float]:
    """Positions of n evenly spaced gridlines, from peaks in a pixel-count profile."""
    peaks, _ = find_peaks(profile, height=0.2 * profile.max(), distance=15)
    best = None
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            for k in range(1, n):
                d = (peaks[j] - peaks[i]) / k
                if d < 15:
                    continue
                for s in range(n):
                    pred = peaks[i] - s * d + d * np.arange(n)
                    hits = np.array([np.min(np.abs(peaks - p)) < tol for p in pred])
                    if best is None or hits.sum() > best[0].sum():
                        best = (hits, pred)
    if best is None:
        raise RuntimeError("no gridlines found")
    hits, pred = best
    if hits.sum() < max(3, n - 2):
        raise RuntimeError(f"only {hits.sum()} of {n} gridlines found")
    idx = np.flatnonzero(hits)
    found = np.array([peaks[np.argmin(np.abs(peaks - pred[i]))] for i in idx])
    slope, icpt = np.polyfit(idx, found, 1)
    fit = icpt + slope * np.arange(n)
    return fit, float(np.std(found - fit[idx]))


def trace(blue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curve centre row for each column; gaps interpolated.

    The printed line is the true curve drawn with a round pen of radius r, so its
    top edge is the curve dilated by a disk. Eroding the top edge with the same disk
    gives the curve back: peaks are kept exactly (narrow spikes aren't widened or
    flattened), and only valleys narrower than the pen get filled."""
    cols = np.flatnonzero(blue.any(axis=0))
    tops = np.array([np.argmax(blue[:, c]) for c in cols], dtype=float)
    x = np.arange(cols.min(), cols.max() + 1)
    top = np.interp(x, cols, tops)
    # Pen radius from the line thickness where the curve is flat.
    runs = np.array([blue[:, c].sum() for c in cols])
    flat = np.abs(np.gradient(tops)) < 0.5
    r = 0.5 * float(np.median(runs[flat])) if flat.any() else 0.0
    k = int(np.ceil(r))
    dx = np.arange(-k, k + 1)
    disk = np.sqrt(np.clip(r**2 - dx**2, 0, None))
    height = -top  # up is positive
    eroded = grey_erosion(height, structure=disk, mode="nearest")
    return x.astype(float), -eroded - 0.5


def digitise(pdf: Path, plot: Plot, tmp: Path) -> np.ndarray:
    page = render(pdf, plot.page, tmp)
    left, top, right, bottom = plot.box
    a = page[top:bottom, left:right]
    blue, grid = masks(a)
    gx, ex = grid_lines(grid.sum(axis=0), len(plot.x_values))
    gy, ey = grid_lines(grid.sum(axis=1), len(plot.y_values))
    cx, cy = trace(blue)
    # Pixel -> data by straight-line fits through the gridlines.
    kx, bx = np.polyfit(gx, plot.x_values, 1)
    ky, by = np.polyfit(gy, plot.y_values, 1)
    data = np.column_stack([kx * cx + bx, ky * cy + by])
    print(
        f"{plot.name}: gridlines fitted to {ex:.1f} px (x) and {ey:.1f} px (y); "
        f"1 px = {abs(kx):.3f} nm, {abs(ky):.4f} in y; {len(data)} points"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(a.shape[1] / 150, a.shape[0] / 150), dpi=150)
    ax.imshow(a.astype(np.uint8))
    ax.plot(cx, cy, "r-", lw=0.6, label="trace")
    for x in gx:
        ax.axvline(x, color="lime", lw=0.4)
    for y in gy:
        ax.axhline(y, color="lime", lw=0.4)
    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    check = ROOT / "output" / "datasheets" / f"datasheet_trace_{plot.name}.png"
    check.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(check)
    plt.close(fig)
    return data


def main() -> None:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "Setup_A_manuals_v2.pdf")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for plot in PLOTS:
            data = digitise(pdf, plot, Path(tmp))
            out = OUT_DIR / f"{plot.name}_spectrum.csv"
            header = (
                f"Digitised from {plot.source}.\n"
                "Traced from a 300 dpi render of a scanned page by tools/digitise_datasheets.py;\n"
                "accuracy about +-1 px (see the printed scale). Not a measurement made in this practical.\n"
                f"{plot.x_label},{plot.y_label}"
            )
            np.savetxt(out, data, delimiter=",", fmt="%.4f", header=header, comments="# ", encoding="utf-8")
            print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
