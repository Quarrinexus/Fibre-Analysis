"""Section 5 (an amplifier in each arm, FP laser only) analysis of the 2026-11-11 main run.

Run from fibre_analysis/:
    uv run python tools/section5_figures.py

Writes output/section5/section5_summary.csv, output/section5/amp2_fp_equal_gain_summary_main.png and the
report figure ../figures/fig14_amplifier_each_arm_contrast.png (paper style, matching fig11).

Prediction: with the arms matched in total intensity (signal + ASE), the Young limit is ~1 and the
contrast is limited by the ASE share, C_pred = S / (S + N) = 1 - N / I_total, where N is the
laser-off total at the same currents. Laser-off ASE overestimates the ASE with the laser on (gain
saturation), so C_pred is a lower bound on the ASE-only limit.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from section6_figures import minor, paper_style

from fibre_analysis.io import utf8_output

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE / "output" / "section5"
FIGS = HERE.parent / "figures"

# Matched pairs (I1, I2) in mA, from the FP arm maps (amp2_fp_arm*_main), each arm alone in V.
PAIRS = [
    (39.5, 42.0, 0.05),
    (51.0, 57.5, 0.2),
    (59.5, 69.0, 0.5),
    (67.0, 78.5, 1.0),
    (75.5, 90.5, 2.0),
    (81.0, 99.0, 3.0),
    (86.5, 106.5, 4.0),
]

# Section 4 (one amplifier, amplifier 1, FP 1.31 mW, on the revival): limit from REPORT §4.3.
S4_LIMIT = {
    30: 0.15,
    35: 0.23,
    40: 0.53,
    45: 0.69,
    50: 0.77,
    55: 0.73,
    60: 0.65,
    65: 0.56,
    70: 0.48,
    75: 0.41,
    80: 0.36,
}


def groups(name: str) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = {}
    with open(DATA / f"{name}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(float(row["x"]), []).append(row)
    return out


def main() -> None:
    utf8_output()
    OUT.mkdir(parents=True, exist_ok=True)
    both = groups("amp2_fp_both_main")
    ase = groups("amp2_ase_fp_pairs_main")
    rows = []
    for i1, i2, arm in PAIRS:
        c = [float(r["contrast"]) for r in both[i1]]
        tot = st.mean(float(r["mean"]) for r in both[i1])
        n = st.mean(float(r["mean"]) for r in ase[i1])
        pred = 1 - n / tot
        rows.append(
            dict(
                I1=i1,
                I2=i2,
                arm_V=arm,
                total_V=tot,
                ase_V=n,
                ase_share=n / tot,
                C_mean=st.mean(c),
                C_sd=st.stdev(c),
                C_sem=st.stdev(c) / len(c) ** 0.5,
                C_min=min(c),
                C_max=max(c),
                C_pred=pred,
                ratio=st.mean(c) / pred,
            )
        )
    with open(OUT / "section5_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['I1']:5.1f}/{r['I2']:5.1f}  total {r['total_V']:.3f} V  ASE {r['ase_V']:.3f} V "
            f"({100 * r['ase_share']:.0f}%)  C {r['C_mean']:.3f}±{r['C_sem']:.3f} (sd {r['C_sd']:.3f}, "
            f"max {r['C_max']:.3f})  pred {r['C_pred']:.2f}  C/pred {r['ratio']:.2f}"
        )

    s4 = groups("amp1_fp_both_balance_rev_main")
    x4 = sorted(s4)
    c4 = [st.mean(float(r["contrast"]) for r in s4[x]) for x in x4]
    e4 = [st.stdev([float(r["contrast"]) for r in s4[x]]) / len(s4[x]) ** 0.5 for x in x4]

    paper_style()
    fig, ax = plt.subplots()
    x = [r["I1"] for r in rows]
    ax.errorbar(
        x,
        [r["C_mean"] for r in rows],
        yerr=[r["C_sem"] for r in rows],
        fmt="o",
        color="C0",
        label="Two amplifiers, measured (mean of 10)",
    )
    ax.plot(x, [r["C_max"] for r in rows], "^", mfc="none", color="C0", label="Two amplifiers, highest capture")
    ax.plot(x, [r["C_pred"] for r in rows], "-", color="C0", label=r"Two amplifiers, ASE limit $S/(S+N)$")
    ax.errorbar(x4, c4, yerr=e4, fmt="s", color="C1", label="One amplifier (§4), measured")
    ax.plot(
        sorted(S4_LIMIT), [S4_LIMIT[k] for k in sorted(S4_LIMIT)], "--", color="C1", label="One amplifier (§4), limit"
    )
    ax.set_xlabel("Amplifier 1 current (mA)")
    ax.set_ylabel("Fringe contrast")
    ax.set_ylim(0, 1.3)
    minor(ax)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "amp2_fp_equal_gain_summary_main.png")
    fig.savefig(FIGS / "fig14_amplifier_each_arm_contrast.png")
    print("wrote", OUT / "amp2_fp_equal_gain_summary_main.png")


if __name__ == "__main__":
    main()
