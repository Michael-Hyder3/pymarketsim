"""Holdout table, paired differences and unscreened OOD margins, plus the LaTeX rows for both tables.

usage: python reproduce/stats.py
"""
from __future__ import annotations

import json

from common import RESULTS, mean_se

AGENTS = ["Ours", "ZI", "ZIP", "Cs", "HBL"]
MATCHED = {"pv": "ZI", "sp": "ZIP", "lob": "HBL"}
SCREEN = {"pv": 500, "sp": 500, "lob": 1000}
REG = {"pv": r"$\PV$", "sp": r"$\mathrm{Sp}$", "lob": r"$\LOB$"}
T1 = ["Ours", "ZI", "ZIP", "HBL", "Cs"]
PD = ["ZI", "ZIP", "HBL", "Cs"]


def load(regime, holdout):
    pe = json.loads((RESULTS / f"{regime}_{holdout}.json").read_text())["per_episode"]
    pe["Cs"] = json.loads((RESULTS / f"cs_{regime}.json").read_text())["per_episode"][holdout]
    return pe


def row_stats(regime, holdout):
    pe = load(regime, holdout)
    labels = sorted(pe["Ours"])
    row = {}
    for a in AGENTS:
        m, se = mean_se([pe[a][l] for l in labels])
        e = {"mean": m, "se": se}
        if a != "Ours":
            dm, dse = mean_se([pe["Ours"][l] - pe[a][l] for l in labels])
            e.update(diff=dm, ci=1.96 * dse, pct=100 * dm / m)
            if holdout == "ood":
                rest = labels[SCREEN[regime]:]
                om, _ = mean_se([pe["Ours"][l] for l in rest])
                am, _ = mean_se([pe[a][l] for l in rest])
                e["unscreened_pct"] = 100 * (om - am) / am
        row[a] = e
    top = max(AGENTS, key=lambda a: row[a]["mean"])
    sig = True
    for a in AGENTS:
        if a != top:
            dm, dse = mean_se([pe[top][l] - pe[a][l] for l in labels])
            sig &= dm - 1.96 * dse > 0
    row["bold"] = top if sig else None
    return row


def mark(cell, agent, regime, bold_agent):
    if bold_agent == agent:
        return r"\mathbf{" + cell + "}"
    if agent == MATCHED[regime]:
        return r"\underline{" + cell + "}"
    return cell


def main() -> None:
    out = {}
    t1, pd = [], []
    for r in ("pv", "sp", "lob"):
        out[r] = {"Cs_shade": json.loads((RESULTS / f"cs_{r}.json").read_text())["best"]}
        for h in ("ood", "id"):
            row = out[r][h] = row_stats(r, h)
            print(r, h, "  ".join(f"{a}:{row[a]['mean']:.1f}+-{row[a]['se']:.1f}" for a in T1))
            cells = ["$" + mark("%.1f\\pm %.1f" % (row[a]["mean"], row[a]["se"]), a, r, row["bold"]) + "$" for a in T1]
            t1.append("    %s & %s\n      & " % (REG[r], h.upper()) + "\n      & ".join(cells) + r" \\")
            cells = ["$" + mark("%.1f\\pm%.1f" % (row[a]["diff"], row[a]["ci"]), a, r, None) + "$" for a in PD]
            pd.append("    %s & %s\n      & " % (REG[r], h.upper()) + " & ".join(cells) + r" \\")
        t1.append(r"    \midrule")
        pd.append(r"    \midrule")
    (RESULTS / "final_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    (RESULTS / "table1_rows.tex").write_text("\n".join(t1[:-1]) + "\n")
    (RESULTS / "paired_rows.tex").write_text("\n".join(pd[:-1]) + "\n")
    print("wrote final_stats.json, table1_rows.tex, paired_rows.tex")


if __name__ == "__main__":
    main()
