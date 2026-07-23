"""
Data-driven replacements for the decorative/filler figures, so every figure is
relevant to the project context. Reads the processed Parquet with pandas
(no Spark needed). Outputs to docs/diagrams/.

Run:  .venv/bin/python src/make_relevant_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.utils import PROCESSED_PARQUET, DOCS

DG = DOCS / "diagrams"


def _save(fig, name):
    fig.savefig(DG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}")


def data_scale_card():
    """Intro figure: dataset-at-a-glance infographic (real project stats)."""
    tiles = [
        ("1,294,129", "raw incident records", "#1f4e79"),
        ("1,167,832", "cleaned records", "#2e75b6"),
        ("21", "source columns", "#117864"),
        ("295 MB", "raw CSV size", "#8e44ad"),
        ("2015–2026", "time span", "#c0392b"),
        ("219", "bus operators", "#d68910"),
        ("6", "ML models compared", "#16a085"),
        ("CC0", "open licence", "#2c3e50"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("NYC Bus Breakdown & Delays — Dataset at a Glance",
                 fontsize=14, fontweight="bold", color="#1a1a1a", pad=12)
    for i, (big, small, color) in enumerate(tiles):
        r, c = divmod(i, 4)
        x, y = c, 1 - r
        ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), 0.88, 0.82,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     facecolor=color, edgecolor="none"))
        ax.text(x + 0.5, y + 0.58, big, ha="center", va="center",
                color="white", fontsize=19, fontweight="bold")
        ax.text(x + 0.5, y + 0.26, small, ha="center", va="center",
                color="white", fontsize=9.5, alpha=0.95)
    _save(fig, "fig_intro_datascale.png")


def problem_incidents(df):
    """Problem-statement figure: real scale of breakdowns & delays over time."""
    d = df[(df["year"] >= 2015) & (df["year"] <= 2026)]
    piv = (d.assign(t=d["incident_type"])
             .groupby(["year", "t"]).size().unstack(fill_value=0))
    for col in ["Running Late", "Breakdown"]:
        if col not in piv:
            piv[col] = 0
    piv = piv.sort_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    yrs = piv.index.astype(int).astype(str)
    ax1.bar(yrs, piv["Running Late"], label="Running Late", color="#5b9bd5")
    ax1.bar(yrs, piv["Breakdown"], bottom=piv["Running Late"],
            label="Breakdown", color="#c0392b")
    ax1.set_title("Reported Incidents by Year", fontweight="bold")
    ax1.set_ylabel("Number of incidents"); ax1.set_xlabel("School year (start)")
    ax1.tick_params(axis="x", rotation=45, labelsize=8); ax1.legend()

    tot_bd = int(df["is_breakdown"].sum()); tot = len(df)
    ax2.pie([tot - tot_bd, tot_bd], labels=["Running Late", "Breakdown"],
            colors=["#5b9bd5", "#c0392b"], autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.42))
    ax2.set_title("Incident Type Share\n(1.17M cleaned records)", fontweight="bold")
    _save(fig, "fig_problem_incidents.png")


def datasource_overview():
    """Data-source figure: the 21 source columns grouped by category."""
    groups = [
        ("Temporal", "#1f4e79", ["School_Year", "Occurred_On", "Created_On",
                                  "Informed_On", "Last_Updated_On"]),
        ("Operator & Route", "#117864", ["Bus_Company_Name", "Bus_No", "Route_Number",
                                          "Run_Type", "Schools_Serviced", "Boro"]),
        ("Incident", "#c0392b", ["Reason", "Breakdown_or_Running_Late",
                                 "How_Long_Delayed", "Busbreakdown_ID", "Incident_Number"]),
        ("Passengers & Flags", "#8e44ad", ["Number_Of_Students_On_The_Bus",
                                           "Has_Contractor_Notified_Schools",
                                           "Has_Contractor_Notified_Parents",
                                           "Have_You_Alerted_OPT", "School_Age_or_PreK"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("Data Source — 21 Columns of the NYC 'Bus Breakdown and Delays' Dataset",
                 fontsize=13, fontweight="bold", pad=12)
    for i, (name, color, cols) in enumerate(groups):
        r, c = divmod(i, 2)
        x, y = c, 1 - r
        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), 0.9, 0.86,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     facecolor="white", edgecolor=color, linewidth=2))
        ax.add_patch(plt.Rectangle((x + 0.05, y + 0.72), 0.9, 0.19, facecolor=color))
        ax.text(x + 0.5, y + 0.815, name, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        for k, col in enumerate(cols):
            ax.text(x + 0.11, y + 0.62 - k * 0.115, f"•  {col}", ha="left",
                    va="center", fontsize=8.4, color="#222", family="monospace")
    _save(fig, "fig_datasource_overview.png")


def main():
    df = pd.read_parquet(PROCESSED_PARQUET,
                         columns=["year", "incident_type", "is_breakdown", "delay_minutes"])
    data_scale_card()
    problem_incidents(df)
    datasource_overview()
    print("\nRelevant replacement figures generated.")


if __name__ == "__main__":
    main()
