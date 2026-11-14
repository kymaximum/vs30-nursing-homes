"""The five paper figures. Each function takes the state dict produced by the
pipeline stages and writes one PNG to ``OUT``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from .config import CFG, OUT, rmse, to_points

plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})


def fig1_folds_variogram(labels, tracts, folds):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    tracts.boundary.plot(ax=ax[0], lw=0.1, color="0.8")
    to_points(labels).plot(ax=ax[0], column=folds["fold"], cmap="tab10", markersize=3)
    ax[0].set_title("measured Vs30 sites by spatial fold")
    ax[1].plot(folds["vg_edges"], folds["vg_gamma"], "o-", ms=3)
    ax[1].axvline(folds["vrange_km"], color="crimson", ls="--", label=f"range {folds['vrange_km']:.0f} km")
    ax[1].set(xlabel="lag (km)", ylabel="semivariance")
    ax[1].legend()
    fig.savefig(OUT / "fig1_folds_variogram.png")
    plt.close(fig)


def fig2_predicted_vs_measured(y, is_bay, slope_oof, operational, preds):
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.6), sharex=True, sharey=True)
    panels = [("slope", slope_oof[~is_bay]), ("operational", operational[~is_bay]),
              ("gradient-boosted", preds[~is_bay])]
    for a_, (name, p) in zip(ax, panels):
        a_.scatter(y[~is_bay], p, s=3, alpha=0.4)
        a_.axline((5.5, 5.5), slope=1, color="k", lw=0.6)
        a_.set_title(f"{name}  rmse={rmse(y[~is_bay], p):.3f}")
        a_.set_xlabel("measured ln Vs30")
    ax[0].set_ylabel("predicted")
    fig.savefig(OUT / "fig2_predicted_vs_measured.png")
    plt.close(fig)


def fig3_shap_summary(shap_vals, X):
    import shap
    fig = plt.figure(figsize=(5, 4))
    shap.summary_plot(shap_vals, X, max_display=12, show=False)
    fig.savefig(OUT / "fig3_shap_summary.png")
    plt.close(fig)


def fig4_facility_map(fac, tracts, trace_xy):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    tracts.boundary.plot(ax=ax[0], lw=0.1, color="0.85")
    s = ax[0].scatter(fac["lon"], fac["lat"], c=fac["d_PGA"], s=6, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax[0].plot(trace_xy[:, 0], trace_xy[:, 1], color="k", lw=1)
    ax[0].set(xlim=(-123.2, -121.2), ylim=(36.9, 38.6), title="Δ ln PGA (ML − operational)")
    fig.colorbar(s, ax=ax[0], shrink=0.8)
    nf = fac[fac["near"] & fac["d_PGA"].notna()]
    ax[1].errorbar(nf["rjb"], nf["d_PGA"],
                   yerr=[np.clip(nf["d_PGA"] - nf["d_PGA_lo"], 0, None),
                         np.clip(nf["d_PGA_hi"] - nf["d_PGA"], 0, None)],
                   fmt="o", ms=2, lw=0.4, alpha=0.5)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Rjb (km)", ylabel="Δ ln PGA",
              title=f"site-term difference, 90% interval (Rjb ≤ {CFG['near_km']:.0f} km)")
    fig.savefig(OUT / "fig4_facility_map.png")
    plt.close(fig)


def fig5_siting_and_knowledge(fac, bg, strat):
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    ax[0].hist([fac["disagree"].dropna(), bg["disagree"].dropna()], bins=40, density=True,
               label=["facilities", "background"])
    ax[0].set(xlabel="ln Vs30 disagreement (ML − operational)", ylabel="density")
    ax[0].legend()
    xq = np.arange(len(strat))
    b1 = ax[1].bar(xq - 0.2, strat["width"], 0.4, color="C0", label="90% interval width (ln Vs30)")
    ax1b = ax[1].twinx()
    b2 = ax1b.bar(xq + 0.2, strat["d_label_km_median"], 0.4, color="C1",
                  label="median distance to nearest Vs30 site (km)")
    ax[1].set(xticks=xq, xticklabels=strat.index, xlabel="SVI quartile", ylabel="interval width")
    ax1b.set_ylabel("distance (km)")
    ax[1].set_ylim(0, strat["width"].max() * 1.35)
    ax1b.set_ylim(0, strat["d_label_km_median"].max() * 1.35)
    ax[1].legend(handles=[b1, b2], loc="upper left", fontsize=6.5, frameon=False)
    fig.savefig(OUT / "fig5_siting_and_knowledge.png")
    plt.close(fig)


def fig6_scenario_hazard(fac, tracts, trace_xy):
    """DSHA view: median scenario PGA (operational Vs30) at footprint facilities, and PGA vs Rjb
    under both Vs30 maps with the GMM's +/-1 sigma band."""
    nf = fac[fac["near"] & fac["PGA_op"].notna()].sort_values("rjb")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    tracts.boundary.plot(ax=ax[0], lw=0.1, color="0.85")
    s = ax[0].scatter(nf["lon"], nf["lat"], c=nf["PGA_op"], s=6, cmap="magma_r",
                      norm=matplotlib.colors.LogNorm(vmin=0.02, vmax=1.0))
    ax[0].plot(trace_xy[:, 0], trace_xy[:, 1], color="k", lw=1)
    ax[0].set(xlim=(-123.2, -121.2), ylim=(36.9, 38.6), title="median scenario PGA (g), operational Vs30")
    fig.colorbar(s, ax=ax[0], shrink=0.8)
    ax[1].fill_between(nf["rjb"], nf["PGA_op"] * np.exp(-nf["sig_PGA"]), nf["PGA_op_p84"],
                       color="C0", alpha=0.15, lw=0, label="±1σ (operational)")
    ax[1].plot(nf["rjb"], nf["PGA_op"], ".", ms=3, color="C0", label="operational Vs30")
    ax[1].plot(nf["rjb"], nf["PGA_ml"], ".", ms=3, color="C1", alpha=0.6, label="model Vs30")
    ax[1].set(xlabel="Rjb (km)", ylabel="PGA (g)", yscale="log", title="scenario PGA at facilities")
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.savefig(OUT / "fig6_scenario_hazard.png")
    plt.close(fig)
