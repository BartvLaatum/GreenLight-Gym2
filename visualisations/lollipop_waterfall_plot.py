import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_config

def load_data(args):
    base_path = os.path.join(f"results", args.project, args.mode)

    def load_and_label(folder, model_name):
        # For stochastic mode with uncertainty, adjust the path
        if args.mode == "stochastic" and hasattr(args, 'uncertainty_value') and args.uncertainty_value > 0:
            folder_path = os.path.join(base_path, folder, str(args.uncertainty_value))
        else:
            folder_path = os.path.join(base_path, folder)

        files = [f for f in os.listdir(folder_path) if
                 all(x in f for x in [args.growth_year, args.start_day, args.location]) and f.endswith(".csv")]
        if not files:
            print(f"Warning: No data found for {model_name} in {folder_path}")
            return pd.DataFrame()  # Return an empty DataFrame if no file is found
        filepath = os.path.join(folder_path, files[0])
        data = pd.read_csv(filepath)
        data["model"] = model_name
        return data
    ppo_data = load_and_label("ppo", "PPO")
    sac_data = load_and_label("sac", "SAC")
    rb_data = load_and_label("rb_baseline", "RB Baseline")
    return pd.concat([ppo_data, sac_data, rb_data], ignore_index=True)


def plot_lollipop_delta_vs_baseline(
    costs_df, 
    baseline="RB Baseline", 
    as_percent=False,
    title="Δ vs Baseline",
    out_png="lollipop_vs_baseline.png",
    out_pdf="lollipop_vs_baseline.pdf"
    ):
    """
    Lollipop plot of 'Controller - Baseline' for each metric.
    If as_percent=True, shows percent change relative to baseline.

    costs_df: DataFrame index=metrics, columns=controllers (must include 'baseline')
    """
    metrics = list(costs_df.index)
    controllers = [c for c in costs_df.columns if c != baseline]

    colors = ["#003366", "#A60000"]

    # Build long-form deltas
    rows = []
    for m in metrics:
        base = costs_df.loc[m, baseline]
        for c in controllers:
            val = costs_df.loc[m, c]
            if as_percent:
                delta = np.nan if base == 0 or pd.isna(base) else (val - base) / base * 100.0
            else:
                delta = val - base
            rows.append({"metric": m, "controller": c, "delta": delta})
    ddf = pd.DataFrame(rows)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_positions = np.arange(len(metrics))
    ax.axvline(0, color="black", linewidth=3)

    # spacing between controllers per metric
    offset = 0.2 if len(controllers) == 2 else 0.08

    for i, c in enumerate(controllers):
        sub = ddf[ddf["controller"] == c]
        y = y_positions + (i - (len(controllers)-1)/2) * offset
        # stems
        for (yy, d) in zip(y, sub["delta"]):
            ax.plot([0, d], [yy, yy], linewidth=3, alpha=0.8, c=colors[i])
        # markers
        ax.scatter(sub["delta"], y, s=55, label=c, zorder=4, c=colors[i])
        # labels
        for (d, yy) in zip(sub["delta"], y):
            if pd.isna(d): 
                continue
            txt = f"{d:.1f}%" if as_percent else f"{d:.2f}"
            if d < 0:
                xoffset = -0.25
            else:
                xoffset = 0.25
            ax.text(d+xoffset, yy, txt, fontsize=14, ha="left" if d >= 0 else "right",
                    va="center", fontweight="bold")
    # Add one point to xlim for better spacing
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin-3, xmax + 1)
    fontsize = 24
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metrics, fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xlabel("Change vs baseline" + (" (%)" if as_percent else " (EU/m²)"),
                  fontweight="bold", fontsize=fontsize)
    ax.legend(frameon=False, fontsize=fontsize, loc="upper right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf, dpi=300)
    plt.close(fig)
    return out_png, out_pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost metrics from different models")
    parser.add_argument("--project", type=str, required=True, help="Path to project folder")
    parser.add_argument("--mode", type=str, choices=["deterministic", "stochastic"], required=True, help="Simulation mode")
    parser.add_argument("--uncertainty_value", type=float, help="Parameter uncertainty scale the agent was trained with")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year")
    parser.add_argument("--start_day", type=str, required=True, help="Start day")
    parser.add_argument("--location", type=str, required=True, help="Location")
    args = parser.parse_args()
    # =============== Example dataset (replace with yours) ===============
    # Use the categories from your screenshot. Replace numbers with your actual metrics.
    metrics = ["EPI","Revenue","Heat","Electricity","CO2"]
    costs = load_data(args)
    costs = costs.groupby("model")[["EPI", "Revenue", "Heat costs", "Elec costs", "CO2 costs"]].sum()
    costs.columns = metrics
    costs = costs.T

    # =============== Generate plots ===============

    ll_png, ll_pdf = plot_lollipop_delta_vs_baseline(costs,
                                                     baseline="RB Baseline",
                                                    as_percent=False,
                                                    title="Controller Performance vs RB Baseline")

    # llp_png, llp_pdf = plot_lollipop_delta_vs_baseline(costs, baseline="RB Baseline",
    #                                                 as_percent=True,
    #                                                 title="Δ vs Baseline (percent)")

    ll_png