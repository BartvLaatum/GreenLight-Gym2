import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from matplotlib.backends.backend_svg import FigureCanvasSVG

### Latex font in plots
# Ensure text is converted to paths
plt.rcParams["svg.fonttype"] = "path"  # Converts text to paths
plt.rcParams["text.usetex"] = False    # Use mathtext instead of full LaTeX
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # Use a more Illustrator-friendly font
plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = 8  # General font size
plt.rcParams["axes.labelsize"] = 10  # Axis label size
plt.rcParams["xtick.labelsize"] = 8  # Tick labels
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["font.family"] = "serif"  # Use a journal-friendly font

plt.rcParams["text.usetex"] = False
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# plt.rcParams["text.usetex"] = True
# plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams["axes.linewidth"] = 1.5  # Axis border thickness
plt.rcParams["lines.linewidth"] = 1.5  # Line thickness
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1

plt.rc("axes", unicode_minus=False)

width = 85 * 0.03937  # 85 mm ≈ 3.35 inches
height = width * 0.75  # Adjust aspect ratio (3:2 or 4:3 is ideal)

def load_data(args):
    base_path = os.path.join("data/"+args.project, args.mode)
    # Load PPO data
    ppo_file = [f for f in os.listdir(base_path) if 
                all(x in f for x in ["ppo", args.growth_year, args.start_day, args.location]) and f.endswith(".csv")][0]
    ppo_data = pd.read_csv(os.path.join(base_path, ppo_file))
    ppo_data["model"] = "PPO"

    # Load SAC data
    sac_file = [f for f in os.listdir(base_path) if 
                all(x in f for x in ["sac", args.growth_year, args.start_day, args.location]) and f.endswith(".csv")][0]
    sac_data = pd.read_csv(os.path.join(base_path, sac_file))
    sac_data["model"] = "SAC"

    # Load RB baseline data
    rb_file = [f for f in os.listdir(base_path) if 
               all(x in f for x in ["rb_baseline", args.growth_year, args.start_day, args.location]) and f.endswith(".csv")][0]
    rb_data = pd.read_csv(os.path.join(base_path, rb_file))
    rb_data["model"] = "RB Baseline"

    # Combine all data
    return pd.concat([ppo_data, sac_data, rb_data], ignore_index=True)

def costs_plot(data):
    metrics = ["EPI", "Revenue", "Heat costs", "Elec costs", "CO2 costs"]
    models = data["model"].unique()
    n_metrics = len(metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(width, height),dpi=300)
    bar_width = 0.25
    index = range(n_metrics)
    colors = ["#003366", "#0066CC","#4394E5"]

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [data[data["model"] == model][metric].sum() for metric in metrics]
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])

    # Customize plot
    # ax.set_title("Cost Metrics Comparison Across Models")

    ax.set_ylabel(r"Cost (EU/m$^2$)")
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ["EPI", "Revenue", "Heat", "Electricity", r"CO$_2$"]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()
    # Adjust layout and save
    plt.tight_layout()
    fig.canvas = FigureCanvasSVG(fig)
    plt.savefig(f"figures/{args.project}/{args.mode}/cost_metrics_comparison.svg", format="svg", dpi=300)
    plt.savefig(f"figures/{args.project}/{args.mode}/cost_metrics_comparison.png")
    plt.close()

def violations_plot(data):
    metrics = ["temp_violation", "co2_violation", "rh_violation"]
    models = data["model"].unique()
    n_metrics = len(metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(width, height),dpi=300)
    bar_width = 0.25
    index = range(n_metrics)
    colors = ["#003366", "#0066CC","#4394E5"]
    # ax.set_yscale(?'log')

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [data[data["model"] == model][metric].sum() for metric in metrics]
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])

    # Customize plot
    ax.set_ylabel("Cumulative violation")
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ["Temperature", r"CO$_2$", "Relative Humidity"]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()
    # ax.set_yticks([1e1, 1e2, 1e3])
    # ax.set_ylim(1e1, max(max(values) * 1.1, 1e3))  # Prevent bars from going to zero

    # Adjust layout and save
    plt.tight_layout()
    fig.canvas = FigureCanvasSVG(fig)
    plt.savefig(f"figures/{args.project}/{args.mode}/violations_metrics_comparison.svg", format="svg", dpi=300, metadata={"Creator": "Illustrator"})
    plt.savefig(f"figures/{args.project}/{args.mode}/violations_metrics_comparison.png")
    plt.close()

def main(args):
    data = load_data(args)
    costs_plot(data)
    violations_plot(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost metrics from different models")
    parser.add_argument("--project", type=str, required=True, help="Path to project folder")
    parser.add_argument("--mode", type=str, choices=["deterministic", "stochastic"], required=True, help="Simulation mode")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year")
    parser.add_argument("--start_day", type=str, required=True, help="Start day")
    parser.add_argument("--location", type=str, required=True, help="Location")
    args = parser.parse_args()

    main(args)