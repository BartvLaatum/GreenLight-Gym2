import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
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
    base_path = os.path.join(f"data/{args.project}", args.mode)

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

def costs_plot(data):
    metrics = ["EPI", "Revenue", "Heat costs", "Elec costs", "CO2 costs"]
    print(data.keys())
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
        print(values)
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])

    # Customize plot
    # ax.set_title("Cost Metrics Comparison Across Models")

    ax.set_ylabel(r"Cumulative cost (EU/m$^2$)")
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ["EPI", "Revenue", "Heat", "Electricity", r"CO$_2$"]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()
    # Adjust layout and save
    plt.tight_layout()
    fig.canvas = FigureCanvasSVG(fig)
    # plt.savefig(f"figures/{args.project}/{args.mode}/cost_metrics_comparison.svg", format="svg", dpi=300)
    # plt.savefig(f"figures/{args.project}/{args.mode}/cost_metrics_comparison.png")
    # plt.close()
    plt.show()

def violations_plot(data):
    metrics = ["temp_violation", "co2_violation", "rh_violation"]
    models = data["model"].unique()
    n_metrics = len(metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    bar_width = 0.25
    index = range(n_metrics)
    colors = ["#003366", "#0066CC","#4394E5"]
    # ax.set_yscale(?'log')

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [data[data["model"] == model][metric].sum() for metric in metrics]
        print(values)
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])

    # Customize plot
    ax.set_ylabel("Cumulative penalty")
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ["Temperature", r"CO$_2$", "Relative Humidity"]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()
    # ax.set_yticks([1e1, 1e2, 1e3])
    # ax.set_ylim(1e1, max(max(values) * 1.1, 1e3))  # Prevent bars from going to zero

    # Adjust layout and save
    plt.tight_layout()
    fig.canvas = FigureCanvasSVG(fig)
    # plt.savefig(f"figures/{args.project}/{args.mode}/violations_metrics_comparison.svg", format="svg", dpi=300, metadata={"Creator": "Illustrator"})
    # plt.savefig(f"figures/{args.project}/{args.mode}/violations_metrics_comparison.png")
    # plt.close()
    plt.show()

def main(args):
    data = load_data(args)
    costs_plot(data)
    violations_plot(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost metrics from different models")
    parser.add_argument("--project", type=str, required=True, help="Path to project folder")
    parser.add_argument("--mode", type=str, choices=["deterministic", "stochastic"], required=True, help="Simulation mode")
    parser.add_argument("--uncertainty_value", type=float, help="Parameter uncertainty scale the agent was trained with")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year")
    parser.add_argument("--start_day", type=str, required=True, help="Start day")
    parser.add_argument("--location", type=str, required=True, help="Location")
    args = parser.parse_args()

    if args.mode == "stochastic":
        if args.uncertainty_value is None:
            raise ValueError("Uncertainty value must be provided for stochastic mode.")
    main(args)