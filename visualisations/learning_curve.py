import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
### Latex font in plots
# Ensure text is converted to paths
plt.rcParams["svg.fonttype"] = "path"  # Converts text to paths
plt.rcParams["text.usetex"] = False    # Use mathtext instead of full LaTeX
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # Use a more Illustrator-friendly font
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.family"] = "DejaVu Sans" 

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

WIDTH = 85 * 0.03937  # 85 mm ≈ 3.35 inches
HEIGHT = WIDTH * 0.75  # Adjust aspect ratio (3:2 or 4:3 is ideal)

def load_data(project):
    """Load rollout data for PPO and SAC"""
    load_dir = f"data/{project}/deterministic/"


    # Load data for both algorithms
    ppo_path = load_dir + "ppo_det/rollout.csv"
    sac_path = load_dir +  "sac_det/rollout.csv"
    baseline_path = load_dir + "rb_baseline-201059-Amsterdam.csv"
    
    ppo_data = pd.read_csv(ppo_path)
    sac_data = pd.read_csv(sac_path)
    baseline_data = pd.read_csv(baseline_path)
    return ppo_data, sac_data, baseline_data

def plot_learning_curves(ppo_data, sac_data, baseline):
    """Create learning curve plot"""
    PPO_COLOR = "#003366"
    SAC_COLOR = "#A60000"
    baseline_rewards = baseline["Rewards"].sum()

    fig, ax = plt.subplots(1, figsize=(WIDTH, HEIGHT), dpi=300)

    # Plot PPO data
    ax.plot(ppo_data['global step'], ppo_data['train reward'], label='PPO', color=PPO_COLOR, alpha=0.8)
    # Plot SAC data
    ax.plot(sac_data['global step'], sac_data['train reward'], label='SAC', color=SAC_COLOR, alpha=0.8)

    ax.hlines(baseline_rewards,ppo_data['global step'].iloc[0], ppo_data['global step'].iloc[-1], label="RB Baseline", linestyle="--", color=c, alpha=0.8)

    ax.set_xlabel('Total time steps')
    ax.set_ylabel('Cumulative reward')
    ax.legend()
    plt.tight_layout()
    # Save the plot
    fig.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig('learning_curves.svg', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot learning curves for PPO and SAC')
    parser.add_argument('--project', type=str, required=True, help='Project name')
    # parser.add_argument('--group', type=str, required=True, help='Group name')
    args = parser.parse_args()
    
    # Load data
    ppo_data, sac_data, baseline = load_data(args.project)

    # Create plot
    plot_learning_curves(ppo_data, sac_data, baseline)

if __name__ == "__main__":
    main()
