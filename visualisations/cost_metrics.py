import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 24
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.grid"] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 4   # Default for all spines
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 4  # Thicker major x-ticks
plt.rcParams['xtick.major.width'] = 2  # Thicker major x-
plt.rcParams['ytick.major.size'] = 4  
plt.rcParams['ytick.major.width'] = 2 
plt.rc('axes', unicode_minus=False)


def load_data(args):
    base_path = os.path.join("data/"+args.project, args.mode)
    # Load PPO data
    ppo_file = [f for f in os.listdir(base_path) if 
                all(x in f for x in ['ppo', args.growth_year, args.start_day, args.location]) and f.endswith('.csv')][0]
    ppo_data = pd.read_csv(os.path.join(base_path, ppo_file))
    ppo_data['model'] = 'PPO'

    # Load SAC data
    sac_file = [f for f in os.listdir(base_path) if 
                all(x in f for x in ['sac', args.growth_year, args.start_day, args.location]) and f.endswith('.csv')][0]
    sac_data = pd.read_csv(os.path.join(base_path, sac_file))
    sac_data['model'] = 'SAC'

    # Load RB baseline data
    rb_file = [f for f in os.listdir(base_path) if 
               all(x in f for x in ['rb_baseline', args.growth_year, args.start_day, args.location]) and f.endswith('.csv')][0]
    rb_data = pd.read_csv(os.path.join(base_path, rb_file))
    rb_data['model'] = 'RB Baseline'

    # Combine all data
    return pd.concat([ppo_data, sac_data, rb_data], ignore_index=True)

def costs_plot(data):
    metrics = ['EPI', 'Revenue', 'Heat costs', 'Elec costs', 'CO2 costs']
    models = data['model'].unique()
    n_metrics = len(metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = range(n_metrics)
    colors = ["#003366", "#0066CC","#4394E5"]

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [data[data['model'] == model][metric].sum() for metric in metrics]
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])

    # Customize plot
    # ax.set_title('Cost Metrics Comparison Across Models')

    ax.set_ylabel(r'Cost (EU/m$^2$)')
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ['EPI', 'Revenue', 'Heat', 'Electricity', r'CO$_2$']
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('cost_metrics_comparison.png')
    plt.close()

def violations_plot(data):
    metrics = ["temp_violation", "co2_violation", "rh_violation"]
    models = data['model'].unique()
    n_metrics = len(metrics)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.25
    index = range(n_metrics)
    colors = ["#003366", "#0066CC","#4394E5"]

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [data[data['model'] == model][metric].sum() for metric in metrics]
        ax.bar([x + i * bar_width for x in index], values, bar_width, 
               label=model, color=colors[i])


    # Customize plot
    ax.set_ylabel('Violation')
    ax.set_xticks([x + bar_width for x in index])
    xlabels = ["Temperature", r"CO$_2$", "Relative Humidity"]
    ax.set_xticklabels(xlabels, rotation=0)
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('violations_metrics_comparison.png')
    plt.close()



def main(args):
    data = load_data(args)
    costs_plot(data)
    violations_plot(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot cost metrics from different models')
    parser.add_argument('--project', type=str, required=True, help='Path to project folder')
    parser.add_argument('--mode', type=str, choices=['deterministic', 'stochastic'], required=True, help='Simulation mode')
    parser.add_argument('--growth_year', type=str, required=True, help='Growth year')
    parser.add_argument('--start_day', type=str, required=True, help='Start day')
    parser.add_argument('--location', type=str, required=True, help='Location')
    args = parser.parse_args()

    main(args)