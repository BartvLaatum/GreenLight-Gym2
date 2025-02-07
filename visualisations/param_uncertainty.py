import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

WIDTH = 87.5 * 0.03937  # 85 mm ≈ 3.35 inches
HEIGHT = WIDTH * 0.75  # Adjust aspect ratio (3:2 or 4:3 is ideal)

def load_data(project, mode, algorithm, growth_year, start_day, location):
    """
    Load and organize data from CSV files based on specified parameters.
    This function reads CSV files from a hierarchical directory structure organized by project,
    mode, algorithm, and noise levels. It filters files based on growth year, start day, and
    location parameters.
    Parameters
    ----------
    project : str
        The name of the project directory
    mode : str
        The mode directory name within the project
    algorithm : str
        The algorithm directory name
    growth_year : str
        Filter parameter for the growth year in filenames
    start_day : str
        Filter parameter for the start day in filenames
    location : str
        Filter parameter for the location in filenames
    Returns
    -------
    dict
        A nested dictionary where:
        - First level keys are noise levels (as strings)
        - Values are pandas DataFrames containing the data from matching CSV files
    Notes
    -----
    - The function assumes a specific directory structure: data/project/mode/algorithm/noise_level/
    - Only processes the first matching CSV file found for each noise level
    - Noise levels are sorted numerically
    - Prints warning messages if files are not found or if no matching CSV exists
    Example
    -------
    >>> data = load_data("project1", "mode1", "algo1", "2023", "day1", "weatherlocationA")
    """
    base_path = os.path.join("data", project, mode, algorithm)
    noise_levels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    noise_levels.sort(key=lambda x: float(x))
    data_dict = {}
    # Initialize dictionary to store DataFrames for each noise level and file
    data_dict = {noise_level: {} for noise_level in noise_levels}
    # Get all CSV files from the first noise level folder (assuming same files in all folders)
    for noise_level in noise_levels:
        folder_path = os.path.join(base_path, noise_level)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') 
                    and growth_year in f 
                    and start_day in f 
                    and location in f]
        
        if csv_files:
            try:
                data_dict[noise_level] = pd.read_csv(os.path.join(folder_path, csv_files[0]))
            except FileNotFoundError:
                print(f"File not found: {os.path.join(folder_path, csv_files[0])}")
                continue
        else:
            print(f"No matching CSV file found in {folder_path}")
    return data_dict

def plot_cumulative_reward(final_rewards, col2plot):

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    # ax.plot(final_rewards.index, final_rewards[f"Cumulative {col2plot}"], "o-", label=col2plot)
    ax.errorbar(final_rewards.index, final_rewards[f"Cumulative {col2plot}"], yerr=final_rewards[f"std {col2plot}"], fmt="o-", capsize=0)
    ax.set_xlabel(r"Noise Level $(\sigma)$")
    ax.set_ylabel(col2plot)
    plt.show()

def compute_cumulative_metrics(data_dict):
    columns_to_sum = [
        'cFruit', 'Rewards', 'EPI', 'Revenue', 'Heat costs', 'CO2 costs',
        'Elec costs', 'temp_violation', 'co2_violation', 'rh_violation'
    ]

    # Process each noise level's data
    for noise_level, data in data_dict.items():
        # Group by episode and compute cumsum within each episode
        for col in columns_to_sum:
            if col in data.columns:
                data[f'cumsum {col}'] = data.groupby('episode')[col].cumsum()

    # Create final rewards dataframe with columns for mean and std
    final_rewards = pd.DataFrame(index=data_dict.keys())
    for noise_level, data in data_dict.items():
        for col in columns_to_sum:
            if f'cumsum {col}' in data.columns:
                # Get the last value for each episode
                episode_finals = data.groupby('episode')[f'cumsum {col}'].last()
                # Calculate mean and std
                final_rewards.loc[noise_level, f'Cumulative {col}'] = episode_finals.mean()
                final_rewards.loc[noise_level, f'std {col}'] = episode_finals.std()

    return final_rewards

def main(args):
    data_dict = load_data(args.project, args.mode, args.algorithm, args.growth_year, args.start_day, args.location)
  
    final_rewards = compute_cumulative_metrics(data_dict)
    plot_cumulative_reward(final_rewards, col2plot="Rewards")
    plot_cumulative_reward(final_rewards, col2plot="EPI")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost metrics from different models")
    parser.add_argument("--project", type=str, required=True, help="Path to project folder")
    parser.add_argument("--mode", type=str, choices=["deterministic", "stochastic"], required=True, help="Simulation mode")
    parser.add_argument("--algorithm" , type=str, required=True, help="RL algorithm to use")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year")
    parser.add_argument("--start_day", type=str, required=True, help="Start day")
    parser.add_argument("--location", type=str, required=True, help="Location")
    args = parser.parse_args()

    main(args)

