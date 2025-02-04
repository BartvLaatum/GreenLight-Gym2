import argparse

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 20

plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.grid"] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 4   # Default for all spines
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.major.size'] = 4  # Thicker major x-ticks
plt.rcParams['xtick.major.width'] = 2  # Thicker major x-
plt.rcParams['ytick.major.size'] = 4  
plt.rcParams['ytick.major.width'] = 2 
plt.rc('axes', unicode_minus=False)
PPO_COLOR = "#003366"
SAC_COLOR = "#A60000"

def state_plot(days2plot, time_steps, dt):
    # Plot key state variables over time for all controllers
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex='col', sharey='row')

    constraints = [(15, 34), (50, 85), (400, 1600)]

    end_idx = int(days2plot * (86400/dt))
    variables = ['temp_air', 'rh_air', 'co2_air']
    labels = [r"Temperature ($^\circ$C)", "Relative Humidity (%)", r"CO$_2$ (ppm)"]
    
    for i, (var, label) in enumerate(zip(variables, labels)):
        # Left column (first time period)
        axes[i,0].plot(time_steps[:end_idx], ppo_df[var][:end_idx], color=PPO_COLOR, label="PPO", alpha=0.8)
        axes[i,0].plot(time_steps[:end_idx], sac_df[var][:end_idx], color=SAC_COLOR, label="SAC", alpha=0.8)
        axes[i,0].axhline(y=constraints[i][0], color='gray', linestyle='--', alpha=0.5)
        axes[i,0].axhline(y=constraints[i][1], color='gray', linestyle='--', alpha=0.5)
        axes[i,0].set_ylabel(label)
        
        # Right column (second time period)
        axes[i,1].plot(time_steps[-end_idx:], ppo_df[var][-end_idx:], color=PPO_COLOR, label="PPO", alpha=0.8)
        axes[i,1].plot(time_steps[-end_idx:], sac_df[var][-end_idx:], color=SAC_COLOR, label="SAC", alpha=0.8)
        axes[i,1].axhline(y=constraints[i][0], color='gray', linestyle='--', alpha=0.5)
        axes[i,1].axhline(y=constraints[i][1], color='gray', linestyle='--', alpha=0.5)

    # Format x-axis to show hours
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%D%H'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=6))

    # Set x-label for bottom plots only
    axes[2,0].set_xlabel("Time of day (hours)")
    axes[2,1].set_xlabel("Time of day (hours)")

    axes[0,0].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb project name")
    parser.add_argument("--stochastic", action="store_true", help="Whether to run the experiment in stochastic mode")
    parser.add_argument("--ppo_name", type=str, required=True, help="Name of the ppo model to visualize")
    parser.add_argument("--sac_name", type=str, required=True, help="Name of the sac model to visualize")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year to visualize")
    parser.add_argument("--start_day", type=str, required=True, help="Start day of the year to visualize")
    parser.add_argument("--location", type=str, required=True, help="location to visualize")
    parser.add_argument("--n_days2plot", type=int, required=True, help="Number of days to visualize")
    args = parser.parse_args()

    # load dir
    if args.stochastic:
        load_dir = f"data/{args.project}/stochastic"
    else:
        load_dir = f"data/{args.project}/deterministic"

    # File paths
    ppo_file = f"{load_dir}/ppo-{args.ppo_name}-{args.growth_year}{args.start_day}-{args.location}.csv"
    sac_file = f"{load_dir}/sac-{args.sac_name}-{args.growth_year}{args.start_day}-{args.location}.csv" 
    rb_file = f"{load_dir}/rb_baseline-{args.growth_year}{args.start_day}-{args.location}.csv"

    # Load data
    ppo_df = pd.read_csv(ppo_file)[:-1]
    sac_df = pd.read_csv(sac_file)[:-1]
    rb_df = pd.read_csv(rb_file)
    dt = 900

    # Convert start_day and create timestamps
    start_date = pd.to_datetime(f"{args.growth_year}-01-01") + pd.Timedelta(days=int(args.start_day))
    time_steps = [start_date + pd.Timedelta(seconds=int(i*dt)) for i in range(len(ppo_df))]

    # Display basic information about the datasets
    ppo_df.head(), sac_df.head(), rb_df.head()
    state_plot(args.n_days2plot, time_steps, dt)


        # Plot control actions over time for all controllers
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    # Plot boiler usage
    axes[0, 0].plot(time_steps, ppo_df["uBoil"], label="PPO", alpha=0.8)
    axes[0, 0].plot(time_steps, sac_df["uBoil"], label="SAC", alpha=0.8)
    axes[0, 0].plot(time_steps, rb_df["uBoil"], label="Rule-Based", alpha=0.8)
    axes[0, 0].set_ylabel("Boiler Usage (uBoil)")
    axes[0, 0].set_title("Boiler Usage Over Time")
    axes[0, 0].legend()

    # Plot CO2 injection
    axes[0, 1].plot(time_steps, ppo_df["uCo2"], label="PPO", alpha=0.8)
    axes[0, 1].plot(time_steps, sac_df["uCo2"], label="SAC", alpha=0.8)
    axes[0, 1].plot(time_steps, rb_df["uCo2"], label="Rule-Based", alpha=0.8)
    axes[0, 1].set_ylabel("CO2 Injection (uCo2)")
    axes[0, 1].set_title("CO2 Injection Over Time")

    # Plot thermal screen usage
    axes[1, 1].plot(time_steps, ppo_df["uThScr"], label="PPO", alpha=0.8)
    axes[1, 1].plot(time_steps, sac_df["uThScr"], label="SAC", alpha=0.8)
    axes[1, 1].plot(time_steps, rb_df["uThScr"], label="Rule-Based", alpha=0.8)
    axes[1, 1].set_ylabel("Thermal Screen Usage (uThScr)")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_title("Thermal Screen Usage Over Time")

    plt.tight_layout()
    # plt.show()


        # Plot control actions over time for all controllers
    fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
    # Plot temperature
    axes.plot(time_steps, ppo_df["cFruit"], label="PPO", alpha=0.8)
    axes.plot(time_steps, sac_df["cFruit"], label="SAC", alpha=0.8)
    axes.plot(time_steps, rb_df["cFruit"], label="Rule-Based", alpha=0.8)
    axes.set_ylabel("Fruit weight DM (mg/m^2)")
    # axes.set_title("Greenhouse Air Temperature Over Time")
    axes.legend()
    plt.tight_layout()


    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # Heatmap bins
    bins_x = np.linspace(ppo_df["temp_air"].min(), ppo_df["temp_air"].max(), 50)
    bins_y = np.linspace(ppo_df["uBoil"].min(), ppo_df["uBoil"].max(), 50)

    # Function to create heatmaps
    def plot_heatmap(ax, df, title):
        heatmap, xedges, yedges = np.histogram2d(df["temp_air"], df["uBoil"], bins=[bins_x, bins_y])
        im = ax.imshow(heatmap.T, origin="lower", aspect="auto", 
                      extent=[bins_x.min(), bins_x.max(), bins_y.min(), bins_y.max()], 
                      cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Indoor Temperature (°C)")
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Frequency')

    # Plot heatmaps for each controller
    plot_heatmap(axes[0], ppo_df, "PPO Controller")
    plot_heatmap(axes[1], sac_df, "SAC Controller")
    plot_heatmap(axes[2], rb_df, "Rule-Based Controller")

    # Common y-label
    axes[0].set_ylabel("Boiler Usage (uBoil)")

    plt.tight_layout()
    plt.show()
