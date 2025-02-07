import pandas as pd
import matplotlib.pyplot as plt

# Load the data
states_pipe = pd.read_csv("data/AgriControl/comparison/gl_gym/states_no_growPipe2009.csv")
led_data = pd.read_csv("data/AgriControl/comparison/matlab/states_no_growPipe2009.csv")
# weather_pipe = pd.read_csv("data/AgriControl/comparison/gl_gym/weather_pipe2009.csv").iloc[:states_pipe.shape[0]]

# Get common columns between the two dataframes
# common_cols = ["co2Air"]
common_cols = states_pipe.columns
n_cols = len(common_cols)
fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))

# Plot each column
for i, col in enumerate(common_cols):
    ax = axes[i]
    ax.plot(states_pipe[col].values[:288], label='GL-Gym', alpha=0.7)
    ax.plot(led_data[col].values[:288], label='GL-Matlab', alpha=0.7)
    ax.set_title(f'{col}')
    ax.legend()
    ax.grid(True)

fig.savefig("figures/AgriControl/comparison/states_pipeinput_led_comparison.png")
# axes[9].plot(weather_pipe["tPipe"].values[:288], label='GL-Gym', alpha=0.7)

# Load control data
controls_pipe = pd.read_csv("data/AgriControl/comparison/gl_gym/controls_pipe2009.csv").iloc[:states_pipe.shape[0]]

# Create figure for control comparison
fig2, axes = plt.subplots(6, 1, figsize=(12, 8))
matlab_cols = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
for i, col in enumerate(controls_pipe.columns[:]):
    ax = axes[i]
    ax.plot(controls_pipe[col].values[:288], label='GL-Gym', alpha=0.7)
    ax.set_title(col)
    ax.legend()
    ax.grid(True)
fig2.tight_layout()
fig2.savefig("figures/AgriControl/comparison/controls.png")

# fig3, axes = plt.subplots(6, 1, figsize=(8, 12))
# for i in range(6):
#     axes[i].plot(weather_pipe.values[:288, i],  label='GL-Gym', alpha=0.7)
#     axes[i].set_title('Air temperature')
#     axes[i].legend()
#     axes[i].grid(True)

# fig3.tight_layout()
# fig3.savefig("figures/AgriControl/comparison/weather.png")
