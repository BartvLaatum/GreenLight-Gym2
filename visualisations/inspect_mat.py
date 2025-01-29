import pandas as pd
import matplotlib.pyplot as plt

# Load the data
states_pipe = pd.read_csv("data/comparison/states_pipeinput.csv")
led_data = pd.read_csv("data/comparison/gl_gym_states.csv")

# Get common columns between the two dataframes
# common_cols = ["co2Air"]
common_cols = states_pipe.columns
n_cols = len(common_cols)
fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))

# Plot each column
for i, col in enumerate(common_cols):
    ax = axes[i]
    ax.plot(states_pipe[col].values, label='GL-Gym', alpha=0.7)
    ax.plot(led_data[col].values, label='GL-Matlab', alpha=0.7)
    ax.set_title(f'{col}')
    ax.legend()
    ax.grid(True)

fig.savefig("data/comparison/states_pipeinput_led_comparison.png")

# Load control data
controls_pipe = pd.read_csv("data/comparison/controls_pipeinput.csv").iloc[:states_pipe.shape[0]]
glgym_controls = pd.read_csv("data/comparison/gl_gym_states.csv")

pipe_weather = pd.read_csv("data/comparison/weather_pipeinput.csv").values[:,:10]
gl_gym_weather = pd.read_csv("data/comparison/gl_gym_weather.csv").values[:,:10]

print(pipe_weather.shape, gl_gym_weather.shape)

# Create figure for control comparison
fig2, axes = plt.subplots(6, 1, figsize=(12, 8))
matlab_cols = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
for i, col in enumerate(controls_pipe.columns[:]):
    ax = axes[i]
    ax.plot(controls_pipe[col].values, label='GL-Gym', alpha=0.7)
    ax.plot(glgym_controls[matlab_cols[i]].values, label='GL-Matlab', alpha=0.7)
    ax.set_title(col)
    ax.legend()
    ax.grid(True)
fig2.tight_layout()
fig2.savefig("data/comparison/controls_comparison.png")

fig3, axes = plt.subplots(6, 1, figsize=(8, 12))
for i in range(6):
    axes[i].plot(pipe_weather[:, i], label='GL-Gym', alpha=0.7)
    axes[i].plot(gl_gym_weather[:, i], label='GL-Matlab', alpha=0.7)
    axes[i].set_title('Air temperature')
    axes[i].legend()
    axes[i].grid(True)

fig3.tight_layout()
fig3.savefig("data/comparison/weather_comparison.png")
