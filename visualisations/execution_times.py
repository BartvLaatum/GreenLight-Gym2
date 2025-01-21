import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def convert_to_fps(times):
    n_steps = 10 * 24 * 12
    time_per_step = times / n_steps
    return 1 / time_per_step

# Read the CSV files
cpp_times = pd.read_csv("data/run_times/cpp_model.csv")
glplus_times = pd.read_csv("data/run_times/glplus.csv")
gl_matlab_times = pd.read_csv("data/run_times/gl_matlab.csv")

cpp_fps = convert_to_fps(cpp_times["elapsed_time"])
glplus_fps = convert_to_fps(glplus_times["elapsed_time"])
gl_matlab_fps = convert_to_fps(gl_matlab_times["elapsed_time"])

# Calculate means and standard deviations
cpp_mean = cpp_fps.mean()
cpp_std = cpp_fps.std()
glplus_mean = glplus_fps.mean()
glplus_std = glplus_fps.std()
gl_matlab_mean = gl_matlab_fps.mean()
gl_matlab_std = gl_matlab_fps.std()


# Create bar plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
models = ["GL-Matlab", "GL-Python", "GL-Gym"]
means = [gl_matlab_mean, glplus_mean, cpp_mean]
stds = [gl_matlab_std, glplus_std,cpp_std ]

colors = cmc.berlin_r

colors = [colors(0.2), colors(0.3), colors(0.8)]
bars = ax.bar(models, means, yerr=stds, color=colors)

# Customize plot
ax.set_ylabel("Steps per second")
ax.set_yscale("log")

# Add value labels on top of bars
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height/2,
#             f'{height:.0f}',
#             ha='center', va='bottom', color='black')

print("factor of speed up over GL Matlab: ", cpp_mean/gl_matlab_mean)
print("factor of speed up over GL+: ", cpp_mean/glplus_mean)
plt.savefig('figures/execution_time/step_per_second.eps', format='eps', bbox_inches='tight')

plt.tight_layout()
plt.show()

