import matplotlib.pyplot as plt
import matplotlib as mpl
from Phase_Space_Excitability import plot_limit_excite
from Time_Series_Excitability import plot_timeseries_excite
from Phase_Space_Oscillation import plot_limit_cycle
from Time_Series_Oscillation import plot_timeseries
from phase_space_FHN_arrows import streamlined_phase_space_FHN

# Create the figure with constrained layout
size=0.5
fig = plt.figure(constrained_layout=True, figsize=(7,4))

# Define the mosaic layout for 3 columns
# 'A', 'B', and 'C' will represent subplots in the first column
# 'D', 'E', and 'F' will represent subplots in the second column
# 'G' will represent the big subplot in the third column spanning all rows
mosaic = [
    ['A', 'B', 'E'],
    ['C', 'D', 'E']
]

# Define the gridspec keyword arguments
# The width ratios ensure the third column (the big subplot) is larger
gridspec_kw = {'width_ratios': [2, 2, 4]}

# Create subplots based on the defined mosaic and gridspec
axs = fig.subplot_mosaic(mosaic, gridspec_kw=gridspec_kw)

plot_timeseries_excite(ax=axs['A'], plot=False)
plot_timeseries(ax=axs['B'], plot=False)
plot_limit_excite(ax=axs['C'], plot=False)
plot_limit_cycle(ax=axs['D'], plot=False)
streamlined_phase_space_FHN(ax=axs['E'], plot=False)


# Set titles for each subplot
axs['A'].set_title('a', loc='left', pad=10)
axs['B'].set_title('b', loc='left', pad=10)
axs['C'].set_title('c', loc='left', pad=10)
axs['D'].set_title('d', loc='left', pad=10)
axs['E'].set_title('e', loc='left', pad=10)

# axs['C'].legend(['Trajectory', 'x=0', 'y=0', 'Fixed Point', 'Start Point'], loc='upper left', bbox_to_anchor=(0.7, 0.8))


# Display the figure
# plt.tight_layout()
mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\FHN\FHN_time_ps_exc_osc.png")

plt.show()
