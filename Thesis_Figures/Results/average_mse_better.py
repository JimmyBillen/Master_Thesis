# this is data of ReLU tau=7.5 with [16,16] configuration: [0.0090307540940318, 0.0066327655162435, 0.004016635304371, 0.0059448090486287, 0.0080418255373777]
                                                        #: mean: 1.98e-03
                                                        #: sigmoid fit: 6.85e-04

                                                         
import matplotlib.pyplot as plt
import matplotlib as mpl

# medium= tau7.5
# strong= tau100

medium_real_values = [0.0090307540940318, 0.0066327655162435, 0.004016635304371, 0.0059448090486287, 0.0080418255373777]
medium_mean_values = 1.98e-03
medium_fit_values = 6.85e-04

strong_real_values = [0.0026061691272206, 0.0020732187296547, 0.003033925587614, 0.0034380245064068, 0.0033031189774134]
strong_mean_values = 1.42e-03
strong_fit_values = 9.33e-05

plt.figure(figsize=(3,2))

plt.scatter([-1]*len(medium_real_values), medium_real_values, color='C7', alpha=0.8, edgecolors='none', label='exact')
plt.scatter([-1], [1.98e-03], color='C0', label='mean')
plt.scatter([-1], [6.85e-04], color='C1', label='fit')

plt.scatter([1]*len(medium_real_values), strong_real_values, color='C7', alpha=0.8, edgecolors='none')
plt.scatter([1], [strong_mean_values], color='C0')
plt.scatter([1], [strong_fit_values], color='C1')

# plt.legend(['separate', 'mean', 'fit'])
plt.yscale('log')
plt.xticks([-1,1], ['7.5', '100'])
plt.xlim(-3,3)
plt.xlabel(r"Time-Scale Separation ($\tau$)", labelpad=-3)
plt.ylabel('Nullcline Error', labelpad=-2)
plt.legend(loc='upper left', bbox_to_anchor=[-0.05, 1.2], ncols=3, columnspacing=0.2,handletextpad=0.1, frameon=False)
plt.subplots_adjust(top=0.905,
bottom=0.175,
left=0.2,
right=0.985,
hspace=0.2,
wspace=0.2)

mpl.rc("savefig", dpi=300)
plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Benchmark\comparing_mse_methods_different_tau.png')


plt.show()