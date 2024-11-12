# Plots performance in function of the time-scale seperation (tau), for neural networks using min-max normalization
# ReLU, 2 layers, 4, 8 or 16 nodes. Visualizes Nullcline Error and Pearson Correlation Coefficient

import seaborn as sns
import matplotlib.pyplot as plt

# Data
tau = [1, 2, 5, 7.5, 10, 20, 40, 60, 80, 100]
# tau = [1/epsilon for epsilon in tau]

tau1_pcc = [0.96, 0.89, 0.79]
tau_2_pcc = [0.33, 0.20, 0.49]
tau_5_pcc = [0.18, 0.03, 0.09]
tau7_5_pcc = [0.24, 0.41, 0.46]
tau_10_pcc = [0.34, 0.51, 0.15]
tau_20_pcc = [0.50, 0.57, 0.28]
tau_40_pcc = [0.66, 0.59, 0.11]
tau_60_pcc = [0.73, 0.47, 0.02]
tau_80_pcc = [0.82, 0.29, 0.02]
tau_100_pcc = [0.85, 0.52, 0.14]

# MEDIAN: nullcline error from: specific_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu'], plot_option='1')

tau1_mse = [0.00031639285874495, 0.00010429361189184934, 4.72627015122903*10**-5]
tau2_mse = [0.10047021465991421, 0.0828604933802227, 0.05869697316881595]
tau5_mse = [0.3564313456995779, 0.17459837340477086, 0.0453827876930595]
tau7_5_mse = [0.5756316387322697, 0.05205400636486285, 0.01382994466696515]
tau10_mse = [0.2131559948086851, 0.03583109871163275, 0.006919229694493]
tau20_mse = [0.11354634896132745, 0.014973468725776, 0.00525951738815155]
tau40_mse = [0.034013464296126554, 0.007334053895431699, 0.00276174137755735]
tau60_mse = [0.0220665235857927, 0.00587050943629985, 0.00324709672517055]
tau80_mse = [0.0160305020026192, 0.0052106054791731, 0.00424999359544775]
tau100_mse = [0.018055084963948552, 0.00482930397203825, 0.00342511461057305]

taus_df = []
nodes = ['[4,4]', '[8,8]','[16,16]']
nodes_df = []
for tau_val in tau:
    taus_df.extend([tau_val]*3)
    nodes_df.extend(nodes)

pcc_list = [tau1_pcc, tau_2_pcc, tau_5_pcc, tau7_5_pcc, tau_10_pcc, tau_20_pcc, tau_40_pcc, tau_60_pcc, tau_80_pcc, tau_100_pcc]
pcc_df = []
for pcc_values in pcc_list:
    pcc_df.extend(pcc_values)

mse_list = [tau1_mse, tau2_mse, tau5_mse, tau7_5_mse, tau10_mse, tau20_mse, tau40_mse, tau60_mse, tau80_mse, tau100_mse]
mse_df = []
for mse_values in mse_list:
    mse_df.extend(mse_values)

# Create DataFrame
import pandas as pd
df = pd.DataFrame({
    'tau': taus_df,
    'nodes': nodes_df,
    'pcc': pcc_df,
    'mse': mse_df
})
# print(len(taus_df), len(nodes_df), len())

# Plot PCC
fig, ax1 = plt.subplots(figsize=(6,3))
default_palette = sns.color_palette(n_colors=3)
sns.lineplot(data=df, x='tau', y='pcc', hue='nodes', marker='o', markersize=8, ax=ax1, palette=default_palette, alpha=0.7)
# sns.scatterplot(data=df, x='tau', y='pcc', hue='nodes', marker='o', ax=ax1, palette=default_palette, alpha=0.7)

ax1.set_xlabel('Tau (log scale)')
ax1.set_ylabel('PCC')
ax1.set_xscale("log")
ax1.set_xticks(tau)
ax1.set_xticklabels(tau, rotation=90)
# ax1.set_xticklabels([1], rotation=90, ha='left')
# ax1.set_xticklabels([2], rotation=90, ha='right')

# plot MSE
ax2 = ax1.twinx()
sns.lineplot(data=df, x='tau', y='mse', hue='nodes', marker='s', markersize=8, ax=ax2, palette='pastel', linestyle='dashed')
# sns.scatterplot(data=df, x='tau', y='mse', hue='nodes', marker='s', ax=ax2, palette='pastel', linestyle='dashed')
ax2.set_ylabel('Nullcline Error (log scale)')
ax2.set_yscale('log')
ax2.invert_yaxis()

ax1.legend(loc='upper left', bbox_to_anchor=[0.16, 1.03], title='PCC', frameon=False)
ax2.legend(loc='upper left', title='Error', bbox_to_anchor=[0.4, 1.03], frameon=False)
# plt.legend(title='Label')
# plt.legend([],[], frameon=False)
# plt.grid(True)
print('PCC vs Tau')
plt.tight_layout()
plt.subplots_adjust(top=0.984,
bottom=0.191,
left=0.086,
right=0.895,
hspace=0.2,
wspace=0.2)


plt.show()

