import seaborn as sns
import matplotlib.pyplot as plt


# Data
# tau = [1, 7.5, 20, 100]
# tau1_pcc = [0.52, 0.25, -0.14]
# tau7_5_pcc = [0.24, 0.41, 0.46]
# tau_10_pcc = [0.34, 0.51, 0.15]
# tau_20_pcc = [0.49, 0.57, 0.28]
# tau_100_pcc = [0.84, 0.49, 0.12]


# Data
tau = [1, 2, 5, 7.5, 10, 20, 40, 60, 80, 100]
# tau = [1/epsilon for epsilon in tau]
# tau1_pcc = [0.52, 0.25, -0.14]
# tau_2_pcc = [0.44, 0.22, 0.54]
# tau_5_pcc = [0.18, 0.03, 0.09]
# tau7_5_pcc = [0.24, 0.41, 0.46]
# tau_10_pcc = [0.34, 0.51, 0.15]
# tau_20_pcc = [0.49, 0.57, 0.28]
# tau_40_pcc = [0.65, 0.58, 0.09]
# tau_60_pcc = [0.72, 0.46, 0.01]
# tau_80_pcc = [0.81, 0.28, 0.0]
# tau_100_pcc = [0.84, 0.49, 0.12]

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

tau1_mse = [0.00031639285874495, 0.00010429361189184934, 4.72627015122903e-05]
tau2_mse = [0.10047021465991421, 0.0828604933802227, 0.05869697316881595]
tau5_mse = [0.3564313456995779, 0.17459837340477086, 0.0453827876930595]
tau7_5_mse = [0.5756316387322697, 0.05205400636486285, 0.01382994466696515]
tau_10_mse = [0.2131559948086851, 0.03583109871163275, 0.006919229694493]
tau_20_mse = [0.11354634896132745, 0.014973468725776, 0.00525951738815155]
tau_40_mse = [0.034013464296126554, 0.007334053895431699, 0.00276174137755735]
tau_60_mse = [0.0220665235857927, 0.00587050943629985, 0.00324709672517055]
tau_80_mse = [0.0160305020026192, 0.0052106054791731, 0.00424999359544775]
tau_100_mse = [0.018055084963948552, 0.00482930397203825, 0.00342511461057305]

pcc_per_layer = [list(t) for t in zip(tau1_pcc, tau_2_pcc, tau_5_pcc,tau7_5_pcc, tau_10_pcc, tau_20_pcc, tau_40_pcc, tau_60_pcc, tau_80_pcc, tau_100_pcc)]
nodes_4 = pcc_per_layer[0]
nodes_8 = pcc_per_layer[1]
nodes_16 = pcc_per_layer[2]

mse_per_layer = [list(r) for r in zip(tau1_mse, tau2_mse, tau5_mse, tau7_5_mse, tau_10_mse, tau_20_mse, tau_40_mse, tau_60_mse, tau_80_mse, tau_100_mse)]
mse_4 = mse_per_layer[0]
mse_8 = mse_per_layer[1]
mse_16 = mse_per_layer[2]


# Create DataFrame
import pandas as pd
df = pd.DataFrame({
    'tau': tau,
    '[4,4]': nodes_4,
    '[8,8]': nodes_8,
    '[16,16]': nodes_16
})

df_mse = pd.DataFrame({
    'tau': tau,
    '[4,4]': mse_4,
    '[8,8]': mse_8,
    '[16,16]': mse_16
})


# Plot PCC
fig, ax1 = plt.subplots(figsize=(10,6))
default_palette = sns.color_palette(n_colors=3)
sns.lineplot(data=df, x='tau', y='[4,4]', marker='o', markersize=8, label='[4,4]', ax=ax1, palette=default_palette)
sns.lineplot(data=df, x='tau', y='[8,8]', marker='o', markersize=8, label='[8,8]', ax=ax1, palette=default_palette)
sns.lineplot(data=df, x='tau', y='[16,16]', marker='o', markersize=8, label='[16,16]', ax=ax1, palette=default_palette)
ax1.set_xlabel('Tau')
ax1.set_ylabel('PCC')
ax1.set_xticks(tau)

# plot MSE
# a2 = ax1.twinx()
# pastel_palette = sns.color_palette("pastel", n_colors=3)
# print(pastel_palette)
# sns.lineplot(data=df_mse, x='tau', y='[4,4]', marker='o', markersize=8, label='[4,4]', ax=ax1, palette=pastel_palette[:3])
# sns.lineplot(data=df_mse, x='tau', y='[8,8]', marker='o', markersize=8, label='[8,8]', ax=ax1, palette=pastel_palette[:3])
# sns.lineplot(data=df_mse, x='tau', y='[16,16]', marker='o', markersize=8, label='[16,16]', ax=ax1, palette=pastel_palette[:3])

# plt.legend(title='Label')
plt.grid(True)
plt.title('PCC vs Tau')
plt.show()
