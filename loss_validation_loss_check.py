"""
I'm seeing the problem in my training files that my validation loss seems to be higher than my normal loss,
I will check here if that is always the case or not
"""
import os
from settings import TAU
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np

absolute_path = os.path.dirname(__file__)
relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
csv_name = os.path.join(absolute_path, relative_path)
df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

df['ratio'] = df['loss'] / df['validation'] # since we expect validation > loss => loss/validation < 1

count_validation_greater_than_loss = (df['validation'] > df['loss']).sum()
print(count_validation_greater_than_loss, "out of a total", len(df['validation']))
print("This means", 100*count_validation_greater_than_loss/len(df['validation']), '%')

# plt.plot(df['ratio'], marker='o', linestyle='-')
# plt.plot(df['ratio'])
# plt.xlabel('Index')  # Assuming you want the index on x-axis
# plt.ylabel('Ratio (A/B)')
# plt.title('Ratio of Column A to Column B')
# plt.plot([1]*len(df['ratio']))
# plt.show()

mean = []
std = []
epochs = []

df['log_loss'] = np.log10(df['loss'])
df['log_validation'] = np.log10(df['validation'])

for epoch in range(0, 500):
    print(epoch)
    df_epoch = df[df['epoch'] == epoch].copy()
    df_epoch['ratio'] = df_epoch['log_loss'] / df_epoch['log_validation']

    mean_ratio = df_epoch['ratio'].mean()
    std_ratio = df_epoch['ratio'].std()

    mean.append(mean_ratio)
    std.append(std_ratio)
    epochs.append(epoch)

epochs = np.array(epochs)
mean = np.array(mean)
std = np.array(std)

plt.plot(epochs, mean, label='logloss/logvalidation')
plt.fill_between(epochs, mean - std, mean + std, color='grey', alpha=0.4)
# plt.ylim(0, 1.5)

plt.show()
