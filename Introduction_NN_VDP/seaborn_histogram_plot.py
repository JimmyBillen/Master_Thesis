import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate random data for three parameters
num_points = 100
parameter_1 = np.random.lognormal(mean=-3.5, sigma=1.2, size=num_points)
parameter_2 = np.random.lognormal(mean=-4, sigma=2, size=num_points)
# parameter_3 = np.random.lognormal(mean=-4, sigma=1, size=num_points)

# Combine data into a DataFrame
data = {
    'Hyperparameters 1': parameter_1,
    'Hyperparameters 2': parameter_2
}
df = pd.DataFrame(data)

# Plotting
sns.boxplot(data=df)
sns.stripplot(data=df) #x="normalization_method", y="MSE", hue="normalization_method", hue_order=hue_order, order=x_order, palette='tab10')

plt.ylabel('Nullcline Error')
plt.yscale('log')
plt.show()
