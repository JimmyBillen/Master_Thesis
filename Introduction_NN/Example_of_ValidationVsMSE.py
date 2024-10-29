# Example of log - log scale (for further understanding thesis)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Generate logarithmically spaced data points
x_log = np.logspace(-3, 2, 20)
y_linear = 2 * x_log  # Example linear function

# Add noise to y-values
noise_scale = 0.7 * x_log
y_noisy = y_linear + np.random.normal(scale=noise_scale, size=len(x_log))

# Plotting
sns.scatterplot(x=x_log, y=y_noisy)
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points on a Linear Line with Big Noise (Log Scale)')
plt.show()
