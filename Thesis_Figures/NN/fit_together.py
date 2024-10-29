from convergence import plot_loss
from under_over_fitting import plot_fitting
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# Plot the first subplot
plot_fitting(axs[0])
axs[0].set_title('a',loc='left')

# Plot the second subplot
plot_loss(axs[1])
axs[1].set_title('b', loc='left')



plt.tight_layout()


plt.show()
