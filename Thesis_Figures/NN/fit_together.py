from convergence import plot_loss
from under_over_fitting import plot_fitting
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# Plot the first subplot
plot_fitting(axs[0])
axs[0].set_title('a',loc='left')

# Plot the second subplot
plot_loss(axs[1])
axs[1].set_title('b', loc='left')



plt.tight_layout()

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\NN\fittings.png")


plt.show()
