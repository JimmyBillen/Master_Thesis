import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

xmin = -10
xmax = 10
ymin = -1.1
ymax = 1.05
# Define the range of x values
x = np.linspace(xmin, xmax, 1000)

# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Define the tanh activation function
def tanh(x):
    return np.tanh(x)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Plot the ReLU activation function
# plt.figure(figsize=(6, 6))

fig, axs = plt.subplots(ncols=3, figsize=(7,2.5))


axs[0].axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

axs[0].plot(x, relu(x), color='blue')
# plt.xlabel('x')
# plt.ylabel('ReLU(x)')
# plt.set_title('ReLU Activation Function')
# plt.grid(True)
axs[0].set_xlim(-1, 1)
axs[0].set_ylim(ymin, ymax)

axs[0].set_yticks([-1, 0, 1])
axs[0].set_xticks([])

axs[0].set_title('ReLU')

# plt.axvline(0,color='black', linewidth=2)

# plt.show()


# Plot the sigmoid activation function
# plt.figure(figsize=(6, 6))
axs[1].plot(x, sigmoid(x), color='green')
# plt.xlabel('x')
# plt.ylabel('sigmoid(x)')
# plt.set_title('Sigmoid Activation Function')
# plt.grid(True)
axs[1].set_xlim(xmin, xmax)
axs[1].set_ylim(ymin, ymax)

axs[1].axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
axs[1].axhline(1, color='black', linewidth=1, linestyle='--', alpha=0.5)

axs[1].set_xticks([])
axs[1].set_yticks([-1, 0, 1])

axs[1].set_title('Sigmoid')

# Plot the tanh activation function
# plt.figure(figsize=(6, 6))
axs[2].plot(x, tanh(x), color='red')
# plt.xlabel('x')
# plt.ylabel('tanh(x)')
# plt.set_title('Tanh Activation Function')
# plt.grid(True)
axs[2].set_xlim(xmin, xmax)
axs[2].set_ylim(ymin, ymax)
axs[2].set_xticks([])
axs[2].set_yticks([-1, 0, 1])


axs[2].axhline(-1, color='black', linewidth=1, linestyle='--', alpha=0.5)
axs[2].axhline(1, color='black', linewidth=1, linestyle='--', alpha=0.5)

axs[2].set_title('Tanh')

# plt.show()

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\NN\activation_functions.png")

plt.tight_layout()
plt.show()
