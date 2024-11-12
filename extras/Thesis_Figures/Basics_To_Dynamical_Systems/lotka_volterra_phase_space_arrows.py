import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Parameters for the Lotka-Volterra model
alpha = 1.1   # Prey birth rate
beta = 0.4    # Predation rate
gamma = 0.4   # Predator death rate
delta = 0.1   # Predator reproduction rate

# Define the Lotka-Volterra equations
def lotka_volterra(X, Y):
    dx = alpha * X - beta * X * Y
    dy = delta * X * Y - gamma * Y
    return dx, dy

# Create a grid of points
x = np.linspace(0, 8, 20)
y = np.linspace(0, 5, 20)
X, Y = np.meshgrid(x, y)

# Compute the direction vectors at each point
DX, DY = lotka_volterra(X, Y)

# Normalize the direction vectors to make all arrows equal in length
magnitude = np.sqrt(DX**2 + DY**2)
DX = DX / magnitude
DY = DY / magnitude

# Plotting the phase space
fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(X, Y, DX, DY, color='C0')

# Add nullclines for better visualization
nullcline_x = gamma / delta
nullcline_y = alpha / beta
ax.axhline(nullcline_y, color='cyan', linestyle='--', linewidth=2, label=r'$y=\alpha/ \beta$')
ax.axvline(nullcline_x, color='lime', linestyle='--', linewidth=2, label=r'$x=\gamma/\delta$')

# Set labels and title
ax.set_xlabel('Prey population')
ax.set_ylabel('Predator population')
ax.set_title('Phase Space of the Lotka-Volterra Model')

# Set limits for the axes
ax.set_xlim(0, 8)
ax.set_ylim(0, 5)

plt.xticks([gamma/delta, 0], [r'$\gamma/\delta$', 0])
plt.yticks([alpha/beta, 0], [r'$\alpha/ \beta$', 0])
plt.scatter([gamma/delta], [alpha/beta], label='Fixed Point', color='C0', s=60, zorder=3)
plt.legend(framealpha=1, title='Nullcline', loc='upper right')

# Show the plot
# plt.grid(True)
plt.tight_layout()

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\BasicsToDynamicalSystems\LV_phaseportrait_arrows.png")

plt.show()
