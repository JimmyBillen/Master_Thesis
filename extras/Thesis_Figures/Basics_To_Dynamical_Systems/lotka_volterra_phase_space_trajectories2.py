import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Parameters for the Lotka-Volterra model
alpha = 1.1   # Prey birth rate
beta = 0.4    # Predation rate
gamma = 0.4   # Predator death rate
delta = 0.1   # Predator reproduction rate

alpha=1
beta=3
gamma=2
delta=4

# Define the Lotka-Volterra equations
def lotka_volterra(X, Y):
    dx = alpha * X - beta * X * Y
    dy = delta * X * Y - gamma * Y
    return dx, dy

def x_dot(x, y):
    return alpha * x - beta * x * y

def y_dot(x,y):
    return -gamma * y + delta*x*y


# Create a grid of points
xmax = 1
ymax = 0.8
x = np.linspace(0, xmax, 15)
y = np.linspace(0, ymax, 15)
X, Y = np.meshgrid(x, y)

# Compute the direction vectors at each point
DX, DY = lotka_volterra(X, Y)

# Normalize the direction vectors to make all arrows equal in length
magnitude = np.sqrt(DX**2 + DY**2)
DX = DX / magnitude
DY = DY / magnitude

# Plotting the phase space
fig, ax = plt.subplots(figsize=(4, 3))
plt.quiver(X,Y,DX,DY,color='grey') 

# num_steps = 1_000_000
# x_values = np.zeros(num_steps)
# y_values = np.zeros(num_steps)
# x_values[0]=1.2
# y_values[0]=1.3
# h=0.00001
# for i in range(1, num_steps):
#     x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
#     y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
# plt.plot(x_values, y_values, color='C0')
# print('done 1')

# num_steps = 1_000_000
# x_values = np.zeros(num_steps)
# y_values = np.zeros(num_steps)
# x_values[0]=1
# y_values[0]=1
# h=0.00001
# for i in range(1, num_steps):
#     x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
#     y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
# plt.plot(x_values, y_values, color='C1')
# print('done 2')

# num_steps = 1_000_000
# x_values = np.zeros(num_steps)
# y_values = np.zeros(num_steps)
# x_values[0]=1.1
# y_values[0]=1.1
# h=0.00001
# for i in range(1, num_steps):
#     x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
#     y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
# plt.plot(x_values, y_values, color='C2')
# print('done 3')

# num_steps = 1_000_000
# x_values = np.zeros(num_steps)
# y_values = np.zeros(num_steps)
# x_values[0]=gamma/delta
# y_values[0]=0.05
# h=0.00001
# for i in range(1, num_steps):
#     x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
#     y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
# plt.plot(x_values, y_values, color='C3')

# num_steps = 1_000_000
# x_values = np.zeros(num_steps)
# y_values = np.zeros(num_steps)
# x_values[0]=gamma/delta
# y_values[0]=0.025
# h=0.00001
# for i in range(1, num_steps):
#     x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
#     y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
# plt.plot(x_values, y_values, color='C3')


num_steps = 1_000_000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=gamma/delta
y_values[0]=0.2
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values, color='C3')

num_steps = 1_000_000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=gamma/delta
y_values[0]=0.1
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values, color='C3')


num_steps = 1_000_000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=gamma/delta
y_values[0]=0.15
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values, color='C3', label='Trajectory')

# Add nullclines for better visualization
nullcline_x = gamma / delta
nullcline_y = alpha / beta
ax.axhline(nullcline_y, color='xkcd:bright aqua', linestyle='--', linewidth=2, label=r'$y=\alpha/ \beta$')
ax.axvline(nullcline_x, color='lime', linestyle='--', linewidth=2, label=r'$x=\gamma/\delta$')

# Set labels and title
ax.set_xlabel(r'Prey population $x$')
ax.set_ylabel(r'Predator population $y$')
ax.set_title('Phase Space of the Lotka-Volterra Model')

# Set limits for the axes
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)

plt.xticks([gamma/delta, 0], [r'$\gamma/\delta$', 0])
plt.yticks([alpha/beta, 0], [r'$\alpha/ \beta$', 0])
plt.scatter([gamma/delta], [alpha/beta], label='Fixed Point', color='black', s=60, zorder=3)
# plt.legend(framealpha=1, loc='upper right')

# ax.text(0.1, 0.1, 'Test', color='black', 
#         bbox=dict(facecolor='grey', edgecolor='green', boxstyle='round'))
ax.text(0.05, 0.6, r'$\dot{y}<0$'+',\n'+r'$\dot{x}<0$', color='black', 
        bbox=dict(facecolor='w', edgecolor='black', boxstyle='round', alpha=0.8))
ax.text(0.85, 0.6, r'$\dot{y}>0$'+',\n'+r'$\dot{x}<0$', color='black', 
        bbox=dict(facecolor='w', edgecolor='black', boxstyle='round', alpha=0.8))
ax.text(0.05, 0.04, r'$\dot{y}<0$'+',\n'+r'$\dot{x}>0$', color='black', 
        bbox=dict(facecolor='w', edgecolor='black', boxstyle='round', alpha=0.8))
ax.text(0.85, 0.05, r'$\dot{y}>0$'+',\n'+r'$\dot{x}>0$', color='black', 
        bbox=dict(facecolor='w', edgecolor='black', boxstyle='round', alpha=0.8))
print('x:gammadelta', gamma/delta,'y:alphabeta', alpha/beta)

# Show the plot
# plt.grid(True)
plt.tight_layout()

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\BasicsToDynamicalSystems\LV_phaseportrait_trajectory2.png")

plt.show()
