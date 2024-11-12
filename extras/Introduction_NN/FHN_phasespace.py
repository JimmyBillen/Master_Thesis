# Van der Pol Oscillator using Lienard plane and mu>>1
# (Is an introductory to understanding the FitzHugh–Nagumo model)

import numpy as np
import matplotlib.pyplot as plt

R = 0.1
I = 10
TAU = 7.5
A = 0.7
B = 0.8


def F(x):
    return x**3 / 3 - x

def limit_cycle(mu, num_points=10000, t_end=50, x_init=0, y_init=0):
    dt = t_end / num_points
    t = np.linspace(0, t_end, num_points)

    x = np.zeros(num_points)
    y = np.zeros(num_points)

    x[0] = x_init  # Initial condition for x
    y[0] = y_init    # Initial condition for y

    for i in range(1, num_points):
        x_dot = x[i-1] - ((x[i-1])**3)/3 - y[i-1] + R * I
        y_dot = (x[i-1]+A-B*y[i-1]) / TAU

        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt

    return x, y

plt.figure(figsize=(6, 6))

# Set the values of mu
mu = 1
x_inits = [1, 2.5, -2, 0.5]
y_inits = [2, 1, 0, 1.5]

# Plot the limit cycle for each value of mu
for x,y in zip(x_inits, y_inits):
    x_lc, y_lc = limit_cycle(mu, num_points=10000, t_end=50, x_init=x, y_init=y)

    plt.plot(x_lc, y_lc, '-')
x = np.linspace(-4,4,1001)
# plt.plot(x, x-(x**3)/3 + R*I, '--', color = "lime", label = r"$\dot{y}=0$")
# plt.plot(x, (x+A)/B, '--', color = "cyan", label = r"$\dot{x}=0$")
plt.scatter(x_inits, y_inits, color='grey', alpha=0.6, label='Start point')

# plt.scatter(0,0, color='black', label='Fixed point')

# dots = [-2,2]
# dots_null = [((i**3)/3) - i for i in dots]
# plt.plot(dots, dots_null, 'bo')

xmin = -3
xmax = 3
ymin = -1
ymax = 3
x = np.linspace(xmin, xmax, 20)
y = np.linspace(ymin, ymax, 20)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

U = X-(X**3)/3 - Y + R*I  # Example vector dependent on x and y
V = (X + A - B*Y)/TAU  # Example vector dependent on x and y

U = U / np.sqrt((U**2+V**2))
V = V / np.sqrt((U**2 + V**2))

# Plot the phase space
plt.quiver(X, Y, U, V, scale=28, color='grey', alpha=0.6)
''' Zet quiver aan om alle pijltjes te zien'''

plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.axhline(0, color='black', linewidth=1, linestyle='--')


# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(r'Two-Dimensional Phase Space of Van Der Pol Oscillator for $\mu$=1')
plt.xlim(-3,3)
plt.ylim(-1, 3)
plt.grid(True)
# plt.legend()
plt.show()
