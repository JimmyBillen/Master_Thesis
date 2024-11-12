# Task 4.3 of PMCS using Lienard plane and mu>>1
# Van der Pol Oscillator
# (Is an introductory to understanding the FitzHugh–Nagumo model)

import numpy as np
import matplotlib.pyplot as plt

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
        x_dot = y[i-1]
        y_dot = -x[i-1] - mu * y[i-1] * (x[i-1]**2 -1)

        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt

    return x, y

plt.figure(figsize=(6, 6))

# Set the values of mu
mu = 1
x_inits = [1, 0, -1, 3]
y_inits = [0, 3, -4, -1]

# Plot the limit cycle for each value of mu
for x,y in zip(x_inits, y_inits):
    x_lc, y_lc = limit_cycle(mu, num_points=10000, t_end=50, x_init=x, y_init=y)

    plt.plot(x_lc, y_lc, '-')
x = np.linspace(-4,4,1001)
plt.plot(x, -x/(mu*(x**2-1)), '--', color = "lime", label = r"$\dot{y}=0$")
plt.plot(x, np.zeros(len(x)), '--', color = "cyan", label = r"$\dot{x}=0$")
plt.scatter(x_inits, y_inits, color='grey', alpha=0.6, label='Start point')

plt.scatter(0,0, color='black', label='Fixed point')

# dots = [-2,2]
# dots_null = [((i**3)/3) - i for i in dots]
# plt.plot(dots, dots_null, 'bo')

xmin = -4
xmax = 4
ymin = -5
ymax = 5
x = np.linspace(xmin, xmax, 20)
y = np.linspace(ymin, ymax, 20)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

U = Y  # Example vector dependent on x and y
V = -X-mu * (X**2-1) * Y  # Example vector dependent on x and y

# Normalize the direction vectors to make all arrows equal in length
magnitude = np.sqrt(U**2 + V**2)
DX = U / magnitude
DY = V / magnitude

# Plot the phase space
plt.quiver(X, Y, DX, DY, color='grey', alpha=0.6)
''' Zet quiver aan om alle pijltjes te zien'''


plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Two-Dimensional Phase Space of Van Der Pol Oscillator for $\mu$=1')
plt.xlim(-4,4)
plt.ylim(-5, 5)
plt.grid(True)
plt.legend()
plt.show()
