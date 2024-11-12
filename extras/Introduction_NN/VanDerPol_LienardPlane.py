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
        x_dot = mu * (y[i-1] - F(x[i-1]))
        y_dot = -1 / mu * x[i-1]

        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt

    return x, y

# Set the values of mu
mu_values = 1
x_inits = [1, 0, -1, 4]
y_inits = [0, 3, -4, -1]

# Plot the limit cycle for each value of mu
for x,y in zip(x_inits, y_inits):
    x_lc, y_lc = limit_cycle(1, num_points=10000, t_end=50, x_init=x, y_init=y)

    plt.plot(x_lc, y_lc, '-', label=f'(x0,y0)=({x},{y})')
x = np.linspace(-2.5,2.5,1000)
plt.plot(x, (1/3)*x**3-x, '--', color = "lime", label = r"$y=x^3/3-x$")

dots = [-2,2]
# dots_null = [((i**3)/3) - i for i in dots]
# plt.plot(dots, dots_null, 'bo')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Limit Cycle and Cubic Nullcline')
plt.grid(True)
plt.legend()
plt.show()
