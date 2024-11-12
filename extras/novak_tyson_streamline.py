import numpy as np
import matplotlib.pyplot as plt

# Constants (make sure to define these appropriately based on your specific problem)
Kd = 1
Et = 1
k2 = 1
p=4
k1 = kdx = kdy = 0.05

Km = 0.1 * Kd

Ki = 2 / Kd

S = 1

print( k2 * Et / Kd)
ksy = 1

# Define the functions
def x_dot(x_val, y_val):
    return k1 * S * (Kd**p / (Kd**p + y_val**p)) - x_val * kdx

def y_dot(x_val, y_val):
    return ksy * x_val - kdy * y_val - (k2 * Et) * (y_val) / (Km + y_val + Ki * y_val**2)

# Create a grid of x and y values
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

# Compute the velocities
U = x_dot(X, Y)
V = y_dot(X, Y)

# Create the phase-space plot
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')
plt.title('Phase-Space Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.grid(True)
plt.show()
