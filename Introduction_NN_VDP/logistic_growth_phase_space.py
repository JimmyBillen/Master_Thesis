# phase portrait of logistic growth

import numpy as np
import matplotlib.pyplot as plt

def xdot(x, r, K):
    return r * x * (1 - x / K)    

# Parameters
K = 5
r = 0.5

# Define the range of x values
x = np.linspace(0, 1.2*K, 100)

# Calculate the derivative dx/dt
xdots = xdot(x, r, K)

# Plot the direction field with vectors
plt.figure(figsize=(8, 6))
plt.plot(x, xdots, color='blue', label='dx/dt = rx(1-x/K)')
plt.axvline(x=5, color='gray', linestyle='--', label='Carrying Capacity K')


# Add vectors
x_vector = np.linspace(0, 1.2*K, 15)
for x_vec in x_vector:  # Adjust step size a)s needed
    plt.arrow(x_vec, 0, xdot(x_vec, r, K)/2, 0, width=0.005, head_width=0.023, head_length=0.07, fc='black', ec='black')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.title('Phase Portrait: Vector Field for Logistic Equation\n at Carrying Capacity (K) of 5 and Growth Rate (r) of 0.5')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
