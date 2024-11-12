# phase space of Lotka-Volterra model (predator-prey model)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

# Define the range of x and y values
xmin = 0
xmax = 1.8

x = np.linspace(xmin, xmax, 10)
y = np.linspace(xmin, xmax, 10)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

alpha=1
beta=3
gamma=2
delta=4

# Define the vectors for the phase space
def x_dot(x, y):
    return alpha * x - beta * x * y

def y_dot(x,y):
    return -gamma * y + delta*x*y


U = alpha*X-beta*X*Y  # Example vector dependent on x and y
V = -gamma*Y + delta*X *Y  # Example vector dependent on x and y

# Plot the phase space
plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U, V, scale=40, color='grey', alpha=0.6)

plt.plot(0,0, marker=MarkerStyle('o', fillstyle='left'), color='black')
# plt.scatter(0,0, marker=MarkerStyle('o', fillstyle='right'), facecolors='none', edgecolors='black')


plt.scatter(gamma/delta,alpha/beta, color='black')

num_steps = 1000000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=1.2
y_values[0]=1.3
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values)
print('done 1')

num_steps = 1000000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=1
y_values[0]=1
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values)
print('done 2')

num_steps = 1000000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=1.1
y_values[0]=1.1
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values)
print('done 3')

num_steps = 1000000
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
x_values[0]=1
y_values[0]=0.33
h=0.00001
for i in range(1, num_steps):
    x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
    y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
plt.plot(x_values, y_values)

plt.xlabel('Prey')
plt.ylabel('Predator')
plt.title(f'Two-Dimensional Phase Space of Lotka-Volterra Model\n'+r'with $\alpha=1$, $\beta=3$, $\gamma=2$ and $\delta=4$')
plt.grid(True)
# plt.legend(loc='upper right')
plt.show()

# time-series
times = np.arange(0, num_steps*h, h)
plt.plot(times, x_values, label='Prey')
plt.plot(times, y_values, label='Predator')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title(f'Population Dynamics in Time for Lotka-Volterra Model\n'+r'with $\alpha=1$, $\beta=3$, $\gamma=2$ and $\delta=4$')
plt.legend()
plt.show()