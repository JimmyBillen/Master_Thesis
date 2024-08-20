import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0,color='black', linewidth=2)

x = np.linspace(-5, 0, 1000)
plt.plot(x, len(x)*[0], color='blue', linewidth=3)

plt.plot([0,0], [1,0], 'b--', linewidth=4)

x = np.linspace(0, 5, 1000)
plt.plot(x, [1]*len(x), color='blue', linewidth=3)

# Add labels and title
plt.xlabel('x')
plt.ylabel('Binary Step Function')
plt.title('Plot of Binary Step Activation Function')

# Show plot
plt.grid(True)
plt.show()


# ===> with theshold <===

# Define the range of x values


# Plot the binary step function
plt.figure(figsize=(8, 6))
# Plot the x and y axes
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0,color='black', linewidth=2)

x = np.linspace(-5, 2, 1000)
plt.plot(x, len(x)*[0], color='blue', linewidth=3)

plt.plot([2,2], [1,0], 'b--', linewidth=4)

x = np.linspace(2, 5, 1000)
plt.plot(x, [1]*len(x), color='blue', linewidth=3)


# Add labels, title, and threshold text
plt.xlabel('x')
plt.ylabel('Binary Step Function')
plt.title('Plot of Binary Step Activation Function with Threshold of 2')

# Show plot
plt.grid(True)
plt.show()
