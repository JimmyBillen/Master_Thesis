import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define your data points
x_data = np.loadtxt('othernullcline_x')
y_data = np.loadtxt('othernullcline_y')

# x_data = np.loadtxt('other_nullcline_mean_x')
# y_data = np.loadtxt('other_nullcline_mean_y')

# Define the 4th order polynomial function
def third_order_poly(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit the curve to the data
popt, pcov = curve_fit(third_order_poly, x_data, y_data)

# Extract the coefficients
a_fit, b_fit, c_fit, d_fit = popt

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = third_order_poly(x_fit, a_fit, b_fit, c_fit, d_fit)

# Plot the data points and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('3rd Order Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients
print("Fitted coefficients:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)
print("d =", d_fit)


# Define the 4th order polynomial function
def fourth_order_poly(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Fit the curve to the data
popt, pcov = curve_fit(fourth_order_poly, x_data, y_data)

# Extract the coefficients
a_fit, b_fit, c_fit, d_fit, e_fit = popt

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = fourth_order_poly(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit)

# Plot the data points and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('4th Order Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients
print("Fitted coefficients:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)
print("d =", d_fit)
print("e =", e_fit)

def fifth_order_poly(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# Fit the curve to the data
popt, pcov = curve_fit(fifth_order_poly, x_data, y_data)

# Extract the coefficients
a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = popt

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = fifth_order_poly(x_fit, a_fit, b_fit, c_fit, d_fit, e_fit, f_fit)

# Plot the data points and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('5th Order Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients
print("Fitted coefficients:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)
print("d =", d_fit)
print("e =", e_fit)
print("f =", f_fit)
