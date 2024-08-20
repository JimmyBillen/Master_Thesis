# nvm: used https://www.desmos.com/calculator/r5mxkdci6k

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve

# # Define each equation as a function of y (with x as a parameter)

# def eq1(y, x):
#     return 1.13*x - 0.9*y - 1.17*x**3 - 0.55*x**2*y**2

# def eq2(y, x):
#     return 0.69*x - 0.32*y - 0.13*x*y**2

# def eq3(y, x):
#     return 1.13*x - 0.93*y - 1.17*x**3

# def eq4(y, x):
#     return 0.69*x - 0.32*y

# # Set up the x range
# x_values = np.linspace(-3, 3, 600)

# # Initialize lists for storing x and y values for each equation
# x_plot1, y_plot1 = [], []
# x_plot2, y_plot2 = [], []
# x_plot3, y_plot3 = [], []
# x_plot4, y_plot4 = [], []

# # Initial guess for fsolve
# initial_guess = 0

# # Calculate y values for each x for all four equations
# for x in x_values:
#     y1 = fsolve(eq1, initial_guess, args=(x))
#     y2 = fsolve(eq2, initial_guess, args=(x))
#     y3 = fsolve(eq3, initial_guess, args=(x))
#     y4 = fsolve(eq4, initial_guess, args=(x))
    
#     # Only append real solutions
#     if np.isreal(y1):
#         x_plot1.append(x)
#         y_plot1.append(y1[0])
    
#     if np.isreal(y2):
#         x_plot2.append(x)
#         y_plot2.append(y2[0])
    
#     if np.isreal(y3):
#         x_plot3.append(x)
#         y_plot3.append(y3[0])
    
#     if np.isreal(y4):
#         x_plot4.append(x)
#         y_plot4.append(y4[0])

# # Plot all curves
# plt.figure(figsize=(10, 8))
# plt.plot(x_plot1, y_plot1, 'b-', label=r'$1.13x - 0.9y - 1.17x^{3} - 0.55x^{2}y^{2} = 0$')
# plt.plot(x_plot2, y_plot2, 'r-', label=r'$0.69x - 0.32y - 0.13xy^{2} = 0$')
# plt.plot(x_plot3, y_plot3, 'g-', label=r'$1.13x - 0.93y - 1.17x^{3} = 0$')
# plt.plot(x_plot4, y_plot4, 'm-', label=r'$0.69x - 0.32y = 0$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plots of the Equations')
# # plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.axhspan(-10, -0.6, color='grey', alpha=0.5)
# plt.axhspan(0.6, 10, color='grey', alpha=0.5)

# plt.axvspan(-10, -1, color='grey', alpha=0.5)
# plt.axvspan(1, 10, color='grey', alpha=0.5)

# plt.ylim(-1.5, 1.5)

# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define each equation as a function of y (with x as a parameter)
def eq1(y, x):
    return 1.13*x - 0.9*y - 1.17*x**3 - 0.55*x**2*y**3

def eq2(y, x):
    return 0.69*x - 0.32*y - 0.13*x*y**2

def eq3(y, x):
    return 1.13*x - 0.93*y - 1.17*x**3

def eq4(y, x):
    return 0.69*x - 0.32*y

# Set up the x range with smaller steps for increased accuracy
x_values = np.linspace(-2, 2, 50)

# Initialize lists for storing x and y values for each equation
x_plot1, y_plot1 = [], []
x_plot2, y_plot2 = [], []
x_plot3, y_plot3 = [], []
x_plot4, y_plot4 = [], []

# Use a small number for the tolerance in fsolve for higher accuracy
tolerance = 1e-6

# Calculate y values for each x for all four equations using multiple initial guesses
for x in x_values:
    # Use multiple initial guesses to find different roots if they exist
    y_guesses = np.linspace(-1, 1, 5)
    
    for initial_guess in y_guesses:
        y1 = fsolve(eq1, initial_guess, args=(x), xtol=tolerance)
        y2 = fsolve(eq2, initial_guess, args=(x), xtol=tolerance)
        y3 = fsolve(eq3, initial_guess, args=(x), xtol=tolerance)
        y4 = fsolve(eq4, initial_guess, args=(x), xtol=tolerance)
        
        # Only append real solutions and avoid duplicates
        if np.isreal(y1) and y1 not in y_plot1:
            x_plot1.append(x)
            y_plot1.append(y1[0])
        
        if np.isreal(y2) and y2 not in y_plot2:
            x_plot2.append(x)
            y_plot2.append(y2[0])
        
        if np.isreal(y3) and y3 not in y_plot3:
            x_plot3.append(x)
            y_plot3.append(y3[0])
        
        if np.isreal(y4) and y4 not in y_plot4:
            x_plot4.append(x)
            y_plot4.append(y4[0])

# Plot all curves
plt.figure(figsize=(3, 3))
plt.plot(x_plot1, y_plot1, 'b-', label=r'system 1', linestyle='dashed')
plt.plot(x_plot2[0::2][0::2][0::3], y_plot2[0::2][0::2][0::3], 'blue', linestyle='dashed')
plt.plot(x_plot3, y_plot3, 'orange', label=r'system 2', linestyle='-.')
plt.plot(x_plot4, y_plot4, 'orange', linestyle='-.')
plt.xlabel(r'$x$', labelpad=-1)
plt.ylabel(r'$y$', labelpad=-2)
plt.title('Phase Space')
# plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# plt.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3)
# plt.fill_between(x, y1, y2, where=(y1 < y2), color='C1', alpha=0.3)

def compute_system_2_dynamics(x0,y0):
    """
    Compute the dynamics of the FitzHugh-Nagumo model using Euler's method.

    Returns:
    time (array): Array of time values.
    v_values (array): Array of membrane potential values.
    w_values (array): Array of recovery variable values.

    Notes:
    - The FitzHugh-Nagumo model describes the dynamics of excitable media.
    - It consists of two coupled differential equations for the membrane potential (v)
      and the recovery variable (w).
    - The equations are integrated using Euler's method.
    - The integration parameters depend on the value of TAU (want same points per period).
    """
    # Initial conditions 
    # x0 = 1.0  # Initial value of v
    # y0 = 2.0  # Initial value of w

    t0 = 0.0
    t_end = 65.5
    num_steps = 15000

    

    # Create arrays to store the values
    time = np.linspace(t0, t_end, num_steps + 1) # +1 to work as expected
    h = (t_end - t0) / num_steps
    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)

    # Initialize the values at t0
    x_values[0] = x0
    y_values[0] = y0

    # Implement Euler's method
    for i in range(1, num_steps + 1):
        x_values[i] = x_values[i - 1] + h * eq3(y_values[i - 1], x_values[i - 1])
        y_values[i] = y_values[i - 1] + h * eq4(y_values[i - 1], x_values[i - 1])

    return time, x_values, y_values

t, x, y = compute_system_2_dynamics(1,0)
plt.plot(x,y, color='C3')

plt.xlim(-2,2)
plt.ylim(-2.1,2.1)

plt.legend(ncols=2, markerscale=0.6, columnspacing=0.8, framealpha=0.9)
plt.tight_layout()
plt.subplots_adjust(top=0.917,
bottom=0.134,
left=0.154,
right=0.965,
hspace=0.2,
wspace=0.2)

import matplotlib as mpl
mpl.rc("savefig", dpi=300)
plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Uniqueness\DifferentNullclinesOutsideLC.png')


plt.show()
