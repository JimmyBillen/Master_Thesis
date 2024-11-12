# Similar limit cycle has different behavior after perturbation (Thesis Section 3.4)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl

# Define each equation as a function of y (with x as a parameter)
def eq1(y, x):
    return 1.13*x - 0.9*y - 1.17*x**3 - 0.55*x**2*y**3

def eq2(y, x):
    return 0.69*x - 0.32*y - 0.13*x*y**2

def eq3(y, x):
    return 1.13*x - 0.93*y - 1.17*x**3

def eq4(y, x):
    return 0.69*x - 0.32*y


def compute_system_1_dynamics(x0,y0):
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
        x_values[i] = x_values[i - 1] + h * eq1(y_values[i - 1], x_values[i - 1])
        y_values[i] = y_values[i - 1] + h * eq2(y_values[i - 1], x_values[i - 1])
    

    return time, x_values, y_values

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


time1, x_1, y_1 = compute_system_1_dynamics(1,2)
time2, x_2, y_2 = compute_system_2_dynamics(1,2)

fig, axs = plt.subplots(1,2)
fig.set_figheight(3)
fig.set_figwidth(6)


xmin = -4
xmax = 4
ymin = -4
ymax = 4
x = np.linspace(xmin, xmax, 20)
y = np.linspace(ymin, ymax, 20)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

U = 1.13*X - 0.9*Y - 1.17*X**3 - 0.55*(X**2)*(Y**3)  # Example vector dependent on x and y
V = 0.69*X - 0.32*Y - 0.13*X*Y**2

DU = U / np.sqrt((U**2+V**2))
DV = V / np.sqrt((U**2 + V**2))

# Plot the phase space
# plt.quiver(X, Y, DU, DV, scale=28, color='grey', alpha=0.6)
''' Zet quiver aan om alle pijltjes te zien'''
c = axs[0].streamplot(X, Y, U, V, density=0.8, linewidth=None, color='grey', minlength=0.1, zorder=0) 
c.lines.set_alpha(0.4)
for x in axs[0].get_children():
    if type(x)==matplotlib.patches.FancyArrowPatch:
        x.set_alpha(0.4) # or x.set_visible(False)

axs[0].plot(x_1, y_1, color='C3')
axs[0].scatter(x_1[0], y_1[0], color='C3')
time1_2, x_1_2, y_1_2 = compute_system_1_dynamics(1, 3)
axs[0].plot(x_1_2, y_1_2, color='C3')
axs[0].scatter(x_1_2[0], y_1_2[0], color='C3')
axs[0].set_xlim([-4,4])
axs[0].set_ylim([-4,4])


# ++++++++++++++++++++++++++++ second +++++++++++++++++++++ 

xmin = -4
xmax = 4
ymin = -4
ymax = 4
x = np.linspace(xmin, xmax, 20)
y = np.linspace(ymin, ymax, 20)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

U = 1.13*X - 0.93*Y - 1.17*X**3 # Example vector dependent on x and y
V = 0.69*X - 0.32*Y

DU = U / np.sqrt((U**2+V**2))
DV = V / np.sqrt((U**2 + V**2))

# Plot the phase space
# plt.quiver(X, Y, DU, DV, scale=28, color='grey', alpha=0.6)
''' Zet quiver aan om alle pijltjes te zien'''
c = axs[1].streamplot(X, Y, U, V, density=0.8, linewidth=None, color='grey', minlength=0.1, zorder=0) 
c.lines.set_alpha(0.4)
for x in axs[1].get_children():
    if type(x)==matplotlib.patches.FancyArrowPatch:
        x.set_alpha(0.4) # or x.set_visible(False)



axs[1].plot(x_2, y_2, color='C3')
axs[1].scatter(x_2[0], y_2[0], color='C3')
time2_2, x_2_2, y_2_2 = compute_system_2_dynamics(1, 3)
axs[1].plot(x_2_2, y_2_2, color='C3')
axs[1].scatter(x_2_2[0], y_2_2[0], color='C3')
axs[1].set_xlim([-4,4])
axs[1].set_ylim([-4,4])
axs[0].set_title("System 1")
axs[1].set_title("System 2")

axs[0].set_xlabel(r"$x$", labelpad=-1)
axs[1].set_xlabel(r"$x$", labelpad=-1)
axs[0].set_ylabel(r"$y$", labelpad=-2)

axs[0].set_xticks([-4,0,4])
axs[0].set_yticks([-4,0,4])
axs[1].set_xticks([-4,0,4])
axs[1].set_yticks([-4,0,4])


plt.subplots_adjust(top=0.91,
bottom=0.12,
left=0.075,
right=0.99,
hspace=0.2,
wspace=0.2)


plt.show()



