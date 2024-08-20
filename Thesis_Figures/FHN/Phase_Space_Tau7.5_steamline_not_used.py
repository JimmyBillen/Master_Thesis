# Task 4.3 of PMCS using Lienard plane and mu>>1
# Van der Pol Oscillator
# (Is an introductory to understanding the FitzHugh–Nagumo model)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
        x_dot = y[i-1]
        y_dot = -x[i-1] - mu * y[i-1] * (x[i-1]**2 -1)

        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt

    return x, y

# plt.figure(figsize=(6, 4))
plt.figure(figsize=(6.4*0.8, 4.8*0.8*0.9))

# Set the values of mu
mu = 1
# x_inits = [1, 0, -1, 3]
# y_inits = [0, 3, -4, -1]
# Plot the limit cycle for each value of mu
# for x,y in zip(x_inits, y_inits):
#     x_lc, y_lc = limit_cycle(mu, num_points=10000, t_end=50, x_init=x, y_init=y)
#     plt.plot(x_lc, y_lc, '-')

x_lc, y_lc = limit_cycle(mu, num_points=10000, t_end=50, x_init=2, y_init=0)
plt.plot(x_lc, y_lc, c='C3', zorder=3, label='Limit Cycle')


x = np.linspace(-4,4,1001)
plt.plot(x, -x/(mu*(x**2-1)), '--', color = 'lime', label = r"$\dot{y}=0$", linewidth=2)
plt.plot(x, np.zeros(len(x)), '--', color = "xkcd:bright aqua", label = r"$\dot{x}=0$", linewidth=2)


# plt.plot(x, np.zeros(len(x))+1, '--', color = "cyan", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))+2, '--', color = "tab:cyan", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))+3, '--', color = "xkcd:bright aqua", label = r"$\dot{x}=0$", linewidth=2)

# plt.plot(x, np.zeros(len(x))-1, color = "C0", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))-2, color = "C0", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))-3, color = "C0", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))-1, '--', color = "cyan", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))-2, '--', color = "tab:cyan", label = r"$\dot{x}=0$", linewidth=2)
# plt.plot(x, np.zeros(len(x))-3, '--', color = "xkcd:bright aqua", label = r"$\dot{x}=0$", linewidth=2)


# plt.plot(x, -x/(mu*(x**2-1)), '--', color = '#00cc99', label = r"$\dot{y}=0$", linewidth=2) #caribean
# plt.plot(x, np.zeros(len(x)), '--', color = "#CEA2FD", label = r"$\dot{x}=0$", linewidth=2) #

# plt.plot(x, -x/(mu*(x**2-1))+1, '--', color = '#4ecb8d', label = r"$\dot{y}=0$", linewidth=2) #caribean
# plt.plot(x, np.zeros(len(x))+1, '--', color = "#c701ff", label = r"$\dot{x}=0$", linewidth=2) #

# plt.plot(x, -x/(mu*(x**2-1)), color = 'C1', label = r"$\dot{y}=0$")
# plt.plot(x, -x/(mu*(x**2-1)), '--', color = '#00cc99', label = r"$\dot{y}=0$", linewidth=2) #caribean
# plt.plot(x, np.zeros(len(x)), color = "C1", label = r"$\dot{x}=0$", linewidth=2) #Purple (bright)
# plt.plot(x, np.zeros(len(x)), '--', color = "#c701ff", label = r"$\dot{x}=0$", linewidth=2) #Purple (bright)
# plt.plot(x, -x/(mu*(x**2-1)), '--', color = '#f9e858', label = r"$\dot{y}=0$", linewidth=2) #yellow

# plt.plot(x, -x/(mu*(x**2-1))+1, color = 'C0', label = r"$\dot{y}=0$", linewidth=2) #yellow
# plt.plot(x, -x/(mu*(x**2-1))+1, '--', color = '#f9e858', label = r"$\dot{y}=0$", linewidth=2) #yellow

# plt.plot(x, -x/(mu*(x**2-1))+2, color = 'C0', label = r"$\dot{y}=0$", linewidth=2) #yellow
# plt.plot(x, -x/(mu*(x**2-1))+2, '--', color = 'cyan', label = r"$\dot{y}=0$", linewidth=2) #yellow

# plt.plot(x, -x/(mu*(x**2-1))-1, color = 'C2', label = r"$\dot{y}=0$", linewidth=2) #yellow
# plt.plot(x, -x/(mu*(x**2-1))-1, '--', color = 'lime', label = r"$\dot{y}=0$", linewidth=2) #yellow




# plt.scatter(x_inits, y_inits, color='grey', alpha=0.6, label='Start point')


plt.scatter(0,0, color='black', label='Fixed point', zorder=5)

# dots = [-2,2]
# dots_null = [((i**3)/3) - i for i in dots]
# plt.plot(dots, dots_null, 'bo')

# xmin = -4
# xmax = 4
# ymin = -5
# ymax = 5
xmin = -2.5
xmax = 2.5
ymin = -3
ymax = 3

x = np.linspace(xmin, xmax, 20)
y = np.linspace(ymin, ymax, 20)

# Create a grid of x and y values
X, Y = np.meshgrid(x, y)

# U = X-X^3/3  # Example vector dependent on x and y
# V = -X-mu * (X**2-1) * Y  # Example vector dependent on x and y


U = X-(X**3)/3 - Y + R*I  # Example vector dependent on x and y
V = (X + A - B*Y)/TAU  # Example vector dependent on x and y

# Normalize the direction vectors to make all arrows equal in length
magnitude = np.sqrt(U**2 + V**2)
DX = U / magnitude
DY = V / magnitude

# Plot the phase space
# plt.quiver(X, Y, DX, DY, color='grey', alpha=0.6)
c = plt.streamplot(X,Y,U,V, density=0.8, linewidth=None, color='grey', minlength=0.8, zorder=0) 
c.lines.set_alpha(0.6)

''' Zet quiver aan om alle pijltjes te zien'''


plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation='horizontal')
plt.title(r'Two-Dimensional Phase Space of Van Der Pol Oscillator')
plt.xlim(xmin,xmax)
plt.ylim(ymin, ymax)
# plt.grid(True)
plt.legend(framealpha=0.9, loc='lower right')
plt.tight_layout()

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\BasicsToDynamicalSystems\VanDerPol_streamline.png")

plt.show()
