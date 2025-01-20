import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
from novak_tyson_solver import compute_novak_dynamics
from data_generation_exploration.FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from create_NN_FHN import calculate_derivatives
import numpy as np
from settings import TAU, R, I
import matplotlib as mpl

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


def interpolate(x1,y1,x2,y2, interpol_x=0):
    """
    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    print(rico)
    y = rico * (interpol_x-x1) + y1
    return y

def nullcline_xdot_ifo_y(yval):
    # ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 
    return ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 

def nullcline_ydot_ifo_y(yval):
    return +(kdy/ksy)*yval + (k2*Et/ksy)* ((yval)/(Km+yval+Ki * yval**2))

def nullcline_xdot_ifo_x(xval):
    return  ( ((k1 * S * Kd**p)/ (kdx * xval)) - Kd**p )**(1/p)

def connecting_curve(xval, dotx):
    return ( ((k1 * S * Kd **p)/(dotx+kdx*xval)) - Kd**p)**(1/p)


# def connecting_curve(y_const, doty):
#     return nullcline_y(y_const)/ksy + doty/ksy

# X -> W
# Y -> V

time, x_values, y_values = compute_novak_dynamics() # assigning v->v, w->v see heads-up above.
x_dot_t_data = np.array(calculate_derivatives(time, x_values))
y_dot_t_data = np.array(calculate_derivatives(time, y_values))

zoomed = False
if zoomed:
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, x_values)[-2]
    index_2 = find_local_maxima(time, x_values)[-1]

    x_values_short = x_values[index_1:index_2]
    u_t_data_short = y_values[index_1:index_2]
    v_dot_t_data_short = y_dot_t_data[index_1:index_2]

# x_values_again = x_values.copy()
# v_dot_t_data_again = v_dot_t_data.copy()


# 
fig = plt.figure()
fig.set_figheight(2)
fig.set_figwidth(6)


# Create 3D plot
ax = fig.add_subplot(1,2,2, projection='3d')

# Plot data points
# ax.scatter(x_values, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(x_values, np.zeros(len(x_values)), nullcline_vdot(x_values), c='b', marker='o')
# ax.plot(y_values, y_dot_t_data, x_values, c='r', alpha=0.7)
# ax.plot(y_values, x_dot_t_data, x_values, c='r', alpha=0.7)
ax.plot(x_values, x_dot_t_data, y_values, c='r', alpha=0.7)

# ax.scatter(y_values, x_dot_t_data, x_values, c='r', alpha=0.03, s=1)

# ax.plot(x_values, np.zeros(len(x_values)), nullcline_vdot(x_values), c='b', alpha=0.7)

min_z = np.min(y_values)
max_z = np.max(y_values)

nullcline_val = nullcline_xdot_ifo_x(x_values)

valid_indices = (nullcline_val >= min_z) & (nullcline_val <= max_z)

x_values_nullcline_plot = x_values[valid_indices]
nullcline_val_plot = nullcline_val[valid_indices]

ax.plot(x_values_nullcline_plot, np.zeros(len(x_values_nullcline_plot)), nullcline_val_plot, c='b', alpha=0.7)


# connect the points with 'C4' color:


# X -> W/U
# Y -> V

# x_values = u_t_data
# y_values = u_dot_t_data
# Specific x value to search for
specific_y = 2
tolerance = 0.1  # Define a tolerance level for how close the x values should be

# Step 1: Find indices of x values that are within the tolerance level
indices = np.where(np.abs(y_values - specific_y) <= tolerance)[0]

# Step 2: Extract the corresponding y values and x values
filtered_x_dot_values = x_dot_t_data[indices]
filtered_y_values = y_values[indices]

# Step 3: Find the highest and lowest y values
if len(filtered_y_values) > 0:
    max_y = np.max(filtered_y_values)
    min_y = np.min(filtered_y_values)
    max_y_x_dot = filtered_x_dot_values[np.argmax(filtered_y_values)]
    min_y_x_dot = filtered_x_dot_values[np.argmin(filtered_y_values)]
    print(min_y, max_y, min_y_x_dot, max_y_x_dot)

    connect_dotx = np.linspace(min_y_x_dot, max_y_x_dot, 100)
    # ax.plot(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx), '--',color='C4')


find_specific_v_value = True
if find_specific_v_value:
    min_index_pos_list = []
    min_index_neg_list = []
    values_approx = np.linspace(min(x_values), max(x_values), 7)
    for value in values_approx:
        diff = [abs(val - value) for val in y_values]

        # Find the index of the minimum absolute difference where b is greater than 0
        min_index_pos = min((i for i, val in enumerate(y_dot_t_data) if val > 0), key=lambda x: diff[x])
        min_index_pos_list.append(min_index_pos)
        # Find the index of the minimum absolute difference where b is less than 0
        min_index_neg = min((i for i, val in enumerate(y_dot_t_data) if val < 0), key=lambda x: diff[x])
        min_index_neg_list.append(min_index_neg)
        print('indices', min_index_neg_list)

for min_index_neg, min_index_pos in zip(min_index_neg_list, min_index_pos_list):
    min_y, max_y = y_values[min_index_neg], y_values[min_index_pos]
    mindot_y, maxdot_y = y_dot_t_data[min_index_neg], y_dot_t_data[min_index_pos]

    # print(mindot_y, max_y_x_dot)
    # mindot_y = -0.1449
specific_y = 0.4
# mindot_y, maxdot_y = -0.0224, 0.0294
mindot_y, maxdot_y = -0.224, 0.0418
connect_dotx = np.linspace(mindot_y, maxdot_y, 1000)
print(mindot_y, maxdot_y)
ax.plot(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx), '--',color='C4')
# ax.scatter(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx),color='C4', s=1)


# ax.set_zlim(0, 4)
ax.set_xlim(0, 0.7)
ax.set_ylim(-0.032, 0.0418)

# x = np.linspace(min(x_values), max(x_values), 2) qdf
# # z = np.linspace(min(y_values), max(u_t_data), 2)
# X, Z = np.meshgrid(x, z)
# Y = np.zeros_like(X)  # Set y = 0
# ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')

# Set labels and title
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\dot{x}$')
ax.set_zlabel(r'$y$')
# ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}')
ax.set_title('a', loc='left')

ax.view_init(elev=40, azim=60)

plt.show()