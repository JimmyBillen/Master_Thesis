# This script visualizes the construction of the cubic nullcline by connecting
# the same values of v of the limit cycle, the curve connecting these two points follows
# from the differential equation. As described in Section 3.2 and 4.5 of the thesis, as used there as well.

# The program is executed when the script is run as a standalone program. Without needing additional input

import matplotlib.pyplot as plt
import numpy as np

from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_vdot
from create_NN_FHN import calculate_derivatives
from settings import R, I

def interpolate(x1,y1,x2,y2, interpol_x=0):
    """
    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    print(rico)
    y = rico * (interpol_x-x1) + y1
    return y

time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

zoomed = True
if zoomed:
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    v_t_data_short = v_t_data[index_1:index_2]
    u_t_data_short = u_t_data[index_1:index_2]
    v_dot_t_data_short = v_dot_t_data[index_1:index_2]

fig = plt.figure()
fig.set_figheight(2)
fig.set_figwidth(6)

# Create 3D plot
ax = fig.add_subplot(1,3,1, projection='3d')

# Plot data points
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)

x = np.linspace(min(v_t_data), max(v_t_data), 2)
z = np.linspace(min(u_t_data), max(u_t_data), 2)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)  # Set y = 0
ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

ax.set_xlabel(r'$v$', labelpad=-8)
ax.set_ylabel(r'$\dot{v}$', labelpad=-5)
ax.set_zlabel(r'$w$', labelpad=-8)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-3)
ax.tick_params(axis='z', which='major', pad=-3)

ax.set_title('a', loc='left', pad=-10)

ax.view_init(elev=40, azim=60)

############################################################################## 2 
ax = fig.add_subplot(1, 3, 2, projection='3d')

ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)
# ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', alpha=0.7)

ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

find_specific_v_value = True
if find_specific_v_value:
    min_index_pos_list = []
    min_index_neg_list = []
    values_approx = np.linspace(min(v_t_data), max(v_t_data), 50)
    values_approx = [0.5]
    for value in values_approx:
        diff = [abs(val - value) for val in v_t_data]

        # Find the index of the minimum absolute difference where b is greater than 0
        min_index_pos = min((i for i, val in enumerate(v_dot_t_data) if val > 0), key=lambda x: diff[x])
        min_index_pos_list.append(min_index_pos)
        # Find the index of the minimum absolute difference where b is less than 0
        min_index_neg = min((i for i, val in enumerate(v_dot_t_data) if val < 0), key=lambda x: diff[x])
        min_index_neg_list.append(min_index_neg)

w_interpol = interpolate(v_dot_t_data[min_index_neg], u_t_data[min_index_neg], v_dot_t_data[min_index_pos], u_t_data[min_index_pos])
ax.scatter(v_t_data[min_index_neg], 0, w_interpol, color='C2', zorder=5)
vmin, vplus = v_t_data[min_index_neg], v_t_data[min_index_pos]
vdotmin, vdotplus = v_dot_t_data[min_index_neg], v_dot_t_data[min_index_pos]
ax.plot([vmin, vplus], [vdotmin, vdotplus], [vmin-(1/3)*(vmin)**3-vdotmin+R*I, vplus-(1/3)*(vplus)**3-vdotplus+R*I], '--',color='C4')

ax.set_xlabel(r'$v$', labelpad=-8)
ax.set_ylabel(r'$\dot{v}$', labelpad=-5)
ax.set_zlabel(r'$w$', labelpad=-10)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-3)
ax.tick_params(axis='z', which='major', pad=-3)

ax.set_title('b', loc='left', pad=-10)

ax.view_init(elev=45, azim=65)

############################################## 3
ax = fig.add_subplot(1, 3, 3, projection='3d')

ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)

v_t_data = v_t_data_short
v_dot_t_data = v_dot_t_data_short

find_specific_v_value = True
if find_specific_v_value:
    min_index_pos_list = []
    min_index_neg_list = []
    values_approx = np.linspace(min(v_t_data), max(v_t_data), 7)
    for value in values_approx:
        diff = [abs(val - value) for val in v_t_data]

        # Find the index of the minimum absolute difference where b is greater than 0
        min_index_pos = min((i for i, val in enumerate(v_dot_t_data) if val > 0), key=lambda x: diff[x])
        min_index_pos_list.append(min_index_pos)
        # Find the index of the minimum absolute difference where b is less than 0
        min_index_neg = min((i for i, val in enumerate(v_dot_t_data) if val < 0), key=lambda x: diff[x])
        min_index_neg_list.append(min_index_neg)
        print('indices', min_index_neg_list)

for min_index_neg, min_index_pos in zip(min_index_neg_list, min_index_pos_list):
    w_interpol = interpolate(v_dot_t_data[min_index_neg], u_t_data[min_index_neg], v_dot_t_data[min_index_pos], u_t_data[min_index_pos])
    # ax.scatter(v_t_data[min_index_neg], 0, w_interpol, color='lime', zorder=5)
    vmin, vplus = v_t_data[min_index_neg], v_t_data[min_index_pos]
    print('vmin', vmin)
    vdotmin, vdotplus = v_dot_t_data[min_index_neg], v_dot_t_data[min_index_pos]
    ax.plot([vmin, vplus], [vdotmin, vdotplus], [vmin-(1/3)*(vmin)**3-vdotmin+R*I, vplus-(1/3)*(vplus)**3-vdotplus+R*I], '--',color='C4')

ax.plot(v_t_data, np.zeros(len(v_t_data))+np.ones(len(v_t_data))*0.0001, nullcline_vdot(v_t_data), c='C2', alpha=0.7)

ax.set_xlabel(r'$v$', labelpad=-8)
ax.set_ylabel(r'$\dot{v}$', labelpad=-5)
ax.set_zlabel(r'$w$', labelpad=-8)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-3)
ax.tick_params(axis='z', which='major', pad=-3)

ax.set_title('c', loc='left', pad=-10)

ax.view_init(elev=29, azim=57)

plt.tight_layout()
plt.subplots_adjust(top=1.0,
bottom=0.0,
left=0.045,
right=0.989,
hspace=0.24,
wspace=0.177)

plt.show()
