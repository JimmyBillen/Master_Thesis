# This script visualizes the construction of the linear nullcline by connecting
# the same values of v of the limit cycle, the curve connecting these two points follows
# from the differential equation. As described in Section 3.2 and 4.5 of the thesis.

# The program is executed when the script is run as a standalone program. Without needing additional input

import matplotlib.pyplot as plt
import numpy as np

from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_wdot
from create_NN_FHN import calculate_derivatives
from settings import TAU, A, B

def interpolate(x1,y1,x2,y2, interpol_x=0):
    """
    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    print(rico)
    y = rico * (interpol_x-x1) + y1
    return y

time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v
u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

ZOOMED = True
if ZOOMED:
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    v_t_data_short = v_t_data[index_1:index_2]
    u_t_data_short = u_t_data[index_1:index_2]
    u_dot_t_data_short = u_dot_t_data[index_1:index_2]

fig = plt.figure()
fig.set_figheight(2)
fig.set_figwidth(6)

ax = fig.add_subplot(111, projection='3d')

x = np.linspace(min(v_t_data), max(v_t_data), 2)
z = np.linspace(min(u_t_data), max(u_t_data), 2)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)  # Set y = 0
ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')

ax.plot(v_t_data, u_dot_t_data, u_t_data, c='r', alpha=0.7)

v_t_data = v_t_data_short
u_dot_t_data = u_dot_t_data_short

find_specific_v_value = True
if find_specific_v_value:
    min_index_pos_list = []
    min_index_neg_list = []
    values_approx = np.linspace(-0.672,0.832, 5)
    for value in values_approx:
        diff = [abs(val - value) for val in v_t_data]

        # Find the index of the minimum absolute difference where b is greater than 0
        min_index_pos = min((i for i, val in enumerate(u_dot_t_data) if val > 0), key=lambda x: diff[x])
        min_index_pos_list.append(min_index_pos)
        # Find the index of the minimum absolute difference where b is less than 0
        min_index_neg = min((i for i, val in enumerate(u_dot_t_data) if val < 0), key=lambda x: diff[x])
        min_index_neg_list.append(min_index_neg)
        print('indices', min_index_neg_list)


for min_index_neg, min_index_pos in zip(min_index_neg_list, min_index_pos_list):
    w_interpol = interpolate(u_dot_t_data[min_index_neg], u_t_data[min_index_neg], u_dot_t_data[min_index_pos], u_t_data[min_index_pos])
    vmin, vplus = v_t_data[min_index_neg], v_t_data[min_index_pos]
    print('vmin', vmin)
    wdotmin, wdotplus = u_dot_t_data[min_index_neg], u_dot_t_data[min_index_pos]
    ax.plot([vmin, vplus], [wdotmin, wdotplus], [(vmin+A-TAU*wdotmin)/B, (vplus+A-TAU*wdotplus)/B], '--',color='C4')

nullcline_v_values = np.linspace(-0.672,0.832)
ax.plot(nullcline_v_values, np.zeros(len(nullcline_v_values)), nullcline_wdot(nullcline_v_values), c='C2', alpha=0.7)

ax.set_xlabel(r'$v$')
ax.set_ylabel(r'$\dot{w}$')
ax.set_zlabel(r'$w$')

ax.set_title('c', loc='left')

ax.view_init(elev=29, azim=57)

plt.tight_layout()
plt.subplots_adjust(left=0.09, bottom=0.208, right=0.954, wspace = 0.707)

plt.show()
