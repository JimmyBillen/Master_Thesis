import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from create_NN_FHN import calculate_derivatives
import numpy as np
from settings import TAU, R, I
import matplotlib as mpl

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

# v_t_data_again = v_t_data.copy()
# v_dot_t_data_again = v_dot_t_data.copy()


# 
fig = plt.figure()
fig.set_figheight(2)
fig.set_figwidth(6)


# Create 3D plot
ax = fig.add_subplot(1,3,1, projection='3d')

# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)
# ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', alpha=0.7)

x = np.linspace(min(v_t_data), max(v_t_data), 2)
z = np.linspace(min(u_t_data), max(u_t_data), 2)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)  # Set y = 0
ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

# Set labels and title
ax.set_xlabel(r'$v$', labelpad=-8)
ax.set_ylabel(r'$\dot{v}$', labelpad=-5)
ax.set_zlabel(r'$w$', labelpad=-8)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-3)
ax.tick_params(axis='z', which='major', pad=-3)

# ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}')
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


##############################################
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


# ax.set_xticks([])
# ax.set_yticks([0])
# ax.set_zticks([])

ax.set_xlabel(r'$v$', labelpad=-8)
ax.set_ylabel(r'$\dot{v}$', labelpad=-5)
ax.set_zlabel(r'$w$', labelpad=-8)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-3)
ax.tick_params(axis='z', which='major', pad=-3)


# ax.legend()
ax.set_title('c', loc='left', pad=-10)

ax.view_init(elev=29, azim=57)


plt.tight_layout()
plt.subplots_adjust(top=1.0,
bottom=0.0,
left=0.045,
right=0.989,
hspace=0.24,
wspace=0.177)

mpl.rc("savefig", dpi=300)
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\FHN\3d_phase_space_nullcline_cubic2.png")


plt.show()





















'''



"""
I feel like in 3d it interpolates linearly to where dot = 0
Will try to confirm findings by interpolating linearly
"""

import pandas as pd

df = pd.DataFrame({
    'v': v_t_data,
    'vdot': v_dot_t_data,
    'w': u_t_data
})

df_pos = df[df['vdot'] >= 0]
df_neg = df[df['vdot'] < 0]

def find_closest_v(v, df):
    closest_v = df.iloc[(df['v']-v).abs().argsort()[0]]['v']
    return closest_v

def interpolate(x1,y1,x2,y2, interpol_x=0):
    """
    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    y = rico * (interpol_x-x1) + y1
    return y

# Looking what values will be when trying linear method (so connecting them)
v_values_linear = []
w_values_linear = []


# Iterate over each row in df_pos
for index, row in df_pos.iterrows():
    v = row['v']
    vdot = row['vdot']
    w = row['w']
    
    # Find the closest value of 'v' in df_neg
    closest_v_neg = find_closest_v(v, df_neg)
    
    # Retrieve the corresponding 'vdot' and 'w' values from df_neg
    closest_row_neg = df_neg[df_neg['v'] == closest_v_neg]
    vdot_neg = closest_row_neg['vdot'].iloc[0]
    w_neg = closest_row_neg['w'].iloc[0]
    
    # Perform your desired operation with these values
    # if 0.99 < v < 1.01:
        # print(f"For v={v} in df_pos, corresponding vdot={vdot} and w={w}. Closest v in df_neg is {closest_v_neg}, with vdot={vdot_neg} and w={w_neg}")
        # print(interpolate(vdot, w, vdot_neg, w_neg, interpol_x=0))

    v_values_linear.append(v)
    w_at_vdotzero = interpolate(vdot, w, vdot_neg, w_neg, interpol_x=0)
    w_values_linear.append(w_at_vdotzero)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)
ax.plot(v_values_linear, np.zeros(len(v_values_linear)), w_values_linear, c='orange', alpha=1)
ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='blue', linestyle='dotted' ,alpha=1)

# Set labels and title
ax.set_xlabel('v')
ax.set_ylabel('vdot')
ax.set_zlabel('w')
ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}')
plt.show()
'''