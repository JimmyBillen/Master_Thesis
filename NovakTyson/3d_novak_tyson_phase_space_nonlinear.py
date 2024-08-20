import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
from novak_tyson_time import compute_novak_dynamics
import numpy as np
from derivate import calculate_derivatives, find_local_maxima

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
    Linearly interpolates between the given values at xdot=interpol_x

    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    print(rico)
    y = rico * (interpol_x-x1) + y1
    return y

def nullcline_xdot_ifo_y(yval):
    'x(y), nullcline xdot=0'
    # ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 
    return ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 

def nullcline_ydot_ifo_y(yval):
    'x(y), nullcline ydot=0'
    return +(kdy/ksy)*yval + (k2*Et/ksy)* ((yval)/(Km+yval+Ki * yval**2))

def nullcline_xdot_ifo_x(xval):
    'y(x), nullcline xdot=0'
    return  ( ((k1 * S * Kd**p)/ (kdx * xval)) - Kd**p )**(1/p)

def connecting_curve(xval, dotx):
    'connects same values of x using the connecting curve. Here for xdot=0'
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
    index_1 = find_local_maxima(time, x_values)[-2]
    index_2 = find_local_maxima(time, x_values)[-1]

    x_values_short = x_values[index_1:index_2]
    u_t_data_short = y_values[index_1:index_2]
    v_dot_t_data_short = y_dot_t_data[index_1:index_2]

# x_values_again = x_values.copy()
# v_dot_t_data_again = v_dot_t_data.copy()


# 
fig = plt.figure()
# fig.set_figheight(2)
# fig.set_figwidth(6)
fig.set_figheight(7)
fig.set_figwidth(7)


# Create 3D plot
ax = fig.add_subplot(1,1,1, projection='3d')


# Create 3D plot
# ax = fig.add_subplot(1,2,2, projection='3d')

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
mindot_y, maxdot_y = -0.224, 0.05

connect_dotx = np.linspace(mindot_y, maxdot_y, 1000)
print(mindot_y, maxdot_y)
# ax.plot(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx), '--',color='C4')
ax.plot(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx),color='C4')

# ax.scatter(np.ones(len(connect_dotx))*specific_y, connect_dotx, connecting_curve(specific_y, connect_dotx),color='C4', s=1)






def find_closest_v(v, df):
    min_diff = float('inf')  # Start with an infinitely large difference
    closest_v = None         # Placeholder for the closest value
    
    # Iterate over each value in the 'v' column of the DataFrame
    for index, row in df.iterrows():
        current_v = row['x']
        diff = abs(current_v - v)
        
        # Update the closest_v if the current difference is smaller
        if diff < min_diff:
            min_diff = diff
            closest_v = current_v
    
    return closest_v


# def find_closest_v(v, df):
#     closest_v = df.iloc[(df['v']-v).abs().argsort()[0]]['v']
#     return closest_v

import pandas as pd
df = pd.DataFrame({
    'x': x_values[50:2500:5],
    'xdot': x_dot_t_data[50:2500:5],
    'y': y_values[50:2500:5]
})


df_pos = df[df['xdot'] >= 0]
df_neg = df[df['xdot'] < 0]

v_values_linear = []
w_values_linear = []

# Iterate over each row in df_pos
for index, row in df_pos.iterrows():
    v = row['x']
    vdot = row['xdot']
    w = row['y']
    
    # Find the closest value of 'v' in df_neg
    closest_v_neg = find_closest_v(v, df_neg)
    
    # Retrieve the corresponding 'vdot' and 'w' values from df_neg
    closest_row_neg = df_neg[df_neg['x'] == closest_v_neg]
    vdot_neg = closest_row_neg['xdot'].iloc[0]
    w_neg = closest_row_neg['y'].iloc[0]
    
    # Perform your desired operation with these values
    # if 0.99 < v < 1.01:
        # print(f"For v={v} in df_pos, corresponding vdot={vdot} and w={w}. Closest v in df_neg is {closest_v_neg}, with vdot={vdot_neg} and w={w_neg}")
        # print(interpolate(vdot, w, vdot_neg, w_neg, interpol_x=0))

    v_values_linear.append(v)
    w_at_vdotzero = interpolate(vdot, w, vdot_neg, w_neg, interpol_x=0)
    w_values_linear.append(w_at_vdotzero)
    print(index/3000)


# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')
ax.plot(v_values_linear, np.zeros(len(v_values_linear)), w_values_linear, c='orange', alpha=1)










# ax.set_zlim(0, 4)
ax.set_xlim(0, 0.7)
ax.set_ylim(-0.032, 0.0418)
ax.set_zlim(-0.05,4)

# Set labels and title
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\dot{x}$')
ax.set_zlabel(r'$y$')

ax.view_init(elev=40, azim=60)

plt.show()