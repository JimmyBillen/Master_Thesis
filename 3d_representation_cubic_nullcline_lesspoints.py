# In Settings document: set less_points to desired amount
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from create_NN_FHN import calculate_derivatives
import numpy as np
from settings import TAU, NUM_OF_POINTS

time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

time_real, v_t_data_real, u_t_data_real = compute_fitzhugh_nagumo_dynamics(15000)

zoomed = False
if zoomed:
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    time = time[index_1: index_2]
    v_t_data = v_t_data[index_1:index_2]
    u_t_data = u_t_data[index_1:index_2]
    v_dot_t_data = v_dot_t_data[index_1:index_2]
    plt.plot(time, v_t_data)
    plt.show()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)
ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', alpha=0.7)

x = np.linspace(min(v_t_data), max(v_t_data), 2)
z = np.linspace(min(u_t_data), max(u_t_data), 2)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)  # Set y = 0

ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')

# Set labels and title
ax.set_xlabel('v')
ax.set_ylabel('vdot')
ax.set_zlabel('w')
ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}')

plt.show()

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
df_pos_later = df_pos.copy()
df_neg = df[df['vdot'] < 0]

# def find_closest_v(v, df):
#     tussenin3= (df['v']-v).abs()
#     tussenin2 = tussenin3.argsort()
#     print(tussenin2)
#     tussensin = tussenin2[0]
#     closest_v = df.iloc[tussensin]['v']
#     return closest_v

def find_closest_v(v, df):
    if 'v' not in df.columns or df.empty:
        raise ValueError("The DataFrame must contain a 'v' column with at least one value.")
    
    # Calculate the absolute differences
    abs_diff = (df['v'] - v).abs()
    
    # Find the index of the minimum difference
    closest_index = abs_diff.idxmin()
    
    # Return the closest value from the 'v' column
    closest_v = df.at[closest_index, 'v']
    return closest_v


def interpolate(x1,y1,x2,y2, interpol_x=0):
    """
    x corresponds to vdot
    y corresponds to w/u
    """
    rico = (y2-y1)/(x2-x1)
    print(rico)
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


# Looking what values will be when trying linear method (so connecting them)
v_values_linear_neg = []
w_values_linear_neg = []

# Iterate over each row in df_pos
for index, row in df_neg.iterrows():
    v = row['v']
    vdot = row['vdot']
    w = row['w']
    
    # Find the closest value of 'v' in df_neg
    closest_v_pos = find_closest_v(v, df_pos)
    
    # Retrieve the corresponding 'vdot' and 'w' values from df_neg
    closest_row_pos = df_pos[df_pos['v'] == closest_v_pos]
    vdot_pos = closest_row_pos['vdot'].iloc[0]
    w_pos = closest_row_pos['w'].iloc[0]
    
    # Perform your desired operation with these values
    # if 0.99 < v < 1.01:
        # print(f"For v={v} in df_pos, corresponding vdot={vdot} and w={w}. Closest v in df_neg is {closest_v_neg}, with vdot={vdot_neg} and w={w_neg}")
        # print(interpolate(vdot, w, vdot_neg, w_neg, interpol_x=0))

    v_values_linear_neg.append(v)
    w_at_vdotzero = interpolate(vdot, w, vdot_pos, w_pos, interpol_x=0)
    w_values_linear_neg.append(w_at_vdotzero)




zoomed_real = True
if zoomed_real:
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time_real, v_t_data_real)[-2]
    index_2 = find_local_maxima(time_real, v_t_data_real)[-1]


    u_dot_t_data_real = np.array(calculate_derivatives(time_real, u_t_data_real))
    v_dot_t_data_real = np.array(calculate_derivatives(time_real, v_t_data_real))

    time_real = time_real[index_1: index_2]
    v_t_data_real = v_t_data_real[index_1:index_2]
    u_t_data_real = u_t_data_real[index_1:index_2]
    v_dot_t_data_real = v_dot_t_data_real[index_1:index_2]
    plt.plot(time_real, v_t_data_real)
    plt.show()





# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')


def split_segments(x, y, threshold=3):
    # Initializing lists to store segments of x and y
    segments_x = []
    segments_y = []

    # Temporary lists to hold current segment values
    current_segment_x = [x[0]]
    current_segment_y = [y[0]]

    # Iterate through the x and y lists
    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > threshold:
            # If the difference exceeds the threshold, start a new segment
            segments_x.append(current_segment_x)
            segments_y.append(current_segment_y)
            current_segment_x = [x[i]]
            current_segment_y = [y[i]]
        else:
            # Otherwise, continue with the current segment
            current_segment_x.append(x[i])
            current_segment_y.append(y[i])

    # Append the last segment
    segments_x.append(current_segment_x)
    segments_y.append(current_segment_y)

    return segments_x, segments_y

segments_v_linear, segments_w_linear = split_segments(v_values_linear, w_values_linear, threshold=3)
for v_values_linear_split, w_values_linear_split in zip(segments_v_linear, segments_w_linear):
    ax.scatter(v_values_linear_split, np.zeros(len(v_values_linear_split)), w_values_linear_split, c='orange', alpha=1, s=3)

segments_v_linear_opposite, segments_w_linear_opposite = split_segments(v_values_linear_neg, w_values_linear_neg, threshold=3)
for v_values_linear_split_neg, w_values_linear_split_neg in zip(segments_v_linear_opposite, segments_w_linear_opposite):
    ax.scatter(v_values_linear_split_neg, np.zeros(len(v_values_linear_split_neg)), w_values_linear_split_neg, c='orange', alpha=1, s=3)
# ax.plot(v_values_linear, np.zeros(len(v_values_linear)), w_values_linear, c='orange', alpha=1)


ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7)
# ax.plot(v_values_linear, np.zeros(len(v_values_linear)), w_values_linear, c='orange', alpha=1)
ax.plot(v_t_data_real, np.zeros(len(v_t_data_real)), nullcline_vdot(v_t_data_real), c='blue', linestyle='dotted' ,alpha=1)


# Set labels and title
ax.set_xlabel('v')
ax.set_ylabel('vdot')
ax.set_zlabel('w')
ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}, numpoints={NUM_OF_POINTS}')
plt.show()

# calculate MSE between orange and blue
def calculate_mean_squared_error(y1, y2):
    return np.sum(np.square(np.array(y1) - np.array(y2))) / len(y1)

v_values_predict = v_values_linear + v_values_linear_neg
w_values_predict = w_values_linear + w_values_linear_neg
real_w_values_at_v_values_predict = nullcline_vdot(np.array(v_values_predict))
print("Mean squared error is:", calculate_mean_squared_error(w_values_predict, real_w_values_at_v_values_predict))

plt.plot(v_t_data_real, nullcline_vdot(v_t_data_real))
plt.show()
plt.scatter(time_real, v_t_data_real)
plt.show()
'''
# error
nullcline_vdot_data = nullcline_vdot(v_t_data)
thinned_nullcline = np.array([nullcline_vdot_data[list(v_t_data).index(value)] for value in v_values_linear])
w_values_linear = np.array(w_values_linear)
# plt.plot(v_values_linear, thinned_nullcline, label='thinned real data')
# plt.plot(v_values_linear, w_values_linear, label='linear predicted data')
plt.plot(v_values_linear, thinned_nullcline - w_values_linear, label='real minus linear predict')
plt.legend()
plt.show()
mse = calculate_mean_squared_error(thinned_nullcline, w_values_linear)
print(mse, "value of 10**-8!!!!")
'''

'''
# show intersect all
fig = plt.figure() # do this in another way: by calculating the values by interpolation
ax = fig.add_subplot(111, projection='3d')
x = v_t_data.copy()[::6] # 6 is much, especially in the fast parts
y = v_dot_t_data.copy()[::6]
z = u_t_data.copy()[::6]
ax.scatter(x, y, z, c='b', marker='o', label='All Points')
print(len(x), len(v_t_data))
# Connect points where x < 0 to each other
for i in range(len(x)):
    print((i/len(x))*100, end='\r')
    if y[i] < 0:
        for j in range(len(x)):
            if y[j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Points in Phase Space')
# ax.set_ylim(-0.001, 0.001)

plt.show()
'''