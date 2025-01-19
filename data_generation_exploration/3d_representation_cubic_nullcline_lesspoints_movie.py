# Three dimensional phase space representation of in- and output of variables for the cubic nullcline
# 1) Limit cycle and vdot = 0 plane
# 2) 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Example data
from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from model_building.create_NN_FHN import calculate_derivatives
import numpy as np
from settings import TAU, R, I, NUM_OF_POINTS

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


zoomed = False
if zoomed:
    """ Only looks at one period around LC without transients"""
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    v_t_data = v_t_data[index_1:index_2]
    u_t_data = u_t_data[index_1:index_2]
    v_dot_t_data = v_dot_t_data[index_1:index_2]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points
# ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
# ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')

# shuffle values
train_val_split = True
if train_val_split:
    num_samples = len(v_dot_t_data)
    num_validation_samples = int(num_samples*0.2)
    indices = np.arange(num_samples).astype(int)
    np.random.shuffle(indices)

    v_t_data = v_t_data[indices]
    v_dot_t_data = v_dot_t_data[indices]
    u_t_data = u_t_data[indices]

    train_vt = v_t_data[num_validation_samples:]
    train_vdot = v_dot_t_data[num_validation_samples:]
    train_ut = u_t_data[num_validation_samples:]
    val_vt = v_t_data[:num_validation_samples]
    val_vdot = v_dot_t_data[:num_validation_samples]
    val_ut = u_t_data[:num_validation_samples]

find_specific_v_value = False #connecting the same v-value, and putting a marker in the vdot=0 plane
if find_specific_v_value:
    min_index_pos_list = []
    min_index_neg_list = []
    values_approx = np.linspace(min(v_t_data), max(v_t_data), 50)
    for value in values_approx:
        diff = [abs(val - value) for val in v_t_data]

        # Find the index of the minimum absolute difference where b is greater than 0
        min_index_pos = min((i for i, val in enumerate(v_dot_t_data) if val > 0), key=lambda x: diff[x])
        min_index_pos_list.append(min_index_pos)
        # Find the index of the minimum absolute difference where b is less than 0
        min_index_neg = min((i for i, val in enumerate(v_dot_t_data) if val < 0), key=lambda x: diff[x])
        min_index_neg_list.append(min_index_neg)

# plane vdot=0:
x = np.linspace(min(v_t_data), max(v_t_data), 2)
z = np.linspace(min(u_t_data), max(u_t_data), 2)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)  # Set y = 0
# Plot the plane


def update1(frame):
    ax.clear()
    # ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # the limit cycle
    # ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', alpha=0.7) # the nullcline
    ax.plot_surface(X, Y, Z, alpha=0.5, color='grey')


    if find_specific_v_value:
        for min_index_neg, min_index_pos in zip(min_index_neg_list, min_index_pos_list):
            # ax.plot([v_t_data[min_index_neg], v_t_data[min_index_pos]], [v_dot_t_data[min_index_neg], v_dot_t_data[min_index_pos]], [u_t_data[min_index_neg], u_t_data[min_index_pos]], color='gold')
            w_interpol = interpolate(v_dot_t_data[min_index_neg], u_t_data[min_index_neg], v_dot_t_data[min_index_pos], u_t_data[min_index_pos])
            ax.scatter(v_t_data[min_index_neg], 0, w_interpol, color='orange')
            vmin, vplus = v_t_data[min_index_neg], v_t_data[min_index_pos]
            vdotmin, vdotplus = v_dot_t_data[min_index_neg], v_dot_t_data[min_index_pos]
            ax.plot([vmin, vplus], [vdotmin, vdotplus], [vmin-(1/3)*(vmin)**3-vdotmin+R*I, vplus-(1/3)*(vplus)**3-vdotplus+R*I], color='blue')
    if train_val_split:
        ax.scatter(train_vt, train_vdot, train_ut, c='r') # the train limit cycle
        ax.scatter(val_vt, val_vdot, val_ut, c='g') # the validation limit cycle
    else:
        ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # the limit cycle

    ax.set_xlabel('v')
    ax.set_ylabel('vdot')
    ax.set_zlabel('w')
    ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}, num points {NUM_OF_POINTS}')
    ax.view_init(elev=30, azim=frame)

# Set labels and title
print("Start animation")
ani = animation.FuncAnimation(fig, update1, frames=np.arange(30,360+30, 1), interval=30)


plt.show()

# saving to m4 using ffmpeg writer 
writervideo = animation.FFMpegWriter(fps=60) 
name = input("Name for file? ")
if not not name: # enter when giving an input
    ani.save(f'{name}.mp4', writer=writervideo) 

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
    print(rico)
    y = rico * (interpol_x-x1) + y1
    return y

# Looking what values will be when trying linear method
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

# => Interpolation done, now visualization:

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update1(frame):
    ax.clear()
    # Plot data points
    # ax.scatter(v_t_data, v_dot_t_data, u_t_data, c='r', marker='o')
    # ax.scatter(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='b', marker='o')
    ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # limit cycle
    ax.plot(v_values_linear, np.zeros(len(v_values_linear)), w_values_linear, c='orange', alpha=1) # guess fit
    ax.plot(v_t_data, np.zeros(len(v_t_data)), nullcline_vdot(v_t_data), c='blue', linestyle='dotted' ,alpha=1) # cubic nullcline
    
    ax.set_xlabel('v')
    ax.set_ylabel('vdot')
    ax.set_zlabel('w')
    ax.set_title(f'3D Plot of w vs. v and vdot for tau={TAU}')
    ax.view_init(elev=30, azim=frame)

# Set labels and title
print("Start animation")
ani = animation.FuncAnimation(fig, update1, frames=np.arange(30,360+30, 1), interval=30)

plt.show()

# saving to m4 using ffmpeg writer 
writervideo = animation.FFMpegWriter(fps=60) 
name = input("Name for file? ")
if not not name: # enter when giving an input
    ani.save(f'{name}.mp4', writer=writervideo) 


# 3) Showing (cubic nullcline vs prediction)-mean squared error
fig, ax = plt.subplots(nrows=1, ncols=2)

nullcline_vdot_data = nullcline_vdot(v_t_data)
thinned_nullcline = np.array([nullcline_vdot_data[list(v_t_data).index(value)] for value in v_values_linear])
w_values_linear = np.array(w_values_linear)
ax[0].plot(v_values_linear, w_values_linear, label='linear predicted data', color='orange')
ax[0].plot(v_values_linear, thinned_nullcline, label='thinned real data', color='blue', linestyle='dotted')
ax[0].plot(v_values_linear, thinned_nullcline - w_values_linear, label='Error Prediction', color='grey')
ax[0].set_title('Real nullcline and prediction with error')
ax[0].legend(loc='upper left')

mse = calculate_mean_squared_error(thinned_nullcline, w_values_linear)
formatted_mse = "{:.2e}".format(mse)

print("mse value", formatted_mse)
ax[1].plot(v_values_linear, thinned_nullcline - w_values_linear, label='Error Prediction', color='grey')
ax[1].set_title(f'Error between real and prediction nullcline \n with mse:{formatted_mse}')
ax[1].legend()

plt.suptitle(f'Real and Predicted Nullclines with Error for tau={TAU}')
plt.show()
