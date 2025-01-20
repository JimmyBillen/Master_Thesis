# This python script attempts to uncover the cubic nullcline in 3d by using triangular fit by
# scipy.interpolate.griddata
# It fits the surface within the boundary of the limit cycle, at the cross section with the vdot=0 plane we recover the nullcline.
#     * Results
#         Able to recover nullcline with sufficient accuracy
#         Fitted nicely, but still very specific (very linear) and not neccesarily generalizable
#         Does not need to iterate: just apply once
# Discussed in Section 3.2 of the thesis.

# The program is executed when the script is run as a standalone program. Without needing additional input


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.animation as animation

from data_generation_exploration.FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from data_generation_exploration.FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from model_building.create_NN_FHN import calculate_derivatives

import sys
sys.path.append('../../Master_Thesis') # needed to import settings
from settings import TAU

"""
This python script tried to fit the cubic nullcline in 3d using 
Using triangular fit by scipy.interpolate.griddata
    Fitted nicely, but still very specific and not neccesarily performing

    Does not need to iterate: just apply once
"""
# load data
time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

# consider one limit cycle only
zoomed = True
if zoomed:
    """ Only looks at one period around LC without transients"""
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    v_t_data = v_t_data[index_1:index_2]
    u_t_data = u_t_data[index_1:index_2]
    v_dot_t_data = v_dot_t_data[index_1:index_2]

# change of variables
v_values = v_t_data
vd_values = v_dot_t_data
y_values = u_t_data

# Step 1: Define the grid
v_min, v_max = min(v_values), max(v_values)
vd_min, vd_max = min(vd_values), max(vd_values)
grid_resolution = 200  # Adjust as needed based on desired resolution
v_grid = np.linspace(v_min, v_max, grid_resolution)
vd_grid = np.linspace(vd_min, vd_max, grid_resolution)

# Step 2: Initialize the solution grid
y_grid = np.zeros((grid_resolution, grid_resolution))

# Interpolate y values onto the grid points of the solution space
y_grid_interpolated = griddata((v_values, vd_values), y_values, (v_grid[None, :], vd_grid[:, None]), method='linear') #(initial guess), full interpolation

# Replace corresponding grid points on y_grid with interpolated values
y_grid[~np.isnan(y_grid_interpolated)] = y_grid_interpolated[~np.isnan(y_grid_interpolated)] #(will interpolate where needed?)

# boundary conditions:
# Find the indices of the closest values in the grid to your given values of v and vd
v_indices = np.abs(v_grid[:, None] - v_values).argmin(axis=0)
vd_indices = np.abs(vd_grid[:, None] - vd_values).argmin(axis=0)

# Set boundary conditions at the specified grid points
y_grid[vd_indices, v_indices] = y_values # z_grid[y_val, x_val] because z_grid[row, column], so z_grid[:,0]=1, sets 1 to all the values of the first row


# Plot the solution
"""As we know, the convention of imshow is that the origin is located on the top left corner, x-axis pointing downward and y-axis pointing rightward.8 jun 2016
To avoid this: set origin = lower """
plt.imshow(y_grid, cmap='jet', origin='lower', extent=[v_min, v_max, vd_min, vd_max])
plt.colorbar(label='Solution')
plt.title(f'Heatmap using scipy.interpolation.griddata:\ntau {TAU}, gridresolution {grid_resolution}')
plt.xlabel('v')
plt.ylabel('vdot')
plt.show()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3) Plotting the slice at vdot=0
# Find the index of vd_grid where vd = 0
vd_zero_index = np.abs(vd_grid - 0).argmin()

# Extract the slice where vd = 0 from the solution grid
# y_slice = y_grid[:, vd_zero_index]
y_slice = y_grid[vd_zero_index, :]
v_slice = v_grid

# mse
mse = calculate_mean_squared_error(y_slice, nullcline_vdot(v_slice))
formatted_mse = "{:.2e}".format(mse)

# Create meshgrid for v and vd values
V, VD = np.meshgrid(v_grid, vd_grid)

surf = ax.plot_surface(V, VD, y_grid, cmap='viridis', alpha=0.7)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)       

def update(frame):
    ax.clear()
    # Plot the surface
    surf = ax.plot_surface(V, VD, y_grid, cmap='viridis', alpha=0.7)

    # show limit cycle as well on plot
    ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # the limit cycle

    # Plot y as a function of v
    ax.plot(v_slice, np.zeros(len(v_slice)), y_slice)
    ax.set_xlabel('v')
    ax.set_ylabel('vdot')
    ax.set_zlabel('w')
    ax.set_title(f'tau {TAU} Nullcline Prediction scipy.griddata:\ngridres.{grid_resolution}, mse {formatted_mse}')
    # ax.set_gid(True)
    # Add color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.view_init(elev=30, azim=frame)

ani = animation.FuncAnimation(fig, update, frames=np.arange(30,360+30, 1), interval=30)
plt.show()

writervideo = animation.FFMpegWriter(fps=60) 
name = input("Name for file? ")
if not not name: # enter when giving an input
    ani.save(f'{name}.mp4', writer=writervideo) 


# Plot y as a function of v
plt.plot(v_slice, y_slice, label='predicted nullcline')
plt.plot(v_slice, nullcline_vdot(v_slice), label='real nullcline')
plt.xlabel('v')
plt.ylabel('vdot')
# plt.title(f'tau {TAU} Nullcline Prediction using Relaxation:\ngridres.{grid_resolution}, iter.{iteration},\nmse {formatted_mse}, accuracy {tolerance}')
plt.title(f'tau {TAU} Nullcline Prediction scipy.griddata:\ngridres.{grid_resolution}, mse {formatted_mse}')
plt.grid(True)
plt.show()
