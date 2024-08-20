import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from FitzHugh_Nagumo_ps import nullcline_vdot, calculate_mean_squared_error
from create_NN_FHN import calculate_derivatives
from settings import TAU

"""
This python script tried to fit the cubic nullcline in 3d using 
1) Solving Laplace Equation using Relaxation Method (https://physics.stackexchange.com/questions/310447/explanation-of-relaxation-method-for-laplaces-equation)
    * Performance
        Did not perform as wanted

    * Method
        Iterates over the grid changing values
        Re-initialise to boundary conditions

    * Initialization choice
        grid resolution
        y_grid initial condition
        accuracy
        iterations

"""
# inputs
time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

zoomed = True
if zoomed:
    """ Only looks at one period around LC without transients"""
    from FitzHugh_Nagumo_t import find_local_maxima
    index_1 = find_local_maxima(time, v_t_data)[-2]
    index_2 = find_local_maxima(time, v_t_data)[-1]

    v_t_data = v_t_data[index_1:index_2]
    u_t_data = u_t_data[index_1:index_2]
    v_dot_t_data = v_dot_t_data[index_1:index_2]

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
y_grid = np.zeros((grid_resolution, grid_resolution)) # initial condition can be changed as well

init_condition_is_griddata_interpol = True
if init_condition_is_griddata_interpol:
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

# before doing optimization
"""As we know, the convention of imshow is that the origin is located on the top left corner, x-axis pointing downward and y-axis pointing rightward.8 jun 2016"""
plt.imshow(y_grid, cmap='jet', origin='lower', extent=[v_min, v_max, vd_min, vd_max])
plt.colorbar(label='Solution')
plt.title('Initialization of grid: heatmap')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for v and vd values
V, VD = np.meshgrid(v_grid, vd_grid)

# Plot the surface
surf = ax.plot_surface(V, VD, y_grid, cmap='viridis', alpha=0.7)
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # the limit cycle

# Add labels and title
ax.set_xlabel('v')
ax.set_ylabel('vdot')
ax.set_zlabel('y')
plt.title('3D Plot of Initialization')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Show the plot
plt.show()

# Set convergence parameters
max_iterations = 50000  # Maximum number of iterations
tolerance = 5e-5  # Convergence tolerance
# Perform relaxation iterations
for iteration in range(max_iterations):
    print("Iteration:", 100*(iteration)/max_iterations, end="\r")
    max_change = 0.0
    index_place = (-1,-1)
    # y_grid[vd_indices, v_indices] = y_values # Noticed better performance when re-initialising boundary conditions in for loop
    for i in range(1, len(vd_grid) - 1):
        for j in range(1, len(v_grid) - 1):            
            old_y = y_grid[i, j]
            # Update y_grid
            y_grid[i, j] = 0.25 * (y_grid[i+1, j] + y_grid[i-1, j] + y_grid[i, j+1] + y_grid[i, j-1])
            # Calculate change in y value
            y_grid[vd_indices, v_indices] = y_values # re-initialise b.c.
            change = abs(y_grid[i, j] - old_y)
            if change > max_change:
                max_change = change
                index_place = (i,j)
    # Check convergence
    if max_change < tolerance:
        print(f"Convergence achieved after {iteration + 1} iterations.")
        break
    else:
        print(max_change, index_place)
else:
    print("Maximum number of iterations reached without convergence.")

# Plot the solution
plt.imshow(y_grid, cmap='jet', origin='lower', extent=[v_min, v_max, vd_min, vd_max])
plt.colorbar(label='Solution')
plt.title(f'Solution to Laplace Equation with Boundary Conditions\ntau {TAU}, gridres{grid_resolution}, accuracy{tolerance}')
plt.xlabel('v')
plt.ylabel('vdot')
plt.show()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for v and vd values
V, VD = np.meshgrid(v_grid, vd_grid)

# Plot the surface
surf = ax.plot_surface(V, VD, y_grid, cmap='viridis', alpha=0.7)

# Add labels and title
ax.set_xlabel('v')
ax.set_ylabel('vdot')
ax.set_zlabel('w')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


# show limit cycle as well on plot
ax.plot(v_t_data, v_dot_t_data, u_t_data, c='r', alpha=0.7) # the limit cycle

# 3) Plotting the slice at vdot=0
# Assuming y_grid, v_grid, and vd_grid are initialized and defined

# Find the index of vd_grid where vd = 0
vd_zero_index = np.abs(vd_grid - 0).argmin()

# Extract the slice where vd = 0 from the solution grid
# y_slice = y_grid[:, vd_zero_index]
y_slice = y_grid[vd_zero_index, :]
v_slice = v_grid

# mse
mse = calculate_mean_squared_error(y_slice, nullcline_vdot(v_slice))
formatted_mse = "{:.2e}".format(mse)

# Plot y as a function of v
plt.plot(v_slice, np.zeros(len(v_slice)), y_slice)
plt.xlabel('v')
plt.ylabel('vdot')
plt.title(f'tau {TAU} Laplace Relaxation:\ngridres.{grid_resolution}, iter.{iteration},\nmse {formatted_mse}, accuracy {tolerance}')
plt.grid(True)
plt.show()

# Plot y as a function of v
plt.plot(v_slice, y_slice, label='predicted nullcline')
plt.plot(v_slice, nullcline_vdot(v_slice), label='real nullcline')
plt.xlabel('v')
plt.ylabel('vdot')
plt.title(f'tau {TAU} Nullcline Prediction using Relaxation:\ngridres.{grid_resolution}, iter.{iteration},\nmse {formatted_mse}, accuracy {tolerance}')
plt.grid(True)
plt.show()