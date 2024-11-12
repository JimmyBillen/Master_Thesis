import numpy as np
import matplotlib.pyplot as plt

""" x = MRNA, y = protein (assen gewisseld in de paper)"""

# 
def nullcline_and_boundary(option, amount_of_points):
    # nullclines_per_option = find_boundary_nullclines()
    nullclines_per_option = 0

    if option == 'option_1':
        bound_nullcline = nullclines_per_option['option_1']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        # nullcline = nullcline_wdot(q)
    if option == 'option_2':
        bound_nullcline = nullclines_per_option['option_2']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        # nullcline = nullcline_wdot_inverse(q)
    if option == 'option_3':
        # bound_nullcline = nullclines_per_option['option_3']
        # q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        q = np.linspace(0.128, 0.652, amount_of_points)
        nullcline = nullcline_x_ifo_x(q)
    if option == 'option_4':
        bound_nullcline = nullclines_per_option=['option_4']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        nullcline = np.zeros(len(q))    # just give zero, don't trust MSE values
        print("MSE values of option_4 cannot be trusted")
    return q, nullcline

def calculate_mean_squared_error(real_data: np.ndarray, generated_data: np.ndarray):
    """
    Calculate the Mean Squared Error (MSE) between real_data and generated_data.

    Parameters:
    - real_data: Array of real data.
    - generated_data: Array of generated data.

    Returns:
    - MSE value.
    """
    # boundbox: leftunder [-0.6235, 0.09566] [0.773182, 1.842041]
    print(generated_data)
    print("   generated --- real    ")
    print(real_data)
    if generated_data.shape!=real_data.shape:
        assert ValueError(f'The shapes of {generated_data} and {real_data} are not the same.')

    return np.sum( np.square(generated_data - real_data)) / len(real_data)

# Constants
Kd = 1
Et = 1
k2 = 1
p = 4
k1 = kdx = kdy = 0.05
Km = 0.1 * Kd
Ki = 2 / Kd
S = 1
ksy = 1

def x_dot(x_val, y_val):
    return k1 * S * (Kd**p / (Kd**p + y_val**p)) - x_val * kdx

def y_dot(x_val, y_val):
    return ksy * x_val - kdy * y_val - (k2 * Et) * (y_val) / (Km + y_val + Ki * y_val**2)

def nullcline_x_ifo_y(yval):
    """x(y), nullcline xdot=0"""
    # ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 
    return ((k1*S)/kdx)*((Kd**p)/(Kd**p+yval**p)) 

def nullcline_x_ifo_x(xval):
    return ( (k1*S*Kd**p)/(kdx*xval) -Kd**p )**(1/p)

def nullcline_y_ifo_x(yval):
    """x(y), nullcline ydot=0"""
    return +(kdy/ksy)*yval + (k2*Et/ksy)* ((yval)/(Km+yval+Ki * yval**2))

def rk4_step(f, g, x, y, h):
    k1_x = h * f(x, y)
    k1_y = h * g(x, y)
    
    k2_x = h * f(x + 0.5 * k1_x, y + 0.5 * k1_y)
    k2_y = h * g(x + 0.5 * k1_x, y + 0.5 * k1_y)
    
    k3_x = h * f(x + 0.5 * k2_x, y + 0.5 * k2_y)
    k3_y = h * g(x + 0.5 * k2_x, y + 0.5 * k2_y)
    
    k4_x = h * f(x + k3_x, y + k3_y)
    k4_y = h * g(x + k3_x, y + k3_y)
    
    x_next = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y_next = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
    
    return x_next, y_next

def compute_novak_dynamics_rk4(num_steps=15000):
    x0 = 0.65178
    y0 = 0.65178
    t0 = 0
    t_end = 365.85
    time = np.linspace(t0, t_end, num_steps + 1)
    h = (t_end - t0) / num_steps
    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    x_values[0] = x0
    y_values[0] = y0

    for i in range(1, num_steps + 1):
        x_values[i], y_values[i] = rk4_step(x_dot, y_dot, x_values[i - 1], y_values[i - 1], h)
    
    return time, x_values, y_values

# Create a high-resolution grid of x and y values
x = np.linspace(0, 1, 2000)
y = np.linspace(0, 4, 2000)
X, Y = np.meshgrid(x, y)

# Compute the velocities
U = x_dot(X, Y)
V = y_dot(X, Y)

# Get the trajectory from the numerical solution
time, xvals, yvals = compute_novak_dynamics_rk4()

# Create the phase-space plot with higher resolution

if __name__ == "__main__":

    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V, density=0.8, linewidth=1, arrowsize=1, arrowstyle='->', broken_streamlines=False)
    # limit cycle
    plt.plot(xvals, yvals, label='Trajectory', linewidth=3, color='red')
    # Nullclines    
    yvals = np.linspace(0, 4, 2000)
    plt.plot(nullcline_x_ifo_y(yvals), yvals, label="xdot", linewidth=3, alpha=0.8)
    plt.plot(nullcline_y_ifo_x(yvals), yvals, label='ydot', linewidth=3, alpha=0.8)

    # xvals = np.linspace(0, 1,2000)
    # plt.plot(xvals, nullcline_x_ifo_x(xvals), linewidth=5, color='purple')
    # Plotting
    plt.title('Phase-Space y-x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 1])
    plt.ylim([0, 4])
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the time series
# def plot_timeseries():
#     time, xvals, yvals = compute_novak_dynamics_rk4()
#     plt.figure(figsize=(10, 5))
#     plt.plot(time, xvals, label='x(t)')
#     plt.plot(time, yvals, label='y(t)')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.title('Novak-Tyson Dynamics')
#     plt.grid()
#     plt.show()

# if __name__ == "__main__":
#     plot_timeseries()
