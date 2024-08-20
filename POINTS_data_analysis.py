# (just run for figure)
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

num_points_list1 = [600, 800, 1400, 1800, 2200, 2600, 3000, 3400, 3800, 4200, 4600] 
num_points_list2 = [100, 200, 300, 400, 500, 700, 900, 1100, 1200, 1300, 1500, 1600, 1700, 1900, 2000, 2100, 2300, 2400, 2500, 2700, 2800, 2900, 3100, 3200, 3300, 3500, 3600, 3700, 3900, 4000, 4100, 4300, 4400, 4500, 4700, 4800, 4900, 6000, 7000, 8000, 9000, 11000, 12000, 13000, 14000]
num_points_list = num_points_list1+num_points_list2

times = []
val_error_logged = []
pccs = []
mse_mean_relu = []

for num_points in num_points_list:
    input_file_path = f"results_changing_amount_of_points_{num_points}.json"
    with open(input_file_path, "r") as json_file:
        results = json.load(json_file)

    times.append(results["elapsed time"])
    val_error_logged.append(results["validation log10"])
    mse_mean_relu.append(results["MSE mean relu"])
    pccs.append(results["PCC"])


print("de totale tijd van simulatie van ", len(times), "duurde", sum(times), "s of uur:", sum(times)/3600)

points = [1000, 5000, 10000, 15000]
val_error = [10**-2, 10**(-2.35), 10**(-2.2), 10**(-2.2)]
mse_error_mean_relu = [1.13*10**-2, 2.43*10**-3, 1.6*10**-3, 1.42*10**-3]

num_points_list_log = np.log(num_points_list)
points_log = np.log(points)

# +++++++++++ extra +++++++++

plot_separetely = False
if plot_separetely:
    plt.scatter(num_points_list, times)
    plt.title("times") # 7k jumps out due to a break in calculations (closed laptop)
    plt.show()

    plt.scatter(points_log, np.log10(val_error))
    plt.scatter(num_points_list_log, val_error_logged)
    plt.title("val error logged")
    plt.show()

    plt.scatter(points_log, mse_error_mean_relu)
    plt.scatter(num_points_list_log, mse_mean_relu)
    plt.title("mse mean relu")
    plt.show()

    plt.scatter(num_points_list_log, pccs)
    plt.title("pcc")
    plt.show()

plot_next_to_eachother = False
if plot_next_to_eachother:
    fig, ax = plt.subplots(1,4)
    fig.set_figheight(4)
    fig.set_figwidth(12)

    times_minute = np.array(times)/60
    ax[0].scatter(num_points_list, times_minute, color='C1')
    ax[0].set_ylim(0, times_minute[-1]+times_minute[-1]*0.05)
    ax[0].set_xticks([100, 5000, 10000, 14000])
    ax[0].set_title("Training Time 40 Runs (min)") # 7k jumps out due to a break in calculations (closed laptop)

    ax[1].scatter(points, np.log10(val_error), color='C1')
    ax[1].scatter(num_points_list, val_error_logged, color='C1')
    ax[1].set_xscale("log")
    ax[1].set_title(r"$\log10$(Mean Validation Error)")

    ax[2].scatter(points, np.log10(mse_error_mean_relu), color='C1')
    ax[2].scatter(num_points_list, np.log10(mse_mean_relu), color='C1')
    ax[2].set_xscale("log")
    ax[2].set_title(r"$\log10$(Mean MSE ReLU)")

    ax[3].scatter(num_points_list, pccs, color='C1')
    ax[3].set_title("PCC")
    ax[3].set_xscale("log")


    plt.tight_layout()
    plt.show()


# ++++++++++++++++++++ GOOD FIGURE +++++++++++++++++

val_error = 10**np.array(val_error_logged)

fig, ax = plt.subplots(1,2)
fig.set_figheight(3)
fig.set_figwidth(6)

# to get the legend pretty
ax[0].scatter([10000000], [10000000000], color='C1', label='Validation', s=15, edgecolor='None')
ax[0].scatter([10000000], [100000000000], color='C2', label='Nullcline', s=15, edgecolor='None')
ax[0].legend(loc='lower left', framealpha=0.6)

print("lalal", val_error)
ax[0].scatter(num_points_list, val_error, color='C1', label='Validation', alpha=0.5, s=18, edgecolor='None')
ax[0].scatter(num_points_list, mse_mean_relu, color='C2', label='Nullcline', alpha=0.5, s=18, edgecolor='None')
ax[0].set_yscale("log")
ax[0].set_xscale('log')
ax[0].set_ylabel("Error (log scale)")
ax[0].set_xlabel("Number of Points (log scale)")

ax[1].scatter(num_points_list, pccs, color='C0', s=25, alpha=0.5, edgecolor='None')
ax[1].set_ylabel("PCC")
ax[1].set_xlabel("Number of Points (log scale)")
ax[1].set_xscale("log")

from scipy.stats import linregress
""" for MSE"""
print("Nullcline Error")
x = np.array(num_points_list)
y = np.array(mse_mean_relu)

# Segments for interpolation
segment1 = (x >= 10**2) & (x <= 10**3)
segment2 = (x >= 10**3) & (x <= 15000)

# Linear interpolation for segment 1
x1 = x[segment1]
y1 = y[segment1]
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log10(x1), np.log10(y1))

# Linear interpolation for segment 2
x2 = x[segment2]
y2 = y[segment2]
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log10(x2), np.log10(y2))

# Calculate fitted lines
y1_fit = slope1 * np.log10(x1) + intercept1
y2_fit = slope2 * np.log10(x2) + intercept2

# Plotting
ax[0].plot(x1, 10**y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='seagreen')
ax[0].plot(x2, 10**y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='seagreen')

# Print the slope coefficients
print(f"Slope for segment 10^2 - 10^3: {slope1:.2f}")
print(f"Slope for segment 10^3 - 10^4: {slope2:.2f}")

""" for VAL"""
print('Validation Error')
x = np.array(num_points_list)
y = np.array(val_error)

# Segments for interpolation
segment1 = (x >= 10**2) & (x <= 10**3)
segment2 = (x >= 10**3) & (x <= 15000)

# Linear interpolation for segment 1
x1 = x[segment1]
y1 = y[segment1]
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log10(x1), np.log10(y1))

# Linear interpolation for segment 2
x2 = x[segment2]
y2 = y[segment2]
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log10(x2), np.log10(y2))

# Calculate fitted lines
y1_fit = slope1 * np.log10(x1) + intercept1
y2_fit = slope2 * np.log10(x2) + intercept2

# Plotting
# ax[0].plot(x1, y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='m')
# ax[0].plot(x2, y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='m')

ax[0].plot(x1, 10**y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='darkorange')
ax[0].plot(x2, 10**y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='darkorange')

ax[0].set_xlim(78, 18000)
ax[0].set_ylim(10**(-3.3), 10**(-0.92))

# Print the slope coefficients
print(f"Slope for segment 10^2 - 10^3: {slope1:.2f}")
print(f"Slope for segment 10^3 - 10^4: {slope2:.2f}")


plt.tight_layout()
plt.subplots_adjust(top=0.99,
bottom=0.165,
left=0.11,
right=0.99,
hspace=0.19,
wspace=0.266)


mpl.rc("savefig", dpi=300)
plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Time_resolution\error_pcc_fitted_vs_resolution.png')

plt.show()



# old 

# fig, ax = plt.subplots(1,2)
# fig.set_figheight(3)
# fig.set_figwidth(6)

# # to get the legend pretty
# ax[0].scatter([10000000], [10000000000], color='C1', label='Validation', s=15, edgecolor='None')
# ax[0].scatter([10000000], [100000000000], color='C2', label='Nullcline', s=15, edgecolor='None')
# ax[0].legend(loc='lower left', framealpha=0.6)


# ax[0].scatter(num_points_list, val_error_logged, color='C1', label='Validation', alpha=0.5, s=18, edgecolor='None')
# ax[0].scatter(num_points_list, np.log10(mse_mean_relu), color='C2', label='Nullcline', alpha=0.5, s=18, edgecolor='None')
# ax[0].set_xscale('log')
# ax[0].set_ylabel("Error")
# ax[0].set_xlabel("Number of Points (log scale)")

# ax[1].scatter(num_points_list, pccs, color='C0', s=25, alpha=0.5, edgecolor='None')
# ax[1].set_ylabel("PCC")
# ax[1].set_xlabel("Number of Points (log scale)")
# ax[1].set_xscale("log")

# from scipy.stats import linregress
# # +++ for MSE +++
# print("Nullcline Error")
# x = np.array(num_points_list)
# y = np.log10(mse_mean_relu)

# # Segments for interpolation
# segment1 = (x >= 10**2) & (x <= 10**3)
# segment2 = (x >= 10**3) & (x <= 15000)

# # Linear interpolation for segment 1
# x1 = x[segment1]
# y1 = y[segment1]
# slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log10(x1), y1)

# # Linear interpolation for segment 2
# x2 = x[segment2]
# y2 = y[segment2]
# slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log10(x2), y2)

# # Calculate fitted lines
# y1_fit = slope1 * np.log10(x1) + intercept1
# y2_fit = slope2 * np.log10(x2) + intercept2

# # Plotting
# ax[0].plot(x1, y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='seagreen')
# ax[0].plot(x2, y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='seagreen')

# # Print the slope coefficients
# print(f"Slope for segment 10^2 - 10^3: {slope1:.2f}")
# print(f"Slope for segment 10^3 - 10^4: {slope2:.2f}")

# # +++ for VAL +++
# print('Validation Error')
# x = np.array(num_points_list)
# y = np.array(val_error_logged)

# # Segments for interpolation
# segment1 = (x >= 10**2) & (x <= 10**3)
# segment2 = (x >= 10**3) & (x <= 15000)

# # Linear interpolation for segment 1
# x1 = x[segment1]
# y1 = y[segment1]
# slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log10(x1), y1)

# # Linear interpolation for segment 2
# x2 = x[segment2]
# y2 = y[segment2]
# slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log10(x2), y2)

# # Calculate fitted lines
# y1_fit = slope1 * np.log10(x1) + intercept1
# y2_fit = slope2 * np.log10(x2) + intercept2

# # Plotting
# # ax[0].plot(x1, y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='m')
# # ax[0].plot(x2, y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='m')

# ax[0].plot(x1, y1_fit, '-', label=f'Segment 1 fit: slope={slope1:.2f}', color='darkorange')
# ax[0].plot(x2, y2_fit, '-', label=f'Segment 2 fit: slope={slope2:.2f}', color='darkorange')

# ax[0].set_xlim(78, 18000)
# ax[0].set_ylim(-3.3, -0.92)

# # Print the slope coefficients
# print(f"Slope for segment 10^2 - 10^3: {slope1:.2f}")
# print(f"Slope for segment 10^3 - 10^4: {slope2:.2f}")


# plt.tight_layout()
# plt.subplots_adjust(top=0.99,
# bottom=0.165,
# left=0.11,
# right=0.99,
# hspace=0.19,
# wspace=0.266)


# mpl.rc("savefig", dpi=300)
# plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Time_resolution\error_pcc_fitted_vs_resolution.png')

# plt.show()

