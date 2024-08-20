import matplotlib.pyplot as plt

points = [1, 5, 10, 15]
val_error = [10**-2, 10**(-2.35), 10**(-2.2), 10**(-2.2)]
mse_error_mean_relu = [1.13*10**-2, 2.43*10**-3, 1.6*10**-3, 1.42*10**-3]
mse_error_sigmoid = [4.54*10**-4, 9.33*10**-5]

plt.plot(points, val_error)
plt.title("val error")
plt.show()

plt.plot(points, mse_error_mean_relu)
plt.title("mean mse relu")
plt.show()

plt.plot(points[-2:], mse_error_sigmoid)
plt.title("mse sigmoid fit on relu")
plt.show()