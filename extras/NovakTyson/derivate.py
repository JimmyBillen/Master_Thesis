# calculating derivatives using finite difference method
def forward_difference(x_values, y_values, begin=0, end=None):
    if end is None:
        end = len(x_values)-1
    derivatives = [(y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i]) for i in range(begin, end)]
    return derivatives

def centered_difference(x_values, y_values, begin=1, end=None):
    if end is None:
        end=len(x_values)-1
    derivatives = [(y_values[i + 1] - y_values[i - 1]) / (x_values[i + 1] - x_values[i - 1]) for i in range(begin, end)]
    return derivatives

def backward_difference(x_values, y_values, begin=1, end=None):
    if end is None:
        end=len(x_values)
    derivatives = [(y_values[i] - y_values[i - 1]) / (x_values[i] - x_values[i - 1]) for i in range(begin, end)]
    return derivatives

def calculate_derivatives(values, h):
    forward_deriv = forward_difference(values, h, begin=0, end = len(values)-1)
    backward_deriv = backward_difference(values, h, begin=len(values)-1, end=len(values))

    return forward_deriv + backward_deriv

def find_local_maxima(x, y, variable=""):
    local_maxima_index = []
    maxima_x_value = []
    local_maxima_value = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_index.append(i)
            maxima_x_value.append(x[i])
            local_maxima_value.append(y[i])
    print('local maxima happens for t-values:', maxima_x_value)
