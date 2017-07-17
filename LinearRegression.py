import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math as math

# Display Log Values
isLoggingOn = True

# Total number of data points
count_data_points = 10

# Mean value of the noise - set to 0 for adding noise (alternatively can make 1 if you want to multiply to
# generate noise)
mu = 0

# Standard deviation of noise
sigma = 0.2

# Generate noise to introduce for each data point
noise = np.random.normal(mu, sigma, count_data_points)

# Pattern the sample data will follow
target_data_pattern = "np.sin(2*np.pi*x)"
# target_data_pattern = "4*x**3-2*x**2"

# Order of the regression (x, x**2, x**3, etc).
# Create the range of orders that should be tested
order_range = np.linspace(1, 9, 9)

# Array to store a RegressionSolution class for each degree order in order_range
arr_regression_solutions = []

# Plotting Variables
# Range of the data on the x-axis
x_axis_range = 1


# Class to hold the contents of the regression solution
class RegressionSolution:
    def __init__(self, regression_expression, order_val, arr_weights, train_rms_val, test_rms_val):
        self.regression_expression = regression_expression
        self.order = order_val
        self.weights = arr_weights
        self.training_root_mean_square = train_rms_val
        self.testing_root_mean_square = test_rms_val


# Generate random sample data points given some pre-determined pattern function
def generate_data(d_size, d_range, d_pattern):
    t_data = []
    for j in range(0, d_size):
        # Pick random floating point number as the x values
        x = np.random.uniform(0, d_range)
        # Evaluate y values
        y = eval(d_pattern)
        # Apply noise, adding noise because noise distribution is centered around 0 (negative and positive values)
        y = y + noise[j]
        t_data.append([x, y])
    return t_data


# Generate the an array containing the values to be summed in the error function
# Example form: (1/2)*((w1 + w2*x1 - t1)^2 + (w1 + w2*x2 - t2)^2) ...
def generate_error_function(arr_weights, arr_data):
    expression = ""
    for j in range(0, len(arr_weights)):
        if j == 0:
            expression = expression + str(arr_weights[j])
        if j == 1:
            expression = expression + " + " + str(arr_weights[j]) + " * {0}"
        if j > 1:
            expression = expression + " + " + str(arr_weights[j]) + " * {0}**" + str(j)

    arr_inner_error_funcs = []
    for j in range(0, len(arr_data)):
        arr_inner_error_funcs.append(
            "(" + expression.format(arr_data[j][0]) + " - " + str(arr_data[j][1]) + ")**2")

    # Generate the fully expanded error function (expanding out the summation)
    err_func = ""
    for j in range(0, len(arr_inner_error_funcs)):
        if j == 0:
            err_func = err_func + ".5 * (" + arr_inner_error_funcs[j]
        else:
            err_func = err_func + " + " + arr_inner_error_funcs[j]
    err_func = err_func + ")"

    return err_func


# Generate the full regression expression
def generate_regression_expression(arr_values):
    output_expression = ""
    for j in range(0, len(arr_values)):
        if j == 0:
            output_expression = output_expression + str(arr_values[j])
        if j == 1:
            output_expression = output_expression + " + " + str(arr_values[j]) + " * x "
        if j > 1:
            output_expression = output_expression + " + " + str(arr_values[j]) + " * x**" + str(j)
    return output_expression


# Plots the outputs and data from the linear regression process
def plot_solutions(d_range, expression, t_pattern, t_data):
    # Extract the x and y values into their own arrays
    training_x, training_y = zip(*t_data)

    x = np.linspace(0, d_range, num=50)
    y = eval(expression)
    plt.figure(figsize=(20, 10))
    # Plot the regression
    plt.subplot(1, 2, 1)
    plt.plot(x, y, color='r')

    # Plot the training data
    plt.plot(x, eval(t_pattern), color='g')
    plt.scatter(training_x, training_y)
    min_y = min(training_y) - (abs(max(training_y)) * .5)
    max_y = max(training_y) + (max(training_y) * .5)
    plt.ylim([min_y, max_y])

    # Plot the noise
    plt.subplot(1, 2, 2)
    count, bins, ignored = plt.hist(noise, 30, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')

    fig, ax1 = plt.subplots()
    ax1.plot(order_range, arr_train_rms, 'b-')
    ax1.set_xlabel('Degrees of Freedom')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('train', color='b')
    ax1.tick_params('y', colors='b')
    plt.ylim([0, 2])

    ax2 = ax1.twinx()
    ax2.plot(order_range, arr_test_rms, color='r')
    ax2.set_ylabel('test', color='r')
    ax2.tick_params('y', colors='r')
    plt.ylim([0, 2])

    plt.show()


training_data = generate_data(count_data_points, x_axis_range, target_data_pattern)
test_data = generate_data(count_data_points, x_axis_range, target_data_pattern)

for o in range(0, len(order_range)):

    # Declare symbols
    weights = sp.symbols('w0:%d' % (order_range[o] + 1))

    # Create the error function with weight variables and training data values
    error_function = generate_error_function(weights, training_data)

    # Evaluate the derivatives of the error function with respect to both w0 and w1
    derivative_arr_w = []
    for i in range(0, len(weights)):
        derivative_arr_w.append(sp.diff(error_function, weights[i]))
        # print("Derivative of error function with respect to w" + str(i) + ":")
        # print(derivative_arr_w[i])
        # print("")

    # Solve the system of equations for w0 and w1
    system_solution = sp.linsolve(derivative_arr_w, weights)

    # Extract the w0, w1 values from the linear solution set
    arr_weight_values = []
    for i in range(0, len(weights)):
        arr_weight_values.append(list(system_solution)[0][i])

    # Generate the regression expression
    output_regression = generate_regression_expression(arr_weight_values)

    # Evaluate the root-mean-square error
    # Erms=(2*E(w*)/N)**(1/2)
    train_error = eval(generate_error_function(arr_weight_values, training_data))
    train_erms = math.sqrt((2*train_error)/count_data_points)
    test_error = eval(generate_error_function(arr_weight_values, test_data))
    test_erms = math.sqrt((2*test_error)/count_data_points)

    if isLoggingOn:
        print("Values for all w when solving the system of derivatives set to 0:")
        print(system_solution)
        print("")
        print("Equation for the regression:")
        print("y = " + output_regression)
        print("")
        print("Training - Root-mean-square error for M = " + str(order_range[o]) + ":")
        print(train_erms)
        print("")
        print("Testing - Root-mean-square error for M = " + str(order_range[o]) + ":")
        print(test_erms)
        print("")

    arr_regression_solutions.append(
        RegressionSolution(output_regression, order_range[o], weights, train_erms, test_erms))

# TODO: Implement sortable object
best_solution = None
arr_test_rms = []
arr_train_rms = []
for i in range(0, len(arr_regression_solutions)):
    arr_test_rms.append(arr_regression_solutions[i].testing_root_mean_square)
    arr_train_rms.append(arr_regression_solutions[i].training_root_mean_square)
    if best_solution is None:
        best_solution = arr_regression_solutions[i]
    elif arr_regression_solutions[i].testing_root_mean_square < best_solution.testing_root_mean_square:
        best_solution = arr_regression_solutions[i]

print("Best Order: " + str(best_solution.order))
print("Best Root Mean Square: " + str(best_solution.testing_root_mean_square))

# Plot out all of the solutions and data
plot_solutions(x_axis_range, best_solution.regression_expression, target_data_pattern, training_data)
