import GPy
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return -np.sin(3*x) - x**2 + 0.7*x 

# Define the bounds of the search space
bounds = np.array([[0, 2]])

# Initialize samples
X_sample = np.random.uniform(0, 2, (5, 1))
Y_sample = objective_function(X_sample)

# Gaussian process with Matern kernel as surrogate model
kernel = GPy.kern.Matern52(input_dim=1)
gp = GPy.models.GPRegression(X_sample, Y_sample, kernel)

for _ in range(25):
    # Update Gaussian process with existing samples
    gp.optimize(messages=False)
    gp.optimize_restarts(num_restarts=10, verbose=False)
    
    # Thompson Sampling: Sample a function from the GP and find its maximum
    y_samples = gp.posterior_samples_f(bounds, size=1)
    x_next = bounds[np.argmax(y_samples)]
    
    # Obtain next objective function value
    y_next = objective_function(x_next)
    
    # Add new sample to existing samples
    X_sample = np.vstack((X_sample, x_next))
    Y_sample = np.vstack((Y_sample, y_next))
    gp.set_XY(X_sample, Y_sample)

# Plot the objective function and samples
plt.figure()
plt.plot(np.linspace(0, 2, 400), objective_function(np.linspace(0, 2, 400)), "y--", lw=2, label="Objective Function")
plt.scatter(X_sample, Y_sample, c="r", s=50, zorder=10, label="Samples")
plt.legend()
plt.show()
