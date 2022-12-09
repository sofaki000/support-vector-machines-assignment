from scipy.optimize import minimize_scalar

def objective_function(x):
    return 3 * x ** 4 - 2 * x + 1


res = minimize_scalar(objective_function)

print(res)