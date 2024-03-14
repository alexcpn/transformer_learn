import numpy as np

# Define a simple quadratic function f(x) = x^T A x + b^T x + c
# where A is a positive definite matrix (which makes the function convex),
# b is a vector, and c is a scalar.
A = np.array([[2, -1], [-1, 2]], dtype=float)
b = np.array([-3, -4], dtype=float)
c = 0

# Compute the gradient (first derivatives)
def gradient(x):
    return A @ x + b

# Compute the Hessian (second derivatives, which is just A in this simple quadratic case)
Hessian = A

# The inverse Hessian is used in Newton's method to find the minimum of the function.
# For our simple function, the Hessian is constant, so we can invert it directly.
inverse_Hessian = np.linalg.inv(Hessian)

# Use Newton's method to find the minimum of the function.
# Start with an initial guess for x.
x_old = np.array([0, 0], dtype=float)

# Newton's update rule: x_new = x_old - H^-1 * gradient
x_new = x_old - inverse_Hessian @ gradient(x_old)

print(x_old, x_new)
