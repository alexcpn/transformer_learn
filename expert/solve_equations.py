# filename: solve_equations.py
import sympy as sp

# Define the variables
a, b, c = sp.symbols('a b c')

# Define the equations
eq1 = sp.Eq(a + b, 5)
eq2 = sp.Eq(2*a - c, 3)
eq3 = sp.Eq(b + 2*c, 10)

# Solve the system of equations
solution = sp.solve((eq1, eq2, eq3), (a, b, c))
print(solution)