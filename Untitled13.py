#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from pulp import *

def solve_SIS_problem(n, m, q, A, C, beta):
    # Create the problem variable to contain the problem data
    problem = LpProblem("Short_Integer_Solution", LpMinimize)

    # Variables
    x_plus = LpVariable.dicts("x_plus", (range(n), range(q)), 0, 1, LpBinary)
    x_minus = LpVariable.dicts("x_minus", (range(n), range(q)), 0, 1, LpBinary)
    u = LpVariable.dicts("u", range(n), 0, None, LpInteger)
    v = LpVariable.dicts("v", range(n), 0, None, LpInteger)

    # Objective function
    problem += lpSum(u[j] + v[j] for j in range(n)), "Minimize_u_plus_v"

    # Constraints
    for j in range(n):
        problem += lpSum(x_plus[j][k] + x_minus[j][k] for k in range(q)) == 1, f"One_x_{j}"
        problem += u[j] == lpSum(k * x_plus[j][k] for k in range(q)), f"Define_u_{j}"
        problem += v[j] == lpSum(x_minus[j][k] for k in range(q)), f"Define_v_{j}"
        problem += u[j] + v[j] <= q - 1, f"Upper_bound_q_{j}"
        problem += u[j] + v[j] <= beta, f"Upper_bound_beta_{j}"

    # SIS constraints
    for i in range(m):
        problem += lpSum(A[i][j] * (lpSum(k * x_plus[j][k] for k in range(q)) - lpSum(k * x_minus[j][k] for k in range(q))) for j in range(n)) == C[i] * q, f"SIS_condition_{i}"

    # Non-zero vector constraint
    problem += lpSum(u[j] + v[j] for j in range(n)) >= 1, "Non_zero_vector"

    # Solve the problem
    problem.solve()

    # Results
    solution_status = LpStatus[problem.status]
    solution_values = {
        'z': [value(lpSum(k * x_plus[j][k] - k * x_minus[j][k] for k in range(q))) for j in range(n)],
        'u': [value(u[j]) for j in range(n)],
        'v': [value(v[j]) for j in range(n)]
    }

    return solution_status, solution_values


n = 5
m = 3
q = 13
A = np.random.randint(-5, 5, (m, n))
C = np.random.randint(-10, 10, m)
beta = q - 1
status, values = solve_SIS_problem(n, m, q, A, C, beta)
print("Status:", status)
print("Values:", values)


# In[ ]:





# In[ ]:




