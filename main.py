import numpy as np
import scipy as sp
from ortools.linear_solver import pywraplp
import cvxopt
import pulp
import math
from lp_solver import LPSolver

c = np.array([-2, 3, -6, -1])
Aeq = np.array([[2, 1, -2, 1], [1, 2, 3, 4]])
beq = np.array([24, 25])
A = np.array([[1, 2, 4, 0], [-1, 1, -2, 0]])
b = np.array([22, -10])

lp = LPSolver(A, b, Aeq, beq, c)

l = lp.solve_linpro()
print(f'-------linpro {l}')
l1 = lp.solve_cvxopt()
print(f'-------cvxopt {l1}')
l2 = lp.solve_ortools()
print(f'-------solve_ortools {l2}')
l3 = lp.solve_pulp()
print(f'-------solve_pulp {l3}')
