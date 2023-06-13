
import numpy as np
import scipy as sp
from ortools.linear_solver import pywraplp
import cvxopt
import os
import pulp
import sys


class LPSolver:

    """
    Linear Programming Problem Solver
    cx->max
    ax<=b
    a_eq=b_eq
    x>=0
    """
    def __init__(self, a, b, a_eq, b_eq, c):
        self.a = a
        self.b = b
        self.a_eq = a_eq
        self.b_eq = b_eq
        self.c = c

    def solve_linpro(self):
        bounds = []
        for i in range(len(self.c)):
            bounds.append((0, None))

        solution = sp.optimize.linprog(
            c=self.c, A_ub=self.a, b_ub=self.b,
            A_eq=self.a_eq, b_eq=self.b_eq,
            bounds=bounds)
        if solution.success:
            return solution.x, solution.fun
        else:
            print('The solver could not find an optimal solution.')

    def solve_ortools(self):
        solution = pywraplp.Solver('Max', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
        x_string = ['x'+str(i) for i in range(1, len(self.c)+1)]
        x = [solution.NumVar(0, solution.infinity(), i) for i in x_string]

        for r in range(len(self.b_eq)):
            solution.Add(sum(self.a_eq[r][u] * x[u]
                             for u in range(len(x))) == self.b_eq[r])

        for r in range(len(self.b)):
            solution.Add(sum(self.a[r][u] * x[u]
                             for u in range(len(x))) <= self.b[r])

        solution.Maximize(sum(-self.c[i] * x[i]
                              for i in range(len(x))))

        if solution.Solve() == pywraplp.Solver.OPTIMAL:
            x_optimal = np.array([])
            for i in x:
                x_optimal = np.append(x_optimal, i.solution_value())
            return x_optimal, -solution.Objective().Value()
        else:
            print('The solver could not find an optimal solution.')

    def solve_cvxopt(self):
        e = - np.eye(len(self.c))
        a = np.concatenate([self.a, self.a_eq])
        b = np.concatenate([self.b, self.b_eq, np.zeros(len(self.c))])
        a = np.concatenate([a, e])
        a = cvxopt.matrix(a.astype(np.double))
        b = cvxopt.matrix(b.astype(np.double))
        c = cvxopt.matrix(self.c.astype(np.double))

        solution = cvxopt.solvers.lp(c, a, b)

        if solution['status'] == 'optimal':
            x_optimal = np.array([])
            for variable in solution['x']:
                x_optimal = np.append(x_optimal, variable)
            return x_optimal, solution['primal objective']
        else:
            print('The solver could not find an optimal solution.')

    def solve_pulp(self):
        solver = pulp.LpProblem(sense=pulp.LpMinimize)
        x = {i: pulp.LpVariable(name=f"x{i}", lowBound=0)
             for i in range(len(self.c))}

        for u in range(len(self.b)):
            solver += pulp.lpSum(self.a[u][r] * x[r]
                                 for r in range(len(x))) <= self.b[u]

        for u in range(len(self.b_eq)):
            solver += pulp.lpSum(self.a_eq[u][r] * x[r]
                                 for r in range(len(x))) == self.b_eq[u]

        solver += (pulp.lpSum(self.c[i] * x[i] for i in range(len(x))))
        #solver.solve()
        #solver.writeLP("lp_problem.lp")
        if solver.solve():
            x_optimal = np.array([])
            for variable in solver.variables():
                if (variable.varValue != None):
                    x_optimal = np.append(x_optimal, variable.varValue)
            return x_optimal, np.sum(self.c * x_optimal)
        else:
            print('The solver could not find an optimal solution.')
