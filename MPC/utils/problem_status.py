import cvxpy as cp

def problem_status(problem):
    '''
    Check the status of the optimization problem and handle accordingly    
    '''

    # Check the solution status and handle accordingly
    if problem.status == cp.OPTIMAL:
        return # Feasible solution found
    elif problem.status == cp.INFEASIBLE:
        # Infeasible problem
        print("The problem is infeasible.")
    elif problem.status == cp.UNBOUNDED:
        # Unbounded problem
        print("The problem is unbounded.")
    elif problem.status == cp.SOLVER_ERROR:
        # Solver error (did not converge)
        print("The solver did not converge.")
    else:
        # Other cases
        print("The problem did not reach an optimal solution.")
    