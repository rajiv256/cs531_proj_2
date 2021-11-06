import math
import pulp as pl
from configs import SOLVER_TYPE

def create_affine_expression(coeffs, variables):
    assert len(coeffs) == len(
        variables), f"lengths of coeffs and var_names doesn't match in `create_affine_expression` in pulp_utils:L7"
    n = len(coeffs)
    affine = pl.LpAffineExpression([(variables[i], coeffs[i]) for i in range(n)])
    return affine


def create_constraint(coeffs, variables, sense, name, rhs):
    """Creates a constraint based on the args

    Args:
        coeffs: coefficients of the constraints
        vars: Names of the vars
        sense: +1, 0, -1 based on >=, ==, <= respectively. Or we can use pl.LpConstraintLE
        rhs: numerical value of the rhs

    Returns:

    """
    assert len(coeffs) == len(
        variables), f"lengths of coeffs and var_names doesn't match in `create_constraint` in pulp_utils:L26"
    lhs = create_affine_expression(coeffs, variables)
    constraint = pl.LpConstraint(e=lhs, sense=sense, name=name, rhs=rhs)
    return constraint


def create_variable(name, lower_bound=-math.inf, upper_bound=math.inf):
    var = pl.LpVariable(name=name, lowBound=lower_bound, upBound=upper_bound)
    return var


def optimize_lp(c, A_ub, b_ub, objective=pl.LpMaximize, solver=None):
    """
    optimizes the LP based on the LP objective.
    Args:
        c:
        A_ub:
        b_ub:
        objective:

    Returns:
        obj_value: Value of the objective function
        values: Value of the individual variable

    """
    model = pl.LpProblem("OptimizeModel", objective)

    var_names = ['x_' + str(i) for i in range(len(c))]
    variables = [pl.LpVariable(name=var_names[i]) for i in range(len(var_names))]
    constraints = [create_constraint(A_ub[i], variables, sense=pl.LpConstraintLE, name='co_' + str(i), rhs=b_ub[i]) for
                   i in range(len(A_ub))]
    for constraint in constraints:
        model.addConstraint(constraint)
    obj = pl.lpDot(c, variables)
    model += obj
    status = model.solve(solver)
    assert status == pl.LpStatusOptimal, "LP not optimized!!"
    values = [pl.value(variable) for variable in variables]
    obj_value = pl.value(obj)
    return obj_value, values

