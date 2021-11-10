import math

import pulp as pl
from tqdm import tqdm


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


def optimize_lp(c, A_ub, b_ub, objective=pl.LpMaximize, solver=None, bounds=[]):
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
    print(len(bounds), len(var_names))
    variables = [
        pl.LpVariable(name=var_names[i], lowBound=bounds[i][0], upBound=
        bounds[i][1]) for i in range(len(var_names))]
    # constraints = [create_constraint(A_ub[i], variables, sense=pl.LpConstraintLE, name='co_' + str(i), rhs=b_ub[i]) for i in range(len(A_ub))]
    # for constraint in constraints:
    #     model.addConstraint(constraint)
    # constraints = []
    # xs = []
    # for i in tqdm(range(len(A_ub))):
    #     xs.append((A_ub[i], variables, i, b_ub[i]))
    # with ThreadPoolExecutor() as executor:
    #     for constraint in tqdm(executor.map(lambda x: create_constraint(x[0], x[1], sense=pl.LpConstraintLE, name='cos_'+str(x[2]), rhs=x[3]), xs)):
    #         constraints.append(constraint)
    # for constraint in tqdm(constraints):
    #     model.addConstraint(constraint)
    for i in tqdm(range(len(A_ub))):
        affine = pl.LpAffineExpression([(variables[j], A_ub[i][j]) for j in range(len(variables)) if A_ub[i][j]!=0.0])
        model += affine <= b_ub[i]

    obj = pl.lpDot(c, variables)
    model += obj
    print('Objective added.')
    status = model.solve(solver)
    assert status==pl.LpStatusOptimal, "LP not optimized!!"
    values = [pl.value(variable) for variable in variables]
    obj_value = pl.value(obj)
    print('Solved.')
    return obj_value, values

