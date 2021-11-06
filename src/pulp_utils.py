import math

import pulp as pl


def create_affine_expression(coeffs, var_names):
    assert len(coeffs) == len(
        var_names), f"lengths of coeffs and var_names doesn't match in `create_affine_expression` in pulp_utils:L7"
    n = len(coeffs)
    X = [pl.LpVariable(var_names[i]) for i in range(n)]
    affine = pl.LpAffineExpression([(X[i], coeffs[i]) for i in range(n)])
    return affine


def create_constraint(coeffs, var_names, sense, rhs):
    """Creates a constraint based on the args

    Args:
        coeffs: coefficients of the constraints
        vars: Names of the vars
        sense: +1, 0, -1 based on >=, ==, <= respectively. Or we can use pl.LpConstraintLE
        rhs: numerical value of the rhs

    Returns:

    """
    assert len(coeffs) == len(
        var_names), f"lengths of coeffs and var_names doesn't match in `create_constraint` in pulp_utils:L26"
    lhs = create_affine_expression(coeffs, var_names)
    constraint = pl.LpConstraint(lhs, sense=sense, rhs=rhs)
    return constraint


def create_variable(name, lower_bound=-math.inf, upper_bound=math.inf):
    var = pl.LpVariable(name=name, lowBound=lower_bound, upBound=upper_bound)
    return var
