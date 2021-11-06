import pulp as pl


def create_affine_expression(coeffs, var_names):
    assert len(coeffs) == len(var_names)
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
    assert len(coeffs) == len(var_names)
    lhs = create_affine_expression(coeffs, var_names)
    constr = pl.LpConstraint(lhs, sense=sense, rhs=rhs)
    return constr


def test_affine_expression(coeffs=[1, 2, 3], var_names=['x_0', 'x_1', 'x_2']):
    print(f'coeffs: {coeffs} | var_names: {var_names}')
    affine = create_affine_expression(coeffs, var_names)
    print(affine)


def test_constraint(coeffs=[1, 2, 3], var_names=['x_0', 'x_1', 'x_2'], sense=pl.LpConstraintLE, rhs=1):
    print(f'coeffs: {coeffs} | var_names: {var_names} | sense: {sense} | rhs: {rhs}')
    constraint = create_constraint(coeffs, var_names, sense, rhs)
    print(constraint)


def test_0(solver_type):
    model = pl.LpProblem("Example", pl.LpMaximize)
    print(solver_type)
    solver = pl.getSolver(solver_type)
    _var = pl.LpVariable('a', 0, 1)
    _var2 = pl.LpVariable('a2', 0, 2)
    model += _var + _var2 <= 3
    model += _var + _var2
    x = _var + _var2
    status = model.solve(solver)
    print(pl.value(_var), pl.value(_var2))
    print(pl.value(x))
    return status


if __name__ == "__main__":
    # Get the available solvers
    av = pl.listSolvers(onlyAvailable=True)
    print(av)

    # Take the first available solver
    status = test_0(av[0])
    print(status)

    test_affine_expression()

    test_constraint()
