"""Tools to make grids of experimental conditions."""
from functools import singledispatch
from itertools import product
from typing import Sequence

import pandas as pd

from autora.state.delta import Result, State, wrap_to_use_state
from autora.variable import Variable, VariableCollection


@singledispatch
def grid_pool(s, **___):
    """
    Function to create a sequence of conditions sampled from a grid of independent variables.

    Depending on the type of the first argument, this will return a different result-type.

    """
    raise NotImplementedError(
        "grid_pool doesn't have an implementation for %s (type=%s)" % (s, type(s))
    )


@grid_pool.register(State)
def grid_pool_on_state(s: State) -> State:
    """

    Args:
        s: a State object with a `variables` field.

    Returns: a State object updated with the new conditions.

    Examples:
        >>> from autora.state.bundled import StandardState
        >>> s = StandardState(variables=VariableCollection(
        ...     independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2", allowed_values=[3, 4])]))
        >>> grid_pool(s)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        StandardState(..., conditions=
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4, ...)

    """

    return wrap_to_use_state(grid_pool_on_variables)(s)


@grid_pool.register(list)
@grid_pool.register(tuple)
def grid_pool_on_ivs(ivs: Sequence[Variable]) -> pd.DataFrame:
    """
    Creates exhaustive pool from discrete values using a Cartesian product of sets

    Args:
        ivs: Variable objects, each of which has an attribute `allowed_values`
            containing a sequence of values.

    Returns: A pd.DataFrame with the exhaustive pool of allowed values



    Examples:
        >>> grid_pool([Variable("x", allowed_values=[1,2])])
           x
        0  1
        1  2

        >>> grid_pool([Variable("x", allowed_values=[1,2]),
        ...            Variable("y", allowed_values=["a","b"])])
           x  y
        0  1  a
        1  1  b
        2  2  a
        3  2  b

        >>> grid_pool([Variable("x", allowed_values=[1,2]),
        ...            Variable("y", allowed_values=["a","b"]),
        ...            Variable("z", allowed_values=[3.0,4.0])])
           x  y    z
        0  1  a  3.0
        1  1  a  4.0
        2  1  b  3.0
        3  1  b  4.0
        4  2  a  3.0
        5  2  a  4.0
        6  2  b  3.0
        7  2  b  4.0


    """
    # Get allowed values for each IV
    l_iv_values = []
    l_iv_names = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"gridsearch_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)
        l_iv_names.append(iv.name)

    # Return Cartesian product of all IV values
    pool = product(*l_iv_values)
    result = pd.DataFrame(pool, columns=l_iv_names)

    return result


@grid_pool.register(VariableCollection)
def grid_pool_on_variables(variables: VariableCollection) -> Result:
    """Creates exhaustive pool of conditions given a definition of variables with allowed_values.

    Args:
        variables: a VariableCollection with `independent_variables` – a sequence of Variable
            objects, each of which has an attribute `allowed_values` containing a sequence of
            values.

    Returns: a Result / Delta object with the conditions as a pd.DataFrame in the `conditions` field

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        With one independent variable "x", and some allowed values, we get exactly those values
        back when running the executor:
        >>> grid_pool(VariableCollection(
        ...     independent_variables=[Variable(name="x", allowed_values=[1, 2, 3])]
        ... ))["conditions"]
           x
        0  1
        1  2
        2  3

        The allowed_values must be specified:
        >>> grid_pool(VariableCollection(independent_variables=[Variable(name="x")]))
        Traceback (most recent call last):
        ...
        AssertionError: gridsearch_pool only supports independent variables with discrete...

        With two independent variables, we get the cartesian product:
        >>> grid_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2", allowed_values=[3, 4]),
        ... ]))["conditions"]
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4

        If any of the variables have unspecified allowed_values, we get an error:
        >>> grid_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ]))
        Traceback (most recent call last):
        ...
        AssertionError: gridsearch_pool only supports independent variables with discrete...


        We can specify arrays of allowed values:
        >>> grid_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]))["conditions"]
                 x  y     z
        0    -10.0  3  20.0
        1    -10.0  3  21.0
        2    -10.0  3  22.0
        3    -10.0  3  23.0
        4    -10.0  3  24.0
        ...    ... ..   ...
        2217  10.0  4  26.0
        2218  10.0  4  27.0
        2219  10.0  4  28.0
        2220  10.0  4  29.0
        2221  10.0  4  30.0
        <BLANKLINE>
        [2222 rows x 3 columns]

    """
    conditions = grid_pool_on_ivs(variables.independent_variables)
    return Result(conditions=conditions)
