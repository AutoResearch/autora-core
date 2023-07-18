"""Tools to make grids of experimental conditions."""
from itertools import product

import pandas as pd

from autora.state.delta import Result, State, wrap_to_use_state
from autora.variable import VariableCollection


def grid_pool_on_state(s: State, **kwargs) -> State:
    """
    Create an exhaustive pool of conditions.

    Args:
        s: a State object with a `variables` field.

    Returns: a State object updated with the new conditions

    Examples:
        >>> from autora.state.bundled import StandardState
        >>> from autora.variable import Variable, VariableCollection
        >>> s = StandardState(variables=VariableCollection(
        ...     independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2", allowed_values=[3, 4])]))
        >>> grid_pool_on_state(s)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        StandardState(..., conditions=
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4, ...)

    """

    return wrap_to_use_state(grid_pool)(s, **kwargs)


def grid_pool(variables: VariableCollection) -> Result:
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
        back when running the experimentalist:
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
        AssertionError: grid_pool only supports independent variables with discrete...

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
        AssertionError: grid_pool only supports independent variables with discrete...


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
    ivs = variables.independent_variables
    # Get allowed values for each IV
    l_iv_values = []
    l_iv_names = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"grid_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)
        l_iv_names.append(iv.name)

    # Return Cartesian product of all IV values
    pool = product(*l_iv_values)
    conditions = pd.DataFrame(pool, columns=l_iv_names)

    return Result(conditions=conditions)
