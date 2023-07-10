""""""
from itertools import product
from typing import Sequence

import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.variable import Variable, VariableCollection


def grid_pool(ivs: Sequence[Variable]) -> product:
    """
    Low level function to create an exhaustive pool from discrete values
    using a Cartesian product of sets.
    """
    # Get allowed values for each IV
    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"grid_pool requires allowed_values to be set, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)


def grid_pool_from_variables(variables: VariableCollection) -> pd.DataFrame:
    """

    Args:
        variables: the description of all the variables in the AER experiment.

    Returns: a Result / Delta object with the conditions as a pd.DataFrame in the `conditions` field

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        With one independent variable "x", and some allowed values, we get exactly those values
        back when running the executor:
        >>> grid_pool_from_variables(variables=VariableCollection(
        ...     independent_variables=[Variable(name="x", allowed_values=[1, 2, 3])]
        ... ))
        {'conditions':    x
        0  1
        1  2
        2  3}

        The allowed_values must be specified:
        >>> grid_pool_from_variables(
        ...     variables=VariableCollection(independent_variables=[Variable(name="x")]))
        Traceback (most recent call last):
        ...
        AssertionError: grid_pool requires allowed_values to be set...

        With two independent variables, we get the cartesian product:
        >>> grid_pool_from_variables(variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2", allowed_values=[3, 4]),
        ... ]))["conditions"]
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4

        If any of the variables have unspecified allowed_values, we get an error:
        >>> grid_pool_from_variables(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ]))
        Traceback (most recent call last):
        ...
        AssertionError: grid_pool requires allowed_values to be set...


        We can specify arrays of allowed values:
        >>> grid_pool_from_variables(
        ...     variables=VariableCollection(independent_variables=[
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
    raw_conditions = grid_pool(variables.independent_variables)
    iv_names = [v.name for v in variables.independent_variables]
    conditions = pd.DataFrame(raw_conditions, columns=iv_names)
    return Result(conditions=conditions)


grid_pool_executor = wrap_to_use_state(grid_pool_from_variables)
grid_pool_executor.__doc__ = """

Args:
    state: a [autora.state.delta.State][] with a `variables` field
    kwargs: ignored

Returns: the [autora.state.delta.State][] with an updated `conditions` field.

Examples:
    >>> from autora.state.delta import State
    >>> from autora.variable import VariableCollection, Variable
    >>> from dataclasses import dataclass, field
    >>> import pandas as pd
    >>> import numpy as np

    We define a state object with the fields we need:
    >>> @dataclass(frozen=True)
    ... class S(State):
    ...     variables: VariableCollection = field(default_factory=VariableCollection)
    ...     conditions: pd.DataFrame = field(default_factory=pd.DataFrame,
    ...                                      metadata={"delta": "replace"})

    With one independent variable "x", and some allowed values:
    >>> s = S(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x", allowed_values=[1, 2, 3])
    ... ]))

    ... we get exactly those values back when running the executor:
    >>> grid_pool_executor(s).conditions
       x
    0  1
    1  2
    2  3

    The allowed_values must be specified:
    >>> grid_pool_executor(
    ...     S(variables=VariableCollection(independent_variables=[Variable(name="x")])))
    Traceback (most recent call last):
    ...
    AssertionError: grid_pool requires allowed_values to be set...

    With two independent variables, we get the cartesian product:
    >>> t = S(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x1", allowed_values=[1, 2]),
    ...         Variable(name="x2", allowed_values=[3, 4]),
    ... ]))
    >>> grid_pool_executor(t).conditions
       x1  x2
    0   1   3
    1   1   4
    2   2   3
    3   2   4

    If any of the variables have unspecified allowed_values, we get an error:
    >>> grid_pool_executor(S(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x1", allowed_values=[1, 2]),
    ...         Variable(name="x2"),
    ... ])))
    Traceback (most recent call last):
    ...
    AssertionError: grid_pool requires allowed_values to be set...


    We can specify arrays of allowed values:
    >>> u = S(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
    ...         Variable(name="y", allowed_values=[3, 4]),
    ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
    ... ]))
    >>> grid_pool_executor(u).conditions
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

    If you require a different type than the pd.DataFrame, then you can instruct the State object
    to convert it (if you have a constructor for the desired type which is compatible with the
    DataFrame):

    We define a state object with the fields we need:
    >>> from typing import Optional
    >>> @dataclass(frozen=True)
    ... class T(State):
    ...     variables: VariableCollection = field(default_factory=VariableCollection)
    ...     conditions: Optional[np.array] = field(default=None,
    ...                                      metadata={"delta": "replace", "converter": np.asarray})

    >>> t = T(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x", allowed_values=[1, 2, 3])
    ... ]))

    The returned DataFrame is converted into the array format:
    >>> grid_pool_executor(t).conditions
    array([[1],
           [2],
           [3]]...)

    This also works for multiple variables:
    >>> t = T(
    ... variables=VariableCollection(independent_variables=[
    ...     Variable(name="x1", allowed_values=[1, 2]),
    ...     Variable(name="x2", allowed_values=[3, 4]),
    ... ]))

    >>> grid_pool_executor(t).conditions
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]]...)
"""
