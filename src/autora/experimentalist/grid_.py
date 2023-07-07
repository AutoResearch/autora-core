""""""
from itertools import product
from typing import Sequence, Type

import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.variable import Variable, VariableCollection


@wrap_to_use_state
def experimentalist(variables: VariableCollection, format: Type = pd.DataFrame):
    """

    Args:
        variables:
        format: the output type required

    Returns:

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     variables: VariableCollection = field(default_factory=VariableCollection)
        ...     conditions: pd.DataFrame = field(default_factory=pd.DataFrame,
        ...                                      metadata={"delta": "replace"})

        >>> s = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=[1, 2, 3])
        ... ]))

        >>> experimentalist(s).conditions  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           x
        0  1
        1  2
        2  3

        >>> t = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2", allowed_values=[3, 4]),
        ... ]))
        >>> experimentalist(t).conditions
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4

        >>> u = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]))
        >>> experimentalist(u).conditions
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
    if format is pd.DataFrame:
        iv_names = [v.name for v in variables.independent_variables]
        conditions = format(raw_conditions, columns=iv_names)

    else:
        raise NotImplementedError

    return Result(conditions=conditions)


def grid_pool(ivs: Sequence[Variable]):
    """
    Pipeline function to create an exhaustive pool from discrete values
    using a Cartesian product of sets.
    """
    # Get allowed values for each IV
    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"grid_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)
