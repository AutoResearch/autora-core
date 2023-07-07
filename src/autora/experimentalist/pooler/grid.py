""""""
from itertools import product
from typing import Sequence, Type

import numpy as np
import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.variable import Variable, VariableCollection


@wrap_to_use_state
def experimentalist(variables: VariableCollection, fmt: Type = pd.DataFrame):
    """

    Args:
        variables:
        fmt: the output type required

    Returns:

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

        ... we get exactly those values back when running the experimentalist:
        >>> experimentalist(s).conditions
           x
        0  1
        1  2
        2  3

        The allowed_values must be specified:
        >>> experimentalist(
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
        >>> experimentalist(t).conditions
           x1  x2
        0   1   3
        1   1   4
        2   2   3
        3   2   4

        If any of the variables have unspecified allowed_values, we get an error:
        >>> experimentalist(S(
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

        The output can be in several formats. The default is pd.DataFrame.
        Alternative: `np.recarray`:
        >>> experimentalist(s, fmt=np.recarray).conditions
        rec.array([(1,), (2,), (3,)],
                  dtype=[('x', '<i...')])

        >>> experimentalist(t, fmt=np.recarray).conditions
        rec.array([(1, 3), (1, 4), (2, 3), (2, 4)],
                  dtype=[('x1', '<i...'), ('x2', '<i...')])

        Alternative: `np.array` (without field names):
        >>> experimentalist(t, fmt=np.array).conditions
        array([[1, 3],
               [1, 4],
               [2, 3],
               [2, 4]])

    """
    raw_conditions = grid_pool(variables.independent_variables)

    iv_names = [v.name for v in variables.independent_variables]
    if fmt is pd.DataFrame:
        conditions = pd.DataFrame(raw_conditions, columns=iv_names)
    elif fmt is np.recarray:
        conditions = np.core.records.fromrecords(
            list(raw_conditions), names=iv_names
        )  # type: ignore
    elif fmt is np.array:
        conditions = np.array(list(raw_conditions))
    else:
        raise NotImplementedError("fmt=%s is not supported" % (fmt))

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
            f"grid_pool requires allowed_values to be set, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)
