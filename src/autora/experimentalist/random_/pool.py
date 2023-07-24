from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from autora.state.delta import Result, State, wrap_to_use_state
from autora.variable import ValueType, Variable, VariableCollection


def _state(
    s: State,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
    **kwargs,
) -> State:
    """
    Create a sequence of conditions randomly sampled from independent variables.

    Args:
        s: a State object with the desired fields
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a State object updated with the new conditions

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

        With one independent variable "x", and some allowed_values:
        >>> s = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=range(10))
        ... ]))

        ... we get some of those values back when running the experimentalist:
        >>> _state(s, random_state=1).conditions
           x
        0  4
        1  5
        2  7
        3  9
        4  0

        With one independent variable "x", and a value_range:
        >>> t = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", value_range=(-5, 5))
        ... ]))

        ... we get a sample of the range back when running the experimentalist:
        >>> _state(t, random_state=1).conditions
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> _state(
        ...     S(variables=VariableCollection(independent_variables=[Variable(name="x")])))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...

        With two independent variables, we get independent samples on both axes:
        >>> t = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=range(1, 5)),
        ...         Variable(name="x2", allowed_values=range(1, 500)),
        ... ]))
        >>> _state(t, num_samples=10, replace=True, random_state=1).conditions
           x1   x2
        0   2  434
        1   3  212
        2   4  137
        3   4  414
        4   1  129
        5   1  205
        6   4  322
        7   4  275
        8   1   43
        9   2   14

        If any of the variables have unspecified allowed_values, we get an error:
        >>> _state(S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ])))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...


        We can specify arrays of allowed values:
        >>> u = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]))
        >>> _state(u, random_state=1).conditions
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0
    """
    return wrap_to_use_state(_result)(
        s, num_samples=num_samples, random_state=random_state, replace=replace, **kwargs
    )


def _result(
    variables: VariableCollection,
    num_samples: int,
    random_state: Optional[int],
    replace: bool,
):
    """
    Create a sequence of conditions randomly sampled from independent variables.

    Args:
        variables: the description of all the variables in the AER experiment.
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a Result / Delta object with the conditions as a pd.DataFrame in the `conditions` field

    """
    conditions = _base(
        variables=variables,
        num_samples=num_samples,
        random_state=random_state,
        replace=replace,
    )
    return Result(conditions=conditions)


def _base(
    variables: VariableCollection,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    """
    Create a sequence of conditions randomly sampled from independent variables.

    Args:
        variables: the description of all the variables in the AER experiment.
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: the generated conditions as a dataframe

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        With one independent variable "x", and some allowed_values we get some of those values
        back when running the experimentalist:
        >>> _base(
        ...     VariableCollection(
        ...         independent_variables=[Variable(name="x", allowed_values=range(10))
        ... ]), random_state=1)
           x
        0  4
        1  5
        2  7
        3  9
        4  0


        ... with one independent variable "x", and a value_range,
        we get a sample of the range back when running the experimentalist:
        >>> _base(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", value_range=(-5, 5))
        ... ]), random_state=1)
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> _base(VariableCollection(independent_variables=[Variable(name="x")]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...

        With two independent variables, we get independent samples on both axes:
        >>> _base(VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=range(1, 5)),
        ...         Variable(name="x2", allowed_values=range(1, 500)),
        ... ]), num_samples=10, replace=True, random_state=1)
           x1   x2
        0   2  434
        1   3  212
        2   4  137
        3   4  414
        4   1  129
        5   1  205
        6   4  322
        7   4  275
        8   1   43
        9   2   14

        If any of the variables have unspecified allowed_values, we get an error:
        >>> _base(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...


        We can specify arrays of allowed values:

        >>> _base(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]), random_state=1)
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0


    """
    rng = np.random.default_rng(random_state)

    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(
                iv.allowed_values, size=num_samples, replace=replace
            )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_samples)

        else:
            raise ValueError(
                "allowed_values or [value_range and type==REAL] needs to be set for "
                "%s" % (iv)
            )

    return pd.DataFrame(raw_conditions)


# Option 2:
on_state = _state
to_result = _result
raw = _base


# Option 3:
run = _state
run_on_state = _state
