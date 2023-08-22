"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`
so that $n$ processes $f_i$ on states $S$ can be represented as
$$f_n(...(f_1(f_0(S))))$$

These are special cases of the [autora.state.delta.on_state][] function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, TypeVar

import pandas as pd
from sklearn.base import BaseEstimator

from autora.state.delta import Delta, State, on_state
from autora.variable import VariableCollection

X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")
StateFunction = Callable[[State], State]


@dataclass(frozen=True)
class StandardState(State):
    """
    Examples:
        The state can be initialized emtpy
        >>> from autora.state.delta import Delta
        >>> from autora.variable import VariableCollection, Variable
        >>> s = StandardState()
        >>> s
        StandardState(variables=None, conditions=None, experiment_data=None, models=[])

        The `variables` can be updated using a `Delta`:
        >>> dv1 = Delta(variables=VariableCollection(independent_variables=[Variable("1")]))
        >>> s + dv1
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='1',...)

        ... and are replaced by each `Delta`:
        >>> dv2 = Delta(variables=VariableCollection(independent_variables=[Variable("2")]))
        >>> s + dv1 + dv2
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='2',...)

        The `conditions` can be updated using a `Delta`:
        >>> dc1 = Delta(conditions=pd.DataFrame({"x": [1, 2, 3]}))
        >>> (s + dc1).conditions
           x
        0  1
        1  2
        2  3

        ... and are replaced by each `Delta`:
        >>> dc2 = Delta(conditions=pd.DataFrame({"x": [4, 5]}))
        >>> (s + dc1 + dc2).conditions
           x
        0  4
        1  5

        Datatypes other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> dc3 = Delta(conditions=np.core.records.fromrecords([(8, "h"), (9, "i")], names="n,c"))
        >>> (s + dc3).conditions
           n  c
        0  8  h
        1  9  i

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        Column names should usually be provided.
        >>> dc4 = Delta(conditions=[(6,), (7,)])
        >>> (s + dc4).conditions
           0
        0  6
        1  7

        Datatypes which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(conditions="not compatible with pd.DataFrame")
        Traceback (most recent call last):
        ...
        ValueError: ...

        Experiment data can be updated using a Delta:
        >>> ded1 = Delta(experiment_data=pd.DataFrame({"x": [1,2,3], "y": ["a", "b", "c"]}))
        >>> (s + ded1).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c

        ... and are extended with each Delta:
        >>> ded2 = Delta(experiment_data=pd.DataFrame({"x": [4, 5, 6], "y": ["d", "e", "f"]}))
        >>> (s + ded1 + ded2).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        4  5  e
        5  6  f

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        >>> ded3 = Delta(experiment_data=pd.DataFrame([(7, "g"), (8, "h")]))
        >>> (s + ded3).experiment_data
           0  1
        0  7  g
        1  8  h

        If there are already data present, the column names must match.
        >>> (s + ded2 + ded3).experiment_data
             x    y    0    1
        0  4.0    d  NaN  NaN
        1  5.0    e  NaN  NaN
        2  6.0    f  NaN  NaN
        3  NaN  NaN  7.0    g
        4  NaN  NaN  8.0    h

        `experiment_data` other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> ded4 = Delta(
        ...     experiment_data=np.core.records.fromrecords([(1, "a"), (2, "b")], names=["x", "y"]))
        >>> (s + ded4).experiment_data
           x  y
        0  1  a
        1  2  b

        `experiment_data` which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(experiment_data="not compatible with pd.DataFrame")
        Traceback (most recent call last):
        ...
        ValueError: ...

        `models` can be updated using a Delta:
        >>> from sklearn.dummy import DummyClassifier
        >>> dm1 = Delta(models=[DummyClassifier(constant=1)])
        >>> dm2 = Delta(models=[DummyClassifier(constant=2), DummyClassifier(constant=3)])
        >>> (s + dm1).models
        [DummyClassifier(constant=1)]

        >>> (s + dm1 + dm2).models
        [DummyClassifier(constant=1), DummyClassifier(constant=2), DummyClassifier(constant=3)]

        The last model is available under the `model` property:
        >>> (s + dm1 + dm2).model
        DummyClassifier(constant=3)

        If there is no model, `None` is returned:
        >>> print(s.model)
        None

        `models` can also be updated using a Delta with a single `model`:
        >>> dm3 = Delta(model=DummyClassifier(constant=4))
        >>> (s + dm1 + dm3).model
        DummyClassifier(constant=4)

        As before, the `models` list is extended:
        >>> (s + dm1 + dm3).models
        [DummyClassifier(constant=1), DummyClassifier(constant=4)]

        No coercion or validation occurs with `models` or `model`:
        >>> (s + dm1 + Delta(model="not a model")).models
        [DummyClassifier(constant=1), 'not a model']


    """

    variables: Optional[VariableCollection] = field(
        default=None, metadata={"delta": "replace"}
    )
    conditions: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "extend", "converter": pd.DataFrame}
    )
    models: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend", "aliases": {"model": lambda model: [model]}},
    )

    @property
    def model(self):
        """Alias for the last model in the `models`."""
        try:
            return self.models[-1]
        except IndexError:
            return None


def state_fn_from_estimator(estimator: BaseEstimator) -> StateFunction:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.

    Examples:
        Initialize a function which operates on the state, `state_fn` and runs a LinearRegression.
        >>> from sklearn.linear_model import LinearRegression
        >>> state_fn = state_fn_from_estimator(LinearRegression())

        Define the state on which to operate (here an instance of the `StandardState`):
        >>> from autora.state.standard import StandardState
        >>> from autora.variable import Variable, VariableCollection
        >>> import pandas as pd
        >>> s = StandardState(
        ...     variables=VariableCollection(
        ...         independent_variables=[Variable("x")],
        ...         dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [1,2,3], "y":[3,6,9]})
        ... )

        Run the function, which fits the model and adds the result to the `StandardState`
        >>> state_fn(s).model.coef_
        array([[3.]])

    """

    @on_state()
    def theorist(
        experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs
    ):
        ivs = [v.name for v in variables.independent_variables]
        dvs = [v.name for v in variables.dependent_variables]
        X, y = experiment_data[ivs], experiment_data[dvs]
        new_model = estimator.set_params(**kwargs).fit(X, y)
        return Delta(model=new_model)

    return theorist


def state_fn_from_x_to_y_fn_df(f: Callable[[X], Y]) -> StateFunction:
    """Wrapper for experiment_runner of the form $f(x) \rarrow y$, where `f` returns just the $y$
    values, with inputs and outputs as a DataFrame or Series with correct column names.

    Examples:
        The conditions are some x-values in a StandardState object:
        >>> from autora.state.standard import StandardState
        >>> s = StandardState(conditions=pd.DataFrame({"x": [1, 2, 3]}))

        The function can be defined on a DataFrame (allowing the explicit inclusion of
        metadata like column names).
        >>> def x_to_y_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = pd.Series(2 * c["x"] + 1, name="y")
        ...     return result

        We apply the wrapped function to `s` and look at the returned experiment_data:
        >>> state_fn_from_x_to_y_fn_df(x_to_y_fn)(s).experiment_data
           x  y
        0  1  3
        1  2  5
        2  3  7

        We can also define functions of several variables:
        >>> def xs_to_y_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = pd.Series(c["x0"] + c["x1"], name="y")
        ...     return result

        With the relevant variables as conditions:
        >>> t = StandardState(conditions=pd.DataFrame({"x0": [1, 2, 3], "x1": [10, 20, 30]}))
        >>> state_fn_from_x_to_y_fn_df(xs_to_y_fn)(t).experiment_data
           x0  x1   y
        0   1  10  11
        1   2  20  22
        2   3  30  33
    """

    @on_state()
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        y = f(x, **kwargs)
        experiment_data = pd.DataFrame.merge(x, y, left_index=True, right_index=True)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def state_fn_from_x_to_xy_fn_df(f: Callable[[X], XY]) -> StateFunction:
    """Wrapper for experiment_runner of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe.

    Examples:
        The conditions are some x-values in a StandardState object:
        >>> from autora.state.standard import StandardState
        >>> s = StandardState(conditions=pd.DataFrame({"x": [1, 2, 3]}))

        The function can be defined on a DataFrame, allowing the explicit inclusion of
        metadata like column names.
        >>> def x_to_xy_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = c.assign(y=lambda df: 2 * df.x + 1)
        ...     return result

        We apply the wrapped function to `s` and look at the returned experiment_data:
        >>> state_fn_from_x_to_xy_fn_df(x_to_xy_fn)(s).experiment_data
           x  y
        0  1  3
        1  2  5
        2  3  7

        We can also define functions of several variables:
        >>> def xs_to_xy_fn(c: pd.DataFrame) -> pd.Series:
        ...     result = c.assign(y=c.x0 + c.x1)
        ...     return result

        With the relevant variables as conditions:
        >>> t = StandardState(conditions=pd.DataFrame({"x0": [1, 2, 3], "x1": [10, 20, 30]}))
        >>> state_fn_from_x_to_xy_fn_df(xs_to_xy_fn)(t).experiment_data
           x0  x1   y
        0   1  10  11
        1   2  20  22
        2   3  30  33

    """

    @on_state()
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experiment_data = f(x, **kwargs)
        return Delta(experiment_data=experiment_data)

    return experiment_runner
