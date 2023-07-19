"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`
so that $n$ processes $f_i$ on states $S$ can be represented as
$$f_n(...(f_1(f_0(S))))$$

These are special cases of the [autora.state.delta.wrap_to_use_state][] function.
"""
from __future__ import annotations

from typing import Callable, Iterable, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autora.experimentalist.pipeline import Pipeline
from autora.state.delta import Delta, State, wrap_to_use_state
from autora.variable import VariableCollection

S = TypeVar("S")
X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")
Executor = Callable[[State], State]


def state_fn_from_estimator(estimator: BaseEstimator) -> Executor:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.

    Examples:
        Initialize a function which operates on the state, `state_fn` and runs a LinearRegression.
        >>> from sklearn.linear_model import LinearRegression
        >>> state_fn = state_fn_from_estimator(LinearRegression())

        Define the state on which to operate (here an instance of the `StandardState`):
        >>> from autora.state.bundled import StandardState
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

    @wrap_to_use_state
    def theorist(
        experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs
    ):
        ivs = [v.name for v in variables.independent_variables]
        dvs = [v.name for v in variables.dependent_variables]
        X, y = experiment_data[ivs], experiment_data[dvs]
        new_model = estimator.set_params(**kwargs).fit(X, y)
        return Delta(model=new_model)

    return theorist


def state_fn_from_x_to_y_fn_df(f: Callable[[X], Y]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow y$, where `f` returns just the $y$
    values, with inputs and outputs as a DataFrame or Series with correct column names.

    Examples:
        The conditions are some x-values in a StandardState object:
        >>> from autora.state.bundled import StandardState
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

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        y = f(x, **kwargs)
        experiment_data = pd.DataFrame.merge(x, y, left_index=True, right_index=True)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def state_fn_from_x_to_xy_fn_df(f: Callable[[X], XY]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe.

    Examples:
        The conditions are some x-values in a StandardState object:
        >>> from autora.state.bundled import StandardState
        >>> s = StandardState(conditions=pd.DataFrame({"x": [1, 2, 3]}))

        The function can be defined on a DataFrame, allowing the explicit inclusion of
        metadata like column names.
        >>> def x_to_xy_fn_df(c: pd.DataFrame) -> pd.Series:
        ...     result = c.assign(y=lambda df: 2 * df.x + 1)
        ...     return result

        We apply the wrapped function to `s` and look at the returned experiment_data:
        >>> state_fn_from_x_to_xy_fn_df(x_to_xy_fn_df)(s).experiment_data
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

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experiment_data = f(x, **kwargs)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def state_fn_from_pipeline(pipeline: Pipeline) -> Executor:
    """Wrapper for experimentalists of the form $f() \rarrow x$, where `f`
    returns both $x$ and $y$ values in a complete dataframe."""

    @wrap_to_use_state
    def experimentalist(params):
        conditions = pipeline(**params)
        if isinstance(conditions, (pd.DataFrame, np.ndarray, np.recarray)):
            conditions_ = conditions
        elif isinstance(conditions, Iterable):
            conditions_ = np.array(list(conditions))
        else:
            raise NotImplementedError("type `%s` is not supported" % (type(conditions)))
        return Delta(conditions=conditions_)

    return experimentalist
