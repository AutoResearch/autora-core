"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`.
so that $n$ processes $f_i$ on states $S$ can be represented as
$$f_n(...(f_1(f_0(S))))$$
"""
from __future__ import annotations

import dataclasses
import inspect
from functools import wraps
from typing import Callable, TypeVar

import pandas as pd
from autora.variable import VariableCollection
from sklearn.base import BaseEstimator

from .state import Delta, State

S = TypeVar("S")
X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")
Executor = Callable[[State], State]


def wrap_to_use_state(f):
    """Decorator to make target `f` into a function on a `State` and `**kwargs`.

    This wrapper makes it easier to pass arguments to a function from a State.

    It was inspired by the pytest "fixtures" mechanism.

    Args:
        f:

    Returns:

    Examples:
        >>> from autora.workflow_ import State, Delta
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        The output must be a `Delta` object.
        >>> from autora.workflow_ import Delta
        >>> @wrap_to_use_state
        ... def experimentalist(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return Delta(conditions=new_conditions)

        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> experimentalist(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @wrap_to_use_state
        ... def theorist(experimental_data: pd.DataFrame, variables: VariableCollection, **kwargs):
        ...     ivs = [v.name for v in variables.independent_variables]
        ...     dvs = [v.name for v in variables.dependent_variables]
        ...     X, y = experimental_data[ivs], experimental_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return Delta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class T(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experimental_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experimental_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> t_prime = theorist(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))

        Arguments from the state can be overridden by passing them in as keyword arguments (kwargs):
        >>> theorist(t, experimental_data=pd.DataFrame({"x": [0,1,2,3], "y": [12,13,14,15]}))\\
        ...     .model.intercept_
        array([12.])

        ... and other arguments supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist(t, fit_intercept=False).model.intercept_
        0.0

        Any parameters not provided by the state must be provided by default values or by the
        caller. If the default is specified:
        >>> @wrap_to_use_state
        ... def experimentalist(conditions, offset=25):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then it need not be passed.
        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[26, 27, 28, 29])

        If a default isn't specified:
        >>> @wrap_to_use_state
        ... def experimentalist(conditions, offset):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then calling the experimentalist without it will throw an error:
        >>> experimentalist(S(conditions=[1,2,3,4]))
        Traceback (most recent call last):
        ...
        TypeError: experimentalist() missing 1 required positional argument: 'offset'

        ... which can be fixed by passing the argument as a keyword to the wrapped function.
        >>> experimentalist(S(conditions=[1,2,3,4]), offset=2)
        S(conditions=[3, 4, 5, 6])

    """
    # Get the set of parameter names from function f's signature
    parameters_ = set(inspect.signature(f).parameters.keys())

    @wraps(f)
    def _f(state_: S, /, **kwargs) -> S:
        # Get the parameters needed which are available from the state_.
        # All others must be provided as kwargs or default values on f.
        assert dataclasses.is_dataclass(state_)
        from_state = parameters_.intersection(
            {f.name for f in dataclasses.fields(state_)}
        )
        arguments_from_state = {k: getattr(state_, k) for k in from_state}
        arguments = dict(arguments_from_state, **kwargs)
        delta = f(**arguments)
        new_state = state_ + delta
        return new_state

    return _f


def theorist_from_estimator(estimator: BaseEstimator) -> Executor:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.
    """

    @wrap_to_use_state
    def theorist(
        experimental_data: pd.DataFrame, variables: VariableCollection, **kwargs
    ):
        ivs = [v.name for v in variables.independent_variables]
        dvs = [v.name for v in variables.dependent_variables]
        X, y = experimental_data[ivs], experimental_data[dvs]
        new_model = estimator.set_params(**kwargs).fit(X, y)
        return Delta(model=new_model)

    return theorist


def experiment_runner_from_x_to_y_function(f: Callable[[X], Y]) -> Executor:
    """Wrapper for experimentalists of the form $f(x) \rarrow y$, where `f` returns just the $y$
    values"""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        y = f(x, **kwargs)
        experimental_data = pd.DataFrame.merge(x, y, left_index=True, right_index=True)
        return Delta(experimental_data=experimental_data)

    return experiment_runner


def experiment_runner_from_x_to_xy_function(f: Callable[[X], XY]) -> Executor:
    """Wrapper for experimentalists of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe."""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experimental_data = f(x, **kwargs)
        return Delta(experimental_data=experimental_data)

    return experiment_runner
