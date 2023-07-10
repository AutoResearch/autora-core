"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`.
so that $n$ processes $f_i$ on states $S$ can be represented as
$$f_n(...(f_1(f_0(S))))$$
"""
from __future__ import annotations

import dataclasses
import inspect
from functools import wraps
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


def theorist_from_estimator(estimator: BaseEstimator) -> Executor:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.
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


def experiment_runner_from_x_to_y_function(f: Callable[[X], Y]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow y$, where `f` returns just the $y$
    values"""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        y = f(x, **kwargs)
        experiment_data = pd.DataFrame.merge(x, y, left_index=True, right_index=True)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def experiment_runner_from_x_to_xy_function(f: Callable[[X], XY]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe."""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experiment_data = f(x, **kwargs)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def experimentalist_from_pipeline(pipeline: Pipeline) -> Executor:
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


class WrapperInputError(Exception):
    pass


def wrap_to_use_state_general(
    f: Callable, input_state_mapping: dict, output_field_name: str
):
    """
    General wrapper for a function to use a state.

    Args:
        f: The function to wrap
        input_state_mapping: A dictionary that maps the keyword arguments of the function
            to the state fields
        output_field_name: A string that maps the return value of f to the state field

    Examples:
        >>> from autora.state.delta import State, Delta
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We define an arbitrary function
        >>> def experimentalist(x):
        ...     new_conditions = [c + 10 for c in x]
        ...     return new_conditions

        We can now use the wrapper to get back a function that works on a state. Here, we need to
        map the input arguments and the return value of the function to the fields of the state:
        >>> experimentalist_on_state = wrap_to_use_state_general(
        ...                                                 f=experimentalist,
        ...                                                 input_state_mapping={'x':'conditions'},
        ...                                                 output_field_name='conditions')

        >>> experimentalist_on_state(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> experimentalist_on_state(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        We can use other states with different fields:
        >>> @dataclass(frozen=True)
        ... class T(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experiment_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )

        And define other functions:
        >>> def theorist(x: pd.DataFrame, v: VariableCollection, **kwargs):
        ...     ivs = [v.name for v in v.independent_variables]
        ...     dvs = [v.name for v in v.dependent_variables]
        ...     X, y = x[ivs], x[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return new_model

        Again, we use the wrapper:
        >>> theorist_on_state = wrap_to_use_state_general(
        ...                                         f=theorist,
        ...                                         input_state_mapping={
        ...                                             'x':'experiment_data',
        ...                                             'v':'variables'},
        ...                                         output_field_name='model')

        >>> t_prime = theorist_on_state(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))


        Other arguments are still supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist_on_state(t, fit_intercept=False).model.intercept_
        0.0

        Any parameters not provided by the state can still be provided to the wrapped function
        >>> def experimentalist(x, offset):
        ...     new_conditions = [c + offset for c in x]
        ...     return new_conditions

        >>> experimentalist_on_state_ = wrap_to_use_state_general(
        ...                                                 f=experimentalist,
        ...                                                 input_state_mapping={'x':'conditions'},
        ...                                                 output_field_name='conditions')


        ... then it need to be passed.
        >>> experimentalist_on_state_(S(conditions=[1,2,3,4]), offset=25)
        S(conditions=[26, 27, 28, 29])
    """
    # Create reverse mapping to fetch function argument names from state field names
    reversed_mapping = {v: k for k, v in input_state_mapping.items()}
    parameters_ = set(inspect.signature(f).parameters.keys())

    # Validation checks
    missing_func_params = set(input_state_mapping.keys()).difference(parameters_)
    if missing_func_params:
        raise ValueError(
            f"The following keys in input_state_mapping are not parameters of the function: "
            f"{missing_func_params}"
        )

    @wraps(f)
    def _f(state_: State, /, **kwargs) -> State:
        # Validation checks
        assert dataclasses.is_dataclass(state_)
        state_fields = {field.name for field in dataclasses.fields(state_)}
        missing_state_fields = set(input_state_mapping.values()).difference(
            state_fields
        )
        if missing_state_fields:
            raise WrapperInputError(
                f"The following values in input_state_mapping are not fields of the state: "
                f"{missing_state_fields}. This error likely originates when wrapping the original "
                f"function to use the state."
            )
        if output_field_name not in state_fields:
            raise WrapperInputError(
                f"output_field_name of '{output_field_name}' is not a field of the state. "
                f"This error likely originates when wrapping the original function to use the "
                f"state."
            )

        # Get the parameters needed which are available from the state_
        from_state = {
            reversed_mapping.get(field.name, field.name): getattr(state_, field.name)
            for field in dataclasses.fields(state_)
            if reversed_mapping.get(field.name, field.name) in parameters_
        }

        # Combine the arguments from state and kwargs
        arguments = {**kwargs, **from_state}

        # Call function with arguments
        result = f(**arguments)

        # Map function output to Delta
        delta_args = {
            output_field_name: result
        }  # mapping directly to output_field_name
        delta: Delta = Delta(**delta_args)

        # Return the updated state
        new_state = state_ + delta
        return new_state

    return _f
