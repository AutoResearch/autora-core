"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`"""
from typing import Callable, TypeVar

import pandas as pd
from autora.variable import VariableCollection
from sklearn.base import BaseEstimator

from .executor import wrap_to_use_state
from .state import Delta, State

Executor = Callable[[State], State]

S = TypeVar("S")
X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")


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
